"""Latency and energy per stage of sd-turbo (512x512, fp16, batch 1) on one GPU.

Latency: CUDA events around the text encoder, each UNet call and the VAE
decode, for --images images per step count after --warmup images.

Energy: the NVML energy counter changes every 100 ms and under-counts when it
is read faster than about 2 Hz (see nvml-energy-probes), while one image takes
well under 100 ms. So no stage is measured image by image. Each stage is
replayed with the inputs captured from a real image, in a loop that lasts about
--block-seconds, and the counter is read once per block: the reading that ends
one block starts the next one. The whole pipeline is measured the same way, so
the stages can be checked against it.
"""
import argparse
import csv
import json
import os
import time

import diffusers
import numpy as np
import pynvml
import torch
from diffusers import AutoPipelineForText2Image

PROMPT = "a photo of a red fox in the snow, natural light, detailed"
STAGES = ("text_encoder", "unet", "vae")

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--images", type=int, default=200)
parser.add_argument("--blocks", type=int, default=15, help="energy blocks per stage and step count")
parser.add_argument("--block-seconds", type=float, default=3.0)
parser.add_argument("--idle-seconds", type=float, default=30.0)
parser.add_argument("--settle-seconds", type=float, default=15.0, help="wait after loading the model, before idle")
parser.add_argument("--out", default="results")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)


def energy_mj():
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)


def telemetry():
    return {
        "sm_mhz": pynvml.nvmlDeviceGetClockInfo(gpu, pynvml.NVML_CLOCK_SM),
        "mem_mhz": pynvml.nvmlDeviceGetClockInfo(gpu, pynvml.NVML_CLOCK_MEM),
        "temp_c": pynvml.nvmlDeviceGetTemperature(gpu, pynvml.NVML_TEMPERATURE_GPU),
        "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(gpu) / 1000,
        "clock_reasons": hex(pynvml.nvmlDeviceGetCurrentClocksEventReasons(gpu)),
    }


pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.set_progress_bar_config(disable=True)

# Each stage is wrapped to record CUDA events and keep the inputs of its last
# call. `timing` is off during the energy blocks so the wrappers cost nothing.
timing = True
events = []  # [stage, start event, end event] for the current image
captured = {}  # stage -> (args, kwargs) of its last call
originals = {}


def wrap(obj, attr, stage):
    fn = getattr(obj, attr)
    originals[stage] = fn

    def timed(*a, **kw):
        if not timing:
            return fn(*a, **kw)
        captured[stage] = (a, kw)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*a, **kw)
        end.record()
        events.append((stage, start, end))
        return out

    setattr(obj, attr, timed)


wrap(pipe.text_encoder, "forward", "text_encoder")
wrap(pipe.unet, "forward", "unet")
wrap(pipe.vae, "decode", "vae")


def generate(steps, seed, output_type="pt"):
    return pipe(PROMPT, num_inference_steps=steps, guidance_scale=0.0, height=512, width=512,
                generator=torch.Generator("cuda").manual_seed(seed), output_type=output_type)


with torch.inference_mode():
    time.sleep(args.settle_seconds)
    e0, t0 = energy_mj(), time.perf_counter()
    time.sleep(args.idle_seconds)
    e1, t1 = energy_mj(), time.perf_counter()
    idle_w = (e1 - e0) / 1e3 / (t1 - t0)
    run = {"gpu": pynvml.nvmlDeviceGetName(gpu), "driver": pynvml.nvmlSystemGetDriverVersion(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "diffusers": diffusers.__version__,
           "idle_w": idle_w, "idle_seconds": t1 - t0, "idle_telemetry": telemetry(), "args": vars(args)}
    print(f"idle {idle_w:.1f} W over {t1 - t0:.1f} s")

    max_steps = max(args.steps)
    image_rows, block_rows = [], []
    for steps in args.steps:
        for i in range(args.warmup):
            generate(steps, seed=i)
        torch.cuda.synchronize()

        for i in range(args.images):
            events.clear()
            start = time.perf_counter()
            generate(steps, seed=i)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - start) * 1e3
            ms = {stage: [] for stage in STAGES}
            for stage, a, b in events:
                ms[stage].append(a.elapsed_time(b))
            row = {"steps": steps, "image": i, "text_encoder_ms": ms["text_encoder"][0]}
            for k in range(max_steps):
                row[f"unet{k + 1}_ms"] = ms["unet"][k] if k < len(ms["unet"]) else ""
            row.update({"vae_ms": ms["vae"][0], "wall_ms": wall_ms, **telemetry()})
            image_rows.append(row)
        rows = [r for r in image_rows if r["steps"] == steps]
        wall = np.median([r["wall_ms"] for r in rows])
        print(f"{steps} step(s): {wall:.1f} ms per image (median of {len(rows)})")

        # Calls per block, from the median latency of each stage in this run.
        per_call_ms = {
            "text_encoder": np.median([r["text_encoder_ms"] for r in rows]),
            "unet": np.median([r[f"unet{k + 1}_ms"] for r in rows for k in range(steps)]),
            "vae": np.median([r["vae_ms"] for r in rows]),
            "pipeline": wall,
        }
        work = {stage: (lambda fn=originals[stage], c=captured[stage]: fn(*c[0], **c[1])) for stage in STAGES}
        work["pipeline"] = lambda: generate(steps, seed=0)
        calls = {stage: max(1, round(args.block_seconds * 1e3 / per_call_ms[stage])) for stage in work}

        generate(steps, seed=0, output_type="pil").images[0].save(f"{args.out}/sample_{steps}step.png")
        timing = False
        torch.cuda.synchronize()
        e_prev, t_prev = energy_mj(), time.perf_counter()
        # ponytail: stage order is fixed in each round; shuffle it if order effects show up between rounds.
        for b in range(args.blocks):
            for stage, fn in work.items():
                for _ in range(calls[stage]):
                    fn()
                torch.cuda.synchronize()
                e, t = energy_mj(), time.perf_counter()
                joules, seconds, n = (e - e_prev) / 1e3, t - t_prev, calls[stage]
                block_rows.append({
                    "steps": steps, "block": b, "stage": stage, "calls": n, "seconds": seconds,
                    "joules": joules, "ms_per_call": seconds / n * 1e3, "j_per_call": joules / n,
                    "dyn_j_per_call": (joules - idle_w * seconds) / n, "mean_w": joules / seconds,
                    **telemetry()})
                e_prev, t_prev = e, t
        timing = True
        for stage in work:
            js = [r["j_per_call"] for r in block_rows if r["steps"] == steps and r["stage"] == stage]
            print(f"  {stage:12} {calls[stage]:5} calls/block  {np.median(js):.3f} J per call")

with open(f"{args.out}/images.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(image_rows[0]))
    w.writeheader()
    w.writerows(image_rows)
with open(f"{args.out}/blocks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(block_rows[0]))
    w.writeheader()
    w.writerows(block_rows)
with open(f"{args.out}/run.json", "w") as f:
    json.dump(run, f, indent=2)
