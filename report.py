"""Summarize results/ into a table (results/summary.md) and a stacked bar chart.

Energy per image of a stage = its energy per call times its calls per image
(1 for the text encoder and the VAE, the step count for the UNet). "Other" is
the whole pipeline minus the three stages: the scheduler, kernel launches and
the gaps between kernels, which the replays of single stages do not contain.
"""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

OUT = "results"
STAGES = ("text_encoder", "unet", "vae")
LABELS = {"text_encoder": "Text encoder", "unet": "UNet", "vae": "VAE decode"}
COLORS = {"text_encoder": "#2a78d6", "unet": "#eb6834", "vae": "#1baf7a"}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def read(name):
    with open(f"{OUT}/{name}") as f:
        return list(csv.DictReader(f))


def quartiles(values):
    return np.percentile([float(v) for v in values], [25, 50, 75])


images, blocks = read("images.csv"), read("blocks.csv")
run = json.load(open(f"{OUT}/run.json"))
steps_list = sorted({int(r["steps"]) for r in images})


def block_values(steps, stage, field):
    return [float(r[field]) for r in blocks if int(r["steps"]) == steps and r["stage"] == stage]


lines = [
    f"{run['gpu']}, driver {run['driver']}, torch {run['torch']} (CUDA {run['cuda']}), diffusers {run['diffusers']}.",
    f"Idle: {run['idle_w']:.1f} W over {run['idle_seconds']:.0f} s. Median [IQR] over images (latency) or blocks (energy).",
    "",
    "| steps | ms/image | J/image total | J/image dynamic | mean W | text encoder | UNet | VAE | other |",
    "|---|---|---|---|---|---|---|---|---|",
]
dynamic = {}  # steps -> {stage: dynamic J per image}
for steps in steps_list:
    ms = quartiles([r["wall_ms"] for r in images if int(r["steps"]) == steps])
    total = quartiles(block_values(steps, "pipeline", "j_per_call"))
    dyn = quartiles(block_values(steps, "pipeline", "dyn_j_per_call"))
    watts = quartiles(block_values(steps, "pipeline", "mean_w"))
    calls = {"text_encoder": 1, "unet": steps, "vae": 1}
    parts = {s: np.median(block_values(steps, s, "dyn_j_per_call")) * calls[s] for s in STAGES}
    parts["other"] = dyn[1] - sum(parts.values())
    dynamic[steps] = parts
    share = " | ".join(f"{parts[s] / dyn[1]:.0%}" for s in (*STAGES, "other"))
    lines.append(f"| {steps} | {ms[1]:.1f} [{ms[0]:.1f}-{ms[2]:.1f}] | {total[1]:.2f} [{total[0]:.2f}-{total[2]:.2f}]"
                 f" | {dyn[1]:.2f} [{dyn[0]:.2f}-{dyn[2]:.2f}] | {watts[1]:.0f} | {share} |")

lines += ["", "Stability over all energy blocks (min / median / max):", ""]
for field in ("sm_mhz", "mem_mhz", "temp_c", "power_limit_w"):
    v = [float(r[field]) for r in blocks]
    lines.append(f"- {field}: {min(v):.0f} / {np.median(v):.0f} / {max(v):.0f}")
lines.append(f"- clock_reasons seen: {', '.join(sorted({r['clock_reasons'] for r in blocks}))}")
open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(6.4, 4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
x = np.arange(len(steps_list))
bottom = np.zeros(len(steps_list))
for stage in STAGES:
    heights = np.array([dynamic[s][stage] for s in steps_list])
    ax.bar(x, heights, 0.5, bottom=bottom, color=COLORS[stage], label=LABELS[stage],
           edgecolor=SURFACE, linewidth=2)
    bottom += heights
measured = [sum(dynamic[s].values()) for s in steps_list]  # the three stages plus "other"
ax.hlines(measured, x - 0.32, x + 0.32, color=INK, linewidth=2, label="Whole pipeline, measured")
for xi, m in zip(x, measured):
    ax.text(xi + 0.34, m, f"{m:.1f} J", ha="left", va="center", color=INK, fontsize=9)
ax.set_xticks(x, [f"{s} step{'s' if s > 1 else ''}" for s in steps_list], color=INK)
ax.set_ylabel("Dynamic energy per image (J)", color=MUTED)
ax.set_title("sd-turbo 512x512 fp16, energy above idle per stage", color=INK, fontsize=11, loc="left")
ax.tick_params(colors=MUTED)
ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/energy_per_stage.png", dpi=150)
