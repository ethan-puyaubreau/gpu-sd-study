NVIDIA GeForce RTX 3080 Ti, driver 610.88, torch 2.11.0+cu128 (CUDA 12.8), diffusers 0.43.0.
Idle: 108.3 W over 30 s. Median [IQR] over images (latency) or blocks (energy).

| steps | ms/image | J/image total | J/image dynamic | mean W | text encoder | UNet | VAE | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 111.9 [111.5-112.9] | 35.81 [35.23-36.28] | 23.69 [22.90-24.19] | 322 | 3% | 29% | 62% | 6% |
| 2 | 145.6 [145.1-147.6] | 45.56 [45.31-46.13] | 30.17 [29.84-30.73] | 321 | 3% | 46% | 48% | 3% |
| 4 | 211.1 [210.5-212.2] | 66.91 [65.20-67.54] | 44.04 [42.65-44.98] | 320 | 2% | 63% | 32% | 3% |

Stability over all energy blocks (min / median / max):

- sm_mhz: 1770 / 1950 / 2025
- mem_mhz: 9451 / 9451 / 9451
- temp_c: 41 / 60 / 62
- power_limit_w: 370 / 370 / 370
- clock_reasons seen: 0x1, 0x4
