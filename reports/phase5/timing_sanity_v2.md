# Phase 5 — Corrected Timing and Throttling Sanity Report (v2)

**Date:** 2026-11-26  
**Execution Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM), Driver 595.79, CUDA 12.8  
**Scope:** Accurate NVML bitmask decoding, unit disambiguation, and realistic 3-epoch runtime projections.  
**Source File:** [`reports/phase5/timing_sanity_v2.json`](file:///c:/SMEIPO%20Model/reports/phase5/timing_sanity_v2.json)

---

## 1. Disambiguation of Units

- **Micro-batch:** 1 training example evaluated at `batch_size=1` (forward + backward pass).
- **Optimizer Step:** 8 micro-batches (`gradient_accumulation_steps=8`, corresponding to 8 training examples).
- **Total Training Set Volume:** 3,786 examples.
- **3-Epoch Total Volume:** 11,358 micro-batches (3,786 * 3) = 1,420 optimizer steps.

---

## 2. Accurate NVML Bitmask Decoding

Per standard NVML / nvidia-smi bitmask specifications (`nvmlClocksThrottleReason*`):
- `0x0000000000000001`: `GpuIdle`
- `0x0000000000000002`: `ApplicationsClocksSetting`
- `0x0000000000000004`: `SwPowerCap`
- `0x0000000000000008`: `HwSlowdown`
- `0x0000000000000010`: `SyncBoost`
- `0x0000000000000020`: `SwThermalSlowdown`
- `0x0000000000000040`: `HwThermalSlowdown`
- `0x0000000000000080`: `HwPowerBrakeSlowdown`

### Empirical Sampling Results Across the 4 Timing Runs

| Run Tag | Total Samples | Active Samples (>50% util) | Active Throttle Reasons | Idle Samples (<=50% util) | Idle Throttle Reasons |
|---|---|---|---|---|---|
| `1024_run1` | 39 | 4 | GpuIdle: 100.0% | 35 | GpuIdle: 74.3%, None: 25.7% |
| `1024_run2` | 20 | 1 | None: 100.0% | 19 | GpuIdle: 63.2%, None: 36.8% |
| `1536_run1` | 20 | 8 | None: 62.5%, SwPowerCap: 37.5% | 12 | GpuIdle: 83.3%, None: 16.7% |
| `1536_run2` | 20 | 8 | SwPowerCap: 37.5%, None: 62.5% | 12 | GpuIdle: 83.3%, None: 16.7% |

> [!NOTE]
> Zero instances of `SwThermalSlowdown`, `HwThermalSlowdown`, `HwSlowdown`, or `HwPowerBrakeSlowdown` occurred. During active compute (>50% GPU util), throttling was either `None` (62.5% - 100%) or standard laptop `SwPowerCap` (37.5% at 1536 cap), which represents normal operation within factory power limits.

---

## 3. Benchmarked Step Times and 3-Epoch Projections

### Benchmarked Steady-State Times on Longest Sequences
- `1024_run2`: **1.275 s / micro-batch** (10.20 s / optimizer step). Warmup overhead: 20.675 s on initial compile.
- `1536_run2`: **1.434 s / micro-batch** (11.47 s / optimizer step). Warmup overhead: 1.778 s.
- Difference: 1536 is **12.5% slower** per step on worst-case sequences.

### Projected 3-Epoch Training Duration (11,358 micro-batches / 1,420 optimizer steps)

| Estimation Method | Cap 1024 Rate | Cap 1024 Duration | Cap 1536 Rate | Cap 1536 Duration |
|---|---|---|---|---|
| **Conservative Upper Bound** (worst-case sequences) | 1.275 s / micro-batch | **4.02 hours** | 1.434 s / micro-batch | **4.52 hours** |
| **Realistic Central Estimate** (mean ~306-328 tokens) | ~0.40 s / micro-batch | **1.26 hours** | ~0.45 s / micro-batch | **1.42 hours** |

Both caps easily fit within standard daytime training windows, well below the previously feared 14+ hours.

