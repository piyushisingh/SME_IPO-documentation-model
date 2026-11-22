# Phase 3B Dry Run & Throughput Benchmarks (Worst-Case & Stratified)

**Companion JSON:** [`reports/phase3/dryrun_v2_results.json`](file:///c:/SMEIPO%20Model/reports/phase3/dryrun_v2_results.json)

## 1. Worst-Case Memory Benchmarks (16 Longest Examples)

| Metric | Worst-Case 1024 | Worst-Case 1536 |
| :--- | :---: | :---: |
| **Longest Sequence** | 1021 | 1533 |
| **Optimizer Steps** | 6 | 6 |
| **Step Time (s)** | 14.148 | 12.017 |
| **Torch Allocated (GB)** | 1.965 | 2.413 |
| **Torch Reserved (GB)** | 2.127 | 2.783 |
| **NVIDIA-SMI Peak Used (GB)** | 2.28 | 2.937 |
| **Baseline Other Processes (GB)** | 1.177 | 1.177 |
| **Headroom (GB)** | **5.716** | **5.059** |

## 2. Length-Stratified Throughput & Training Projections (1024 Cap)

- **Throughput:** 0.715 samples/sec (217.76 tokens/sec)
- **Stratified 20-Step Loss:** First 5 mean = 3.2568, Last 5 mean = 0.8196 (Decreased: True)
- *Note:* This is a pipeline readiness check, not evidence of generalized learning.
- **Projected Training Time (3 Epochs, 3,786 examples):**
  - By Token Throughput: **4.44 hours** (~266.4 minutes)
  - Worst-Case Upper Bound: **5.58 hours** (~334.8 minutes)

## 3. V3 vs V4 Batch Configuration Comparison

| Hyperparameter | V3 Setting | Phase 3B Dry Run Setting |
| :--- | :---: | :---: |
| Per-Device Batch Size | 1 | 1 |
| Gradient Accumulation Steps | 8 | 8 |
| Effective Batch Size | 8 | 8 |
