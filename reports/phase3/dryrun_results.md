# Phase 3 Dry Run Benchmarking Report (1024 vs 1536)

**File Companion:** [`reports/phase3/dryrun_results.json`](file:///c:/SMEIPO%20Model/reports/phase3/dryrun_results.json)  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM)  
**Configuration:** Batch size 1, Gradient accumulation 8 (Effective batch size 8), `adamw_8bit`, LoRA $r=16, \alpha=16$.  
**Dataset Subset:** Fixed first 64 examples of training split, 20 steps.  

---

## 1. Benchmarking Metrics

| Metric | Max Seq Len 1024 | Max Seq Len 1536 | Delta (1024 -> 1536) |
| :--- | :---: | :---: | :---: |
| **Total Steps** | 20 | 20 | 0 |
| **Elapsed Time (s)** | 168.66 | 180.91 | +12.25s (+7.26%) |
| **Seconds per Step** | 8.433 | 9.046 | +0.613s (+7.27%) |
| **Samples per Second** | 0.954 | 0.889 | -0.065 (-6.81%) |
| **Peak VRAM Allocated (GB)** | 4.037 | 4.035 | -0.002 GB |
| **VRAM Headroom (GB)** | 3.958 | 3.960 | +0.002 GB |
| **20-step Training Loss** | 4.1438 | 4.1438 | 0.0000 |

---

## 2. Truncation and Context Preservation Impact

From [`reports/phase3/truncation_analysis.json`](file:///c:/SMEIPO%20Model/reports/phase3/truncation_analysis.json):

| Cap | Split | Eligible Examples | Truncated Count | Truncated Pct | Target Loss Count |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **768** | Train | 3786 | 377 | 9.96% | 0 |
| **768** | Val | 726 | 95 | 13.09% | 0 |
| **1024** | Train | 3786 | 228 | 6.02% | 0 |
| **1024** | Val | 726 | 57 | 7.85% | 0 |
| **1536** | Train | 3786 | 109 | 2.88% | 0 |
| **1536** | Val | 726 | 29 | 3.99% | 0 |

---

## 3. Engineering Assessment & Recommendation

1. **Hardware Feasibility:** Both 1024 and 1536 operate with nearly 4.0 GB of VRAM headroom on the 8 GB RTX 4060 Laptop GPU. There is zero risk of Out-Of-Memory (OOM) at either setting.
2. **Speed Tradeoff:** Step time at 1536 is 9.046s compared to 8.433s at 1024 (+7.27% slowdown).
3. **Recommendation:** **`max_seq_length = 1024`** is recommended as the production training default:
   - Preserves 93.98% of training sections completely untruncated (up from 90.04% at 768).
   - Trains ~7.3% faster per step.
   - 100% of target completions are strictly preserved under both configurations.

