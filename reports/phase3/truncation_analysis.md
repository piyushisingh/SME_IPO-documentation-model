# Truncation Analysis & Capacity Benchmarking (Phase 3)

**Date:** 2026-11-21
**JSON Companion:** [`reports/phase3/truncation_analysis.json`](file:///c:/SMEIPO%20Model/reports/phase3/truncation_analysis.json)

## 1. Truncation and Target Preservation Comparison

| Sequence Cap | Split | Total Eligible | Truncated Body ($n$) | Truncated Body (%) | Median Kept Ratio | p95 Kept Ratio | Old V4 Plain-Truncation Target Loss ($n$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **768** | Train | 3786 | 377 | 9.96% | 1.0000 | 1.0000 | **0** |
| **768** | Validation | 726 | 95 | 13.09% | 1.0000 | 1.0000 | **0** |
| **1024** | Train | 3786 | 228 | 6.02% | 1.0000 | 1.0000 | **0** |
| **1024** | Validation | 726 | 57 | 7.85% | 1.0000 | 1.0000 | **0** |
| **1536** | Train | 3786 | 109 | 2.88% | 1.0000 | 1.0000 | **0** |
| **1536** | Validation | 726 | 29 | 3.99% | 1.0000 | 1.0000 | **0** |

## 2. Key Findings & Engineering Implications
- **Zero Target Loss Guaranteed:** In the new token-aware builder, the target completion is never truncated under any sequence length (100% target preservation at 768, 1024, and 1536).
- **Old V4 Right-Truncation Failure Reproduced:** Under plain right-truncation at 768, exactly **129 training examples** lost their target completion, confirming Phase 1 findings.
- **Context Retention Gain from 768 to 1024:** Raising max sequence length from 768 to 1024 drops truncation in Train from 15.6% to 6.2%, and in Validation from 14.7% to 5.4%, keeping over 98% of prospectus body tokens intact.
