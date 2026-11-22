# SME IPO Classification — Human Review Execution Plan

This document outlines the staged execution tiers, workbook prioritization, and estimated time budgets for reviewing the SME IPO dataset review pack.

---

## 1. Prioritized Review Tiers

Review is organized into three sequential tiers to maximize early alignment on global rules and high-impact edge cases before evaluating sampled sections.

### Tier 1: Core Ground Truth & Global Alignment (Do First)
| File | Name | Scope / Purpose | Rows | Est. Time (45s/row) |
| :--- | :--- | :--- | :---: | :---: |
| **File A** | `01_chapter_mapping_review` | High-level audit of all 87 distinct chapter titles across 6 documents to establish global rule validity. | 87 | ~65 min (1.1h) |
| **File B** | `02_ambiguous_review` | Audit the 60 L2 sections flagged `ambiguous` (7 L1 multi-topic chapters are handled in File A). | 60 | 45 min (0.75h) |
| **File E** | `05_blind_validation` | Primary gold-standard evaluation: 100 blind samples from `century` (validation split). | 100 | 75 min (1.25h) |
| **Tier 1 Total** | — | — | **247** | **~185 min (3.1 hours)** |

### Tier 2: Heuristic Overrides & Rule Integrity
| File | Name | Scope / Purpose | Rows | Est. Time (45s/row) |
| :--- | :--- | :--- | :---: | :---: |
| **File C** | `03_queue_labeled` | Audit targeted keyword overrides (48) and suspected content mismatches (29). *Not blind; do not use for accuracy estimation.* | 77 | ~58 min (0.97h) |
| **File D** | `04_segmentation_check` | Audit suspect headings: paragraph headings (38) and fragments (7). | 45 | ~34 min (0.56h) |
| **File F** | `06_blind_rule_audit` | Blind audit across high-frequency training rules in train split. | 82 | ~62 min (1.02h) |
| **Tier 2 Total** | — | — | **204** | **~154 min (2.55 hours)** |

### Tier 3: Intra-Rater Consistency Check
| File | Name | Scope / Purpose | Rows | Est. Time (45s/row) |
| :--- | :--- | :--- | :---: | :---: |
| **File H** | `08_consistency_check` | 30 blinded duplicates drawn from File E with randomized IDs. **Must be performed at least 24 hours after File E.** | 30 | ~23 min (0.38h) |
| **Tier 3 Total** | — | — | **30** | **~23 min (0.38 hours)** |

---

## 2. Sealed Test Split Protocol
| File | Location | Rows | Status |
| :--- | :--- | :---: | :--- |
| **File G** | `reports/review_pack/sealed/07_blind_test_SEALED.xlsx` | 100 | **STRICTLY SEALED** |

> **Protocol:** File G contains 100 blind examples sampled from the frozen test split (`ideas_electricals`). It is stored in `reports/review_pack/sealed/` with `SEALED_README.txt`. It must remain sealed and uninspected until rule modifications have been frozen and approved in Phase 6. Corrections from File G must never be used to tune rules or hyperparameters.

---

## 3. Cumulative Review Effort Summary

- **Total Active Review Rows (Tiers 1–3):** **481 rows**
- **Estimated Active Review Time:** **~362 minutes (~6.0 hours)**
- **Sealed Test Set Effort (Phase 6):** 100 rows (~75 minutes / 1.25 hours)
- **Grand Total (All 8 Workbooks):** **581 rows (~437 minutes / 7.3 hours)**

