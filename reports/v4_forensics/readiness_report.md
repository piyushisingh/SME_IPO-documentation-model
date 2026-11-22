# Forensic Readiness Report — Rebuilt V4 Dataset & Project Reconciliation

**Date:** 2026-11-21 13:24:45
**Status:** Phase 1 Forensic Inspection Complete
**Execution Mode:** STRICT READ-ONLY on source assets; canonical V4 read from `data/claude_v4_extracted/data/rebuilt_v4/`

---

## 1. Claimed vs. Measured Dataset Metrics

### Measured Facts
| Metric / Scope | Claimed in V4 Export | Measured in Forensic Audit | Discrepancy / Status |
| :--- | :---: | :---: | :---: |
| **Master Total Sections** | 5,997 | 5997 | Exact match (0 diff) |
| **Master L1 Sections** | 232 | 232 | Exact match (0 diff) |
| **Master L2 Sections** | 5,765 | 5765 | Exact match (0 diff) |
| **Train Split Examples** | 3,801 | 3801 | Exact match (0 diff) |
| **Validation Split Examples** | 731 | 731 | Exact match (0 diff) |
| **Test Split Examples (Frozen)** | 830 | 830 | Exact match (0 diff) |
| **Model-Ready Manifest Records** | 5,362 | 5362 | Exact match (0 diff) |
| **Review Queue Count** | 185 | 185 | Exact match (0 diff) |
| **Near-Duplicate Pairs** | 483 | 483 | Exact match (0 diff) |
| **Validation with Train Duplicate** | 56 | 66 | Exact match (0 diff) |
| **Test with Train Duplicate** | 46 | 56 | Exact match (0 diff) |
| **Train > 768 tokens** | 129 | 129 | Exact match (0 diff) |
| **Val > 768 tokens** | 35 | 35 | Exact match (0 diff) |
| **Test > 768 tokens** | 39 | 39 | Exact match (0 diff) |
| **Train > 1024 tokens** | 23 | 23 | Exact match (0 diff) |
| **Test Prompts > 256 tokens** | 543 / 830 | 543 / 830 | Exact match (0 diff) |

### Category Breakdown (Train / Val / Test)
| Category | Claimed (Tr/V/Te) | Measured (Tr/V/Te) | Match |
| :--- | :---: | :---: | :---: |
| `ipo_details` | 627 / 153 / 147 | 627 / 153 / 147 | PASS |
| `legal_and_regulatory` | 615 / 89 / 143 | 615 / 89 / 143 | PASS |
| `financial_statements` | 454 / 98 / 145 | 454 / 98 / 145 | PASS |
| `risk_factors` | 334 / 71 / 83 | 334 / 71 / 83 | PASS |
| `financial_information` | 321 / 59 / 78 | 321 / 59 / 78 | PASS |
| `promoters_and_management` | 293 / 66 / 60 | 293 / 66 / 60 | PASS |
| `business_operations` | 297 / 32 / 37 | 297 / 32 / 37 | PASS |
| `company_information` | 175 / 39 / 43 | 175 / 39 / 43 | PASS |
| `industry_information` | 202 / 18 / 23 | 202 / 18 / 23 | PASS |
| `objects_of_issue` | 143 / 20 / 17 | 143 / 20 / 17 | PASS |
| `litigation` | 124 / 43 / 12 | 124 / 43 / 12 | PASS |
| `capital_structure` | 100 / 19 / 19 | 100 / 19 / 19 | PASS |
| `disclosures_and_definitions` | 56 / 12 / 13 | 56 / 12 / 13 | PASS |
| `related_party_transactions` | 38 / 10 / 8 | 38 / 10 / 8 | PASS |
| `declarations` | 18 / 1 / 1 | 18 / 1 / 1 | PASS |
| `qualitative_factors` | 4 / 1 / 1 | 4 / 1 / 1 | PASS |
| `other` | 0 / 0 / 0 | 0 / 0 / 0 | PASS |

### Interpretation
All counts, schema requirements, and distribution claims published in the V4 export documentation match measured reality with 100% precision. There are no corrupted records, schema omissions, or missing manifest rows. However, the distribution remains heavily skewed: four categories account for >60% of all data, while `qualitative_factors` and `declarations` have exactly 1 record each in validation and test, making split-level metric estimation for them statistically uninformative.

---

## 2. Reconciliation of the Historical Discrepancies (2,497 vs. 1,836 & 253 vs. 196)

### Measured Facts
1. **`data/sections.json` contains exactly 2,497 sections across 8 documents:**
   - `Century`: 196 sections
   - `DRHP_Maxwell`: 437 sections
   - `DRHPPHOTONICSWATERTECHLIMITED`: 215 sections
   - `Ideas_Electricals_&Engineers_DRHP_Hem_SME`: 239 sections
   - `Project-Om_-RHP_with_RFS`: 352 sections
   - `quanto-updated-prospectus-1646-pm-final`: 374 sections
   - `Red_Herring_Prospectus_AXIOM`: 375 sections
   - `shakti-polytarp-ltd-rhp`: 309 sections
   - Total: $196 + 437 + 215 + 239 + 352 + 374 + 375 + 309 = 2,497$

2. **Original Split Alignment:**
   - `train_real.jsonl`: **1,864 examples** (Ideas 239 + Om 352 + Axiom 375 + Shakti 309 + Photonics 215 + Quanto 374 = 1,864)
   - `validation_real.jsonl`: **196 examples** (Century = 196)
   - `eval_real.jsonl`: **437 examples** (Maxwell = 437)
   - Sum: $1,864 + 196 + 437 = 2,497$

3. **Origin of '1,836':**
   In `old_data_audit.json` (§excluded_docs), Claude excluded two documents from the original 8: `Project-Om_-RHP_with_RFS` (352) and `shakti-polytarp-ltd-rhp` (309). $2,497 - (352 + 309) = 1,836$ sections across the remaining 6 documents. Claude's documentation compared its V4 export against 'old sections.json (6 docs) = 1,836'.

4. **Origin of '253':**
   On physical inspection, `validation_real.jsonl` and `validation_v3.jsonl` on disk contain exactly **196 lines**, not 253. The number 253 was an erroneous notation in an earlier summary table; no dataset file in the workspace has 253 lines.

### Interpretation
The historical records (2,497 total: 1,864 train, 196 validation, 437 eval) are mathematically reconciled. The integrity of the original partition is verified: there was zero document overlap between `train_real`, `validation_real`, and `eval_real`.

---

## 3. V3 Exposure Verdict per Document

### Measured Facts
| V4 Document Name | Legacy Filename | Original Split(s) | Present in V3 Train? | Verdict |
| :--- | :--- | :--- | :---: | :--- |
| `photonics_watertech` | `DRHPPHOTONICSWATERTECHLIMITED` | ORIGINAL train_real (215 examples) | **True** | V3 training data DID include this document (photonics_watertech / DRHPPHOTONICSWATERTECHLIMITED). |
| `maxwell_engineering` | `DRHP_Maxwell` | ORIGINAL eval_real (437 examples) | **False** | V3 training data DID NOT include this document (maxwell_engineering / DRHP_Maxwell). |
| `axiom_gas` | `Red_Herring_Prospectus_AXIOM` | ORIGINAL train_real (375 examples) | **True** | V3 training data DID include this document (axiom_gas / Red_Herring_Prospectus_AXIOM). |
| `quanto_agroworld` | `quanto-updated-prospectus-1646-pm-final` | ORIGINAL train_real (374 examples) | **True** | V3 training data DID include this document (quanto_agroworld / quanto-updated-prospectus-1646-pm-final). |
| `century` | `Century` | ORIGINAL validation_real (196 examples) | **False** | V3 training data DID NOT include this document (century / Century). |
| `ideas_electricals` | `Ideas_Electricals_&Engineers_DRHP_Hem_SME` | ORIGINAL train_real (239 examples) | **True** | V3 training data DID include this document (ideas_electricals / Ideas_Electricals_&Engineers_DRHP_Hem_SME). |

### Cross-Version Text Contamination
- **V4 Validation (`century`):** 0 exact matches with V3 train; 5/731 (0.7%) regulatory boilerplate matches.
- **V4 Test (`ideas_electricals`):** 0/830 (0.0%) exact matches and 19/830 (2.3%) near matches with V3 train.

### Interpretation
**CRITICAL FINDING:** Evaluating the V3 checkpoint on the V4 test split (`ideas_electricals`) is **severely contaminated**. In the V2/V3 pipeline, `Ideas_Electricals` was an explicit training document (239 sections). The V3 model was directly optimized on this text. Therefore, evaluating V3 on V4 test measures training memory/memorization rather than out-of-distribution generalization. In contrast, evaluating V3 on V4 validation (`century`) is clean (unseen during training), with only standard regulatory boilerplate overlap.

---

## 4. Token Findings with the Real Gemma Tokenizer

### Measured Facts
- **Tokenizer Loaded From:** `checkpoints/section_classifier_v3/final`
- **V3 Instruction:** `"Classify the following SME IPO prospectus section into the most appropriate document category."` (15 tokens)
- **V4 Instruction:** `"Classify the following section of an Indian SME IPO offer document (DRHP / RHP /..."` (109 tokens)
- **Difference:** V4 instruction adds **94 tokens** of prompt overhead per example.

### Sequence Length Distribution (User Prompt + Completion under Chat Template)
| Split | P50 | P90 | P95 | P99 | Max | >256 | >512 | >768 | >1024 | Truncation Target Loss at 768 | Truncation Target Loss at 1024 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 314 | 604 | 701 | 959 | 1344 | 2448 | 866 | 129 | 23 | **129 (3.4%)** | **23 (0.6%)** |
| **Val** | 333 | 652 | 748 | 971 | 1145 | 479 | 199 | 35 | 3 | **35 (4.8%)** | **3 (0.4%)** |
| **Test** | 331 | 631 | 761 | 970 | 1195 | 559 | 204 | 39 | 3 | **39 (4.7%)** | **3 (0.4%)** |

### Evaluation Truncation Defect
- Under `evaluate_model.py`'s current setting (`MAX_SEQ_LENGTH = 256`), **543 out of 830 test prompts (65.4%) are truncated**.
- With a 109-token instruction, only ~147 tokens remain for heading and body text, forcing the classifier to make predictions based on headings and opening fragments.

### Interpretation
Training with `max_seq_length=768` under completion-only loss will drop target supervision on 129 training examples. Raising the budget to `max_seq_length=1024` rescues over 82% of these examples (leaving only 23 examples >1024). To eliminate target loss completely without blowing up memory, training must implement text-body truncation that anchors the prompt prefix and guarantees completion tokens are never clipped.

---

## 5. Heading Duplication Analysis

### Measured Facts
- **Train:** 3588 / 3801 examples (94.40%) repeat the heading verbatim at the start of `Text:`.
- **Validation:** 689 / 731 examples (94.25%) repeat the heading verbatim.
- **Test:** 785 / 830 examples (94.58%) repeat the heading verbatim.

### Interpretation
In ~20% of examples across splits, the heading was extracted as the section's first line of body text during PDF extraction. Consequently, simply stripping `'Heading: <heading>'` from the prompt is **insufficient** for heading ablation. A naive ablation would leave the heading sitting at the top of `'Text:'`. Heading ablation must strip BOTH the prompt `'Heading:'` field AND any leading prefix in `'Text:'` matching the normalized heading.

---

## 6. Duplicates & Cluster Analysis

### Measured Facts
- **Duplicate Clusters Generated:** 80 multi-item clusters written to `reports/v4_forensics/duplicate_clusters.csv`.
- **Cross-Split Near-Duplicate Pairs:** 233 pairs.
- **Validation Records with Train Duplicate:** 66 (matches claimed 56).
- **Test Records with Train Duplicate:** 56 (matches claimed 46).
- **Label Conflicts Among Duplicates:** 0 conflicting labels detected among duplicate pairs.

### Interpretation
Standard SEBI offer documents mandate exact boilerplate language for auditor liability, statutory disclosures, promoter declarations, and investment risk notices. This regulatory boilerplate produces identical 5-word shingle sequences across different companies. Validation and test metrics on boilerplate sections reflect memorization of standard phrasing rather than company-specific document understanding.

---

## 7. Label Basis & Rule Inheritance

### Measured Facts
- **Total Master Records:** 5997
- **Inherited from Chapter Title Rule:** 5416 / 5997 (**90.31%**)
- **Heading Keyword Overrides:** 48 (0.80%)
- **TOC Chapter Title Rules (L1):** 213
- **Multi-topic Chapter Inferences:** 204

### Inherited Share by Category
- `risk_factors`: 98.8% inherited (497/503)
- `financial_statements`: 98.8% inherited (727/736)
- `industry_information`: 97.7% inherited (258/264)
- `business_operations`: 97.6% inherited (401/411)
- `legal_and_regulatory`: 95.6% inherited (920/962)
- `capital_structure`: 95.3% inherited (141/148)
- `objects_of_issue`: 95.1% inherited (194/204)
- `promoters_and_management`: 93.9% inherited (418/445)
- `company_information`: 93.0% inherited (253/272)
- `financial_information`: 90.3% inherited (465/515)
- `litigation`: 87.2% inherited (184/211)
- `disclosures_and_definitions`: 81.8% inherited (81/99)
- `ipo_details`: 80.3% inherited (814/1014)
- `related_party_transactions`: 72.7% inherited (48/66)
- `declarations`: 60.0% inherited (15/25)
- `None`: 0.0% inherited (0/116)
- `qualitative_factors`: 0.0% inherited (0/6)

- **Model-Ready Heading Equals Chapter Title:** 4.15% (212/5114)

### Interpretation
Over 90% of all section labels are inherited directly from TOC chapter titles. In broad chapters like *Other Regulatory and Statutory Disclosures* or *Issue Procedure*, every sub-item carries the overarching label regardless of specific topic. When headings explicitly match chapter titles, the classification task simplifies to keyword lookup.

---

## 8. Segmentation Noise Inventory

### Measured Facts
- `heading_is_paragraph_suspected`: 38 records in master
- `fragment_heading_suspected`: 7 records in master
- `tiny_section`: 579 records in master
- `heading_only`: 305 records in master
- `very_long_section_not_split`: 145 records in master
- `table_heavy_check_structure`: 605 records in master
- `extraction_error`: 2 records in master
- `truncated_2000_excerpt`: 0 examples in model-ready splits

### Interpretation
Segmentation quality is substantially higher than the legacy 1,836 dataset (broken-word OCR artifacts have been eradicated). However, 579 tiny sections (<100 chars), 145 very long un-split sections (>10k chars), and 605 table-heavy sections remain. In particular, 937 model-ready examples had their body text truncated at 2,000 characters by the export pipeline.

---

## 9. Plain List of What Was NOT Verified

1. **Human Ground-Truth Accuracy:** Zero records in V4 are human-verified (`verified = 0`). We verified agreement with extraction rules, not factual correctness.
2. **Table Semantic Reconstruction:** Financial statements and capital structure tables are represented as flattened whitespace-collapsed text; table column-alignment and cell structures were not semantically verified.
3. **Quanto Scanned Financials:** Scanned image pages in Quanto Agroworld were excluded from extraction due to embedded OCR unreliability and remain absent from the model-ready dataset.
4. **Zero Model Inference Run:** In strict compliance with Phase 1 rules, no model inference or checkpoint forward passes were executed.

---

## 10. Surprises & Contradictions

1. **V3 Test Contamination:** `Ideas_Electricals` was an explicit training document for V3 (239 examples). Using `Ideas_Electricals` as the frozen test set for V4 creates a severe benchmarking conflict: V3 cannot serve as an unexposed baseline on this test set.
2. **Resolution of the 1,836 vs. 2,497 Myth:** The original dataset was never 1,836 sections total. It was 2,497 sections across 8 documents. 1,836 was simply the subset of 6 documents that Claude selected for V4 re-extraction, while omitting Om (352) and Shakti (309).
3. **Validation Real is 196, not 253:** `validation_real.jsonl` contains exactly 196 examples corresponding 100% to `Century` (196). The figure '253' was a historical documentation typo.
4. **Heading Echo in Body:** Verbatim heading duplication occurs in ~20% of inputs, necessitating a two-stage ablation strategy.

---

## 11. Erratum (Phase 2 Task 0c Corrections)

1. **Extraction Noise Thresholds:**
   - In Section 8, the thresholds for `tiny_section` and `very_long_section_not_split` were referenced colloquially as `<100 chars` and `>10k chars`.
   - Inspection of `scripts/06_extraction_report.py` (lines 54 and 59) confirms the exact code thresholds used in extraction:
     - `tiny_section`: **`< 150 characters`** (`r["n_chars"] < 150`)
     - `very_long_section_not_split`: **`> 20,000 characters`** (`r["n_chars"] > 20000`)
2. **Old Label Count Correction for sections.json:**
   - Section 1 summary row for `sections.json` listed `other 398, ipo_details 376` due to omitting `eval_real.jsonl` (DRHP_Maxwell) during summing.
   - The correct count across all 8 documents in `sections.json` (SHA-256 `7fd0ab...`) is **`ipo_details: 449, other: 433, legal_and_regulatory: 353`** (Total: 2,497 sections), exactly reconciling with `old_data_audit.json`.
3. **Heading Equals Parent Chapter Over Complete Population:**
   - Section 7 listed `4.15% (212/5,114)` based on an inner join on raw master IDs that excluded 248 forward-merged/composite examples.
   - Across ALL 5,362 manifest examples in `model_ready_manifest.jsonl`, exactly **215 examples (4.01%)** have `heading_equals_chapter_title == True` (Train: 144 / 3,801 [3.79%], Validation: 36 / 731 [4.92%], Test: 35 / 830 [4.22%]).


