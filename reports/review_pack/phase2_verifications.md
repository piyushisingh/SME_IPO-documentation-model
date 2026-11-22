# Phase 2 — Carry-Over Verifications Report

**Date:** 2026-11-21  
**Execution Mode:** STRICT READ-ONLY on source files and datasets  
**Target:** Historical reconciliation, manifest completeness, and extraction script thresholds  

---

## 1. Task 0a: Recomputed Old Label Distribution from `data/sections.json`

### File Integrity
- **Target File:** [`data/sections.json`](file:///c:/SMEIPO%20Model/data/sections.json)
- **SHA-256 Hash:** `7fd0ab6176c53edb932e6de7d7505d1ded99117eb484c0ed3fbe432457d967dd`

### Category Distribution Breakdown

| Category Name | (i) All 8 Documents (Original sections.json) | (ii) Six V4 Documents Only (Excluded Om & Shakti) |
| :--- | :---: | :---: |
| `ipo_details` | **449** | **318** |
| `other` | **433** | **300** |
| `legal_and_regulatory` | **353** | **291** |
| `promoters_and_management` | 254 | 173 |
| `business_operations` | 212 | 159 |
| `financial_information` | 200 | 136 |
| `financial_statements` | 132 | 110 |
| `capital_structure` | 127 | 102 |
| `objects_of_issue` | 69 | 51 |
| `industry_information` | 62 | 39 |
| `risk_factors` | 57 | 44 |
| `company_information` | 55 | 44 |
| `disclosures_and_definitions` | 36 | 30 |
| `litigation` | 33 | 22 |
| `related_party_transactions` | 14 | 10 |
| `qualitative_factors` | 7 | 5 |
| `declarations` | 4 | 2 |
| **Total** | **2,497** | **1,836** |

### Explanation of the Difference with Phase 1 Report
- **The Bundle's Audit Figures:** In `old_data_audit.json` (§global.category_counts_all_8_docs) and `extraction_report.md`, the bundle quoted for all 8 documents: `ipo_details: 449`, `other: 433`, `legal_and_regulatory: 353`. This matches our recomputation of all 8 documents in `sections.json` with 100% precision.
- **The Phase 1 Report Figures (`other: 398`, `ipo_details: 376`):**
  - In Phase 1, when generating the summary row for `sections.json`, the script mistakenly summed only `train_real.jsonl` (which had 385 `other` and 345 `ipo_details`) plus `validation_real.jsonl` (which had 13 `other` and 31 `ipo_details`):
    - $385 + 13 = \mathbf{398}$ (`other`)
    - $345 + 31 = \mathbf{376}$ (`ipo_details`)
  - It inadvertently omitted `eval_real.jsonl` (`DRHP_Maxwell`, 437 records, containing 73 `ipo_details`, 35 `other`, and 109 `legal_and_regulatory`):
    - `ipo_details`: $376 + 73 = \mathbf{449}$
    - `other`: $398 + 35 = \mathbf{433}$
    - `legal_and_regulatory`: $244 + 109 = \mathbf{353}$
  - When all 8 documents (including `eval_real`) are included, the exact numbers match the bundle's audit.

---

## 2. Task 0b: Heading Equals Parent Chapter Title over ALL 5,362 Manifest Examples

### Recomputed Distribution across all 5,362 Model-Ready Examples
Using the pre-computed canonical field `heading_equals_chapter_title` from [`model_ready_manifest.jsonl`](file:///c:/SMEIPO%20Model/data/claude_v4_extracted/data/rebuilt_v4/model_ready_manifest.jsonl):

| Split | Heading Equals Chapter Title ($n$) | Total Examples ($N$) | Percentage (%) |
| :--- | :---: | :---: | :---: |
| **Train** | 144 | 3,801 | 3.79% |
| **Validation** | 36 | 731 | 4.92% |
| **Test** | 35 | 830 | 4.22% |
| **Overall** | **215** | **5,362** | **4.01%** |

### Explanation of Why Phase 1 Used 5,114
- In Phase 1, the script checked heading equality by performing a dictionary lookup on `master_sections.jsonl` keyed by `example_id`.
- In V4, **248 model-ready examples** were derived from forward-merged heading-only records or multi-section consolidations whose `example_id` in the manifest carried a merged suffix (e.g., `century-L2-006-066+1`).
- Because `master_sections.jsonl` contains raw unmerged section IDs, `master_by_id.get(eid)` returned `None` for those 248 composite examples, leaving $5,362 - 248 = \mathbf{5,114}$ checked records.
- Over the complete population of all 5,362 examples in `model_ready_manifest.jsonl`, exactly **215 examples (4.01%)** have a heading equal to the parent chapter title, perfectly confirming Claude's claim of "4.0%".

---

## 3. Task 0c: Extraction Thresholds in Bundle Scripts

### Inspected Scripts
- [`scripts/06_extraction_report.py`](file:///c:/SMEIPO%20Model/data/claude_v4_extracted/data/rebuilt_v4/scripts/06_extraction_report.py)
- [`scripts/09_build_master.py`](file:///c:/SMEIPO%20Model/data/claude_v4_extracted/data/rebuilt_v4/scripts/09_build_master.py)

### Actual Thresholds in Code
1. **`tiny_section`:**
   - Code: `scripts/06_extraction_report.py:54`
   - Condition: `if r["n_chars"] < 150: f.append("tiny_section")`
   - **Actual threshold: `< 150 characters`** (Phase 1 colloquially wrote `<100 chars`).
2. **`very_long_section_not_split`:**
   - Code: `scripts/06_extraction_report.py:59`
   - Condition: `if r["n_chars"] > 20000: f.append("very_long_section_not_split")`
   - **Actual threshold: `> 20,000 characters`** (Phase 1 colloquially wrote `>10k chars`).

