# Phase 2 Review Pack Integrity Audit

**Date:** 2026-11-21  
**Audit Target:** Review Pack v1 artifacts in `reports/review_pack/` and `src/sme_clf/make_review_pack.py`  
**Auditor:** Antigravity AI Assistant  
**Status:** Complete — Issues Identified, INVALID Flags Placed, Regeneration Executed to `reports/review_pack_v2/`

---

## 1. File Integrity Audit Verdict Table

| File | Verdict | Evidence | Action |
| :--- | :---: | :--- | :--- |
| `01_chapter_mapping_review.xlsx` / `.csv` | **VALID** | Contains 87 distinct normalized chapter titles across all 6 documents, covering all 13 specified columns with dropdown validation. | Retain as reference; regenerate into v2 for pack consistency. |
| `02_ambiguous_review.xlsx` / `.csv` | **VALID** | Contains exactly the 60 L2 master records with `review_status="ambiguous"`. All 10 columns and dropdowns present. | Retain as reference; regenerate into v2 for pack consistency. |
| `03_queue_labeled.xlsx` / `.csv` | **VALID** | Contains exactly 77 de-duplicated records with `heading_keyword_overrides_chapter_label` (48) or `suspected_label_content_mismatch` (29). All 11 columns present. | Retain as reference; regenerate into v2 for pack consistency. |
| `04_segmentation_check.xlsx` / `.csv` | **VALID** | Contains exactly 45 records (38 `heading_is_paragraph_suspected` + 7 `fragment_heading_suspected`). Allowed labels strictly `ok, segmentation_error, unsure`. | Retain as reference; regenerate into v2 for pack consistency. |
| `05_blind_validation.xlsx` / `.csv` | **INVALID** | Allocated a flat 7 samples per category across all 14 large categories (e.g. `related_party_transactions` $N=10 \to n=7$ vs `ipo_details` $N=152 \to n=7$) via round-robin increments, violating proportional stratification. Associated manifest omitted $N_{h,\text{before\_filters}}$. Composite IDs missed quality flag lookup. | Mark INVALID in `INVALID_DO_NOT_USE.txt`; regenerate with proportional allocation into `reports/review_pack_v2/`. |
| `06_blind_rule_audit.xlsx` / `.csv` | **INVALID** | While row count (82) and rule sample sizes (4 for $>50$, 2 for $20-50$) are correct, it relies on the incomplete `sampling_manifest.json` and bundled `answer_key.json`. | Mark INVALID in `INVALID_DO_NOT_USE.txt`; regenerate into `reports/review_pack_v2/`. |
| `sealed/07_blind_test_SEALED.xlsx` / `.csv` | **INVALID** | Identical stratification defect as File E (flat 7 samples per category for all 14 categories $\ge 5$, e.g. `related_party_transactions` $N=8 \to n=7$ vs `financial_statements` $N=145 \to n=7$). Manifest omitted $N_{h,\text{before\_filters}}$. | Log invalidity in `reports/test_access_log.jsonl`; place `INVALID_DO_NOT_USE.txt` in `sealed/`; regenerate into `reports/review_pack_v2/sealed/`. |
| `08_consistency_check.xlsx` / `.csv` | **INVALID** | Derived from the flawed File E sample. | Mark INVALID in `INVALID_DO_NOT_USE.txt`; regenerate from corrected File E into `reports/review_pack_v2/`. |
| `sampling_manifest.json` | **INVALID** | Completely omitted `N_h_before_filters` (the raw split category count). Stratum weights did not reflect proportional allocation. | Supersede with complete manifest in `reports/review_pack_v2/sampling_manifest.json`. |
| `private/answer_key.json` | **INVALID** | Contains answer keys for flawed samples E, G, H. | Supersede with corrected answer key in `reports/review_pack_v2/private/answer_key.json`. |
| `review_guide.md` | **VALID** | Contains accurate condensed taxonomy, edge-case resolution rules, and principles. | Retain; copy to `reports/review_pack_v2/` for consistency. |
| `review_plan.md` | **INVALID** | Misgrouped Tier 1 (described as A+B+C+D instead of A+B+E) and miscalculated tier minutes using 90s/45s mixed rates instead of uniform 45s/row. | Rewrite in `reports/review_pack_v2/review_plan.md` with correct tiers and minutes. |

---

## 2. Recomputed Review Tiers and Time Budgets

At the specified uniform rate of **45 seconds per row** (0.75 minutes/row):

| Tier | Included Files | Target / Focus | Rows | Formula | Estimated Time |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Tier 1 (High Priority)** | File A (`01_chapter_mapping_review`)<br>File B (`02_ambiguous_review`)<br>File E (`05_blind_validation`) | High-level taxonomy alignment, ambiguous L2 sections, and blind validation benchmark. | **247** | $87 + 60 + 100$ | **185.25 min (~3.1 hrs)** |
| **Tier 2 (Medium Priority)**| File C (`03_queue_labeled`)<br>File D (`04_segmentation_check`)<br>File F (`06_blind_rule_audit`) | Targeted override audits, segmentation defect inspection, and training rule verification. | **204** | $77 + 45 + 82$ | **153.00 min (~2.55 hrs)** |
| **Tier 3 (Consistency)** | File H (`08_consistency_check`) | Intra-rater agreement audit (30 disguised validation duplicates). | **30** | $30$ | **22.50 min (~0.38 hrs)** |
| **Total Active Review** | **Tiers 1, 2, and 3** | All unsealed reviewer materials. | **481** | $247 + 204 + 30$ | **360.75 min (~6.0 hrs)** |
| **Quarantined / Sealed** | File G (`07_blind_test_SEALED`) | Frozen test benchmark (**Sealed until Phase 6**). | **100** | $100$ | **75.00 min (~1.25 hrs)** |
| **Total Review Pack** | **All 8 Files** | Comprehensive documentation audit suite. | **581** | $481 + 100$ | **435.75 min (~7.3 hrs)** |

### Reconciliation of Previous Tier 1 Count Discrepancy:
In the initial Phase 2 report, Tier 1 was stated as 247 rows with ~185 minutes, but the text mistakenly listed `A, B, C, D`. The actual sum of A+B+C+D is $87 + 60 + 77 + 45 = \mathbf{269}$ rows. The row count 247 was derived from $A (87) + B (60) + E (100) = \mathbf{247}$, but the text description erroneously swapped E for C+D. The corrected tiers now strictly follow the Phase 2 specification: Tier 1 = A + B + E (247 rows, 185 min), Tier 2 = C + D + F (204 rows, 153 min), Tier 3 = H (30 rows, 22.5 min).

