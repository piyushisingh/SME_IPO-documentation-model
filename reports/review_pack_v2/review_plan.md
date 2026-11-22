# SME IPO Section Classification — Review Plan (v2)

**Version:** 2.0 (Phase 2 Corrected Review Pack)  
**Date:** 2026-11-21  
**Protocol:** Tiered human review sequence with strict blinding and test set isolation  
**Unit Rate:** 45 seconds per row (0.75 minutes/row)

---

## 1. Review Tiers and Time Allocation

| Tier | Workbooks Included | Target Scope | Rows | Time (Min) | Est. Hours |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Tier 1 (High Priority)** | `01_chapter_mapping_review`<br>`02_ambiguous_review`<br>`05_blind_validation` | Taxonomy calibration, ambiguous master sections, and blind validation benchmark. | **247** | **185.25 min** | **~3.1 hrs** |
| **Tier 2 (Medium Priority)** | `03_queue_labeled`<br>`04_segmentation_check`<br>`06_blind_rule_audit` | Targeted override inspection, segmentation defect verification, and train rule audit. | **200** | **150.00 min** | **~2.5 hrs** |
| **Tier 3 (Consistency)** | `08_consistency_check` | Intra-rater agreement audit (30 disguised validation duplicates). | **30** | **22.50 min** | **~0.38 hrs** |
| **Total Active Review** | **Tiers 1, 2, and 3** | All unsealed reviewer materials. | **477** | **357.75 min** | **~6.0 hrs** |
| **Quarantined / Sealed** | `sealed/07_blind_test_SEALED` | Frozen test benchmark (**Sealed until Phase 6**). | **100** | **75.00 min** | **~1.25 hrs** |
| **Total Pack** | **All 8 Files** | Complete audit suite. | **577** | **432.75 min** | **~7.2 hrs** |

---

## 2. Execution Guidelines
- **Tier 1 must be completed first** to establish ground truth on chapter titles and baseline validation performance.
- **Tier 3 should be reviewed at least 24 hours after File E** to provide a valid intra-rater consistency measure.
- **File G remains sealed** under `sealed/` and must not be reviewed until model and rule changes are frozen in Phase 6.
