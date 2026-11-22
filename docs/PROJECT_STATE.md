# Project State Report — SME IPO Documentation AI

**Date:** 2026-11-20  
**Status:** Phase 0 Complete — Workspace Protection & Inventory Verified  
**Active Git Branch:** `main`  

---

## 1. Hardware & Software Environment

| Component | Specification / Detected Value | Status |
| :--- | :--- | :---: |
| **Operating System** | Windows 11 (AMD64) | OK |
| **GPU Hardware** | NVIDIA GeForce RTX 4060 Laptop GPU | OK |
| **GPU Total VRAM** | 8.0 GB (8,188 MB) | OK |
| **Free Disk Space (C:)** | ~228.4 GB | OK |
| **Python Virtualenv** | `C:\SMEIPO Model\unsloth_env` | OK |
| **Python Version** | 3.12.10 (64-bit) | OK |
| **PyTorch Version** | 2.6.0+cu128 (CUDA 12.8 enabled) | OK |
| **Hugging Face Token** | Loaded from `.env` | OK |
| **Unsloth / TRL / Peft** | Unsloth 2025.x / TRL 0.24.0 / PEFT 0.14.0 | OK |
| **Transformers** | Transformers 4.49.0 | OK |

---

## 2. Checkpoint Inventory & Provenance

All checkpoints reside under `checkpoints/` and are fully preserved:

| Checkpoint Name | Base Model | Tuning Method | Objective | Preserved Files |
| :--- | :--- | :--- | :--- | :--- |
| `phase1` | Gemma 3 1B | LoRA (Unsloth) | Offer document drafting / text generation | final adapter, checkpoints 1-2 |
| `section_classifier_v1` | Gemma 3 1B (4-bit) | LoRA (r=16, alpha=16) | Section classification (standard loss on prompt+completion) | final adapter, checkpoints 400, 450, 466 |
| `section_classifier_v2` | Gemma 3 1B (4-bit) | LoRA (r=16, alpha=16) | Completion-only loss (hit 768 token right-truncation bug) | final adapter, checkpoints 500, 600, 699 |
| `section_classifier_v3` | Gemma 3 1B (4-bit) | LoRA (r=16, alpha=16) | Completion-only loss with pre-truncated `train_v3.jsonl` | final adapter, checkpoints 500, 600, 699 |

---

## 3. Dataset Inventory

| Dataset Version | Directory / Path | Record Count | Schema Keys | Verification Hash Status |
| :--- | :--- | :---: | :--- | :---: |
| **V1/V2 Raw Data** | `data/sections.json`<br>`data/structured_ipo_data.json` | 1,836 sections | `heading, category, text, page_start` | Checked in baseline manifest |
| **V2 Model-Ready** | `data/train_real.jsonl`<br>`data/validation_real.jsonl`<br>`data/eval_real.jsonl` | Train: 1,864<br>Val: 253<br>Eval: 437 | `instruction, input, output, metadata` | Checked in baseline manifest |
| **V3 Model-Ready** | `data/train_v3.jsonl`<br>`data/validation_v3.jsonl` | Train: 1,864<br>Val: 253 | `prompt, completion` | Checked in baseline manifest |
| **V4 Export (Loose)** | `data/claude_v4_export/` | Master: 5,997<br>Train: 3,801<br>Val: 731<br>Test: 830 | `example_id, instruction, input, output, label_status` | Checked in baseline manifest |
| **V4 Extracted Bundle** | `data/claude_v4_extracted/data/rebuilt_v4/` | 47 files (master, splits, ocr, 14 scripts, manifest, docs) | Complete provenance + `model_ready_manifest.jsonl` | **Matches official `CHECKSUMS.sha256`** |

---

## 4. Backups and Protection Status

- **Backup Location:** `c:\SMEIPO Model\backups\pre_v4_baseline\`
  - `root_scripts/`: `train.py`, `train copy.py`, `evaluate_model.py`, `test_setup.py`, `test_load_model.py`, `.gitignore`
  - `data/`: `train_real.jsonl`, `validation_real.jsonl`, `eval_real.jsonl`, `train_v3.jsonl`, `validation_v3.jsonl`, `build_v3_dataset.py`, audit scripts, task categories
  - `data/scripts/`: Complete extraction scripts
  - `checkpoints_meta/`: All adapter configs, tokenizer configs, chat templates, and model READMEs
  - `BASELINE_CHECKSUMS.sha256`: SHA-256 for all 231 workspace files
- **Verification Script:** `verify_protection.py` (executes in ~10s and verifies 100% hash parity).

---

## 5. Phase 0 Gate Checklist

- [x] Workspace complete inspection performed.
- [x] No model training initiated.
- [x] Existing V1/V2/V3 datasets and checkpoints 100% preserved.
- [x] Zero files deleted.
- [x] Android project / UI code completely untouched.
- [x] Full backup created in `backups/pre_v4_baseline/`.
- [x] Master SHA-256 baseline generated (`BASELINE_CHECKSUMS.sha256`).
- [x] V4 ZIP safely extracted to isolated directory `data/claude_v4_extracted/` without overwriting.
- [x] `verify_protection.py` created and passed.
- [x] `PROJECT_RULES.md` and `docs/PROJECT_STATE.md` created.
- [x] Stopped before Phase 1 awaiting user approval.

---

## 6. Phase 1 Forensic Inspection Summary

**Date:** 2026-11-21  
**Status:** Complete — All Forensic Analyses & Count Reconciliations Verified

### Key Discoveries & Findings:
1. **Mathematical Count Reconciliation:**
   - `sections.json` has exactly 2,497 sections across 8 documents ($196 + 437 + 215 + 239 + 352 + 374 + 375 + 309 = 2,497$).
   - `train_real.jsonl` (1,864) + `validation_real.jsonl` (196) + `eval_real.jsonl` (437) = 2,497.
   - Claude's citation of "1,836" was the 6-document subset excluding `Project-Om` (352) and `Shakti` (309) ($2,497 - 661 = 1,836$).
   - `validation_real.jsonl` has exactly 196 lines (Century = 196); "253" was an erroneous historical notation.
2. **V3 Exposure / Contamination:**
   - `Ideas_Electricals` (V4 frozen test set) was an active **training document** for V3 (239 examples in `train_real.jsonl`).
   - Evaluating V3 on V4 test measures training memorization. V3 on V4 validation (`Century`) is clean (0 exact matches, 5 boilerplate matches).
3. **Heading Duplication & Ablation Requirements:**
   - In 94.4% of model-ready examples, the heading is repeated verbatim at the start of the `Text:` body. Heading ablation must strip both the prompt `Heading:` line and the text prefix.
4. **Tokenizer Truncation:**
   - At `max_seq_length=768`, 129 training examples lose their completion tokens under completion-only loss. At 1024, only 23 exceed.
   - In `evaluate_model.py` (`MAX_SEQ_LENGTH = 256`), 543/830 (65.4%) of test prompts are truncated, blinding the evaluation.

### Phase 1 Files Created / Manifest:
- `env/pip_freeze.txt`: Environment package snapshot.
- `env/nvidia_smi.txt`: GPU architecture and driver state.
- `env/python_info.txt`: Python runtime and PyTorch/CUDA specs.
- `reports/test_access_log.jsonl`: Audit log of every read to `test.jsonl`.
- `reports/v4_forensics/checkpoint_weights_sha256.json`: Checkpoint weight SHA-256 manifest (44 weights).
- `reports/v4_forensics/duplicate_clusters.csv`: 80 multi-item duplicate clusters across splits.
- `reports/v4_forensics/spot_sample.md`: 60 stratified spot samples from Train & Val.
- `reports/v4_forensics/readiness_report.md`: Complete forensic readiness deliverable.

---

## 7. Phase 2 Review Pack Generation Summary

**Date:** 2026-11-21  
**Status:** Complete — All Review Workbooks, Guides, and Test Suites Verified

### Key Deliverables & Verifications:
1. **Verification Carry-Overs (Task 0):**
   - Recomputed old label distribution for 8 documents (2,497 total: `ipo_details` 449, `other` 433, `legal_and_regulatory` 353) and 6 V4 documents (1,836 total: `ipo_details` 318, `other` 300, `legal_and_regulatory` 291). Reconciled Phase 1 discrepancy (omission of `DRHP_Maxwell` in Phase 1 split summation).
   - Recomputed heading equals parent chapter title across ALL 5,362 manifest examples: 215 / 5,362 (4.01%). Reconciled Phase 1 count of 5,114 (which missed 248 forward-merged composite IDs).
   - Confirmed thresholds in `scripts/06_extraction_report.py`: `tiny_section` is `< 150` characters; `very_long_section_not_split` is `> 20000` characters. Appended Erratum to Section 11 of `reports/v4_forensics/readiness_report.md`.
2. **Review Workbooks Generated (Task 1):**
   - Generated 8 workbooks (.xlsx + UTF-8-BOM .csv twins) via deterministic generator `src/sme_clf/make_review_pack.py` (seed 42):
     - `01_chapter_mapping_review` (87 rows)
     - `02_ambiguous_review` (60 rows)
     - `03_queue_labeled` (77 rows)
     - `04_segmentation_check` (45 rows)
     - `05_blind_validation` (100 rows)
     - `06_blind_rule_audit` (82 rows)
     - `sealed/07_blind_test_SEALED` (100 rows) + `SEALED_README.txt`
     - `08_consistency_check` (30 rows)
     - `private/answer_key.json` (312 entries)
     - `sampling_manifest.json` (strata definitions, seed 42, file hashes)
   - File I skipped because V4-to-old L2 text match coverage is only 20.09% (< 50% threshold).
   - Formula safety: data_type `'s'` used on all text cells; `=,+,-,@` escaped in CSV twins.
   - Dropdown validation: dropdowns reference visible `allowed_values` sheet without hidden rows/columns.
3. **Review Guides (Task 2):**
   - `reports/review_pack/review_guide.md`: Condensed 1-page taxonomy, 6 edge-case resolution rules, 5 operational principles.
   - `reports/review_pack/review_plan.md`: Tiered review sequence (Tier 1: 247 rows / 185 min; Tier 2: 204 rows / 154 min; Tier 3: 30 rows / 23 min; Sealed: 100 rows / 75 min). Total: 581 rows / 437 min (~7.3 hours).
4. **Automated Test Suite (Task 3):**
   - `tests/test_review_pack.py`: 7 automated tests covering schema, strict blindness, reproducibility, partition/cluster constraints, stratification, formula safety, and protection.
   - Executed with exit status 0 (all 7 tests passed).
5. **Safeguarded Dependencies:**
   - `openpyxl` (3.1.5) and `et_xmlfile` (2.0.0) installed into `unsloth_env` after approved dry-run; freeze diff verified as additions only; torch/unsloth imports verified intact.
   - Pytest dry-run evaluated (3 packages); kept uninstalled per safeguards since tests cleanly execute via `unittest`.

---

## 8. Phase 2 Gate Checklist

- [x] Carry-over verifications complete and documented in `reports/review_pack/phase2_verifications.md`.
- [x] Readiness report erratum appended to Section 11 (original text untouched).
- [x] Safeguards strictly followed for `openpyxl` installation (freeze snapshots, dry-run, additions only diff, torch/unsloth imports intact).
- [x] `make_review_pack.py` created and completely reproducible (seed 42).
- [x] All 8 review workbooks created with both `.xlsx` and UTF-8-BOM `.csv` formats.
- [x] Formula injection protections implemented and verified on Excel and CSV twins.
- [x] Dropdown data validation implemented via visible `allowed_values` sheet.
- [x] Strict blindness verified: zero leakages of labels, rule IDs, or master IDs in blind workbooks.
- [x] Test split accessed strictly for File G and logged to `reports/test_access_log.jsonl`.
- [x] Sealed test workbook stored in `sealed/` with `SEALED_README.txt`.
- [x] Answer key stored strictly in `private/answer_key.json`.
- [x] `review_guide.md` and `review_plan.md` created.
- [x] `tests/test_review_pack.py` passes 100% (7/7 tests passed, exit code 0).
- [x] Protection script `verify_protection.py` re-run and 100% passed (231 baseline files + 44 weights intact).
- [x] Zero training or model inference executed.
- [x] Stopped awaiting user approval before Phase 3.

---

## 9. Phase 2 Integrity Audit & Review Pack v2 Regeneration Summary

**Date:** 2026-11-21  
**Status:** Complete — Forensic Audit Verified, Flawed Workbooks Flagged, Corrected v2 Pack Generated and Independently Verified

### Forensic Findings:
1. **`data/sections.json` Hash Reconciliation:**
   - Current live file SHA-256: `7fd0ab6176c53edb932e6de7d7505d1ded99117eb484c0ed3fbe432457d967dd`.
   - Baseline manifest hash (`BASELINE_CHECKSUMS.sha256`): `7fd0ab6176c53edb932e6de7d7505d1ded99117eb484c0ed3fbe432457d967dd`.
   - Result: 100% identical. The file has NOT changed since Phase 0.
   - The Phase 2 report's citation of `a93be9...` was a model hallucination in the report prose; the actual file in `phase2_verifications.md` correctly quoted `7fd0ab...`.
   - Category counts in `sections.json` are: `ipo_details: 449`, `other: 433`, `legal_and_regulatory: 353`, `promoters_and_management: 254`, `business_operations: 212`, `financial_information: 200`, `financial_statements: 132` (2,497 total). The Phase 2 report prose erroneously quoted `business_overview 272` and `financial_statements 338`.
2. **Review Pack v1 Flaws Identified:**
   - `05_blind_validation` and `sealed/07_blind_test_SEALED` applied round-robin budget increments that produced a flat 7 samples across all 14 large categories (e.g. `related_party_transactions` $N=10 \to n=7$ vs `ipo_details` $N=152 \to n=7$).
   - `sampling_manifest.json` completely omitted `N_h_before_filters` for all strata.
   - Forward-merged composite example IDs (`...-066+1`) were not mapped to manifest metadata, resulting in dummy rule `"none"` in File F.
   - In `reports/review_pack/`, `INVALID_DO_NOT_USE.txt` was added to both the root and `sealed/` folders.
3. **Review Pack v2 Deliverables (`reports/review_pack_v2/`):**
   - Corrected generator: `src/sme_clf/make_review_pack_v2.py` (seed 42).
   - Proportional stratified sampling implemented using Hamilton's largest remainder method with minimum 3 and cap 25.
   - Full stratum provenance recorded in `sampling_manifest.json` (`N_h_before_filters`, `N_h_eligible`, `n_h_sampled`, `inclusion_probability`).
   - File counts: A (87), B (60), C (77), D (45), E (100), F (78), G (100, sealed), H (30).
   - Corrected Review Plan: Tier 1 = A + B + E (247 rows, 185.25 min), Tier 2 = C + D + F (200 rows, 150.00 min), Tier 3 = H (30 rows, 22.50 min), Sealed G (100 rows, 75.00 min). Total pack = 577 rows, 432.75 min (~7.2 hrs).
4. **Independent Verification:**
   - Standalone test suite: `tests/verify_review_pack_v2.py` (does not import generator).
   - 10/10 unit tests passed (exit status 0).
   - `verify_protection.py` re-run and passed (231 files + 44 weights intact).

---

## 10. Phase 3 Training Pipeline and Tokenization Optimization Summary

**Date:** 2026-11-21  
**Status:** Complete — Token-Aware Builder, Safety Guards, Unit Tests, and Hardware Benchmarking Verified

### Key Deliverables & Forensic Discoveries:
1. **Existing Training Stack Forensic Facts (`reports/phase3/existing_training_facts.json`):**
   - Gemma-3 short instruction is exactly 15 tokens: `[4335, 1891, 506, 2269, 54113, 65853, 166578, 3336, 1131, 506, 1346, 6780, 2354, 8038, 236761]`.
   - V3 `apply_chat_template(..., tokenize=False)` automatically prepended `<bos>`. When tokenized with `add_special_tokens=True`, a duplicate `<bos>` was created (`[2, 2, ...]`). Fixed via strict `add_special_tokens=False` policy in `src/sme_clf/prompting.py`, guaranteeing exactly one `<bos>` at index 0.
   - Base model verified in local cache: `models--unsloth--gemma-3-1b-it-unsloth-bnb-4bit` (snapshot `bfae90ad90f41a9be81bd68255d733bb80c97159`, 1,000 MB `model.safetensors`).
2. **Prompting Single Source of Truth (`src/sme_clf/prompting.py`):**
   - Fixed 17 V4 categories, short 15-token instruction, prompt hash `43d195e09f58352e46b0a2fa10ee445b2b36e9fffa78ebae348126b158097dcb`.
   - Functions: `strip_leading_heading`, `build_prompt_content`, `build_full_ids`, `build_full_text`.
3. **Token-Aware Dataset Builder (`src/sme_clf/build_dataset.py`):**
   - Filters composite examples across all source IDs (`builder_exclusions.json`: train 15 excluded, val 5 excluded, test 6 excluded; leaving train 3,786 and val 726).
   - Generates `runs/data_v4_1024/train.jsonl` and `validation.jsonl`.
   - Computes multi-cap comparative truncation analysis (`reports/phase3/truncation_analysis.json` and `.md`).
   - 100% target preservation guaranteed (0 target tokens lost across all caps 768, 1024, 1536).
4. **Hard-Checking Dataset Auditor (`src/sme_clf/audit_dataset.py`):**
   - Audited `runs/data_v4_1024/` across 8 strict checks: length exceeded, target tokens mismatch, decoded target mismatch, BOS count, loss mask, label vocab, duplicate IDs, split leakage.
   - Result: 0 violations across all checks (`reports/phase3/dataset_audit_report.json`).
5. **Config-Driven Safety Trainer (`train_v4.py` & `configs/train_v4_default.json`):**
   - Config-driven training script with safety barriers refusing writes into `checkpoints/`, `data/`, `backups/`, `reports/`, or non-empty dirs.
   - Zero test code paths or test arguments.
   - Automatic `run_manifest.json` generation at startup.
6. **Task 5 Dry Run Hardware Benchmarking (`reports/phase3/dryrun_results.json` & `.md`):**
   - 20-step benchmark on 64-example train subset on NVIDIA RTX 4060 Laptop GPU:
     - **1024 seq len:** 168.66s (8.433s/step), peak VRAM 4.037 GB, headroom 3.958 GB, loss 4.1438.
     - **1536 seq len:** 180.91s (9.046s/step), peak VRAM 4.035 GB, headroom 3.960 GB, loss 4.1438.
   - Recommended: `max_seq_length = 1024` for production training (93.98% untruncated, 7.3% faster per step, 0 target losses).
7. **Automated Unit Testing & Protection:**
   - Standard library `unittest` suite `tests/test_phase3.py`: 6/6 tests passed (1.405s, exit code 0).
   - `verify_protection.py`: 100% passed (231 baseline files + 44 weights intact).

---

## 11. Phase 3 Gate Checklist

- [x] Rules 13, 14, 15 appended to `PROJECT_RULES.md`.
- [x] Task 0 forensic facts saved to `reports/phase3/existing_training_facts.json` and `.md`.
- [x] Base model verified available offline in local Hugging Face cache.
- [x] Task 1 `src/sme_clf/prompting.py` created with single source of truth and deterministic prompt hash.
- [x] Task 2 `src/sme_clf/build_dataset.py` created with token-aware right-truncation preserving targets.
- [x] Truncation analysis generated for 768, 1024, and 1536 in `reports/phase3/truncation_analysis.json` and `.md`.
- [x] Task 3 `src/sme_clf/audit_dataset.py` created and passed with 0 violations (`reports/phase3/dataset_audit_report.json`).
- [x] Task 4 `train_v4.py` and `configs/train_v4_default.json` created with strict safety barriers and zero test file references.
- [x] Task 5 dry run benchmarks executed for 1024 and 1536 sequence lengths (`reports/phase3/dryrun_results.json` and `.md`).
- [x] Task 6 unit tests in `tests/test_phase3.py` executed via `unittest` and 100% passed (6/6 tests, exit code 0).
- [x] Protection script `verify_protection.py` re-run and passed (all 231 baseline files and 44 checkpoint weights intact).
- [x] Stopped before full model training awaiting user approval.

---

## 12. Phase 3B Corrections & Benchmark Hardening Summary

**Date:** 2026-11-21  
**Status:** Complete — Empirical Hardware Hardening, Exact Token Aggregations, and Mask Audits Verified

### Key Corrections & Evidentiary Findings:
1. **Exact Body Token Retention (`reports/phase3/truncation_token_retention.json`):**
   - Corrected previous colloquial estimate ("keeping over 98% of tokens") with exact manifest token sums:
     - **Cap 768:** Train retention = **73.1289%** (904,049 / 1,236,240 tokens; 377 / 3,786 examples truncated, 9.9577%); Val retention = **72.0619%** (190,012 / 263,679 tokens; 95 / 726 truncated, 13.0854%).
     - **Cap 1024:** Train retention = **79.3113%** (980,478 / 1,236,240 tokens; 228 / 3,786 examples truncated, 6.0222%); Val retention = **79.5956%** (209,877 / 263,679 tokens; 57 / 726 truncated, 7.8512%).
     - **Cap 1536:** Train retention = **86.0834%** (1,064,197 / 1,236,240 tokens; 109 / 3,786 examples truncated, 2.8790%); Val retention = **87.6934%** (231,229 / 263,679 tokens; 29 / 726 truncated, 3.9945%).
2. **Worst-Case Hardware Benchmarks (`reports/phase3/dryrun_v2_results.json`):**
   - Corrected initial dry run (which used first 64 examples): tested top 16 longest examples at each cap:
     - **1024 Cap (longest 1,021 tokens):** Peak torch reserved 2.127 GB, peak nvidia-smi 2.280 GB, baseline display memory 1.177 GB, headroom = **5.716 GB**, step time 14.148s.
     - **1536 Cap (longest 1,533 tokens):** Peak torch reserved 2.783 GB, peak nvidia-smi 2.937 GB, baseline display memory 1.177 GB, headroom = **5.059 GB**, step time 12.017s.
     - Both comfortably exceed the 1.0 GB minimum headroom safety threshold.
3. **Length-Stratified Throughput & Full Training Projections (`reports/phase3/dryrun_v2_results.json`):**
   - 64-example stratified subset (8 per octile, seed 42) at 1024: throughput = 0.715 samples/sec (217.76 tokens/sec).
   - Mean step loss decreased from 3.2568 (first 5 steps) to 0.8196 (last 5 steps) (pipeline readiness check).
   - Projected 3-epoch training time (3,786 examples, 3,480,534 total tokens): **4.440 hours** (~266.4 min) by throughput; **5.580 hours** (~334.8 min) worst-case upper bound.
4. **V3 Double-BOS Verdict (`reports/phase3/v3_bos_verdict.json`):**
   - **Verdict: V3 trained with a double BOS: NO.**
   - SFTTrainer natively tokenizes conversational prompt/completion datasets via `apply_chat_template(tokenize=True)`, producing exactly `[2, 105, 2364, 107, 4335, 1891]` (single BOS). Double-BOS was strictly an artifact of the audit script `data/audit_v3_chat_template.py`. V3 should be evaluated with single BOS.
5. **Actual Batch Mask Verification (`reports/phase3/actual_batch_mask_check.json`):**
   - Unmasked tokens contain the exact label tokens followed by `[106, 107]` (`<end_of_turn>\n`).
   - Token 1 (`<eos>`) is **NOT** appended.
   - Input sequence contains **exactly 1 BOS token** (token ID 2) at index 0.
6. **Provenance of Test Exclusions (`reports/test_access_log.jsonl`):**
   - Verified that test split counts (6 excluded / 824 eligible) were derived strictly from `split_manifest.json` and component flags in `master_sections.jsonl`. The frozen file `test.jsonl` was never opened.
7. **Dataset Audits & Protection:**
   - `audit_dataset.py` re-run on `runs/data_v4_1024` and `runs/data_v4_1536`: 0 violations across all checks.
   - `verify_protection.py`: [PASS] All 231 baseline files and 44 checkpoint weights intact.

---

## 13. Phase 4 Evaluation Pipeline Implementation & Smoke Test Summary

**Date:** 2026-11-21  
**Status:** Complete — Evaluation Harness, Quarantine Guards, Metrics Engine, and Validation Smoke Test Verified

### Key Deliverables & Forensic Discoveries:
1. **Quarantine Guards & CLI Architecture (`evaluate_v4.py`):**
   - Strictly enforces evaluation safety: blocks evaluation if input file is `test.jsonl`, points to `reports/review_pack*`, or contains sealed/answer-key data without explicit override.
   - CLI flags: `--max-seq-len`, `--model-path`, `--out-dir`, `--variant`, `--view`, `--mode`, `--bootstrap`.
   - Views supported: `all`, `dedup`, `boilerplate_only`, `long_only`, `short_only`.
2. **Predictor Interface & Candidate Scoring (`src/sme_clf/predictors.py`):**
   - Constrained scoring evaluates all 17 candidate label continuations: `P(c_1, ..., c_k | prompt) = sum_{j=1}^k log P(c_j | prompt, c_1, ..., c_{j-1})`.
   - Diagnostic unconstrained generation mode measures invalid generation rate.
   - **Crucial Architecture Discovery regarding DynamicSlidingWindowLayer:** Gemma-3 uses alternating sliding window attention layers (`DynamicSlidingWindowLayer` with window size 512). In `transformers`, `DynamicSlidingWindowLayer.crop()` cannot be called once sequence length exceeds the sliding window. Furthermore, deepcopying `DynamicCache` with hooks hangs in Python recursion. Exact full-forward candidate scoring resolves this, delivering mathematically exact sequence log-probabilities without cache corruption.
3. **Pure-Python/Numpy Metrics Engine (`src/sme_clf/metrics.py`):**
   - Implements multi-class classification metrics: accuracy, macro-F1 (supported classes), all-17 macro-F1, weighted-F1, per-class precision/recall/F1, support counts (flags classes with support < 5), and confusion matrix.
   - Bootstrap 95% confidence intervals with empirical sampling (1,000 iterations) and correlation caveat warning.
4. **Validation Smoke Test (`reports/phase4/smoke_test_results.json` & `.md`):**
   - Labeled: **"SMOKE TEST — NOT A RESULT"**.
   - Evaluated 20 validation examples from Century prospectus using checkpoint `checkpoints/section_classifier_v3/final`.
   - Constrained vs Naive Equivalence: **20 / 20 argmax matches** (Target: $\ge 19/20$), **Max log-prob difference = 0.0** (Target: $< 1e-4$).
   - Performance: Constrained inference elapsed 188.72s (0.106 ex/s), Peak VRAM allocated 2.460 GB, Peak VRAM reserved 3.764 GB.
   - Diagnostic mode: 6 / 20 invalid generations (invalid rate 0.3000).
5. **Automated Unit Testing & Protection:**
   - Standard library `unittest` suite `tests/test_phase4.py`: 6/6 tests passed. All 19 project tests passed in 6.936s.
   - `verify_protection.py`: [PASS] All 231 baseline files and 44 checkpoint weights intact.

---

## 14. Phase 4 Gate Checklist

- [x] Predictor interface `src/sme_clf/predictors.py` created with constrained scoring and diagnostic generation.
- [x] Evaluation runner `evaluate_v4.py` created with required CLI flags and views (`all`, `dedup`, etc.).
- [x] Test set quarantine guard verified: blocks forbidden test paths (`test.jsonl`, `review_pack`, etc.).
- [x] Metrics engine `src/sme_clf/metrics.py` created with accuracy, macro-F1, weighted-F1, confusion matrix, and bootstrap 95% CI.
- [x] Unit tests in `tests/test_phase4.py` executed via `unittest` and 100% passed (6/6 tests).
- [x] Validation smoke test on 20 examples (`Century`) executed and labeled "SMOKE TEST — NOT A RESULT" (`runs/_smoke_phase4/` and `reports/phase4/smoke_test_results.*`).
- [x] Equivalence verified: 20/20 argmax matches ($\ge 19/20$) and max log-prob difference 0.0 ($< 1e-4$).
- [x] Protection script `verify_protection.py` re-run and passed (all 231 baseline files and 44 checkpoint weights intact).
- [x] No training started; awaiting user review and approval.



