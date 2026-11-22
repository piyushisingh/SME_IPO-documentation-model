# SME IPO Documentation AI — Project Operational Rules

This document establishes the binding operational rules for developing, training, and evaluating models in the SME IPO Documentation AI workspace. Every agent and contributor must adhere to these rules without exception.

---

## 1. Immutability & Workspace Protection
1. **Never Overwrite or Delete Existing Checkpoints:**
   - `checkpoints/phase1/`
   - `checkpoints/section_classifier_v1/`
   - `checkpoints/section_classifier_v2/`
   - `checkpoints/section_classifier_v3/`
   Any new model training run must write to a distinct, versioned directory (e.g., `checkpoints/section_classifier_v4/`).
2. **Never Overwrite or Mutate Historical Datasets:**
   - V1/V2 datasets: `data/train_real.jsonl`, `data/validation_real.jsonl`, `data/eval_real.jsonl`, `data/sections.json`, `data/structured_ipo_data.json`.
   - V3 datasets: `data/train_v3.jsonl`, `data/validation_v3.jsonl`.
   - V4 source exports: `data/claude_v4_export/`.
3. **Never Delete Project Files:**
   - No deletion of scripts, logs, manifests, or historical assets without explicit user authorization.
4. **Zero Modifications to Android / Mobile UI Code:**
   - Under no circumstances may Android project files, flutter apps, or front-end UI files be edited or modified during backend model/dataset work.
5. **Backups and Verification:**
   - Baseline backups are stored in `backups/pre_v4_baseline/`.
   - Run `python verify_protection.py` before and after structural changes to ensure zero drift.

---

## 2. Sequence Length & Token Management
1. **Guaranteed Completion Preservation:**
   - When training with `completion_only_loss=True`, the target classification label is placed at the end of the sequence. If the sequence is right-truncated, the target is destroyed.
   - Training scripts must enforce `max_seq_length >= 1024` or implement explicit text-body truncation that guarantees the heading and completion tokens are never truncated.
2. **Evaluation Truncation Prohibition:**
   - Evaluation sequence length must NOT be set to `256` (which truncates 65.4% of test inputs).
   - Evaluation sequence length must match the training context window (`max_seq_length >= 1024`) to evaluate actual prospectus content.

---

## 3. Label Quality & Data Governance
1. **Distinguish Rule-Derived vs Human-Verified Labels:**
   - Over 90% of V4 labels are rule-derived by inheriting TOC chapter names.
   - Do not describe evaluation scores as "accuracy against ground truth" — describe them as agreement with heuristic extraction rules until human-verified.
2. **Frozen Test Split:**
   - `data/claude_v4_extracted/data/rebuilt_v4/test.jsonl` (and `data/claude_v4_export/test.jsonl`) is frozen on `ideas_electricals`.
   - Never tune rules, prompts, or hyperparameters on the test set.
3. **Audit Review Queue:**
   - High-risk records in `review_queue.jsonl` (keyword overrides, content mismatches) must be handled transparently before claiming production readiness.

---

## 4. Execution & Phase Gate Governance
1. **Never Start Model Training Without Explicit Approval:**
   - Training runs consume significant GPU compute and time. Always present configuration and data plans for user approval before starting training.
2. **Phase Progression Gating:**
   - Progress through the recovery plan step-by-step (Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4).
   - Do not jump ahead or combine phases without confirmation.



## 5. Mandatory Phase Rules & Governance Protocols
(1) inspect before editing; never invent paths/counts/metrics/results;
(2) never modify or overwrite V1/V2/V3 checkpoints, original datasets or existing scripts;
(3) new experiments write only to new directories;
(4) never delete without my explicit approval;
(5) no training until readiness gates pass AND I approve;
(6) verify every change with tests and report command + exit status;
(7) report exact files created/modified and why;
(8) stop and ask me if something could cause data loss or invalidate an experiment;
(9) no Android/UI changes;
(10) the V4 test file (ideas_electricals) is FROZEN: no script may use it for tuning, model selection, threshold choice or rule development;
(11) separate measured facts from interpretation in every report;
(12) every phase ends with a PASS/FAIL gate checklist.
(13) every number in a chat report must come from a saved script-output file (JSON/CSV/MD under reports/) and the report must name that file; no hand-typed figures or tables;
(14) use Python's unittest for tests (do not install pytest);
(15) any file under reports/review_pack*, answer keys and the sealed folder must never be read by training/evaluation code.
