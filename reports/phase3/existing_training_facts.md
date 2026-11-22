# Existing Training System Technical Facts (Phase 3 Audit)

**Date:** 2026-11-21  
**Scope:** Forensic analysis of `train.py`, `evaluate_model.py`, `data/build_v3_dataset.py`, and `data/audit_v3_chat_template.py`  
**JSON Machine-Readable Companion:** [`reports/phase3/existing_training_facts.json`](file:///c:/SMEIPO%20Model/reports/phase3/existing_training_facts.json)

---

## 1. Base Model & Cache Status
- **Base Model ID:** `unsloth/gemma-3-1b-it`
  - [`train.py:61`](file:///c:/SMEIPO%20Model/train.py#L61): `default="unsloth/gemma-3-1b-it"`
  - [`data/build_v3_dataset.py:19`](file:///c:/SMEIPO%20Model/data/build_v3_dataset.py#L19): `MODEL_NAME = "unsloth/gemma-3-1b-it"`
  - [`data/audit_v3_chat_template.py:16`](file:///c:/SMEIPO%20Model/data/audit_v3_chat_template.py#L16): `MODEL_NAME = "unsloth/gemma-3-1b-it"`
  - [`evaluate_model.py:36`](file:///c:/SMEIPO%20Model/evaluate_model.py#L36): `MODEL_PATH = "checkpoints/section_classifier_v3/final"`
- **Local Cache Verification:**
  - Base model repository is **cached locally** under `C:\Users\piyushi singh\.cache\huggingface\hub\models--unsloth--gemma-3-1b-it`.
  - Snapshot `5b11413a10db4e486ef16a20101fd028f8f2499c` contains all tokenizer, chat template, and config files.
  - 4-bit quant `models--unsloth--gemma-3-1b-it-unsloth-bnb-4bit` snapshot `bfae90ad90f41a9be81bd68255d733bb80c97159` contains `model.safetensors` (1,000 MB).
  - **Network Status:** Fully cached offline; zero network calls or model downloads required.

---

## 2. Model Architecture & LoRA Configuration
- **4-bit Quantization:**
  - [`train.py:292`](file:///c:/SMEIPO%20Model/train.py#L292): `load_in_4bit=True` in `FastLanguageModel.from_pretrained`.
  - [`evaluate_model.py:85`](file:///c:/SMEIPO%20Model/evaluate_model.py#L85): `load_in_4bit=True`.
- **LoRA Hyperparameters:**
  - Rank ($r$): `16` ([`train.py:96`](file:///c:/SMEIPO%20Model/train.py#L96), [`train.py:314`](file:///c:/SMEIPO%20Model/train.py#L314)).
  - Alpha ($\alpha$): `16` ([`train.py:103`](file:///c:/SMEIPO%20Model/train.py#L103), [`train.py:326`](file:///c:/SMEIPO%20Model/train.py#L326)).
  - Dropout: `0` ([`train.py:328`](file:///c:/SMEIPO%20Model/train.py#L328)).
  - Target Modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` ([`train.py:316-324`](file:///c:/SMEIPO%20Model/train.py#L316-L324)).
  - Bias: `"none"` ([`train.py:330`](file:///c:/SMEIPO%20Model/train.py#L330)).
  - Gradient Checkpointing: `"unsloth"` ([`train.py:332`](file:///c:/SMEIPO%20Model/train.py#L332)).
  - Random State: `3407` ([`train.py:334`](file:///c:/SMEIPO%20Model/train.py#L334)).

---

## 3. Training Hyperparameters & Precision
- **Optimizer:** `adamw_8bit` ([`train.py:438`](file:///c:/SMEIPO%20Model/train.py#L438)).
- **Learning Rate:** `1e-4` (0.0001) ([`train.py:131`](file:///c:/SMEIPO%20Model/train.py#L131)).
- **Learning Rate Scheduler:** `cosine` ([`train.py:442`](file:///c:/SMEIPO%20Model/train.py#L442)).
- **Warmup Ratio:** `0.05` ([`train.py:424`](file:///c:/SMEIPO%20Model/train.py#L424)).
- **Weight Decay:** `0.01` ([`train.py:440`](file:///c:/SMEIPO%20Model/train.py#L440)).
- **Epochs:** `3.0` ([`train.py:124`](file:///c:/SMEIPO%20Model/train.py#L124)).
- **Per-Device Train Batch Size:** `1` ([`train.py:110`](file:///c:/SMEIPO%20Model/train.py#L110)).
- **Gradient Accumulation Steps:** `8` ([`train.py:117`](file:///c:/SMEIPO%20Model/train.py#L117)).
- **Effective Batch Size:** $1 \times 8 = 8$.
- **Precision:** `bf16=True` on RTX 4060, `fp16=False` ([`train.py:446-449`](file:///c:/SMEIPO%20Model/train.py#L446-L449)).
- **Seed:** `3407` ([`train.py:444`](file:///c:/SMEIPO%20Model/train.py#L444)).
- **Packing:** `False` (not enabled in SFTConfig).
- **Sequence Length:** `max_seq_length = 768` ([`train.py:89`](file:///c:/SMEIPO%20Model/train.py#L89), [`data/build_v3_dataset.py:21`](file:///c:/SMEIPO%20Model/data/build_v3_dataset.py#L21)).

---

## 4. Example Rendering, BOS Policy & Double-BOS Finding
- **Rendering Function:** `tokenizer.apply_chat_template` using the `gemma-3` template format ([`train.py:298`](file:///c:/SMEIPO%20Model/train.py#L298)).
- **Rendered Format:**
  ```text
  <bos><start_of_turn>user
  {instruction}

  Heading: {heading}

  Section text:
  {body}<end_of_turn>
  <start_of_turn>model
  {category}<end_of_turn>
  ```
- **BOS Handling Discovery:**
  - When `apply_chat_template(messages, tokenize=False)` renders text, it **already includes `<bos>` as literal text at the start**.
  - If tokenized with `add_special_tokens=True`, the tokenizer adds a second `<bos>`, yielding **Double BOS: `[2, 2, 105, 2364, 107]`**.
  - If tokenized with `add_special_tokens=False` or `apply_chat_template(messages, tokenize=True)`, exactly **one `<bos>` is present at index 0: `[2, 105, 2364, 107, 4335]`**.
  - **Enforced Policy:** Exactly one `<bos>` at index 0. All tokenization of rendered templates must pass `add_special_tokens=False`.

---

## 5. Completion-Only Loss Implementation
- **V3 Implementation:**
  - [`train.py:463`](file:///c:/SMEIPO%20Model/train.py#L463): `completion_only_loss=True` in `SFTConfig`.
  - [`train.py:467`](file:///c:/SMEIPO%20Model/train.py#L467): `dataset_text_field=None`.
  - Dataset mapped to `"prompt"` and `"completion"` list of message dicts ([`train.py:206-218`](file:///c:/SMEIPO%20Model/train.py#L206-L218)).
- **TRL 0.24.0 Compatibility Verification:**
  - Verified in `trl/trainer/sft_trainer.py:452`: TRL identifies conversational prompt/completion inputs, creates `completion_mask`, and sets `labels[completion_mask == 0] = -100`.
  - Only the assistant response tokens (`{category}<end_of_turn>\n`) receive active training gradients.

---

## 6. Exact V3 Instruction Text
- **Source:** [`data/train_real.jsonl`](file:///c:/SMEIPO%20Model/data/train_real.jsonl)
- **Text:** `"Classify the following SME IPO prospectus section into the most appropriate document category."`
- **Token Count:** Exactly **15 tokens**.
- **Token IDs:** `[4335, 1891, 506, 2269, 54113, 65853, 166578, 3336, 1131, 506, 1346, 6780, 2354, 8038, 236761]`.
- **Decoded Tokens:** `['Class', 'ify', ' the', ' following', ' SME', ' IPO', ' prospectus', ' section', ' into', ' the', ' most', ' appropriate', ' document', ' category', '.']`.

---

## 7. Discrepancy & Consistency Audit
- **Split Sizes:** Verified that `train_real.jsonl` has 1,864 rows, `validation_real.jsonl` has 196 rows, and `eval_real.jsonl` has 437 rows ($1864 + 196 + 437 = 2,497$).
- **Max Length:** Confirmed V3 trained with `max_seq_length=768`.
- **Evaluation Defect:** Confirmed [`evaluate_model.py:40`](file:///c:/SMEIPO%20Model/evaluate_model.py#L40) mistakenly set `MAX_SEQ_LENGTH = 256`, which truncated 65.4% of evaluation sequences and produced invalid historical metrics.

