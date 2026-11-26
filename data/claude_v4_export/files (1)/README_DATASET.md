# data/rebuilt_v4 — how to use

**Provisional dataset.** Labels are rule-derived from chapter titles and unverified. Do not describe results as accuracy against human truth. No tokenizer audit has been run.

## Master vs model-ready
- **Master (source of truth):** `master_sections.jsonl` — all sections, full text, provenance, flags. Edit/review labels *here* (or in a copy), then regenerate derived files.
- **Model-ready (derived):** `train.jsonl`, `validation.jsonl`, `test.jsonl` — instruction/input/output examples built from the master by `scripts/11_build_model_ready.py`. Provenance for each example is in `model_ready_manifest.jsonl` (join on `example_id`).

## Load (Python)
```python
import json
def load(p): return [json.loads(l) for l in open(p, encoding='utf-8')]
train, val, test = (load(f'data/rebuilt_v4/{s}.jsonl') for s in ('train','validation','test'))
master = load('data/rebuilt_v4/master_sections.jsonl')
```

## Before training
1. Run `python scripts/14_audit_tokenizer.py --model <model> --max-seq-len <N> --data-dir data/rebuilt_v4` and read `tokenizer_audit_report.json`.
2. Review `review_queue.jsonl` and spot-check labels using `annotation_guidelines.md`.
3. Keep `test.jsonl` frozen (Ideas Electricals). See `split_manifest.json` for limits and leave-one-document-out folds.
4. Read `data_quality_report.md` §4–§9.

## Regenerate
Scripts in `scripts/` read the six source PDFs from the upload folder (paths are constants at the top of `01_extract_pages.py` and `03/04`); edit them for your machine. Nothing in this folder modifies your existing project files.
