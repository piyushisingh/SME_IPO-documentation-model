"""
src/sme_clf/audit_dataset.py
Strict, hard-checking audit tool for tokenized V4 datasets.
Exits with code 1 upon any single violation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from transformers import AutoTokenizer

from src.sme_clf.prompting import (
    CATEGORIES_17,
    SHORT_INSTRUCTION,
    PROMPT_VERSION,
    build_full_ids,
)

ROOT = Path(r"c:\SMEIPO Model")
DEFAULT_SNAPSHOT = (
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--unsloth--gemma-3-1b-it" / "snapshots" /
    "5b11413a10db4e486ef16a20101fd028f8f2499c"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Quality & Tokenization Audit")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing dataset files")
    parser.add_argument("--split-manifest", type=str, default="data/claude_v4_extracted/data/rebuilt_v4/split_manifest.json")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--tokenizer-path", type=str, default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--out-report", type=str, default="reports/phase3/dataset_audit_report.json")
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    print("=" * 70)
    print(f"RUNNING DATASET AUDIT ON: {data_dir} (Max Seq Len: {args.max_seq_len})")
    print("=" * 70)

    tok_path = Path(args.tokenizer_path)
    if not tok_path.exists():
        tok_path = DEFAULT_SNAPSHOT
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), local_files_only=True)
    bos_id = tokenizer.bos_token_id

    # Load split manifest for document-split disjointness
    split_man_p = ROOT / args.split_manifest
    split_manifest = json.load(open(split_man_p, encoding="utf-8"))
    split_doc_map = split_manifest.get("documents", {})

    violations = {
        "length_exceeded": 0,
        "target_tokens_mismatch": 0,
        "decoded_target_mismatch": 0,
        "bos_count_violation": 0,
        "loss_mask_violation": 0,
        "label_vocabulary_violation": 0,
        "duplicate_ids": 0,
        "split_document_leakage": 0
    }

    seen_ids = set()
    split_docs_observed = {}
    audit_results = {}
    decoded_unmasked_spans_samples = []

    for split_name in ["train", "validation"]:
        split_p = data_dir / f"{split_name}.jsonl"
        manifest_p = data_dir / f"{split_name}_manifest.jsonl"
        if not split_p.exists():
            print(f"[INFO] Split file {split_p.name} not found. Skipping.")
            continue

        examples = [json.loads(l) for l in open(split_p, encoding="utf-8") if l.strip()]
        manifest_records = [json.loads(l) for l in open(manifest_p, encoding="utf-8") if l.strip()] if manifest_p.exists() else []
        manifest_by_id = {m["example_id"]: m for m in manifest_records}

        print(f"\nAuditing split '{split_name}': {len(examples)} examples...")
        split_docs_observed[split_name] = set()

        for idx, ex in enumerate(examples):
            eid = ex.get("example_id", f"{split_name}_{idx}")
            
            # 1. Duplicate ID check
            if eid in seen_ids:
                violations["duplicate_ids"] += 1
            seen_ids.add(eid)

            m_rec = manifest_by_id.get(eid, {})
            doc_id = m_rec.get("document_id", "")
            if doc_id:
                split_docs_observed[split_name].add(doc_id)
                # Check document belongs to expected split in split_manifest
                expected_split = split_doc_map.get(doc_id)
                if expected_split and expected_split != split_name:
                    violations["split_document_leakage"] += 1

            prompt_msgs = ex["prompt"]
            comp_msgs = ex["completion"]
            target_label = comp_msgs[0]["content"].strip()

            # 2. Label vocabulary check
            if target_label not in CATEGORIES_17:
                violations["label_vocabulary_violation"] += 1

            # 3. Tokenize full sequence via chat template
            messages = prompt_msgs + comp_msgs
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

            full_tokens = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]

            # 4. Length check
            if len(full_tokens) > args.max_seq_len:
                violations["length_exceeded"] += 1

            # 5. BOS count check
            if full_tokens[0] != bos_id or full_tokens.count(bos_id) != 1:
                violations["bos_count_violation"] += 1

            # 6. Loss mask check (exact TRL completion_only_loss mechanism)
            # Labels: -100 for all prompt tokens, actual tokens for completion
            labels = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]
            unmasked = [t for t in labels if t != -100]
            unmasked_decoded = tokenizer.decode(unmasked, skip_special_tokens=False).strip()

            # The unmasked tokens must decode to target label followed by <end_of_turn>
            expected_unmasked_text = f"{target_label}<end_of_turn>"
            if unmasked_decoded != expected_unmasked_text:
                violations["loss_mask_violation"] += 1

            # Check that unmasked tokens end with end_of_turn token
            eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if full_tokens[-2] != eot_id and full_tokens[-1] != eot_id:
                violations["target_tokens_mismatch"] += 1

            # Save 5 sample spans from training split
            if split_name == "train" and len(decoded_unmasked_spans_samples) < 5:
                decoded_unmasked_spans_samples.append({
                    "example_id": eid,
                    "target_label": target_label,
                    "unmasked_token_ids": unmasked,
                    "unmasked_decoded": unmasked_decoded
                })

        audit_results[split_name] = {
            "total_audited": len(examples),
            "documents": sorted(list(split_docs_observed[split_name]))
        }

    # Verify train / validation document disjointness
    train_docs = split_docs_observed.get("train", set())
    val_docs = split_docs_observed.get("validation", set())
    doc_overlap = train_docs & val_docs
    if doc_overlap:
        violations["split_document_leakage"] += len(doc_overlap)

    out_report_p = ROOT / args.out_report
    out_report_p.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "data_dir": str(data_dir),
        "max_seq_len": args.max_seq_len,
        "violations": violations,
        "splits": audit_results,
        "train_validation_document_disjointness": {
            "train_documents": sorted(list(train_docs)),
            "validation_documents": sorted(list(val_docs)),
            "overlap": sorted(list(doc_overlap)),
            "disjoint": len(doc_overlap) == 0
        },
        "training_unmasked_spans_sample": decoded_unmasked_spans_samples
    }
    with open(out_report_p, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 70)
    print("AUDIT RESULTS & VIOLATION SUMMARY:")
    print("=" * 70)
    total_v = sum(violations.values())
    for k, v in violations.items():
        print(f"  {k:<35}: {v}")
    print(f"\nTOTAL VIOLATIONS: {total_v}")

    print("\nSample Decoded Unmasked Spans (Training):")
    for s in decoded_unmasked_spans_samples:
        print(f"  Example {s['example_id']}: label='{s['target_label']}' -> unmasked: {repr(s['unmasked_decoded'])}")

    if total_v > 0:
        print(f"\n[FAIL] Audit failed with {total_v} violations! Exiting with code 1.")
        sys.exit(1)
    else:
        print("\n[PASS] All audit checks passed with ZERO violations.")
        sys.exit(0)

if __name__ == "__main__":
    main()
