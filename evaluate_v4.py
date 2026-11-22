"""
evaluate_v4.py
Unified, production-grade evaluation harness for SME IPO Section Classification.
Enforces strict test set quarantine, prompt single source of truth, dual inference modes
(constrained scoring vs diagnostic generation), and comprehensive metrics reporting.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

DEFAULT_SNAPSHOT = (
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--unsloth--gemma-3-1b-it-unsloth-bnb-4bit" / "snapshots" /
    "bfae90ad90f41a9be81bd68255d733bb80c97159"
)

TEST_ACCESS_LOG = WORKSPACE_ROOT / "reports" / "test_access_log.jsonl"
DUPLICATE_CLUSTERS_CSV = WORKSPACE_ROOT / "reports" / "v4_forensics" / "duplicate_clusters.csv"

def compute_sha256(filepath: Path) -> str:
    if not filepath.is_file():
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def validate_test_guard(data_path: Path, args: argparse.Namespace):
    """
    Enforces strict test set quarantine:
    Refuses any path matching test.jsonl, ideas_electricals, sealed/, review_pack*, or answer_key
    unless explicit confirmation flags are provided.
    """
    p_str = str(data_path.resolve()).lower()
    
    # 1. Review Pack and Answer Key blocks (Strictly forbidden always)
    if "review_pack" in p_str or "sealed" in p_str or "answer_key" in p_str:
        raise PermissionError(
            f"[SAFETY VIOLATION] Evaluator refuses to read review pack / sealed / answer key files: {data_path}"
        )

    # 2. Test Split Detection
    is_test = False
    if "test" in data_path.name.lower():
        is_test = True
    
    # Inspect content or document_id if jsonl
    if data_path.is_file() and data_path.suffix == ".jsonl":
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if "ideas_electricals" in first_line:
                    is_test = True
        except Exception:
            pass

    if is_test:
        if not args.final_test or args.confirm != "I understand the test set is single-use":
            raise PermissionError(
                "[TEST SET QUARANTINE] Access to frozen test set requires BOTH --final-test "
                "AND --confirm \"I understand the test set is single-use\"."
            )

        # Check repeat evaluation on same model hash
        model_hash = compute_sha256(Path(args.model_path) / "adapter_model.safetensors") if Path(args.model_path).exists() else ""
        if TEST_ACCESS_LOG.exists() and model_hash:
            with open(TEST_ACCESS_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("model_file_hash") == model_hash and not args.allow_repeat:
                            raise PermissionError(
                                f"[TEST SET SINGLE-USE VIOLATION] Model {args.model_path} (hash: {model_hash}) "
                                "has already been evaluated on the test set. Pass --allow-repeat to override."
                            )
                    except json.JSONDecodeError:
                        continue

        # Log authorized access
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_path": str(args.model_path),
            "model_file_hash": model_hash,
            "data_hash": compute_sha256(data_path),
            "command": " ".join(sys.argv)
        }
        with open(TEST_ACCESS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SME IPO Section Classification Model (V4)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to fine-tuned model checkpoint/adapter")
    parser.add_argument("--base-model", type=str, default=str(DEFAULT_SNAPSHOT), help="Base model path or Hugging Face ID")
    parser.add_argument("--data-file", type=str, default=None, help="Direct path to evaluation JSONL file")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing dataset splits")
    parser.add_argument("--split-name", type=str, default="validation", help="Split name if data-dir is used")
    parser.add_argument("--max-seq-len", type=int, required=True, help="Maximum sequence length (REQUIRED, no default)")
    parser.add_argument("--variant", type=str, default="full", choices=["full", "no_heading"], help="Prompt variant")
    parser.add_argument("--view", type=str, default="all", choices=["all", "dedup", "boilerplate_only", "long_only", "short_only"], help="Evaluation subset view")
    parser.add_argument("--mode", type=str, default="constrained", choices=["constrained", "generate"], help="Prediction mode")
    parser.add_argument("--prompt-version", type=str, default="v4_short_15tok", help="Prompt version tag")
    parser.add_argument("--out-dir", type=str, required=True, help="Must be a new, empty output directory")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap resamples for 95% CI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation examples (e.g. for smoke test)")
    
    # Test guard arguments
    parser.add_argument("--final-test", action="store_true", help="Permit evaluation on test set")
    parser.add_argument("--confirm", type=str, default=None, help="Explicit confirmation string for test set access")
    parser.add_argument("--allow-repeat", action="store_true", help="Allow repeat test set evaluation on same model")
    
    return parser.parse_args()

def load_evaluation_data(args) -> Tuple[List[dict], Path]:
    if args.data_file:
        data_path = Path(args.data_file)
    elif args.data_dir:
        data_path = Path(args.data_dir) / f"{args.split_name}.jsonl"
    else:
        raise ValueError("Either --data-file or --data-dir + --split-name must be specified!")

    if not data_path.is_file():
        raise FileNotFoundError(f"Evaluation data file not found: {data_path}")

    validate_test_guard(data_path, args)

    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if args.limit:
        examples = examples[:args.limit]

    return examples, data_path

def filter_view(examples: List[dict], view: str, manifest_path: Optional[Path]) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Partitions evaluation examples into views:
    - 'all': All examples
    - 'dedup': Exclude examples whose duplicate cluster also contains a TRAIN example
    - 'boilerplate_only': The excluded duplicate examples
    - 'long_only': Examples whose body was truncated at the chosen cap
    - 'short_only': The rest (untruncated)
    """
    if view == "all":
        return examples, {"view": "all", "count": len(examples)}

    # Read manifest if available to check truncation
    manifest_by_id = {}
    if manifest_path and manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                manifest_by_id[r["example_id"]] = r

    # Read duplicate clusters
    cross_split_dups = set()
    if DUPLICATE_CLUSTERS_CSV.exists():
        import csv
        with open(DUPLICATE_CLUSTERS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # If cluster has both train and val/test
                splits = set(row.get("splits", "").split("+"))
                if "train" in splits and ("validation" in splits or "test" in splits):
                    eids = row.get("example_ids", "").split(";")
                    for eid in eids:
                        cross_split_dups.add(eid.strip())

    filtered = []
    for ex in examples:
        eid = ex.get("example_id", "")
        m = manifest_by_id.get(eid, {})
        is_trunc = m.get("body_truncated", False)
        is_dup_with_train = (eid in cross_split_dups)

        if view == "dedup" and not is_dup_with_train:
            filtered.append(ex)
        elif view == "boilerplate_only" and is_dup_with_train:
            filtered.append(ex)
        elif view == "long_only" and is_trunc:
            filtered.append(ex)
        elif view == "short_only" and not is_trunc:
            filtered.append(ex)

    return filtered, {
        "view": view,
        "count": len(filtered),
        "total_before_filter": len(examples)
    }

def main():
    args = parse_args()

    # Safety: Refuse non-empty out-dir
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"[SAFETY VIOLATION] Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.sme_clf.prompting import (
        CATEGORIES_17,
        SHORT_INSTRUCTION,
        build_prompt_content,
        strip_leading_heading,
    )
    from src.sme_clf.metrics import (
        compute_classification_metrics,
        bootstrap_confidence_interval,
        save_confusion_matrix_csv,
        save_confusion_matrix_png,
    )
    from src.sme_clf.predictors import LLMPredictor

    # Load data
    raw_examples, data_path = load_evaluation_data(args)
    manifest_path = data_path.parent / f"{data_path.stem}_manifest.jsonl"
    
    # Filter view
    eval_examples, view_info = filter_view(raw_examples, args.view, manifest_path)
    print(f"Loaded {len(eval_examples)} examples for view '{args.view}' from {data_path}")

    # Load Model and Tokenizer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import torch

    print(f"Loading base model from: {args.base_model}")
    print(f"Loading adapter from: {args.model_path}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    base_model_path = args.base_model
    if not Path(base_model_path).exists():
        base_model_path = str(DEFAULT_SNAPSHOT)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        local_files_only=True
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    
    # Attach fine-tuned adapter if path provided and not base model
    if Path(args.model_path).exists() and Path(args.model_path).resolve() != Path(base_model_path).resolve():
        model.load_adapter(str(args.model_path))
        
    FastLanguageModel.for_inference(model)

    predictor = LLMPredictor(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )

    predictions = []
    y_true = []
    y_pred = []
    invalid_count = 0

    t0 = time.perf_counter()

    for idx, ex in enumerate(eval_examples):
        eid = ex.get("example_id", f"ex_{idx}")
        doc_id = ex.get("document_id", eid.split("-")[0] if "-" in eid else "unknown")

        # Extract target/gold
        if "completion" in ex and isinstance(ex["completion"], list):
            gold = ex["completion"][0]["content"].strip()
        else:
            gold = ex.get("output") or ex.get("category", "")

        # Extract prompt text
        if "prompt" in ex and isinstance(ex["prompt"], list):
            prompt_content = ex["prompt"][0]["content"]
        else:
            heading = ex.get("heading") or ex.get("original_heading") or ""
            body = ex.get("input") or ex.get("section_text") or ""
            prompt_content = f"{SHORT_INSTRUCTION}\n\n{build_prompt_content(heading, body, variant=args.variant)}"

        if args.mode == "constrained":
            pred, conf, score_vec, raw_lps = predictor.score_candidates_constrained(prompt_content)
            is_valid = True
        else:
            pred, is_valid = predictor.generate_diagnostic(prompt_content)
            conf = None
            score_vec = None
            if not is_valid:
                invalid_count += 1

        y_true.append(gold)
        y_pred.append(pred)

        predictions.append({
            "example_id": eid,
            "document_id": doc_id,
            "gold": gold,
            "pred": pred,
            "valid": is_valid,
            "confidence": conf,
            "score_vector": score_vec,
            "variant": args.variant,
            "view": args.view,
            "model_id": args.model_path
        })

    t1 = time.perf_counter()
    elapsed = t1 - t0
    examples_per_sec = round(len(eval_examples) / max(0.001, elapsed), 3)

    # Save predictions.jsonl
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # Compute Metrics
    metrics = compute_classification_metrics(y_true, y_pred, labels=CATEGORIES_17)
    metrics["view_info"] = view_info
    metrics["speed_examples_per_sec"] = examples_per_sec
    metrics["elapsed_seconds"] = round(elapsed, 2)
    metrics["mode"] = args.mode
    metrics["invalid_count"] = invalid_count
    metrics["invalid_rate"] = round(invalid_count / len(eval_examples), 4)

    # Bootstrap 95% CI
    bootstrap_res = bootstrap_confidence_interval(y_true, y_pred, labels=CATEGORIES_17, n_resamples=args.bootstrap, seed=args.seed)
    metrics["bootstrap_95_ci"] = bootstrap_res

    # Save metrics.json
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save Confusion Matrix CSV & PNG
    cm = np.array(metrics["confusion_matrix"])
    cm_csv_path = out_dir / "confusion_matrix.csv"
    save_confusion_matrix_csv(cm, CATEGORIES_17, cm_csv_path)

    cm_png_path = out_dir / "confusion_matrix.png"
    save_confusion_matrix_png(cm, CATEGORIES_17, cm_png_path)

    # Save Summary MD
    is_smoke = "smoke" in str(out_dir).lower()
    summary_path = out_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        if is_smoke:
            f.write("# SMOKE TEST — NOT A RESULT\n\n")
        f.write("# SME IPO Section Classification Evaluation Summary\n\n")
        f.write(f"- **Model Path:** `{args.model_path}`\n")
        f.write(f"- **Data Source:** `{data_path}`\n")
        f.write(f"- **View:** `{args.view}` ({view_info['count']} examples)\n")
        f.write(f"- **Mode:** `{args.mode}`\n")
        f.write(f"- **Accuracy:** `{metrics['accuracy']}`\n")
        f.write(f"- **Macro-F1 (Supported):** `{metrics['macro_f1_supported']}`\n")
        f.write(f"- **Macro-F1 (All 17):** `{metrics['macro_f1_all_17']}`\n")
        f.write(f"- **Weighted-F1:** `{metrics['weighted_f1']}`\n")
        f.write(f"- **Speed:** `{examples_per_sec} examples/second`\n\n")
        f.write("### Bootstrap 95% Confidence Intervals:\n")
        f.write(f"- Accuracy: `[{bootstrap_res['accuracy_ci_95'][0]}, {bootstrap_res['accuracy_ci_95'][1]}]`\n")
        f.write(f"- Macro-F1: `[{bootstrap_res['macro_f1_ci_95'][0]}, {bootstrap_res['macro_f1_ci_95'][1]}]`\n\n")
        f.write(f"> [!WARNING] {bootstrap_res['correlation_warning']}\n\n")

    print(f"\n[DONE] Evaluation complete. Saved outputs to: {out_dir}")
    print(f"Accuracy: {metrics['accuracy']} | Macro-F1: {metrics['macro_f1_supported']} | Speed: {examples_per_sec} ex/s")

if __name__ == "__main__":
    main()
