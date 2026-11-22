"""
train_v4.py
Production-grade training script for Gemma 3 LoRA SME IPO Section Classifier.
Enforces safety barriers, reproducible seeds, run manifest tracking, and dry-run benchmarking.
"""

import os
import sys
import json
import time
import shutil
import random
import argparse
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Enforce environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

DEFAULT_SNAPSHOT = (
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--unsloth--gemma-3-1b-it-unsloth-bnb-4bit" / "snapshots" /
    "bfae90ad90f41a9be81bd68255d733bb80c97159"
)

PROTECTED_DIRS = [
    Path(r"c:\SMEIPO Model\checkpoints"),
    Path(r"c:\SMEIPO Model\data"),
    Path(r"c:\SMEIPO Model\backups"),
    Path(r"c:\SMEIPO Model\reports"),
    Path(r"c:\SMEIPO Model\src"),
]

def compute_sha256(filepath: Path) -> str:
    if not filepath.is_file():
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "not_a_git_repo"

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def validate_safety(output_dir: Path, is_dry_run: bool = False, resume: bool = False):
    abs_out = output_dir.resolve()
    for prot in PROTECTED_DIRS:
        prot_abs = prot.resolve()
        if abs_out == prot_abs or prot_abs in abs_out.parents:
            raise PermissionError(
                f"[SAFETY VIOLATION] Refusing to write to protected directory: {abs_out}. "
                f"Matches protected path: {prot_abs}"
            )
    
    # Check if inside checkpoints/section_classifier_v*
    if "checkpoints" in abs_out.parts:
        raise PermissionError(
            f"[SAFETY VIOLATION] Writing into checkpoints/ is strictly prohibited: {abs_out}"
        )
        
    if abs_out.exists() and any(abs_out.iterdir()):
        if not is_dry_run and not resume:
            raise FileExistsError(
                f"[SAFETY VIOLATION] Output directory exists and is not empty: {abs_out}. "
                "Specify a new directory or pass --resume."
            )

def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 3 LoRA SME IPO Section Classifier Trainer (V4)")
    parser.add_argument("--config", type=str, default="configs/train_v4_default.json", help="Path to JSON config")
    parser.add_argument("--train-file", type=str, default=None, help="Override train.jsonl path")
    parser.add_argument("--val-file", type=str, default=None, help="Override validation.jsonl path")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--dry-run", action="store_true", help="Run 20-step dry run on 64 examples in runs/_dryrun/")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--epochs", type=float, default=None, help="Override num training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    
    args = parser.parse_args()
    return args

def load_config(config_path: str) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def run_training(config: Dict[str, Any], is_dry_run: bool = False, resume: bool = False):
    set_seed(config.get("seed", 42))

    output_dir = Path(config["output_dir"])
    if is_dry_run:
        output_dir = WORKSPACE_ROOT / "runs" / "_dryrun"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        validate_safety(output_dir, is_dry_run=is_dry_run, resume=resume)
        output_dir.mkdir(parents=True, exist_ok=True)

    train_file = Path(config["train_file"])
    val_file = Path(config["val_file"])
    if not train_file.is_file():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not val_file.is_file():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    # Safety: Refuse if test.jsonl is accidentally passed
    if "test" in train_file.name.lower() or "test" in val_file.name.lower():
        raise ValueError("[SAFETY VIOLATION] Train or validation file cannot be test split!")

    # Pre-record system facts and manifest
    import unsloth
    import torch
    import transformers
    import trl
    import peft

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    manifest = {
        "timestamp_iso": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config": config,
        "is_dry_run": is_dry_run,
        "resumed": resume,
        "train_file": {
            "path": str(train_file),
            "sha256": compute_sha256(train_file)
        },
        "val_file": {
            "path": str(val_file),
            "sha256": compute_sha256(val_file)
        },
        "environment": {
            "python_version": sys.version,
            "os": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
            "gpu_name": gpu_name,
            "total_vram_gb": round(total_vram_gb, 2),
            "transformers_version": transformers.__version__,
            "trl_version": trl.__version__,
            "peft_version": peft.__version__,
            "unsloth_version": unsloth.__version__,
        }
    }

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Recorded run manifest to: {manifest_path}")

    # Load Model and Tokenizer
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    model_path = config.get("model_name_or_path")
    if not model_path or not Path(model_path).exists():
        model_path = str(DEFAULT_SNAPSHOT)

    print(f"Loading base model from: {model_path}")
    max_seq_length = config.get("max_seq_length", 1024)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=config.get("load_in_4bit", True),
        dtype=None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        target_modules=config.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.get("seed", 42),
    )

    # Load datasets
    train_dataset = load_dataset("json", data_files=str(train_file), split="train")
    val_dataset = load_dataset("json", data_files=str(val_file), split="train")

    if is_dry_run:
        train_dataset = train_dataset.select(range(min(64, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(16, len(val_dataset))))
        print(f"[DRY RUN] Subsetting to {len(train_dataset)} train and {len(val_dataset)} val examples.")

    max_steps = 20 if is_dry_run else -1
    num_epochs = 1.0 if is_dry_run else config.get("num_train_epochs", 3.0)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=config.get("learning_rate", 1e-4),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        logging_steps=1 if is_dry_run else config.get("logging_steps", 10),
        save_strategy="no" if is_dry_run else config.get("save_strategy", "epoch"),
        eval_strategy="no" if is_dry_run else config.get("eval_strategy", "epoch"),
        save_total_limit=config.get("save_total_limit", 3),
        optim=config.get("optim", "adamw_8bit"),
        weight_decay=config.get("weight_decay", 0.01),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        seed=config.get("seed", 42),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none",
        max_length=max_seq_length,
        completion_only_loss=True,
        dataset_text_field=None,
        dataloader_num_workers=0,
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    
    t0 = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume)
    t1 = time.perf_counter()

    elapsed_sec = t1 - t0
    peak_vram_bytes = torch.cuda.max_memory_reserved(0) if torch.cuda.is_available() else 0
    peak_vram_gb = peak_vram_bytes / (1024**3)
    vram_headroom_gb = total_vram_gb - peak_vram_gb

    stats = {
        "is_dry_run": is_dry_run,
        "max_seq_length": max_seq_length,
        "total_steps": trainer.state.global_step,
        "elapsed_seconds": round(elapsed_sec, 2),
        "seconds_per_step": round(elapsed_sec / max(1, trainer.state.global_step), 3),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "total_vram_gb": round(total_vram_gb, 2),
        "vram_headroom_gb": round(vram_headroom_gb, 3),
        "train_loss": round(float(train_result.training_loss), 4) if hasattr(train_result, "training_loss") else None,
    }

    print("\n" + "=" * 60)
    print("TRAINING / DRY-RUN METRICS SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60 + "\n")

    if is_dry_run:
        # Write to reports/phase3/dryrun_results_{max_seq_length}.json
        rep_path = WORKSPACE_ROOT / "reports" / "phase3" / f"dryrun_result_{max_seq_length}.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Dry run result written to: {rep_path}")

    return stats

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    if args.train_file:
        cfg["train_file"] = args.train_file
    if args.val_file:
        cfg["val_file"] = args.val_file
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_seq_len:
        cfg["max_seq_length"] = args.max_seq_len
    if args.epochs:
        cfg["num_train_epochs"] = args.epochs
    if args.batch_size:
        cfg["per_device_train_batch_size"] = args.batch_size
    if args.learning_rate:
        cfg["learning_rate"] = args.learning_rate

    run_training(cfg, is_dry_run=args.dry_run, resume=args.resume)

if __name__ == "__main__":
    main()
