"""
Smoke test for the local Gemma 3 1B fine-tuning environment.
Run this once after setup to confirm everything is wired up correctly
before writing any real training code.

Usage:
    python test_setup.py
"""

import os
import sys


def check(label, fn):
    """Run a check, print pass/fail, and return whether it passed."""
    try:
        result = fn()
        print(f"[PASS] {label}: {result}")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        return False


def main():
    print("=" * 60)
    print("Environment smoke test")
    print("=" * 60)

    all_ok = True

    # 1. PyTorch + CUDA
    def torch_check():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — check your PyTorch/CUDA install")
        return f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"

    all_ok &= check("PyTorch + CUDA", torch_check)

    # 2. VRAM sanity check (RTX 4060 laptop should show ~8GB)
    def vram_check():
        import torch
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"{total_gb:.1f} GB total VRAM"

    all_ok &= check("VRAM detected", vram_check)

    # 3. Unsloth import
    def unsloth_check():
        from unsloth import FastLanguageModel  # noqa: F401
        return "imported OK"

    all_ok &= check("Unsloth import", unsloth_check)

    # 4. Hugging Face token loaded from .env
    def hf_token_check():
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not found — check your .env file exists and is loaded")
        return f"token found (starts with {token[:6]}...)"

    all_ok &= check("Hugging Face token (.env)", hf_token_check)

    # 5. datasets / transformers / peft / trl import
    def libs_check():
        import datasets, transformers, peft, trl  # noqa: F401
        return (
            f"datasets {datasets.__version__}, "
            f"transformers {transformers.__version__}, "
            f"peft {peft.__version__}, "
            f"trl {trl.__version__}"
        )

    all_ok &= check("Core libraries", libs_check)

    # 6. PDF parsing libraries (for dataset building)
    def pdf_check():
        import pdfplumber, fitz  # fitz = PyMuPDF
        return "pdfplumber + PyMuPDF OK"

    all_ok &= check("PDF libraries", pdf_check)

    print("=" * 60)
    if all_ok:
        print("All checks passed. Environment is ready for training.")
    else:
        print("Some checks failed — fix these before writing training code.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()