"""
First real load test: pulls Gemma 3 1B and loads it in 4-bit via Unsloth.
This confirms your GPU can actually hold the model, not just that
the libraries import correctly.

Usage:
    python test_load_model.py

Expect the first run to take a few minutes (downloading ~1-2GB of weights).
"""

import time

import torch
from dotenv import load_dotenv

load_dotenv()  # picks up HF_TOKEN from .env


def main():
    print("=" * 60)
    print("Loading Gemma 3 1B in 4-bit via Unsloth...")
    print("=" * 60)

    from unsloth import FastLanguageModel

    start = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3-1b-it",  # instruction-tuned base
        max_seq_length=1024,                  # keep modest for your 8GB card
        load_in_4bit=True,
        dtype=None,                           # let Unsloth pick the best dtype
    )

    elapsed = time.time() - start
    print(f"\nModel loaded in {elapsed:.1f} seconds")

    # Report VRAM actually used
    if torch.cuda.is_available():
        used_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"VRAM allocated: {used_gb:.2f} GB")
        print(f"VRAM reserved:  {reserved_gb:.2f} GB")

    # Quick generation smoke test
    print("\nRunning a quick generation test...")
    FastLanguageModel.for_inference(model)  # enable fast inference mode

    prompt = "List three things a company must disclose in an IPO offer document:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n--- Model output ---")
    print(result)
    print("--- end output ---\n")

    print("=" * 60)
    print("SUCCESS: Gemma 3 1B loads and runs on your GPU.")
    print("You're ready to move on to LoRA setup and real training.")
    print("=" * 60)


if __name__ == "__main__":
    main()