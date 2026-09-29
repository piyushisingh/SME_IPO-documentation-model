"""
LoRA fine-tuning script for the Setu offer-document model.
Works for Gemma 3 1B now, and scales to larger models (4B, 8B) by
changing --model_name and adjusting batch size / sequence length.

Expected dataset format (JSONL, one example per line):
    {"instruction": "Is there any pending litigation against the company?",
     "input": "Company: Acme Textiles Ltd. Forum: Delhi High Court. Amount: Rs 12,00,000. Status: Pending.",
     "output": "There is one pending litigation against the Company before the Delhi High Court involving a claim of Rs 12,00,000, the outcome of which remains uncertain."}

- "instruction" = the plain-language question/fact being asked for
- "input"       = the confirmed facts/context (can be empty string if not needed)
- "output"      = the drafted clause text you want the model to learn to produce

Usage:
    python train.py --dataset data/train.jsonl --output_dir checkpoints/phase1

To resume an interrupted run:
    python train.py --dataset data/train.jsonl --output_dir checkpoints/phase1 --resume
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="unsloth/gemma-3-1b-it",
                    help="Base model to fine-tune. Swap to unsloth/gemma-3-4b-it or "
                         "unsloth/Meta-Llama-3.1-8B-Instruct when scaling up.")
    p.add_argument("--dataset", type=str, required=True,
                    help="Path to your JSONL dataset file.")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Where checkpoints are saved locally, e.g. checkpoints/phase1")
    p.add_argument("--max_seq_length", type=int, default=1024,
                    help="Lower this (e.g. 512) if you hit out-of-memory errors, "
                         "especially when scaling to bigger models.")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--per_device_train_batch_size", type=int, default=2,
                    help="Drop to 1 if you hit out-of-memory, especially on bigger models.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4,
                    help="Effective batch size = per_device_batch_size * this. "
                         "Increase this instead of batch size if VRAM is tight.")
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--save_steps", type=int, default=50,
                    help="Checkpoint frequency. Lower for small datasets, "
                         "higher (e.g. 200-500) once your dataset is large.")
    p.add_argument("--save_total_limit", type=int, default=3,
                    help="Keep only the N most recent local checkpoints to save disk.")
    p.add_argument("--eval_holdout_fraction", type=float, default=0.1,
                    help="Fraction of data held out for eval. See the note in the "
                         "code below about holding out by DOCUMENT, not by row.")
    p.add_argument("--push_to_hub", action="store_true",
                    help="Auto-push every checkpoint to your HF Hub model repo.")
    p.add_argument("--hub_model_id", type=str, default=None,
                    help="e.g. yourusername/gemma3-1b-offerdoc-phase1")
    p.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint in output_dir if one exists.")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer, SFTConfig

    # ---- 1. Load base model in 4-bit ----
    print(f"Loading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    # Gemma's chat template — swap "gemma-3" to "llama-3.1" if you move to Llama later
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # ---- 2. Attach LoRA adapters ----
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # key memory saver, especially for bigger models
        random_state=3407,
    )

    # ---- 3. Load and format dataset ----
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    def format_example(example):
        messages = [
            {"role": "user", "content": f"{example['instruction']}\n\n{example.get('input', '')}".strip()},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(format_example)

    # NOTE: this is a random split. Once your data has multiple examples from the
    # SAME source document, switch to holding out entire documents for eval instead
    # of random rows — otherwise the model can "cheat" by seeing similar examples
    # from the same company during training. See earlier discussion on this.
    split = dataset.train_test_split(test_size=args.eval_holdout_fraction, seed=3407)
    train_dataset, eval_dataset = split["train"], split["test"]
    print(f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")

    # ---- 4. Training configuration ----
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="every_save" if args.push_to_hub else "end",
    )

    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--push_to_hub requires --hub_model_id, e.g. yourusername/gemma3-1b-offerdoc-phase1")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    # ---- 5. Train (with resume support) ----
    resume_checkpoint = None
    if args.resume:
        if os.path.isdir(args.output_dir):
            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                resume_checkpoint = True
                print("Resuming from latest checkpoint in", args.output_dir)
            else:
                print("--resume was set but no checkpoint found, starting fresh.")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ---- 6. Save final adapter locally ----
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final adapter saved to {final_dir}")

    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing final adapter to {args.hub_model_id} ...")
        model.push_to_hub(args.hub_model_id, token=os.getenv("HF_TOKEN"))
        tokenizer.push_to_hub(args.hub_model_id, token=os.getenv("HF_TOKEN"))
        print("Done.")


if __name__ == "__main__":
    main()
