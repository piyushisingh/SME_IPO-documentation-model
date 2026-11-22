"""
SME IPO SECTION CLASSIFIER — V2

Purpose:
    Fine-tune Gemma 3 1B to classify SME IPO prospectus sections.

IMPORTANT V2 CHANGE:
    Training loss is calculated ONLY on the classification completion
    (the category label), not on the input prompt/section text.

Example:

    Prompt:
        Classify this SME IPO prospectus section.

        Heading: CASH FLOW FROM OPERATING ACTIVITIES

        Section text:
        ...

    Completion:
        financial_information

This prevents the model from learning to reproduce prospectus text
instead of learning the classification task.

Dataset:
    data/train_real.jsonl

Validation:
    data/validation_real.jsonl

Final evaluation:
    data/eval_real.jsonl

V2 output:
    checkpoints/section_classifier_v2/
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train the SME IPO Section Classification Model."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/gemma-3-1b-it",
        help="Base model to fine-tune."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Training JSONL dataset."
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        required=True,
        help="Validation JSONL dataset."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints."
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=768,
        help="Maximum sequence length."
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank."
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha."
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Training batch size."
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps."
    )

    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate."
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps."
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum checkpoints to keep."
    )

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to Hugging Face Hub."
    )

    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hugging Face repository ID."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint."
    )

    return parser.parse_args()


# ============================================================
# LOAD DATASET
# ============================================================

def prepare_dataset(dataset):

    """
    Convert the existing instruction/input/output dataset into
    a prompt-completion dataset.

    IMPORTANT:
        prompt = user-side classification request
        completion = category label

    With completion_only_loss=True, only the category label
    contributes to the training loss.
    """

    def convert_example(example):

        instruction = example["instruction"].strip()

        input_text = example.get(
            "input",
            ""
        ).strip()

        output = example["output"].strip()

        prompt_text = (
            f"{instruction}\n\n"
            f"{input_text}"
        ).strip()

        return {
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": output
                }
            ]
        }

    return dataset.map(
        convert_example,
        remove_columns=dataset.column_names
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Imports inside main
    # --------------------------------------------------------

    import torch

    from datasets import load_dataset

    from unsloth import (
        FastLanguageModel,
        is_bfloat16_supported,
    )

    from unsloth.chat_templates import get_chat_template

    from trl import (
        SFTTrainer,
        SFTConfig,
    )

    # --------------------------------------------------------
    # Environment information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SME IPO SECTION CLASSIFIER — V2")
    print("=" * 70)

    print()
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print("Base model:", args.model_name)
    print("Training dataset:", args.dataset)
    print("Validation dataset:", args.eval_dataset)
    print("Output directory:", args.output_dir)

    # --------------------------------------------------------
    # 1. Load base model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 1: LOADING BASE MODEL")
    print("=" * 70)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    # Gemma 3 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    # --------------------------------------------------------
    # 2. Attach LoRA
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 2: ATTACHING LoRA")
    print("=" * 70)

    model = FastLanguageModel.get_peft_model(
        model,

        r=args.lora_r,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

        lora_alpha=args.lora_alpha,

        lora_dropout=0,

        bias="none",

        use_gradient_checkpointing="unsloth",

        random_state=3407,
    )

    # --------------------------------------------------------
    # 3. Load training dataset
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 3: LOADING DATASETS")
    print("=" * 70)

    train_raw = load_dataset(
        "json",
        data_files=args.dataset,
        split="train",
    )

    validation_raw = load_dataset(
        "json",
        data_files=args.eval_dataset,
        split="train",
    )

    print()
    print("Raw training examples:", len(train_raw))
    print("Raw validation examples:", len(validation_raw))

    # --------------------------------------------------------
    # Convert to prompt-completion format
    # --------------------------------------------------------

    train_dataset = prepare_dataset(
        train_raw
    )

    eval_dataset = prepare_dataset(
        validation_raw
    )

    print()
    print("Prepared training examples:", len(train_dataset))
    print("Prepared validation examples:", len(eval_dataset))

    # --------------------------------------------------------
    # Show one example
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMPLE TRAINING EXAMPLE")
    print("=" * 70)

    sample = train_dataset[0]

    print()
    print("PROMPT:")
    print(sample["prompt"])

    print()
    print("COMPLETION:")
    print(sample["completion"])

    # --------------------------------------------------------
    # 4. Training configuration
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 4: TRAINING CONFIGURATION")
    print("=" * 70)

    sft_config = SFTConfig(

        output_dir=args.output_dir,

        per_device_train_batch_size=(
            args.per_device_train_batch_size
        ),

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        num_train_epochs=args.num_train_epochs,

        learning_rate=args.learning_rate,

        warmup_ratio=0.05,

        logging_steps=10,

        save_strategy="steps",

        save_steps=args.save_steps,

        save_total_limit=args.save_total_limit,

        eval_strategy="steps",

        eval_steps=args.save_steps,

        optim="adamw_8bit",

        weight_decay=0.01,

        lr_scheduler_type="cosine",

        seed=3407,

        fp16=not is_bfloat16_supported(),

        bf16=is_bfloat16_supported(),

        report_to="none",

        max_length=args.max_seq_length,

        # ====================================================
        # CRITICAL V2 SETTING
        # ====================================================
        #
        # The model is trained on prompt + completion,
        # BUT LOSS IS CALCULATED ONLY ON COMPLETION.
        #
        # This is the major difference from V1.
        #
        completion_only_loss=True,

        # Use the tokenizer's chat template for
        # conversational prompt/completion examples.
        dataset_text_field=None,

        push_to_hub=args.push_to_hub,

        hub_model_id=args.hub_model_id,

        hub_strategy=(
            "every_save"
            if args.push_to_hub
            else "end"
        ),
    )

    # --------------------------------------------------------
    # Validate Hub arguments
    # --------------------------------------------------------

    if args.push_to_hub and not args.hub_model_id:

        raise ValueError(
            "--push_to_hub requires --hub_model_id"
        )

    # --------------------------------------------------------
    # 5. Create trainer
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 5: CREATING TRAINER")
    print("=" * 70)

    trainer = SFTTrainer(

        model=model,

        processing_class=tokenizer,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        args=sft_config,
    )

    # --------------------------------------------------------
    # 6. Resume support
    # --------------------------------------------------------

    resume_checkpoint = None

    if args.resume:

        if os.path.isdir(args.output_dir):

            checkpoints = [
                d
                for d in os.listdir(args.output_dir)
                if d.startswith("checkpoint-")
            ]

            if checkpoints:

                resume_checkpoint = True

                print()
                print(
                    "Resuming from latest checkpoint in:",
                    args.output_dir
                )

            else:

                print()
                print(
                    "--resume was specified, "
                    "but no checkpoint was found."
                )

    # --------------------------------------------------------
    # 7. Training
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 6: STARTING V2 TRAINING")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print(
        "Training loss will be calculated ONLY on "
        "the category completion."
    )

    print()
    print("This prevents the model from learning "
          "to reproduce section text.")

    print()

    trainer.train(
        resume_from_checkpoint=resume_checkpoint
    )

    # --------------------------------------------------------
    # 8. Save final adapter
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 7: SAVING FINAL LoRA ADAPTER")
    print("=" * 70)

    final_dir = os.path.join(
        args.output_dir,
        "final"
    )

    model.save_pretrained(
        final_dir
    )

    tokenizer.save_pretrained(
        final_dir
    )

    print()
    print("Final adapter saved to:")
    print(final_dir)

    # --------------------------------------------------------
    # 9. Hugging Face Hub
    # --------------------------------------------------------

    if args.push_to_hub and args.hub_model_id:

        print()
        print(
            f"Pushing final adapter to "
            f"{args.hub_model_id} ..."
        )

        model.push_to_hub(
            args.hub_model_id,
            token=os.getenv("HF_TOKEN"),
        )

        tokenizer.push_to_hub(
            args.hub_model_id,
            token=os.getenv("HF_TOKEN"),
        )

        print("Hugging Face upload complete.")

    # --------------------------------------------------------
    # 10. Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V2 TRAINING COMPLETE")
    print("=" * 70)

    print()
    print("Model output:")
    print(final_dir)

    print()
    print("Training dataset:")
    print(args.dataset)

    print()
    print("Validation dataset:")
    print(args.eval_dataset)

    print()
    print("Final evaluation dataset:")
    print("data/eval_real.jsonl")

    print()
    print("NEXT STEP:")
    print(
        "Evaluate V2 on the untouched "
        "data/eval_real.jsonl dataset."
    )

    print()


if __name__ == "__main__":

    main()