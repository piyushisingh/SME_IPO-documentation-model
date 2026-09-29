import torch
from unsloth import FastLanguageModel


BASE_MODEL = "unsloth/gemma-3-1b-it"
TRAINED_MODEL = "checkpoints/phase1/final"


TEST_CASES = [
    {
        "instruction": "Is there any pending litigation against the company?",
        "input": (
            "Company: Acme Textiles Ltd.\n"
            "Forum: Delhi High Court.\n"
            "Amount: Rs 12,00,000.\n"
            "Status: Pending."
        ),
        "expected": (
            "There is one pending litigation against the Company "
            "before the Delhi High Court involving a claim of "
            "Rs 12,00,000."
        ),
    },
    {
        "instruction": "Describe the company's incorporation details.",
        "input": (
            "Company: Acme Textiles Ltd.\n"
            "Date of incorporation: 15 March 2015.\n"
            "Registered office: New Delhi."
        ),
        "expected": (
            "Acme Textiles Ltd. was incorporated on 15 March 2015 "
            "and has its registered office in New Delhi."
        ),
    },
    {
        "instruction": "Describe the company's principal business activity.",
        "input": (
            "Company: Acme Textiles Ltd.\n"
            "Principal business: manufacture and sale of textile products."
        ),
        "expected": (
            "Acme Textiles Ltd. is principally engaged in the "
            "manufacture and sale of textile products."
        ),
    },
]


def load_model(model_path):
    print(f"\nLoading model: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate(model, tokenizer, instruction, input_text):
    messages = [
        {
            "role": "user",
            "content": f"{instruction}\n\n{input_text}",
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=120,
            temperature=0.2,
        )

    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )

    return response.strip()


def run_evaluation(model_name, model, tokenizer):
    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)

    for i, test in enumerate(TEST_CASES, start=1):
        print(f"\n--- TEST CASE {i} ---")

        print("Instruction:")
        print(test["instruction"])

        print("\nInput:")
        print(test["input"])

        output = generate(
            model,
            tokenizer,
            test["instruction"],
            test["input"],
        )

        print("\nModel output:")
        print(output)

        print("\nExpected reference:")
        print(test["expected"])

        print("-" * 70)


def main():
    print("=" * 70)
    print("SME IPO MODEL EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load base model
    # ---------------------------------------------------------
    base_model, base_tokenizer = load_model(BASE_MODEL)

    run_evaluation(
        "BASE GEMMA 3 1B",
        base_model,
        base_tokenizer,
    )

    # Free GPU memory before loading the second model.
    del base_model
    del base_tokenizer
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 2. Load fine-tuned model
    # ---------------------------------------------------------
    trained_model, trained_tokenizer = load_model(TRAINED_MODEL)

    run_evaluation(
        "FINE-TUNED GEMMA + LoRA",
        trained_model,
        trained_tokenizer,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()