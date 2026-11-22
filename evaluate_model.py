"""
SME IPO SECTION CLASSIFIER — FINAL EVALUATION

Purpose:
    Evaluate the fine-tuned Gemma 3 1B LoRA classifier on the
    completely held-out evaluation dataset.

Training:
    data/train_real.jsonl

Validation:
    data/validation_real.jsonl

Final evaluation:
    data/eval_real.jsonl

The model was trained to classify an SME IPO prospectus section
into one of the document categories.

IMPORTANT:
    This script evaluates classification, NOT document generation.
"""

import json
import os
from collections import Counter, defaultdict

import torch
from unsloth import FastLanguageModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "checkpoints/section_classifier_v3/final"

EVAL_FILE = "data/eval_real.jsonl"

MAX_SEQ_LENGTH = 256
MAX_NEW_TOKENS = 12

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

def load_eval_dataset(path):

    examples = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            examples.append(json.loads(line))

    return examples


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING FINE-TUNED SME IPO CLASSIFIER")
    print("=" * 70)

    print()
    print("Model:", MODEL_PATH)
    print("Device:", DEVICE)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


# ============================================================
# PREDICT CATEGORY
# ============================================================

def predict_category(model, tokenizer, instruction, input_text):

    prompt = (
        f"{instruction}\n\n"
        f"{input_text}"
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            use_cache=True,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return response


# ============================================================
# NORMALIZE MODEL OUTPUT
# ============================================================

VALID_CATEGORIES = {
    "ipo_details",
    "risk_factors",
    "disclosures_and_definitions",
    "legal_and_regulatory",
    "promoters_and_management",
    "company_information",
    "financial_information",
    "financial_statements",
    "related_party_transactions",
    "capital_structure",
    "objects_of_issue",
    "qualitative_factors",
    "industry_information",
    "business_operations",
    "litigation",
    "declarations",
    "other",
}


def normalize_prediction(prediction):

    prediction = prediction.strip().lower()

    # Exact match
    if prediction in VALID_CATEGORIES:
        return prediction

    # Remove common punctuation
    cleaned = (
        prediction
        .replace("`", "")
        .replace('"', "")
        .replace("'", "")
        .replace(".", "")
        .replace(",", "")
        .strip()
    )

    if cleaned in VALID_CATEGORIES:
        return cleaned

    # Search for category names inside generated response
    for category in VALID_CATEGORIES:

        if category in cleaned:

            return category

    return "INVALID"


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, tokenizer, examples):

    print()
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print()
    print("Evaluation examples:", len(examples))
    print("Maximum sequence length:", MAX_SEQ_LENGTH)
    print("Maximum generated tokens:", MAX_NEW_TOKENS)

    correct = 0
    invalid = 0

    total = len(examples)

    category_total = Counter()
    category_correct = Counter()

    confusion = defaultdict(Counter)

    invalid_examples = []

    for index, example in enumerate(examples, start=1):

        instruction = example.get("instruction", "")
        input_text = example.get("input", "")

        expected = example.get("output", "").strip().lower()

        prediction_raw = predict_category(
            model,
            tokenizer,
            instruction,
            input_text,
        )

        prediction = normalize_prediction(prediction_raw)

        category_total[expected] += 1

        if prediction == expected:

            correct += 1
            category_correct[expected] += 1

        else:

            confusion[expected][prediction] += 1

            if prediction == "INVALID":

                invalid += 1

                if len(invalid_examples) < 20:

                    invalid_examples.append({
                        "index": index,
                        "expected": expected,
                        "prediction": prediction_raw,
                        "heading": example.get("metadata", {}).get(
                            "heading",
                            "N/A"
                        ),
                    })

        # Progress every 25 examples
        if index % 25 == 0 or index == total:

            accuracy = correct / index * 100

            print(
                f"[{index:>3}/{total}] "
                f"Accuracy: {accuracy:.2f}%"
            )

        # Release temporary tensors
        if DEVICE == "cuda":

            torch.cuda.empty_cache()


    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    accuracy = (
        correct / total * 100
        if total
        else 0
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print()
    print(f"Total examples : {total}")
    print(f"Correct        : {correct}")
    print(f"Incorrect      : {total - correct}")
    print(f"Invalid output : {invalid}")
    print(f"Accuracy       : {accuracy:.2f}%")

    # ========================================================
    # PER-CATEGORY RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("PER-CATEGORY ACCURACY")
    print("=" * 70)

    for category in sorted(category_total):

        total_category = category_total[category]
        correct_category = category_correct[category]

        category_accuracy = (
            correct_category / total_category * 100
            if total_category
            else 0
        )

        print(
            f"{category:35s} "
            f"{correct_category:3d}/{total_category:3d} "
            f"({category_accuracy:6.2f}%)"
        )

    # ========================================================
    # CONFUSION INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print("MISCLASSIFICATION SUMMARY")
    print("=" * 70)

    any_confusion = False

    for expected in sorted(confusion):

        for predicted, count in confusion[expected].most_common():

            any_confusion = True

            print(
                f"Expected: {expected:35s} "
                f"Predicted: {predicted:35s} "
                f"Count: {count}"
            )

    if not any_confusion:

        print("No misclassifications.")

    # ========================================================
    # INVALID OUTPUTS
    # ========================================================

    if invalid_examples:

        print()
        print("=" * 70)
        print("SAMPLE INVALID OUTPUTS")
        print("=" * 70)

        for item in invalid_examples:

            print()
            print("Example:", item["index"])
            print("Heading:", item["heading"])
            print("Expected:", item["expected"])
            print("Model output:", item["prediction"])

    # ========================================================
    # GPU MEMORY
    # ========================================================

    if DEVICE == "cuda":

        allocated = torch.cuda.memory_allocated() / (
            1024 ** 3
        )

        reserved = torch.cuda.memory_reserved() / (
            1024 ** 3
        )

        print()
        print("=" * 70)
        print("GPU MEMORY")
        print("=" * 70)

        print(
            f"Allocated: {allocated:.2f} GB"
        )

        print(
            f"Reserved : {reserved:.2f} GB"
        )

    return accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SME IPO SECTION CLASSIFIER")
    print("FINAL HELD-OUT EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(EVAL_FILE):

        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVAL_FILE}"
        )

    # --------------------------------------------------------
    # Load evaluation data
    # --------------------------------------------------------

    examples = load_eval_dataset(EVAL_FILE)

    print()
    print("Loaded evaluation dataset:")
    print(EVAL_FILE)

    print()
    print("Examples:", len(examples))

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, tokenizer = load_model()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    accuracy = evaluate(
        model,
        tokenizer,
        examples,
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"FINAL ACCURACY: {accuracy:.2f}%"
    )


if __name__ == "__main__":

    main()