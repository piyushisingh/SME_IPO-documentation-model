import json
from pathlib import Path

from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "train_v3.jsonl"
VALIDATION_FILE = BASE_DIR / "validation_v3.jsonl"

MODEL_NAME = "unsloth/gemma-3-1b-it"

MAX_LENGTH = 768


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("V3 CHAT TEMPLATE + TRUNCATION AUDIT")
print("=" * 70)

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print("Tokenizer loaded.")
print()


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                data.append(
                    json.loads(line)
                )

    return data


# ============================================================
# AUDIT DATASET
# ============================================================

def audit_dataset(path):

    examples = load_jsonl(path)

    print()
    print("=" * 70)
    print(f"AUDITING: {path.name}")
    print("=" * 70)

    print()
    print(f"Total examples: {len(examples)}")

    total_over_limit = 0
    category_missing = 0
    category_present = 0

    max_length_seen = 0

    affected_examples = []

    for index, example in enumerate(examples):

        prompt_messages = example["prompt"]
        completion_messages = example["completion"]

        category = completion_messages[0]["content"].strip()

        messages = (
            prompt_messages
            + completion_messages
        )

        # ----------------------------------------------------
        # Actual Gemma chat template
        # ----------------------------------------------------

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        full_length = len(full_tokens)

        max_length_seen = max(
            max_length_seen,
            full_length,
        )

        # ----------------------------------------------------
        # Tokenize exactly as max_length would
        # ----------------------------------------------------

        truncated_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]

        if full_length > MAX_LENGTH:

            total_over_limit += 1

        # ----------------------------------------------------
        # Check category token sequence
        # ----------------------------------------------------

        category_tokens = tokenizer(
            category,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        found = False

        if len(category_tokens) > 0:

            for start in range(
                0,
                len(truncated_tokens)
                - len(category_tokens)
                + 1,
            ):

                if (
                    truncated_tokens[
                        start:start + len(category_tokens)
                    ]
                    == category_tokens
                ):

                    found = True
                    break

        if found:

            category_present += 1

        else:

            category_missing += 1

            if len(affected_examples) < 20:

                affected_examples.append(
                    {
                        "index": index,
                        "category": category,
                        "full_length": full_length,
                        "truncated_length": len(
                            truncated_tokens
                        ),
                    }
                )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print(
        f"Examples within 768 tokens: "
        f"{len(examples) - total_over_limit}"
    )

    print(
        f"Examples over 768 tokens:    "
        f"{total_over_limit}"
    )

    print(
        f"Category preserved:          "
        f"{category_present}"
    )

    print(
        f"Category NOT found:          "
        f"{category_missing}"
    )

    print(
        f"Longest actual sequence:     "
        f"{max_length_seen}"
    )

    if len(examples) > 0:

        print(
            f"Potentially affected:        "
            f"{category_missing / len(examples) * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Affected examples
    # --------------------------------------------------------

    if affected_examples:

        print()
        print(
            "=" * 70
        )
        print(
            "AFFECTED EXAMPLES"
        )
        print(
            "=" * 70
        )

        for item in affected_examples:

            print()

            print(
                f"Example:       {item['index']}"
            )

            print(
                f"Expected:      {item['category']}"
            )

            print(
                f"Full tokens:   {item['full_length']}"
            )

            print(
                f"After cutoff:  {item['truncated_length']}"
            )

    print()


# ============================================================
# RUN
# ============================================================

audit_dataset(TRAIN_FILE)

audit_dataset(VALIDATION_FILE)


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("V3 AUDIT COMPLETE")
print("=" * 70)