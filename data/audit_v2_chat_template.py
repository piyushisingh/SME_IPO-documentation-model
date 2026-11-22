import json
from pathlib import Path

from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "train_real.jsonl"

MODEL_NAME = "unsloth/gemma-3-1b-it"

MAX_LENGTH = 768


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("V2 CHAT TEMPLATE + TRUNCATION AUDIT")
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
# LOAD DATA
# ============================================================

examples = []

with open(TRAIN_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if line:
            examples.append(json.loads(line))


print(f"Total examples: {len(examples)}")
print()


# ============================================================
# AUDIT
# ============================================================

category_preserved = 0
category_lost = 0

affected_examples = []


for index, example in enumerate(examples):

    instruction = example["instruction"].strip()

    input_text = example.get(
        "input",
        ""
    ).strip()

    category = example["output"].strip()


    prompt_text = (
        f"{instruction}\n\n"
        f"{input_text}"
    ).strip()


    # Reproduce V2 conversational structure
    messages = [
        {
            "role": "user",
            "content": prompt_text,
        },
        {
            "role": "assistant",
            "content": category,
        },
    ]


    # Apply the actual Gemma chat template
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


    # Tokenize WITHOUT truncation first
    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]


    full_length = len(full_tokens)


    # Tokenize with the EXACT V2 limit
    truncated_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]


    truncated_length = len(truncated_tokens)


    # --------------------------------------------------------
    # Check whether category text survives
    # --------------------------------------------------------

    # Encode category itself
    category_tokens = tokenizer(
        category,
        add_special_tokens=False,
    )["input_ids"]


    # Search for the category token sequence
    found = False

    if len(category_tokens) > 0:

        for start in range(
            0,
            len(truncated_tokens) - len(category_tokens) + 1
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

        category_preserved += 1

    else:

        category_lost += 1

        if len(affected_examples) < 30:

            affected_examples.append(
                {
                    "index": index,
                    "heading": input_text.split(
                        "Section text:",
                        1
                    )[0].replace(
                        "Heading:",
                        ""
                    ).strip(),
                    "category": category,
                    "full_length": full_length,
                    "truncated_length": truncated_length,
                }
            )


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("RESULTS")
print("=" * 70)

print()

print(
    f"Total examples:             {len(examples)}"
)

print(
    f"Category preserved:         {category_preserved}"
)

print(
    f"Category NOT found:         {category_lost}"
)

percentage = (
    category_lost / len(examples) * 100
)

print(
    f"Category potentially lost:  {percentage:.2f}%"
)


# ============================================================
# AFFECTED EXAMPLES
# ============================================================

print()
print("=" * 70)
print("FIRST AFFECTED EXAMPLES")
print("=" * 70)

for item in affected_examples:

    print()

    print(
        f"Example:       {item['index']}"
    )

    print(
        f"Heading:       {item['heading'][:100]}"
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


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

print()
print("=" * 70)
print("FINAL DIAGNOSIS")
print("=" * 70)

print()

if category_lost > 0:

    print(
        "CONFIRMED: Some category labels do not survive "
        "the 768-token cutoff."
    )

    print(
        "The V2 training setup is therefore training on "
        "examples where the target may be missing."
    )

else:

    print(
        "Category text survived the cutoff in all examples."
    )

    print(
        "The simple truncation hypothesis is NOT confirmed."
    )

print()
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)