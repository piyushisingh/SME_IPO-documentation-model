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
print("V2 TRAINING DATA LENGTH AUDIT")
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


print(f"Total training examples: {len(examples)}")
print()


# ============================================================
# TOKEN LENGTH ANALYSIS
# ============================================================

lengths = []

under_256 = 0
between_256_512 = 0
between_512_768 = 0
over_768 = 0

category_preserved = 0
category_at_risk = 0

long_examples = []


for index, example in enumerate(examples):

    instruction = example["instruction"].strip()

    input_text = example.get(
        "input",
        ""
    ).strip()

    category = example["output"].strip()


    # This recreates the V2 prompt construction
    prompt_text = (
        f"{instruction}\n\n"
        f"{input_text}"
    ).strip()


    # Tokenize the user prompt
    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]


    # Tokenize the category separately
    category_tokens = tokenizer(
        category,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]


    prompt_length = len(prompt_tokens)
    category_length = len(category_tokens)

    total_estimated_length = (
        prompt_length
        + category_length
    )

    lengths.append(total_estimated_length)


    # --------------------------------------------------------
    # Buckets
    # --------------------------------------------------------

    if total_estimated_length < 256:

        under_256 += 1

    elif total_estimated_length < 512:

        between_256_512 += 1

    elif total_estimated_length <= MAX_LENGTH:

        between_512_768 += 1

    else:

        over_768 += 1


    # --------------------------------------------------------
    # Can the category theoretically fit?
    # --------------------------------------------------------

    if total_estimated_length <= MAX_LENGTH:

        category_preserved += 1

    else:

        category_at_risk += 1

        long_examples.append(
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
                "prompt_tokens": prompt_length,
                "category_tokens": category_length,
                "total_tokens": total_estimated_length,
            }
        )


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("TOKEN LENGTH RESULTS")
print("=" * 70)

print()
print(f"Total examples:                 {len(examples)}")

print()
print(f"< 256 tokens:                   {under_256}")
print(f"256 - 511 tokens:               {between_256_512}")
print(f"512 - 768 tokens:               {between_512_768}")
print(f"> 768 tokens:                   {over_768}")

print()
print(f"Category theoretically fits:    {category_preserved}")
print(f"Category potentially truncated: {category_at_risk}")

if len(examples) > 0:

    percentage = (
        category_at_risk / len(examples)
    ) * 100

    print(
        f"Potentially affected:           {percentage:.2f}%"
    )


# ============================================================
# LONGEST EXAMPLES
# ============================================================

print()
print("=" * 70)
print("LONGEST EXAMPLES")
print("=" * 70)

long_examples.sort(
    key=lambda x: x["total_tokens"],
    reverse=True
)


for item in long_examples[:20]:

    print()

    print(
        f"Example index: {item['index']}"
    )

    print(
        f"Heading:       {item['heading'][:120]}"
    )

    print(
        f"Expected:      {item['category']}"
    )

    print(
        f"Prompt tokens: {item['prompt_tokens']}"
    )

    print(
        f"Category tok:  {item['category_tokens']}"
    )

    print(
        f"Total tokens:  {item['total_tokens']}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("DIAGNOSIS")
print("=" * 70)

print()

if category_at_risk > 0:

    print(
        "WARNING: Some training examples are longer than "
        f"{MAX_LENGTH} tokens."
    )

    print(
        "The category label may be lost when the training "
        "sequence is truncated."
    )

    print(
        "This supports the truncation hypothesis."
    )

else:

    print(
        "No training examples exceed the current "
        f"{MAX_LENGTH}-token limit."
    )

    print(
        "The truncation hypothesis is unlikely to be "
        "the main cause."
    )

print()
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)