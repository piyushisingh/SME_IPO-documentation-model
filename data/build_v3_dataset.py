import json
from pathlib import Path

from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_INPUT = BASE_DIR / "train_real.jsonl"
VALIDATION_INPUT = BASE_DIR / "validation_real.jsonl"

TRAIN_OUTPUT = BASE_DIR / "train_v3.jsonl"
VALIDATION_OUTPUT = BASE_DIR / "validation_v3.jsonl"

MODEL_NAME = "unsloth/gemma-3-1b-it"

MAX_LENGTH = 768

# Reserve room for:
# chat template + instruction + heading + category
RESERVED_OUTPUT_TOKENS = 32


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("BUILDING V3 DATASET")
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
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                data.append(
                    json.loads(line)
                )

    return data


# ============================================================
# BUILD ONE EXAMPLE
# ============================================================

def build_example(example):

    instruction = example["instruction"].strip()

    input_text = example.get(
        "input",
        ""
    ).strip()

    category = example["output"].strip()


    # --------------------------------------------------------
    # Extract heading
    # --------------------------------------------------------

    heading = ""

    if "Heading:" in input_text:

        parts = input_text.split(
            "Section text:",
            1
        )

        heading_part = parts[0]

        heading = heading_part.replace(
            "Heading:",
            "",
            1
        ).strip()


    # --------------------------------------------------------
    # Extract section text
    # --------------------------------------------------------

    section_text = input_text

    if "Section text:" in input_text:

        section_text = input_text.split(
            "Section text:",
            1
        )[1].strip()


    # --------------------------------------------------------
    # Build base prompt
    # --------------------------------------------------------

    base_prompt = (
        f"{instruction}\n\n"
        f"Heading: {heading}\n\n"
        f"Section text:\n"
    )


    # --------------------------------------------------------
    # Tokenize fixed portions
    # --------------------------------------------------------

    base_tokens = tokenizer(
        base_prompt,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]


    category_tokens = tokenizer(
        category,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]


    # --------------------------------------------------------
    # Reserve tokens for:
    #
    # category
    # chat template overhead
    # safety margin
    # --------------------------------------------------------

    available_for_section = (
        MAX_LENGTH
        - len(base_tokens)
        - len(category_tokens)
        - RESERVED_OUTPUT_TOKENS
    )


    # Safety fallback
    if available_for_section < 64:

        available_for_section = 64


    # --------------------------------------------------------
    # Tokenize section
    # --------------------------------------------------------

    section_tokens = tokenizer(
        section_text,
        add_special_tokens=False,
        truncation=True,
        max_length=available_for_section,
    )["input_ids"]


    truncated_section = tokenizer.decode(
        section_tokens,
        skip_special_tokens=True,
    )


    # --------------------------------------------------------
    # Final prompt
    # --------------------------------------------------------

    prompt_text = (
        f"{instruction}\n\n"
        f"Heading: {heading}\n\n"
        f"Section text:\n"
        f"{truncated_section}"
    ).strip()


    return {
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": category,
            }
        ],
    }


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    input_path,
    output_path,
):

    examples = load_jsonl(
        input_path
    )

    print()
    print(
        f"Processing: {input_path.name}"
    )

    print(
        f"Input examples: {len(examples)}"
    )


    processed = []


    for example in examples:

        processed.append(
            build_example(example)
        )


    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for example in processed:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


    print(
        f"Output examples: {len(processed)}"
    )

    print(
        f"Saved to: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

process_dataset(
    TRAIN_INPUT,
    TRAIN_OUTPUT,
)

process_dataset(
    VALIDATION_INPUT,
    VALIDATION_OUTPUT,
)


print()
print("=" * 70)
print("V3 DATASET BUILD COMPLETE")
print("=" * 70)