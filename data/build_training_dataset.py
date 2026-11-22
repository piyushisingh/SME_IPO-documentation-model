import json
import random
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# This script is inside the data/ folder,
# so its parent directory is already the data directory.
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "sections.json"

TRAIN_FILE = BASE_DIR / "train_real.jsonl"
VALIDATION_FILE = BASE_DIR / "validation_real.jsonl"
EVAL_FILE = BASE_DIR / "eval_real.jsonl"

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)


# ============================================================
# DOCUMENT-LEVEL SPLIT
# ============================================================

document_names = list(documents.keys())

random.seed(RANDOM_SEED)
random.shuffle(document_names)

total_docs = len(document_names)

if total_docs < 3:
    raise ValueError(
        "At least 3 documents are required for train/validation/evaluation split."
    )


# 6 train / 1 validation / 1 evaluation for 8 documents
train_docs = document_names[:6]
validation_docs = document_names[6:7]
eval_docs = document_names[7:8]


# ============================================================
# CREATE TRAINING EXAMPLES
# ============================================================

def create_examples(doc_names):

    examples = []

    for doc_name in doc_names:

        document = documents[doc_name]

        sections = document.get("sections", [])

        for section in sections:

            heading = section.get("heading", "").strip()
            category = section.get("category", "other")
            text = section.get("text", "").strip()
            page_start = section.get("page_start")

            # Skip unusable sections
            if not heading:
                continue

            if not text:
                continue

            if len(text) < 80:
                continue

            # ------------------------------------------------
            # Classification example
            # ------------------------------------------------

            examples.append({
                "instruction": (
                    "Classify the following SME IPO prospectus section "
                    "into the most appropriate document category."
                ),

                "input": (
                    f"Heading: {heading}\n\n"
                    f"Section text:\n{text}"
                ),

                "output": category,

                "metadata": {
                    "document": doc_name,
                    "heading": heading,
                    "page_start": page_start
                }
            })

    return examples


# ============================================================
# BUILD DATASETS
# ============================================================

train_examples = create_examples(train_docs)

validation_examples = create_examples(validation_docs)

eval_examples = create_examples(eval_docs)


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(path, examples):

    with open(path, "w", encoding="utf-8") as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )


save_jsonl(TRAIN_FILE, train_examples)
save_jsonl(VALIDATION_FILE, validation_examples)
save_jsonl(EVAL_FILE, eval_examples)


# ============================================================
# REPORT
# ============================================================

print("=" * 70)
print("REAL TRAINING DATASET CREATED")
print("=" * 70)

print()
print("Documents:")
print("  Train:", len(train_docs))
print("  Validation:", len(validation_docs))
print("  Evaluation:", len(eval_docs))

print()
print("Sections / examples:")
print("  Train:", len(train_examples))
print("  Validation:", len(validation_examples))
print("  Evaluation:", len(eval_examples))

print()
print("Train documents:")
for doc in train_docs:
    print("  -", doc)

print()
print("Validation document:")
for doc in validation_docs:
    print("  -", doc)

print()
print("Evaluation document:")
for doc in eval_docs:
    print("  -", doc)

print()
print("Saved:")
print(" ", TRAIN_FILE)
print(" ", VALIDATION_FILE)
print(" ", EVAL_FILE)

print()
print("IMPORTANT:")
print("The split was performed at DOCUMENT level to reduce data leakage.")