import json
from pathlib import Path
from collections import Counter


INPUT_FILE = Path("data/sections.json")
OUTPUT_FILE = Path("data/heading_inventory.txt")


# ============================================================
# LOAD SECTIONS
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    documents = json.load(f)


# ============================================================
# COLLECT HEADINGS
# ============================================================

heading_counter = Counter()

heading_examples = {}


for document_name, document in documents.items():

    for section in document.get("sections", []):

        heading = section.get("heading", "").strip()

        category = section.get(
            "category",
            "other"
        )

        if not heading:
            continue

        key = heading.upper()

        heading_counter[key] += 1

        if key not in heading_examples:

            heading_examples[key] = {
                "original": heading,
                "category": category,
                "file": document.get(
                    "source_file"
                ),
                "page": section.get(
                    "page_start"
                )
            }


# ============================================================
# WRITE INVENTORY
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "IPO HEADING INVENTORY\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Unique headings: "
        f"{len(heading_counter)}\n"
    )

    f.write(
        f"Total heading occurrences: "
        f"{sum(heading_counter.values())}\n\n"
    )

    # Most common first
    for number, (
        heading,
        count
    ) in enumerate(
        heading_counter.most_common(),
        start=1
    ):

        example = heading_examples[
            heading
        ]

        f.write(
            f"{number}. {heading}\n"
        )

        f.write(
            f"   Occurrences: {count}\n"
        )

        f.write(
            f"   Current category: "
            f"{example['category']}\n"
        )

        f.write(
            f"   Example file: "
            f"{example['file']}\n"
        )

        f.write(
            f"   Page: "
            f"{example['page']}\n"
        )

        f.write("\n")


# ============================================================
# PRINT SUMMARY
# ============================================================

print("=" * 70)
print("HEADING INVENTORY CREATED")
print("=" * 70)

print(
    f"Unique headings: {len(heading_counter)}"
)

print(
    f"Total heading occurrences: "
    f"{sum(heading_counter.values())}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print("\nTop 30 headings:")

for number, (
    heading,
    count
) in enumerate(
    heading_counter.most_common(30),
    start=1
):

    print(
        f"{number}. {heading} "
        f"({count})"
    )