import json
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "sections.json"
OUTPUT_PATH = BASE_DIR / "other_sections_audit.txt"


# ============================================================
# LOAD SECTIONS.JSON
# ============================================================

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# EXTRACT ALL SECTIONS
# ============================================================

all_sections = []

for company_name, document in data.items():

    if not isinstance(document, dict):
        continue

    source_file = document.get(
        "source_file",
        ""
    )

    sections = document.get(
        "sections",
        []
    )

    if not isinstance(sections, list):
        continue

    for section in sections:

        if not isinstance(section, dict):
            continue

        # Keep the source/company information available
        section_copy = section.copy()

        section_copy["company"] = company_name
        section_copy["source_file"] = source_file

        all_sections.append(section_copy)


# ============================================================
# KEEP ONLY TRUE "OTHER" SECTIONS
# ============================================================

other_sections = []

for section in all_sections:

    category = section.get(
        "category",
        ""
    )

    if category == "other":
        other_sections.append(section)


# ============================================================
# COUNT OTHER HEADINGS
# ============================================================

heading_counts = Counter()

examples = defaultdict(list)

for section in other_sections:

    heading = section.get(
        "heading",
        ""
    ).strip()

    if not heading:
        continue

    heading_counts[heading] += 1

    # Keep maximum 3 examples per heading
    if len(examples[heading]) < 3:

        examples[heading].append({
            "company": section.get(
                "company",
                "Unknown"
            ),

            "source_file": section.get(
                "source_file",
                "Unknown"
            ),

            "page": section.get(
                "page_start",
                "Unknown"
            ),

            "text": section.get(
                "text",
                ""
            )[:500]
        })


# ============================================================
# WRITE AUDIT REPORT
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "OTHER CATEGORY AUDIT\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Total sections in dataset: "
        f"{len(all_sections)}\n"
    )

    f.write(
        f"Total OTHER sections: "
        f"{len(other_sections)}\n"
    )

    f.write(
        f"Unique OTHER headings: "
        f"{len(heading_counts)}\n\n"
    )

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "TOP OTHER HEADINGS\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    for i, (heading, count) in enumerate(
        heading_counts.most_common(),
        1
    ):

        f.write(
            f"{i}. {heading}\n"
        )

        f.write(
            f"   Occurrences: {count}\n"
        )

        for example in examples[heading]:

            f.write(
                f"   Company: "
                f"{example['company']}\n"
            )

            f.write(
                f"   Source file: "
                f"{example['source_file']}\n"
            )

            f.write(
                f"   Page: "
                f"{example['page']}\n"
            )

            text = example["text"].replace(
                "\n",
                " "
            )

            f.write(
                f"   Text: {text}\n"
            )

        f.write(
            "\n"
        )


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print(
    "=" * 70
)

print(
    "OTHER CATEGORY AUDIT COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Total sections in dataset: "
    f"{len(all_sections)}"
)

print(
    f"Total OTHER sections: "
    f"{len(other_sections)}"
)

print(
    f"Unique OTHER headings: "
    f"{len(heading_counts)}"
)

print(
    f"Saved to: {OUTPUT_PATH}"
)