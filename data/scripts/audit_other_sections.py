import json
from collections import Counter

INPUT_PATH = "data/sections.json"
OUTPUT_PATH = "data/other_sections_audit.txt"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

headings = Counter()
examples = {}

for company, document in data.items():
    for section in document.get("sections", []):
        if section.get("category") == "other":
            heading = section.get("heading", "").strip()

            if not heading:
                heading = "[NO HEADING]"

            headings[heading] += 1

            if heading not in examples:
                examples[heading] = {
                    "company": company,
                    "page": section.get("page_start"),
                    "text": section.get("text", "")[:500]
                }

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("OTHER CATEGORY AUDIT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Unique OTHER headings: {len(headings)}\n")
    f.write(f"Total OTHER sections: {sum(headings.values())}\n\n")

    for i, (heading, count) in enumerate(headings.most_common(), 1):
        ex = examples[heading]

        f.write(f"{i}. {heading}\n")
        f.write(f"   Occurrences: {count}\n")
        f.write(f"   Example file: {ex['company']}\n")
        f.write(f"   Page: {ex['page']}\n")
        f.write(f"   Text: {ex['text'].replace(chr(10), ' ')}\n")
        f.write("-" * 80 + "\n")

print("=" * 70)
print("OTHER CATEGORY AUDIT COMPLETE")
print("=" * 70)
print(f"Unique OTHER headings: {len(headings)}")
print(f"Total OTHER sections: {sum(headings.values())}")
print(f"Saved to: {OUTPUT_PATH}")