import re
from pathlib import Path


# Folder containing extracted raw text files
INPUT_FOLDER = Path("data/raw_text")

# Folder where cleaned text will be saved
OUTPUT_FOLDER = Path("data/processed")

# Create output folder if needed
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    # Remove null characters
    text = text.replace("\x00", "")

    # Fix words split across lines because of hyphenation
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove excessive spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive blank lines
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)

    # Remove spaces at the beginning/end of lines
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    # Rebuild document
    text = "\n".join(lines)

    return text.strip()


def process_file(input_file):
    print("=" * 70)
    print(f"Cleaning: {input_file.name}")
    print("=" * 70)

    raw_text = input_file.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    cleaned_text = clean_text(raw_text)

    output_file = OUTPUT_FOLDER / input_file.name

    output_file.write_text(
        cleaned_text,
        encoding="utf-8"
    )

    print(f"Original characters: {len(raw_text):,}")
    print(f"Cleaned characters:  {len(cleaned_text):,}")
    print(f"Saved to: {output_file}")
    print()


def main():

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Input folder not found: {INPUT_FOLDER}"
        )

    text_files = list(INPUT_FOLDER.glob("*.txt"))

    if not text_files:
        print("No text files found in data/raw_text/")
        return

    print("=" * 70)
    print(f"Found {len(text_files)} text files")
    print("=" * 70)

    for text_file in text_files:
        process_file(text_file)

    print("=" * 70)
    print("ALL TEXT FILES CLEANED")
    print("=" * 70)


if __name__ == "__main__":
    main()