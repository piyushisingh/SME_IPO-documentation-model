from pathlib import Path
from pypdf import PdfReader
import re


# ==========================================
# FOLDERS
# ==========================================

PDF_FOLDER = Path("data/pdfs")
RAW_FOLDER = Path("data/raw_text")
PROCESSED_FOLDER = Path("data/processed")


# Create folders if they don't exist
RAW_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):

    # Remove excessive spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Fix words broken across lines
    # Example:
    # compa-
    # ny
    # becomes:
    # company
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove spaces at beginning/end of lines
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


# ==========================================
# PROCESS ONE PDF
# ==========================================

def process_pdf(pdf_path):

    print("\n" + "=" * 70)
    print(f"PROCESSING: {pdf_path.name}")
    print("=" * 70)

    try:

        reader = PdfReader(str(pdf_path))

        print(f"Pages: {len(reader.pages)}")

        all_text = []

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                text = page.extract_text()

                if text:

                    all_text.append(
                        f"\n--- PAGE {page_number} ---\n{text}"
                    )

            except Exception as e:

                print(
                    f"Warning: Could not extract page "
                    f"{page_number}: {e}"
                )

        raw_text = "\n".join(all_text)

        # ------------------------------------------
        # Save raw extracted text
        # ------------------------------------------

        raw_output = RAW_FOLDER / f"{pdf_path.stem}.txt"

        raw_output.write_text(
            raw_text,
            encoding="utf-8"
        )

        # ------------------------------------------
        # Clean text
        # ------------------------------------------

        cleaned_text = clean_text(raw_text)

        # ------------------------------------------
        # Save cleaned text
        # ------------------------------------------

        processed_output = (
            PROCESSED_FOLDER / f"{pdf_path.stem}.txt"
        )

        processed_output.write_text(
            cleaned_text,
            encoding="utf-8"
        )

        print(f"Raw characters:     {len(raw_text):,}")
        print(f"Clean characters:    {len(cleaned_text):,}")

        print(f"Raw saved to:        {raw_output}")
        print(f"Cleaned saved to:    {processed_output}")

        return True

    except Exception as e:

        print(f"ERROR processing {pdf_path.name}")
        print(e)

        return False


# ==========================================
# PROCESS ALL PDFs
# ==========================================

def main():

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:

        print("No PDF files found in:")
        print(PDF_FOLDER)

        return

    print("=" * 70)
    print("SME IPO PDF DATA PROCESSING")
    print("=" * 70)

    print(f"PDFs found: {len(pdf_files)}")

    successful = 0
    failed = 0

    for pdf in pdf_files:

        result = process_pdf(pdf)

        if result:
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")

    print("\nRaw text folder:")
    print(RAW_FOLDER)

    print("\nProcessed text folder:")
    print(PROCESSED_FOLDER)


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    main()