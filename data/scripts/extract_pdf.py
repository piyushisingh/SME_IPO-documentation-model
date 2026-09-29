from pypdf import PdfReader
from pathlib import Path


# Folder containing all IPO PDFs
PDF_FOLDER = Path("data/pdfs")

# Folder where extracted text will be saved
OUTPUT_FOLDER = Path("data/raw_text")

# Create output folder if it does not exist
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def extract_pdf(pdf_path):
    print("=" * 70)
    print(f"Processing: {pdf_path.name}")
    print("=" * 70)

    reader = PdfReader(str(pdf_path))

    print(f"Number of pages: {len(reader.pages)}")

    all_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()

            if text:
                all_text.append(
                    f"\n\n--- PAGE {page_number} ---\n\n{text}"
                )

        except Exception as e:
            print(f"Error on page {page_number}: {e}")

    full_text = "".join(all_text)

    output_file = OUTPUT_FOLDER / f"{pdf_path.stem}.txt"

    output_file.write_text(
        full_text,
        encoding="utf-8"
    )

    print(f"Characters extracted: {len(full_text):,}")
    print(f"Saved to: {output_file}")
    print()


def main():

    if not PDF_FOLDER.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {PDF_FOLDER}"
        )

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    print("=" * 70)
    print(f"Found {len(pdf_files)} PDF files")
    print("=" * 70)

    for pdf_file in pdf_files:
        extract_pdf(pdf_file)

    print("=" * 70)
    print("ALL PDFs PROCESSED")
    print("=" * 70)


if __name__ == "__main__":
    main()