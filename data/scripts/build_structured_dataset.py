import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROCESSED_FOLDER = Path("data/processed")
OUTPUT_FILE = Path("data/sections.json")


# ============================================================
# IMPORTANT IPO HEADINGS
# ============================================================

KNOWN_HEADINGS = [
    "CORPORATE INFORMATION",
    "OUR COMPANY",
    "ABOUT THE COMPANY",
    "OUR BUSINESS",
    "BUSINESS OVERVIEW",
    "INDUSTRY OVERVIEW",
    "INDUSTRY",
    "OUR PRODUCTS",
    "OUR SERVICES",
    "BUSINESS OPERATIONS",

    "HISTORY",
    "HISTORY AND CERTAIN CORPORATE MATTERS",
    "MAJOR MILESTONES",

    "PROMOTERS",
    "PROMOTER GROUP",
    "OUR PROMOTERS",
    "BOARD OF DIRECTORS",
    "DIRECTORS",
    "KEY MANAGERIAL PERSONNEL",
    "SENIOR MANAGEMENT",

    "CAPITAL STRUCTURE",
    "SHARE CAPITAL",
    "SHAREHOLDING PATTERN",

    "FINANCIAL INFORMATION",
    "FINANCIAL STATEMENTS",
    "FINANCIAL POSITION",
    "FINANCIAL PERFORMANCE",
    "KEY FINANCIAL INFORMATION",
    "FINANCIAL AND OPERATING INFORMATION",

    "RISK FACTORS",
    "RISKS",
    "INTERNAL RISK FACTORS",
    "EXTERNAL RISK FACTORS",

    "OBJECTS OF THE ISSUE",
    "OBJECTS OF ISSUE",
    "USE OF PROCEEDS",
    "UTILIZATION OF ISSUE PROCEEDS",
    "UTILISATION OF ISSUE PROCEEDS",

    "THE ISSUE",
    "ISSUE DETAILS",
    "ISSUE STRUCTURE",
    "OFFER DETAILS",
    "FRESH ISSUE",
    "OFFER FOR SALE",

    "LITIGATION",
    "LEGAL PROCEEDINGS",
    "OUTSTANDING LITIGATIONS",
    "MATERIAL LITIGATION",

    "GOVERNMENT AND OTHER APPROVALS",
    "GOVERNMENT APPROVALS",
    "REGULATORY APPROVALS",
    "LICENSES AND APPROVALS",
    "LEGAL AND REGULATORY",

    "MATERIAL CONTRACTS",
    "MATERIAL CONTRACTS AND DOCUMENTS",

    "COMPETITION",
    "COMPETITIVE STRENGTHS",
    "COMPETITIVE LANDSCAPE",

    "CUSTOMERS",
    "SUPPLIERS",
    "MANUFACTURING FACILITIES",
    "PROPERTIES",
    "GEOGRAPHICAL PRESENCE",

    "GENERAL CORPORATE PURPOSES",
    "WORKING CAPITAL",
    "CAPITAL EXPENDITURE",

    "DIVIDEND POLICY",
    "REGULATIONS AND POLICIES",
    "INTELLECTUAL PROPERTY",
]


# ============================================================
# CATEGORY CLASSIFICATION
# ============================================================

CATEGORY_MAP = {

    "company_information": [
        "CORPORATE INFORMATION",
        "OUR COMPANY",
        "ABOUT THE COMPANY",
        "HISTORY",
        "HISTORY AND CERTAIN CORPORATE MATTERS",
        "MAJOR MILESTONES",
    ],

    "promoters_and_management": [
        "PROMOTERS",
        "PROMOTER GROUP",
        "OUR PROMOTERS",
        "BOARD OF DIRECTORS",
        "DIRECTORS",
        "KEY MANAGERIAL PERSONNEL",
        "SENIOR MANAGEMENT",
    ],

    "capital_structure": [
        "CAPITAL STRUCTURE",
        "SHARE CAPITAL",
        "SHAREHOLDING PATTERN",
    ],

    "financial_information": [
        "FINANCIAL INFORMATION",
        "FINANCIAL STATEMENTS",
        "FINANCIAL POSITION",
        "FINANCIAL PERFORMANCE",
        "KEY FINANCIAL INFORMATION",
        "FINANCIAL AND OPERATING INFORMATION",
    ],

    "litigation": [
        "LITIGATION",
        "LEGAL PROCEEDINGS",
        "OUTSTANDING LITIGATIONS",
        "MATERIAL LITIGATION",
    ],

    "risk_factors": [
        "RISK FACTORS",
        "RISKS",
        "INTERNAL RISK FACTORS",
        "EXTERNAL RISK FACTORS",
    ],

    "objects_of_issue": [
        "OBJECTS OF THE ISSUE",
        "OBJECTS OF ISSUE",
        "USE OF PROCEEDS",
        "UTILIZATION OF ISSUE PROCEEDS",
        "UTILISATION OF ISSUE PROCEEDS",
        "GENERAL CORPORATE PURPOSES",
        "WORKING CAPITAL",
        "CAPITAL EXPENDITURE",
    ],

    "ipo_details": [
        "THE ISSUE",
        "ISSUE DETAILS",
        "ISSUE STRUCTURE",
        "OFFER DETAILS",
        "FRESH ISSUE",
        "OFFER FOR SALE",
    ],

    "industry_information": [
        "INDUSTRY OVERVIEW",
        "INDUSTRY",
        "COMPETITION",
        "COMPETITIVE STRENGTHS",
        "COMPETITIVE LANDSCAPE",
    ],

    "legal_and_regulatory": [
        "GOVERNMENT AND OTHER APPROVALS",
        "GOVERNMENT APPROVALS",
        "REGULATORY APPROVALS",
        "LICENSES AND APPROVALS",
        "LEGAL AND REGULATORY",
        "MATERIAL CONTRACTS",
        "MATERIAL CONTRACTS AND DOCUMENTS",
        "REGULATIONS AND POLICIES",
    ],

    "business_operations": [
        "OUR BUSINESS",
        "BUSINESS OVERVIEW",
        "OUR PRODUCTS",
        "OUR SERVICES",
        "BUSINESS OPERATIONS",
        "CUSTOMERS",
        "SUPPLIERS",
        "MANUFACTURING FACILITIES",
        "PROPERTIES",
        "GEOGRAPHICAL PRESENCE",
        "INTELLECTUAL PROPERTY",
    ],
    "financial_statements": [
        "FINANCIAL STATEMENTS",
        "RESTATED FINANCIAL STATEMENTS",
        "RESTATED CONSOLIDATED FINANCIAL STATEMENTS",
        "NOTES FORMING PART OF THE FINANCIAL STATEMENTS",
        "NOTES FORMING PART OF RESTATED CONSOLIDATED FINANCIAL STATEMENTS",
        "NOTES FORMING PART OF THE FINANCIAL STATEMENTS, AS RESTATED",
        "AUDITED FINANCIAL STATEMENTS",
        "STATEMENT OF PROFIT AND LOSS",
        "BALANCE SHEET",
        "CASH FLOW STATEMENT",
        "STATEMENT OF CHANGES IN EQUITY",
    ],

    "declarations": [
        "DECLARATION",
        "DECLARATIONS",
        "DECLARATION BY THE COMPANY",
        "DECLARATION BY THE PROMOTERS",
        "ISSUER DECLARATION",
    ],

    "related_party_transactions": [
        "RELATED PARTY TRANSACTIONS",
        "SUMMARY OF RELATED PARTY TRANSACTIONS",
        "RELATED PARTIES",
        "TRANSACTIONS WITH RELATED PARTIES",
    ],

    "shareholding": [
        "SHAREHOLDING",
        "SHAREHOLDING PATTERN",
        "SHAREHOLDING OF THE PROMOTERS",
        "PROMOTER SHAREHOLDING",
        "CAPITALIZATION",
    ],
}


# ============================================================
# NORMALIZE HEADING
# ============================================================

def normalize_heading(text):

    text = text.strip()

    # Remove common numbering
    text = re.sub(
        r"^(chapter\s*)?\d+[\.\):\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^[IVXLC]+[\.\):\-]\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CHECK WHETHER A LINE IS A REAL HEADING
# ============================================================

def is_heading(line):
    line = line.strip()

    if not line:
        return False

    # --------------------------------------------------
    # BASIC FILTERS
    # --------------------------------------------------

    # Very long lines are normally body text/table content
    if len(line) > 120:
        return False

    normalized = normalize_heading(line)
    upper = normalized.upper()

    # --------------------------------------------------
    # EXACT KNOWN HEADINGS
    # --------------------------------------------------

    if upper in KNOWN_HEADINGS:
        return True

    # --------------------------------------------------
    # OBVIOUS NON-HEADINGS
    # --------------------------------------------------

    # Sentences ending with punctuation
    if line.endswith((".", ",", ";", ":")):
        return False

    # Too many words = probably paragraph/table text
    if len(line.split()) > 12:
        return False

    # Lines containing strong data/table indicators
    blocked_terms = [
        "CIN",
        "ISIN",
        "PAN",
        "GST",
        "EMAIL",
        "WEBSITE",
        "TELEPHONE",
        "PHONE",
        "FAX",
        "DATED",
        "DATE",
        "PAGE",
        "RS.",
        "₹",
        "%",
        "NIL",
    ]

    upper_line = line.upper()

    for term in blocked_terms:
        if term in upper_line:
            return False

    # --------------------------------------------------
    # DON'T ACCEPT DATA-HEAVY LINES
    # --------------------------------------------------

    digit_count = sum(c.isdigit() for c in line)

    if digit_count >= 3:
        return False

    # --------------------------------------------------
    # ALL-CAPS HEADING DETECTION
    # --------------------------------------------------

    letters = re.sub(r"[^A-Za-z]", "", line)

    if len(letters) < 5:
        return False

    uppercase_letters = sum(
        1 for c in letters if c.isupper()
    )

    uppercase_ratio = (
        uppercase_letters / len(letters)
    )

    if uppercase_ratio >= 0.90:

        # Real headings are usually reasonably short
        if len(line) <= 100 and len(line.split()) <= 10:

            # Avoid lines that look like table/data rows
            if "|" in line:
                return False

            # Avoid lines with lots of numbers
            if sum(c.isdigit() for c in line) > 2:
                return False

            return True

    return False

# ============================================================
# CLASSIFY HEADING
# ============================================================

def classify_heading(heading):

    normalized = normalize_heading(heading).upper()

    for category, headings in CATEGORY_MAP.items():

        for known_heading in headings:

            if normalized == known_heading:
                return category

    return "other"


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(file_path):

    print("\n" + "=" * 70)
    print(f"PROCESSING: {file_path.name}")
    print("=" * 70)

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    sections = []

    current_section = None
    current_page = None

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # ----------------------------------------
        # Page marker
        # ----------------------------------------

        page_match = re.match(
            r"---\s*PAGE\s*(\d+)\s*---",
            line,
            flags=re.IGNORECASE
        )

        if page_match:

            current_page = int(
                page_match.group(1)
            )

            continue

        # ----------------------------------------
        # Heading
        # ----------------------------------------

        if is_heading(line):

            # Save previous section
            if current_section:

                current_section["text"] = (
                    "\n".join(
                        current_section["text"]
                    ).strip()
                )

                if len(current_section["text"]) >= 80:

                    sections.append(
                        current_section
                    )

            # Start new section
            heading = normalize_heading(line)

            current_section = {

                "heading": heading,

                "category": classify_heading(
                    heading
                ),

                "page_start": current_page,

                "text": []
            }

        else:

            if current_section:

                current_section["text"].append(line)

    # ----------------------------------------
    # Save final section
    # ----------------------------------------

    if current_section:

        current_section["text"] = (
            "\n".join(
                current_section["text"]
            ).strip()
        )

        if len(current_section["text"]) >= 80:

            sections.append(
                current_section
            )

    print(
        f"Sections detected: {len(sections)}"
    )

    # Category summary
    category_counts = {}

    for section in sections:

        category = section["category"]

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    for category, count in sorted(
        category_counts.items()
    ):

        print(
            f"  {category}: {count}"
        )

    return {

        "source_file": file_path.name,

        "sections": sections,

        "section_count": len(sections)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(
        PROCESSED_FOLDER.glob("*.txt")
    )

    if not files:

        print(
            "ERROR: No processed TXT files found."
        )

        return

    print("=" * 70)
    print("IPO DOCUMENT SECTION EXTRACTION")
    print("=" * 70)

    print(
        f"Documents found: {len(files)}"
    )

    documents = {}

    total_sections = 0

    for file_path in files:

        result = process_document(
            file_path
        )

        documents[file_path.stem] = result

        total_sections += (
            result["section_count"]
        )

    # ----------------------------------------
    # Save
    # ----------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("SECTION EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Documents processed: {len(documents)}"
    )

    print(
        f"Total sections: {total_sections}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()