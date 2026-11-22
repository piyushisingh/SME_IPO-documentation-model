#!/usr/bin/env python3
"""
Phase 3/4: build master_sections.jsonl from extracted_sections.jsonl.

LABELS ARE PROVISIONAL RULE LABELS. No human has verified any of them. Nothing here is LLM-generated.
Rule order of precedence (see annotation_guidelines.md):
  1. structural records (cover/contents)            -> no category, review_status=excluded_structural
  2. L1 chapter title rule (TOC title -> category)   -> L1 label; L2 inherits it when the chapter is single-topic
  3. L2 heading keyword rule (only a few documented overrides, and the only way to label multi-topic chapters)
  4. anything else in a multi-topic chapter          -> category=null, review_status=ambiguous (never 'other')
review_status values: needs_review (rule label, unverified) | ambiguous | extraction_error | excluded_structural.  'verified' is never assigned.
"""
import json, re, collections
from pathlib import Path
OUT = Path("/mnt/user-data/outputs/data/rebuilt_v4")
R = [json.loads(l) for l in open(OUT / "extracted_sections.jsonl", encoding="utf-8")]

CATS = ["ipo_details", "risk_factors", "disclosures_and_definitions", "legal_and_regulatory", "promoters_and_management", "company_information",
        "financial_information", "financial_statements", "related_party_transactions", "capital_structure", "objects_of_issue",
        "qualitative_factors", "industry_information", "business_operations", "litigation", "declarations", "other"]

def key(h):
    h = re.sub(r"^(SECTION\s+)?[IVX]+\s*[.:\-\u2013\u2014]*\s+", "", h.strip(), flags=re.I)
    h = re.sub(r"^[\u2013\u2014\-]+\s*", "", h)
    return re.sub(r"\s+", " ", h.upper()).strip()

# (rule_id, regex on normalised chapter title, category or None, scope, overlap_note)
# scope: single = L2 children inherit the category; multi = L2 labelled only by keyword rules, else ambiguous
CHAPTER_RULES = [
    ("C01", r"^(GENERAL )?DEFINITIONS|^CERTAIN CONVENTIONS|^PRESENTATION OF FINANCIAL|^FORWARD[\s\u2013\-]*LOOKING", "disclosures_and_definitions", "single", None),
    ("C02", r"^RISK FACTOR", "risk_factors", "single", None),
    ("C03", r"^THE (ISSUE|OFFER)$|^(ISSUE|OFFER) (STRUCTURE|PROCEDURE)|^TERMS OF THE (ISSUE|OFFER)", "ipo_details", "single", None),
    ("C04", r"^RESTRICTIONS? ON FOREIGN OWNERSHIP", "legal_and_regulatory", "single", "ipo_details"),
    ("C05", r"^CAPITAL STRUCTURE", "capital_structure", "single", None),
    ("C06", r"^CAPITALI[SZ]ATION STATEMENT", "financial_information", "single", "capital_structure"),
    ("C07", r"^OBJECTS OF THE (ISSUE|OFFER)", "objects_of_issue", "single", None),
    ("C08", r"^BASIS FOR (ISSUE|OFFER) PRICE", "ipo_details", "single", "financial_information"),
    ("C09", r"^STATEMENT OF (POSSIBLE )?(SPECIAL )?TAX BENEFITS", "legal_and_regulatory", "single", "financial_information"),
    ("C10", r"^(OUR )?INDUSTRY|^INDUSTRY OVERVIEW", "industry_information", "single", None),
    ("C11", r"^(OUR )?BUSINESS", "business_operations", "single", None),
    ("C12", r"^KEY .*REGULATIONS", "legal_and_regulatory", "single", "industry_information"),
    ("C13", r"^(OUR )?HISTORY", "company_information", "single", None),
    ("C14", r"^OUR MANAGEMENT", "promoters_and_management", "single", None),
    ("C15", r"^OUR PROMOTERS?", "promoters_and_management", "single", None),
    ("C16", r"^OUR (GROUP COMPAN|SUBSIDIAR)", "company_information", "single", "promoters_and_management"),
    ("C17", r"^SUMMARY OF RELATED PARTY", "related_party_transactions", "single", None),
    ("C18", r"^DIVIDEND POLICY", "financial_information", "single", "capital_structure"),
    ("C19", r"^SUMMARY (OF )?(OUR )?FINANCIAL|^SUMMARY (OF )?CONTINGENT|^OTHER FINANCIAL INFORMATION|^(STATEMENT OF )?FINANCIALS? INDEBTEDNESS|^MANAGEMENT.{0,3}S? DISCUSSION", "financial_information", "single", None),
    ("C20", r"^RESTATED (AUDITED )?FINANCIAL|^STATEMENT OF (ASSETS|CASH FLOW|PROFIT)", "financial_statements", "single", None),
    ("C21", r"^OUTSTANDING LITIGATION", "litigation", "single", None),
    ("C22", r"^GOVERNMENT AND OTHER|^OTHER REGULATORY AND STATUTORY", "legal_and_regulatory", "single", None),
    ("C23", r"^MATERIAL CONTRACTS|^MAIN PROVISIONS|^DESCRIPTION OF EQUITY SHARES", "legal_and_regulatory", "single", None),
    ("C24", r"^DECLARATION", "declarations", "single", None),
    ("C25", r"^GENERAL INFORMATION", None, "multi", None),
    ("C26", r"^ABRIDGED PROSPECTUS", None, "multi", None),
]
STRUCT = ("front_matter", "contents_page")

# L2 heading keyword rules. Applied (a) as overrides inside specific single-topic chapters, (b) as the only labels inside multi-topic chapters.
KW_RULES = [
    ("K01", r"RELATED PARTY|RELATED-PARTY", "related_party_transactions", {"financial_statements", "financial_information"}),
    ("K02", r"QUALITATIVE FACTORS", "qualitative_factors", {"ipo_details"}),
    ("K03", r"QUANTITATIVE FACTORS", "financial_information", {"ipo_details"}),
    ("K04", r"^DECLARATION", "declarations", None),
    ("K05", r"LITIGATION", "litigation", None),
]
# multi-topic chapters only
KW_GENERAL_INFO = [
    ("G01", r"REGISTERED OFFICE|CORPORATE IDENTITY|INCORPORATION|BRIEF SUMMARY", "company_information"),
    ("G02", r"BOARD OF DIRECTORS|KEY MANAGERIAL|COMPANY SECRETARY|COMPLIANCE OFFICER|CHIEF FINANCIAL", "promoters_and_management"),
    ("G03", r"CREDIT RATING|IPO GRADING|BROKERS?|TRUSTEES?|REGISTRAR|BANKERS?|SCSB|SYNDICATE|DEPOSITORY|INTERMEDIAR|MONITORING AGENCY|APPRAISING|UNDERWRIT|MARKET MAKING|BOOK BUILDING|GREEN SHOE|WITHDRAWAL OF THE (ISSUE|OFFER)|GRIEVANCE|EXPERT OPINION|ACCOUNTANT|LEAD MANAGER|LEGAL COUNSEL|BID/ ?(ISSUE|OFFER)|TYPE OF (ISSUE|OFFER)|FILING OF OFFER|INTER-?SE ALLOCATION|STEPS TO BE TAKEN|ILLUSTRATION OF", "ipo_details"),
]
KW_ABRIDGED = [
    ("A01", r"OBJECTS OF THE ISSUE|UTILI[SZ]ATION OF NET PROCEEDS|GENERAL CORPORATE PURPOSE|PREPAYMENT OR REPAYMENT", "objects_of_issue"),
    ("A02", r"^PROMOTERS?$|^A\. PROMOTER|PROMOTER", "promoters_and_management"),
    ("A03", r"SHAREHOLDING", "capital_structure"),
    ("A04", r"LITIGATION", "litigation"),
    ("A05", r"^DECLARATION", "declarations"),
    ("A06", r"PRODUCTS / SERVICES|SEGMENT REPORTING|SWOT|REVENUE CONCENTRATION", "business_operations"),
    ("A07", r"GAAP FINANCIAL MEASURES|NET ASSET VALUE", "financial_information"),
]
FRAGMENT = re.compile(r"^(herring prospectus|corporate|issuer name|stock exchange|allottees)$", re.I)

# independent content-cue check (heuristic second opinion, NOT verification)
CUES = {
    "litigation": r"writ petition|hon.ble|tribunal|criminal|show cause|first information report|\bFIR\b|suit no|\bcourt\b",
    "risk_factors": r"we cannot assure|adversely affect|no assurance|may not be able to|could be adversely|there can be no",
    "financial_statements": r"total assets|net worth|in lakhs|restated|balance sheet|profit after tax|cash flow",
    "objects_of_issue": r"net proceeds|utili[sz]ation|deployment of|working capital requirement",
    "promoters_and_management": r"date of birth|\bDIN\b|brief profile|qualification|designation|term of office|nationality",
    "disclosures_and_definitions": r"shall mean|\bmeans the\b|abbreviation|unless the context",
    "capital_structure": r"paid[- ]up|authori[sz]ed share capital|pre[- ]issue|post[- ]issue",
}
COMPAT = [{"financial_statements", "financial_information"}, {"capital_structure", "ipo_details", "financial_information"},
          {"legal_and_regulatory", "litigation"}, {"promoters_and_management", "company_information"},
          {"risk_factors", "business_operations"}, {"objects_of_issue", "ipo_details"}, {"disclosures_and_definitions", "legal_and_regulatory"},
          {"financial_statements", "related_party_transactions"}, {"financial_information", "related_party_transactions"},
          {"risk_factors", "financial_information"}, {"risk_factors", "legal_and_regulatory"}, {"litigation", "risk_factors"},
          {"promoters_and_management", "related_party_transactions"}]
def compatible(a, b): return a == b or any(a in s and b in s for s in COMPAT)

def chapter_label(title):
    k = key(title)
    for rid, rx, cat, scope, ov in CHAPTER_RULES:
        if re.search(rx, k):
            return rid, cat, scope, ov
    return None, None, None, None

by_id = {r["example_id"]: r for r in R}
for r in R:
    f = set(r["quality_flags"])
    r["normalized_heading"] = re.sub(r"\s+", " ", re.sub(r"[\u2018\u2019]", "'", r["original_heading"])).strip()
    r["category"] = None; r["category_proposed"] = []; r["label_basis"] = None; r["label_rule_id"] = None
    r["label_confidence_tier"] = "none"; r["label_status"] = "no_label"; r["review_status"] = "needs_review"
    r["training_exclusion_reasons"] = []
    if r["level"] == 1:
        r["parent_heading"] = None; r["heading_path"] = [r["original_heading"]]
    else:
        p = by_id[r["parent_section_id"]]
        r["parent_heading"] = p["original_heading"]
        r["heading_path"] = [p["original_heading"]] + ([] if r["heading_source"] == "chapter_preamble" else [r["original_heading"]])

# ---------- L1 labels
for r in R:
    if r["level"] != 1: continue
    f = set(r["quality_flags"])
    if f & set(STRUCT):
        r["review_status"] = "excluded_structural"; r["label_basis"] = "structural_record"; r["training_exclusion_reasons"].append("structural_front_matter_or_contents")
        continue
    rid, cat, scope, ov = chapter_label(r["original_heading"])
    r["chapter_scope"] = scope
    if rid is None:
        r["review_status"] = "ambiguous"; r["label_basis"] = "no_chapter_rule"; r["quality_flags"].append("no_chapter_rule_matched"); continue
    if scope == "multi":
        r["review_status"] = "ambiguous"; r["label_basis"] = "multi_topic_chapter"; r["label_rule_id"] = rid
        r["quality_flags"].append("multi_topic_chapter_no_single_label"); continue
    r["category"] = cat; r["label_basis"] = "toc_chapter_title_rule"; r["label_rule_id"] = rid; r["label_status"] = "unverified_rule_label"
    r["label_confidence_tier"] = "rule_chapter_overlap" if ov else "rule_chapter"
    if ov: r["category_proposed"] = [ov]; r["quality_flags"].append("category_overlap_possible")
    if "scanned_insert_F1_to_F36" in f:
        r["review_status"] = "extraction_error"; r["training_exclusion_reasons"].append("scanned_insert_unreliable_text")

# ---------- L2 labels
for r in R:
    if r["level"] != 2: continue
    p = by_id[r["parent_section_id"]]; pf = set(p["quality_flags"]); h = r["original_heading"]; hu = re.sub(r"\s+", " ", h.upper())
    if "contents_page" in pf or "front_matter" in pf:
        r["review_status"] = "excluded_structural"; r["label_basis"] = "structural_record"; r["training_exclusion_reasons"].append("structural_front_matter_or_contents"); continue
    scope = p.get("chapter_scope")
    r["chapter_scope"] = scope
    if r["heading_source"] == "detected_bold_heading" and FRAGMENT.match(h.strip()):
        r["quality_flags"].append("fragment_heading_suspected")
    if r["heading_source"] == "detected_bold_heading" and len(h) > 300:
        r["quality_flags"].append("heading_is_paragraph_suspected")
    if scope == "single":
        cat, tier, rid, basis = p["category"], p["label_confidence_tier"], p["label_rule_id"], "inherited_from_chapter_rule"
        for kid, rx, kcat, only_parent in KW_RULES:
            if re.search(rx, hu) and r["heading_source"] == "detected_bold_heading" and (only_parent is None or p["category"] in only_parent) and kcat != cat:
                cat, tier, rid, basis = kcat, "rule_heading_keyword", kid, "heading_keyword_override"
                r["category_proposed"] = [p["category"]]; r["quality_flags"].append("heading_keyword_overrides_chapter_label"); break
        r["category"] = cat; r["label_confidence_tier"] = tier; r["label_rule_id"] = rid; r["label_basis"] = basis; r["label_status"] = "unverified_rule_label"
        if p["review_status"] == "extraction_error":
            r["review_status"] = "extraction_error"; r["training_exclusion_reasons"].append("scanned_insert_unreliable_text")
        if "category_overlap_possible" in p["quality_flags"]:
            r["quality_flags"].append("category_overlap_possible"); r["category_proposed"] = list(set(r["category_proposed"]) | set(p["category_proposed"]))
    elif scope == "multi":
        rules = KW_GENERAL_INFO if key(p["original_heading"]).startswith("GENERAL INFORMATION") else KW_ABRIDGED
        hit = next(((rid, c) for rid, rx, c in rules if re.search(rx, hu)), None) if r["heading_source"] == "detected_bold_heading" else None
        if hit:
            r["category"] = hit[1]; r["label_rule_id"] = hit[0]; r["label_basis"] = "heading_keyword_in_multi_topic_chapter"
            r["label_confidence_tier"] = "rule_heading_keyword"; r["label_status"] = "unverified_rule_label"
        else:
            r["review_status"] = "ambiguous"; r["label_basis"] = "multi_topic_chapter_no_keyword_hit"; r["quality_flags"].append("label_ambiguous_multi_topic_chapter")
            r["training_exclusion_reasons"].append("ambiguous_label")
    else:
        r["review_status"] = "ambiguous"; r["label_basis"] = "no_chapter_rule"; r["training_exclusion_reasons"].append("ambiguous_label")
    if r["review_status"] == "needs_review" and r["category"] is None:
        r["review_status"] = "ambiguous"

# ---------- independent content-cue check (heuristic)
CUE_RX = {c: re.compile(rx, re.I) for c, rx in CUES.items()}
for r in R:
    if r["level"] != 2 or r["category"] is None or r["n_chars"] < 400: continue
    t = r["section_text"]
    dens = {c: len(rx.findall(t)) * 1000.0 / len(t) for c, rx in CUE_RX.items()}
    top = max(dens, key=dens.get)
    r["content_cue_top"] = top; r["content_cue_density_per_1000_chars"] = round(dens[top], 2)
    if dens[top] >= 4.0 and not compatible(r["category"], top):
        r["quality_flags"].append("suspected_label_content_mismatch"); r["category_proposed"] = sorted(set(r["category_proposed"]) | {top})

# ---------- objective exclusion reasons (applied later when building model-ready data)
for r in R:
    fl = set(r["quality_flags"])
    if "contains_image_only_page" in fl and r["n_chars"] < 300: r["training_exclusion_reasons"].append("image_only_page_no_text")
    r["training_exclusion_reasons"] = sorted(set(r["training_exclusion_reasons"]))
    r["quality_flags"] = sorted(set(r["quality_flags"]))
    if r["review_status"] == "needs_review" and r["category"] is None: r["review_status"] = "ambiguous"

with open(OUT / "master_sections.jsonl", "w", encoding="utf-8") as fo:
    for r in R: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
c = collections.Counter((r["level"], r["review_status"]) for r in R)
print("records", len(R), dict(c))
print("L1 categories:", dict(collections.Counter(r["category"] for r in R if r["level"] == 1)))
print("L2 categories:", dict(collections.Counter(r["category"] for r in R if r["level"] == 2)))
