#!/usr/bin/env python3
"""Phase 1 (data-only): audit the OLD files that were provided. Read-only. Writes old_data_audit.json.
Scope: sections.json, structured_ipo_data.json, task_categories.json.  No scripts/JSONL/train files were provided."""
import json, re, collections, statistics as st
from pathlib import Path
UP = Path("/mnt/user-data/uploads"); OUT = Path("/mnt/user-data/outputs/data/rebuilt_v4")
MAP = {"Century": "century", "DRHP_Maxwell": "maxwell_engineering", "DRHPPHOTONICSWATERTECHLIMITED": "photonics_watertech",
       "Ideas_Electricals_&Engineers_DRHP_Hem_SME": "ideas_electricals", "Red_Herring_Prospectus_AXIOM": "axiom_gas",
       "quanto-updated-prospectus-1646-pm-final": "quanto_agroworld"}
EXCLUDED = ["Project-Om_-RHP_with_RFS", "shakti-polytarp-ltd-rhp"]
CAT11 = set(json.load(open(UP / "task_categories.json")).keys())
CAT17 = {"ipo_details", "risk_factors", "disclosures_and_definitions", "legal_and_regulatory", "promoters_and_management", "company_information",
         "financial_information", "financial_statements", "related_party_transactions", "capital_structure", "objects_of_issue",
         "qualitative_factors", "industry_information", "business_operations", "litigation", "declarations", "other"}
def an(s): return re.sub(r"[^a-z0-9]", "", s.lower())

old = json.load(open(UP / "sections.json", encoding="utf-8"))
structured = json.load(open(UP / "structured_ipo_data.json", encoding="utf-8"))
res = {"per_doc": {}, "excluded_docs": {k: len(old[k]["sections"]) for k in EXCLUDED}, "global": {}}

allcats = collections.Counter(); heading_cats = collections.defaultdict(collections.Counter)
for k, v in old.items():
    for s in v["sections"]:
        allcats[s.get("category")] += 1
        heading_cats[re.sub(r"\s+", " ", s["heading"].strip().upper())][s.get("category")] += 1
res["global"]["category_counts_all_8_docs"] = dict(allcats.most_common())
res["global"]["categories_not_in_17"] = sorted(c for c in allcats if c not in CAT17)
res["global"]["categories_in_17_never_used"] = sorted(CAT17 - set(allcats))
res["global"]["task_categories_json_keys_n"] = len(CAT11)
res["global"]["task_categories_missing_from_17"] = sorted(CAT17 - CAT11)
res["global"]["headings_with_conflicting_labels"] = sum(1 for h, c in heading_cats.items() if len(c) > 1 and sum(c.values()) > 1)
res["global"]["headings_repeated_ge2"] = sum(1 for h, c in heading_cats.items() if sum(c.values()) > 1)
res["global"]["conflict_examples"] = [{"heading": h, "labels": dict(c)} for h, c in heading_cats.items() if len(c) > 1 and sum(c.values()) >= 3][:12]

for oldk, doc_id in MAP.items():
    secs = old[oldk]["sections"]
    pages = [json.loads(l) for l in open(OUT / "pages" / f"{doc_id}.jsonl", encoding="utf-8")]
    # concatenated alnum text with page offsets
    cat, offs = "", []
    for p in pages:
        offs.append(len(cat)); cat += an(p["text_layout"])
    def page_of(pos):
        lo, hi = 0, len(offs) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offs[mid] <= pos: lo = mid
            else: hi = mid - 1
        return lo + 1
    n = len(secs); lens = [len(s["text"]) for s in secs]
    texts = collections.Counter(an(s["text"]) for s in secs)
    exact_dup = sum(c - 1 for c in texts.values() if c > 1)
    empty = sum(1 for s in secs if len(s["text"].strip()) == 0)
    short = sum(1 for l in lens if l < 100)
    # traceability: can the old text be found in the PDF text, and on which page?
    found = notfound = page_exact = page_other = 0
    deltas = collections.Counter()
    for s in secs:
        key = an(s["text"])[:70]
        if len(key) < 30:
            continue
        occ, start = [], 0
        while len(occ) < 8:
            i = cat.find(key, start)
            if i < 0: break
            occ.append(page_of(i)); start = i + 1
        if not occ:
            notfound += 1; continue
        found += 1
        try: ps = int(s["page_start"])
        except Exception: ps = None
        if ps is not None and ps in occ: page_exact += 1
        else:
            page_other += 1
            if ps is not None: deltas[occ[0] - ps] += 1
    types_ps = collections.Counter(type(s.get("page_start")).__name__ for s in secs)
    cats = collections.Counter(s.get("category") for s in secs)
    # heading present at start of text?
    head_in_text = sum(1 for s in secs if an(s["heading"])[:25] and an(s["text"]).startswith(an(s["heading"])[:25]))
    res["per_doc"][doc_id] = {
        "old_key": oldk, "old_section_count": n, "empty_text": empty, "text_len_lt_100": short,
        "text_len_median": int(st.median(lens)), "text_len_max": max(lens), "text_len_p95": int(sorted(lens)[int(0.95 * n)]),
        "exact_duplicate_text_records": exact_dup, "page_start_python_types": dict(types_ps),
        "text_found_verbatim_in_pdf_text(alnum-normalised, first 70 chars)": found, "text_not_found_in_pdf": notfound,
        "page_start_matches_a_pdf_page_where_text_occurs": page_exact, "page_start_does_not_match": page_other,
        "most_common_pdf_page_minus_old_page_start": deltas.most_common(4),
        "heading_repeated_at_start_of_text": head_in_text, "category_counts": dict(cats.most_common()),
    }

# structured_ipo_data.json
sd = {}
for oldk, doc_id in MAP.items():
    d = structured[oldk]
    items = {c: len(v) for c, v in d.items()}
    lens = [len(x["text"]) for v in d.values() for x in v]
    keys = collections.Counter(tuple(sorted(x.keys())) for v in d.values() for x in v)
    sd[doc_id] = {"categories": len(d), "items_total": sum(items.values()), "items_per_category": items,
                  "text_len_median": int(st.median(lens)) if lens else None, "text_len_max": max(lens) if lens else None,
                  "item_key_sets": [list(k) for k in keys]}
res["structured_ipo_data"] = sd
res["structured_ipo_data_category_keys"] = sorted(next(iter(structured.values())).keys())
(OUT / "old_data_audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps({"global": res["global"]}, indent=1, ensure_ascii=False)[:2500])
for d, r in res["per_doc"].items():
    print(f"\n{d}: n={r['old_section_count']} empty={r['empty_text']} <100ch={r['text_len_lt_100']} median={r['text_len_median']} max={r['text_len_max']} p95={r['text_len_p95']} exact_dups={r['exact_duplicate_text_records']}")
    print(f"   found_in_pdf={r['text_found_verbatim_in_pdf_text(alnum-normalised, first 70 chars)']} not_found={r['text_not_found_in_pdf']} | page_ok={r['page_start_matches_a_pdf_page_where_text_occurs']} page_off={r['page_start_does_not_match']} delta_top={r['most_common_pdf_page_minus_old_page_start']} | page_start types={r['page_start_python_types']}")
    print("   cats:", r["category_counts"])
