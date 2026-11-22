#!/usr/bin/env python3
"""Phase 2 step 6: add objective quality flags to extracted sections, run independent checks, write
extraction_stats.json, extraction_issues.csv and extraction_report.md.  Every number in the report comes from this script."""
import json, re, csv, collections, statistics as st
from pathlib import Path
OUT = Path("/mnt/user-data/outputs/data/rebuilt_v4")
CFG = json.loads((OUT / "scripts/doc_config.json").read_text(encoding="utf-8"))
MANIFEST = {m["doc_id"]: m for m in json.loads((OUT / "source_manifest.json").read_text())}
INV = json.loads((OUT / "scan_inventory.json").read_text())
META = json.loads((OUT / "structure_meta.json").read_text())
OLD = json.loads((OUT / "old_data_audit.json").read_text())
ISSUES_STRUCT = json.loads((OUT / "structure_issues.json").read_text())
R = [json.loads(l) for l in open(OUT / "extracted_sections.jsonl", encoding="utf-8")]
DOCS = list(CFG)

ocr = {}
for d in DOCS:
    f = OUT / "ocr" / f"{d}.jsonl"
    if f.exists():
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); ocr[(d, r["pdf_page"])] = r
scan_kind = {(d, r["pdf_page"]): r["kind"] for d, rows in INV.items() for r in rows}

def tab_line(s):
    return len(re.split(r" {3,}", s.strip())) >= 3
def nws(s): return len(re.sub(r"\s", "", s))

# ---------------------------------------------------------------- page-level facts + char artifacts
pagefacts, art = {}, {}
LIG = re.compile("[\ufb00-\ufb06]"); CID = re.compile(r"\(cid:\d+\)"); MOJ = re.compile("â€|Ã[\x80-\xbf]"); SHY = "\u00ad"
CTRL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f]"); SPLIT = re.compile(r"\b[A-Z] [A-Z]{3,}\b")
for d in DOCS:
    P = [json.loads(l) for l in open(OUT / "pages" / f"{d}.jsonl", encoding="utf-8")]
    a = collections.Counter(); sp = collections.Counter(); pua_pages = []
    for p in P:
        t = p["text_layout"]
        a["ligature_chars"] += len(LIG.findall(t)); a["cid_patterns"] += len(CID.findall(t)); a["mojibake_patterns"] += len(MOJ.findall(t))
        a["soft_hyphens"] += t.count(SHY); a["control_chars"] += len(CTRL.findall(t)); a["replacement_chars_U+FFFD"] += t.count("\ufffd")
        if p["n_pua_chars"]: a["pua_chars"] += p["n_pua_chars"]; pua_pages.append(p["pdf_page"])
        for m in SPLIT.findall(t): sp[m] += 1
    art[d] = {**a, "pua_pages": pua_pages, "kerning_split_word_candidates_total": sum(sp.values()), "kerning_split_top": sp.most_common(6)}
    pagefacts[d] = {"n_pages": len(P), "total_nonws_chars": sum(nws(p["text_layout"]) for p in P),
                    "n_lines_total": sum(len(p["text_layout"].split("\n")) for p in P)}

# ---------------------------------------------------------------- section flags
by_parent = collections.defaultdict(list)
for r in R:
    if r["parent_section_id"]: by_parent[r["parent_section_id"]].append(r)
for r in R:
    f = list(r["quality_flags"])
    lines = [l for l in r["section_text"].split("\n") if l.strip()]
    tr = (sum(tab_line(l) for l in lines) / len(lines)) if lines else 0
    r["tabular_line_ratio"] = round(tr, 3)
    if r["n_chars"] < 150: f.append("tiny_section")
    body = r["section_text"].strip()
    head = r["original_heading"]
    if r["level"] == 2 and r["heading_source"] == "detected_bold_heading" and len(re.sub(r"\s+", "", body)) <= len(re.sub(r"\s+", "", head)) + 2:
        f.append("heading_only")
    if r["n_chars"] > 20000: f.append("very_long_section_not_split")
    if tr >= 0.4 and len(lines) >= 6: f.append("table_heavy_check_structure")
    pg = range(r["start_page"], r["end_page"] + 1)
    kinds = [scan_kind.get((r["document_id"], p)) for p in pg]
    if "image_only" in kinds: f.append("contains_image_only_page")
    if "image_with_text_layer" in kinds: f.append("contains_scanned_page_with_embedded_ocr_layer")
    if any(p in art[r["document_id"]]["pua_pages"] for p in pg): f.append("contains_private_use_glyphs")
    lowocr = [p for p in pg if (r["document_id"], p) in ocr and (ocr[(r["document_id"], p)]["ocr_mean_word_conf"] or 0) < 60]
    if lowocr: f.append("ocr_low_confidence_pages_present")
    if r["level"] == 2 and r["heading_source"] == "chapter_preamble" and r["n_chars"] > 20000: f.append("chapter_has_no_internal_headings_detected")
    r["quality_flags"] = sorted(set(f))
    r["review_status"] = "unlabeled"
with open(OUT / "extracted_sections.jsonl", "w", encoding="utf-8") as fo:
    for r in R: fo.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------- independent conservation checks
cons = {}
for d in DOCS:
    L1 = [r for r in R if r["document_id"] == d and r["level"] == 1]
    L2 = [r for r in R if r["document_id"] == d and r["level"] == 2]
    nws_l1 = sum(nws(r["section_text_raw"]) for r in L1)
    nws_l2 = sum(nws(r["section_text_raw"]) for r in L2) + sum(nws(r["section_text_raw"]) for r in L1 if "contents_page" in r["quality_flags"])
    page_cov = collections.Counter()
    for r in L1:
        for p in r["pages"]: page_cov[p["pdf_page"]] += 1
    # a page may legitimately be shared by two consecutive chapters (chapters begin mid-page)
    missing_pages = [p for p in range(1, pagefacts[d]["n_pages"] + 1) if p not in page_cov]
    ids = [r["example_id"] for r in R if r["document_id"] == d]
    cons[d] = {"nonws_chars_pages": pagefacts[d]["total_nonws_chars"], "nonws_chars_L1_raw": nws_l1, "nonws_chars_L2_raw": nws_l2,
               "L1_equals_pages": nws_l1 == pagefacts[d]["total_nonws_chars"], "L2_equals_pages": nws_l2 == pagefacts[d]["total_nonws_chars"],
               "pages_not_in_any_L1": missing_pages, "duplicate_example_ids": len(ids) - len(set(ids)),
               "company_names_in_doc": sorted({r["company_name"] for r in R if r["document_id"] == d}),
               "source_files_in_doc": sorted({r["source_filename"] for r in R if r["document_id"] == d})}

# ---------------------------------------------------------------- boundary integrity checks
import difflib
LEADER = re.compile(r"(\.{5,}|-{5,}|_{5,}|\u2026{3,})\s*\d{1,3}\s*$")
def _alnum(x): return re.sub(r"[^A-Z0-9]", "", x.upper())
integrity = {}
for d in DOCS:
    leak, nostart = [], []
    for r in R:
        if r["document_id"] != d or r["level"] != 1: continue
        fl = set(r["quality_flags"])
        if not ({"contents_page", "front_matter"} & fl):
            n = sum(1 for l in r["section_text_raw"].split("\n") if LEADER.search(l.rstrip()))
            if n >= 3: leak.append({"example_id": r["example_id"], "leader_lines": n, "heading": r["original_heading"]})
        if r["heading_source"] == "toc_anchored":
            first = [re.sub(r"\s+", " ", l).strip() for l in r["section_text"].split("\n") if l.strip()][:8]
            vs = [r["original_heading"]]
            m = re.match(r"^(?:SECTION\s+)?[IVX]+\s*[.:\-\u2013\u2014]*\s+(.*)$", r["original_heading"], re.I)
            if m: vs.append(m.group(1))
            ok = False
            for l in first:
                a = _alnum(l)
                for v in vs:
                    av = _alnum(v)
                    if av and a and (av in a or a in av and len(a) >= 10 or difflib.SequenceMatcher(None, a[:len(av)+10], av).ratio() >= 0.8):
                        ok = True
            if not ok: nostart.append({"example_id": r["example_id"], "heading": r["original_heading"], "pdf_page": r["start_page"], "first_lines": first[:3], "flags": r["quality_flags"]})
    integrity[d] = {"toc_leak_sections": leak, "toc_anchored_chapters_whose_first_8_lines_lack_the_heading": nostart}

# ---------------------------------------------------------------- risk-factor numbering check
risk = {}
for d in DOCS:
    rf = [r for r in R if r["document_id"] == d and r["level"] == 1 and "RISK FACTOR" in r["original_heading"].upper()]
    if not rf: risk[d] = None; continue
    rf = rf[0]
    kids = by_parent[rf["example_id"]]
    nums = []
    for k in kids:
        m = re.match(r"^(\d{1,3})[.)]\s", k["original_heading"])
        if m: nums.append(int(m.group(1)))
    mx = max(nums) if nums else 0
    risk[d] = {"chapter_example_id": rf["example_id"], "pdf_pages": f"{rf['start_page']}-{rf['end_page']}", "L2_in_chapter": len(kids),
               "numbered_risk_headings_detected": len(nums), "distinct": len(set(nums)), "max_number": mx,
               "numbers_missing_from_1_to_max": len(set(range(1, mx + 1)) - set(nums)),
               "sample_missing": sorted(set(range(1, mx + 1)) - set(nums))[:12]}

# ---------------------------------------------------------------- TOC vs body
toc = {}
for d in DOCS:
    L1 = [r for r in R if r["document_id"] == d and r["level"] == 1 and r.get("toc")]
    how = collections.Counter()
    for r in L1:
        fl = r["quality_flags"]
        how["anchored_on_TOC_page"] += 1 if not any(x.startswith("anchor_by") for x in fl) else 0
        how["anchored_by_forward_scan"] += 1 if "anchor_by_forward_scan" in fl else 0
        how["anchored_by_printed_page_only"] += 1 if "anchor_by_printed_page_only_heading_not_found" in fl else 0
    disc = []
    for r in L1:
        tp, sp = r["toc"]["printed_page"], r["start_printed_page"]
        if tp is not None and sp is not None and sp not in (tp, tp - 1):
            disc.append({"example_id": r["example_id"], "heading": r["original_heading"], "toc_printed": tp, "body_printed": sp, "pdf_page": r["start_page"]})
    toc[d] = {"toc_entries_parsed_total": sum(1 for r in L1) + sum(1 for i in ISSUES_STRUCT if i["doc_id"] == d and i["type"] == "toc_entry_unresolved"),
              "toc_entries_unresolved": sum(1 for i in ISSUES_STRUCT if i["doc_id"] == d and i["type"] == "toc_entry_unresolved"),
              "anchoring": dict(how), "toc_vs_body_page_discrepancies": disc}

# ---------------------------------------------------------------- per-doc stats
stats = {}
for d in DOCS:
    L1 = [r for r in R if r["document_id"] == d and r["level"] == 1]
    L2 = [r for r in R if r["document_id"] == d and r["level"] == 2]
    body_L1 = [r for r in L1 if not ({"front_matter", "contents_page"} & set(r["quality_flags"]))]
    ch = [r["n_chars"] for r in L2]
    fl = collections.Counter(x for r in L2 for x in r["quality_flags"])
    fl1 = collections.Counter(x for r in L1 for x in r["quality_flags"])
    inv = INV[d]
    ocrd = [ocr[(d, r["pdf_page"])] for r in inv if (d, r["pdf_page"]) in ocr]
    stats[d] = {
        "source_filename": CFG[d]["source_filename"], "sha256": MANIFEST[d]["sha256"], "company_name": CFG[d]["company_name"],
        "parts": [{k: p[k] for k in ("part", "document_type", "doc_date", "pdf_start", "pdf_end")} for p in CFG[d]["parts"]],
        "pdf_pages": pagefacts[d]["n_pages"], "pages_with_text_layer": pagefacts[d]["n_pages"] - MANIFEST[d]["n_pages_no_text"],
        "printed_page_numbers_recovered": META[d]["printed_page_found"],
        "header_footer_patterns_detected": META[d]["n_header_footer_patterns"],
        "L1_total": len(L1), "L1_body_chapters_excluding_cover_and_toc": len(body_L1), "L2_total": len(L2),
        "L2_median_chars": int(st.median(ch)), "L2_p95_chars": int(sorted(ch)[int(.95 * len(ch))]), "L2_max_chars": max(ch),
        "L2_flag_counts": dict(fl), "L1_flag_counts": dict(fl1),
        "image_only_pages": [r["pdf_page"] for r in inv if r["kind"] == "image_only"],
        "full_page_image_with_text_layer_pages": [r["pdf_page"] for r in inv if r["kind"] == "image_with_text_layer"],
        "ocr_pages_run": len(ocrd),
        "ocr_mean_conf_over_pages": round(st.mean([o["ocr_mean_word_conf"] for o in ocrd if o["ocr_mean_word_conf"] is not None]), 1) if ocrd else None,
        "ocr_pages_conf_lt_60": sorted(o["pdf_page"] for o in ocrd if (o["ocr_mean_word_conf"] or 0) < 60),
        "char_artifacts": art[d], "conservation": cons[d], "risk_factor_numbering": risk[d], "toc": toc[d], "boundary_integrity": integrity[d],
    }
(OUT / "extraction_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------------------------------------------------------- issues CSV
rows = []
def add(d, ex, pages, typ, sev, detail, action):
    rows.append({"document_id": d, "example_id": ex, "pdf_pages": pages, "issue_type": typ, "status": sev, "detail": detail, "proposed_action": action})
for d in DOCS:
    s = stats[d]
    for p in s["image_only_pages"]:
        o = ocr.get((d, p))
        conf = o["ocr_mean_word_conf"] if o else None
        txt = (o["ocr_text"].strip().replace("\n", " / ")[:80] if o else "")
        if conf is not None and conf < 60:
            add(d, "", p, "image_only_page_ocr_unusable", "confirmed", f"OCR mean conf {conf}; looks like a photograph/graphic (OCR text: {txt!r})", "exclude page text from training; keep as non-text page in provenance")
        else:
            add(d, "", p, "image_only_page_ocr_recovered", "confirmed", f"OCR mean conf {conf}; text {txt!r}", "keep OCR text as separate field; verify manually before use")
    if s["full_page_image_with_text_layer_pages"]:
        pp = s["full_page_image_with_text_layer_pages"]
        ag = [ocr[(d, p)]["embedded_token_agreement_with_ocr"] for p in pp if (d, p) in ocr and ocr[(d, p)]["embedded_token_agreement_with_ocr"] is not None]
        add(d, "", f"{pp[0]}-{pp[-1]}" if len(pp) > 1 else pp[0], "scanned_pages_with_embedded_ocr_layer", "confirmed",
            f"{len(pp)} pages; embedded layer shows OCR-typical errors; token agreement with fresh Tesseract OCR: mean {round(st.mean(ag),3) if ag else 'n/a'} over {len(ag)} pages", "do not trust numeric values in these pages; verify against page image; mark sections needs_review")
    if art[d]["pua_pages"]:
        add(d, "", ",".join(map(str, art[d]["pua_pages"])), "private_use_glyphs", "confirmed", "U+F0D8 (Wingdings arrow bullet) extracted as private-use character", "map to a visible bullet in normalised text (decision in Phase 4); raw kept")
    for dd in toc[d]["toc_vs_body_page_discrepancies"]:
        add(d, dd["example_id"], dd["pdf_page"], "source_toc_page_number_disagrees_with_body", "confirmed", f"{dd['heading']}: TOC says p.{dd['toc_printed']}, body heading at printed p.{dd['body_printed']}", "none (source-document inconsistency); anchored by heading text")
    for r in R:
        if r["document_id"] != d: continue
        if "anchor_by_printed_page_only_heading_not_found" in r["quality_flags"]:
            add(d, r["example_id"], r["start_page"], "chapter_heading_not_found_in_body", "confirmed", f"TOC title '{r['original_heading']}' has no matching heading text; anchored at TOC printed page", "manual check of boundary")
        if "manual_override" in r["quality_flags"]:
            add(d, r["example_id"], f"{r['start_page']}-{r['end_page']}", "manual_structure_override", "confirmed", r.get("evidence", ""), "documented in doc_config.json")
        if r["level"] == 2 and "very_long_section_not_split" in r["quality_flags"]:
            add(d, r["example_id"], f"{r['start_page']}-{r['end_page']}", "very_long_section", "suspected", f"{r['n_chars']} chars under heading '{r['original_heading'][:50]}'", "keep whole (rule 7); link chunks to parent if model-sized chunks are needed; check for merged topics")
    rf = risk[d]
    if rf and rf["numbers_missing_from_1_to_max"]:
        add(d, rf["chapter_example_id"], rf["pdf_pages"], "risk_factor_numbering_gaps", "suspected", f"{rf['numbered_risk_headings_detected']} numbered headings, max #{rf['max_number']}, {rf['numbers_missing_from_1_to_max']} numbers missing (e.g. {rf['sample_missing']})", "inspect: merged risk factors or numbering restarts")
    if art[d]["kerning_split_word_candidates_total"] > 20:
        add(d, "", "", "kerning_split_word_candidates", "suspected", f"{art[d]['kerning_split_word_candidates_total']} matches of '<capital> <CAPS word>' e.g. {art[d]['kerning_split_top'][:3]}", "inspect samples; may be genuine (e.g. 'A B C Ltd') or glyph-gap splits")
    for lk in integrity[d]["toc_leak_sections"]:
        add(d, lk["example_id"], "", "toc_listing_leaked_into_chapter", "confirmed", f"{lk['leader_lines']} TOC-style leader lines in '{lk['heading']}'", "fix anchoring")
    for ns in integrity[d]["toc_anchored_chapters_whose_first_8_lines_lack_the_heading"]:
        add(d, ns["example_id"], ns["pdf_page"], "chapter_does_not_start_with_its_heading", "suspected", f"'{ns['heading']}' first lines: {ns['first_lines']} flags={ns['flags']}", "manual boundary check")
    if not (cons[d]["L1_equals_pages"] and cons[d]["L2_equals_pages"]):
        add(d, "", "", "text_conservation_failed", "confirmed", json.dumps(cons[d]), "fix before use")
with open(OUT / "extraction_issues.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("issues rows:", len(rows))
for d in DOCS:
    s = stats[d]
    print(f"{d:22s} pages={s['pdf_pages']:3d} L1={s['L1_total']:3d} (body {s['L1_body_chapters_excluding_cover_and_toc']:3d}) L2={s['L2_total']:4d} cons_L1={cons[d]['L1_equals_pages']} cons_L2={cons[d]['L2_equals_pages']} pages_missing={cons[d]['pages_not_in_any_L1']} ocr_pages={s['ocr_pages_run']}")
