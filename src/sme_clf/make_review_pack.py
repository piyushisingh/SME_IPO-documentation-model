"""
Complete implementation of make_review_pack.py
Phase 2 Review Pack Generator
Seed: 42
"""

import os
import sys
import json
import csv
import hashlib
import random
import re
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path(r"c:\SMEIPO Model")
CANONICAL_V4 = ROOT / "data" / "claude_v4_extracted" / "data" / "rebuilt_v4"
OUTPUT_DIR = ROOT / "reports" / "review_pack"
SEALED_DIR = OUTPUT_DIR / "sealed"
PRIVATE_DIR = OUTPUT_DIR / "private"
TEST_LOG_PATH = ROOT / "reports" / "test_access_log.jsonl"
CLUSTERS_CSV = ROOT / "reports" / "v4_forensics" / "duplicate_clusters.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEALED_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES_17 = [
    "ipo_details",
    "risk_factors",
    "disclosures_and_definitions",
    "legal_and_regulatory",
    "promoters_and_management",
    "company_information",
    "financial_information",
    "financial_statements",
    "related_party_transactions",
    "capital_structure",
    "objects_of_issue",
    "qualitative_factors",
    "industry_information",
    "business_operations",
    "litigation",
    "declarations",
    "other",
]

CONFIDENCE_VALUES = ["high", "medium", "low"]

def log_test_access(purpose, script_name="make_review_pack.py"):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "purpose": purpose,
        "script": script_name,
        "target": "data/claude_v4_extracted/data/rebuilt_v4/test.jsonl"
    }
    with open(TEST_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sanitize_csv_cell(val):
    if val is None:
        return ""
    val_str = str(val)
    if val_str and val_str[0] in ("=", "+", "-", "@"):
        return "'" + val_str
    return val_str

def write_workbook_and_csv(
    xlsx_path,
    csv_path,
    headers,
    rows,
    allowed_labels,
    reviewer_label_col_idx,
    reviewer_conf_col_idx,
    col_widths=None
):
    wb = openpyxl.Workbook()
    ws_data = wb.active
    ws_data.title = "review_data"
    ws_data.views.sheetView[0].showGridLines = True
    ws_data.freeze_panes = "A2"

    ws_vals = wb.create_sheet(title="allowed_values")
    ws_vals.views.sheetView[0].showGridLines = True
    ws_vals["A1"] = "Allowed Reviewer Labels"
    ws_vals["A1"].font = Font(bold=True)
    for r_idx, val in enumerate(allowed_labels, start=2):
        c = ws_vals.cell(row=r_idx, column=1, value=val)
        c.data_type = 's'

    ws_vals["B1"] = "Allowed Confidence"
    ws_vals["B1"].font = Font(bold=True)
    for r_idx, val in enumerate(CONFIDENCE_VALUES, start=2):
        c = ws_vals.cell(row=r_idx, column=2, value=val)
        c.data_type = 's'

    ws_vals.column_dimensions["A"].width = 30
    ws_vals.column_dimensions["B"].width = 20

    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    for col_idx, h in enumerate(headers, start=1):
        cell = ws_data.cell(row=1, column=col_idx, value=h)
        cell.data_type = 's'
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    num_rows = len(rows)
    for row_idx, r_data in enumerate(rows, start=2):
        for col_idx, val in enumerate(r_data, start=1):
            cell = ws_data.cell(row=row_idx, column=col_idx)
            if val is not None and str(val) != "":
                cell.value = str(val)
                cell.data_type = 's'
            else:
                cell.value = ""
                cell.data_type = 's'

            cell.font = Font(name="Calibri", size=10)
            cell.border = thin_border

            if col_idx in (reviewer_label_col_idx, reviewer_conf_col_idx):
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif col_idx == 1:
                cell.alignment = Alignment(horizontal="center", vertical="top")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

    if num_rows > 0:
        max_r = num_rows + 1
        label_col_letter = get_column_letter(reviewer_label_col_idx)
        conf_col_letter = get_column_letter(reviewer_conf_col_idx)

        dv_label = DataValidation(
            type="list",
            formula1=f"=allowed_values!$A$2:$A${len(allowed_labels)+1}",
            allow_blank=True,
            showDropDown=True
        )
        ws_data.add_data_validation(dv_label)
        dv_label.add(f"{label_col_letter}2:{label_col_letter}{max_r}")

        dv_conf = DataValidation(
            type="list",
            formula1=f"=allowed_values!$B$2:$B${len(CONFIDENCE_VALUES)+1}",
            allow_blank=True,
            showDropDown=True
        )
        ws_data.add_data_validation(dv_conf)
        dv_conf.add(f"{conf_col_letter}2:{conf_col_letter}{max_r}")

    if col_widths:
        for col_idx, w in enumerate(col_widths, start=1):
            col_letter = get_column_letter(col_idx)
            ws_data.column_dimensions[col_letter].width = w
    else:
        for col_idx in range(1, len(headers) + 1):
            col_letter = get_column_letter(col_idx)
            ws_data.column_dimensions[col_letter].width = 25

    wb.save(xlsx_path)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow([sanitize_csv_cell(c) for c in r])

def clean_chapter_title(title):
    t = re.sub(r'^SECTION\s+[IVXLCDM]+\s*[:\-–—]\s*', '', title, flags=re.IGNORECASE)
    t = re.sub(r'^SECTION\s+[IVXLCDM]+\b\s*', '', t, flags=re.IGNORECASE)
    return t.strip().upper()

def extract_body(inp):
    if "Text:\n" in inp:
        return inp.split("Text:\n", 1)[1].strip()
    return inp.strip()

def main():
    print("=" * 70)
    print("STARTING MAKE_REVIEW_PACK (Seed: 42)")
    print("=" * 70)
    random.seed(42)

    # Load master records
    master_p = CANONICAL_V4 / "master_sections.jsonl"
    master_records = [json.loads(l) for l in open(master_p, encoding="utf-8") if l.strip()]
    master_by_id = {m["example_id"]: m for m in master_records}

    # Load model-ready records
    train_p = CANONICAL_V4 / "train.jsonl"
    val_p = CANONICAL_V4 / "validation.jsonl"
    test_p = CANONICAL_V4 / "test.jsonl"
    train_records = [json.loads(l) for l in open(train_p, encoding="utf-8") if l.strip()]
    val_records = [json.loads(l) for l in open(val_p, encoding="utf-8") if l.strip()]
    
    log_test_access("make_review_pack.py: Loading test.jsonl for review pack creation and sealed sampling")
    test_records = [json.loads(l) for l in open(test_p, encoding="utf-8") if l.strip()]

    # Load manifest
    manifest_p = CANONICAL_V4 / "model_ready_manifest.jsonl"
    manifest_records = [json.loads(l) for l in open(manifest_p, encoding="utf-8") if l.strip()]
    manifest_by_id = {m["example_id"]: m for m in manifest_records}

    # Load review queue
    review_q_p = CANONICAL_V4 / "review_queue.jsonl"
    review_q_records = [json.loads(l) for l in open(review_q_p, encoding="utf-8") if l.strip()]

    # Load duplicate clusters
    cluster_by_eid = {}
    if CLUSTERS_CSV.exists():
        with open(CLUSTERS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cluster_by_eid[r["example_id"]] = r["cluster_id"]

    # --------------------------------------------------------------------------
    # FILE A: 01_chapter_mapping_review
    # --------------------------------------------------------------------------
    print("\n--- Generating File A: 01_chapter_mapping_review ---")
    # Identify all distinct normalized chapter titles across the 6 documents
    # Chapter titles come from L1 records in master_sections
    l1_records = [m for m in master_records if m.get("level") == 1 or m.get("level") == "L1"]
    
    # Group by clean title
    chapters = defaultdict(lambda: {
        "rule_ids": Counter(),
        "assigned_cats": Counter(),
        "overlap_alts": set(),
        "docs": set(),
        "master_ids": set(),
        "subheadings": Counter(),
    })

    for l1 in l1_records:
        raw_title = l1.get("original_heading") or l1.get("normalized_heading") or ""
        clean_t = clean_chapter_title(raw_title)
        c_dict = chapters[clean_t]
        c_dict["rule_ids"][l1.get("label_rule_id", "N/A")] += 1
        c_dict["assigned_cats"][l1.get("category", "N/A")] += 1
        c_dict["docs"].add(l1.get("document_id"))
        c_dict["master_ids"].add(l1["example_id"])
        
        # Check alternatives
        prop = l1.get("category_proposed", [])
        for p_cat in prop:
            if p_cat != l1.get("category"):
                c_dict["overlap_alts"].add(p_cat)

    # For each chapter, find all L2 sections under it
    for m in master_records:
        if m.get("level") == 2 or m.get("level") == "L2":
            pid = m.get("parent_section_id")
            parent_l1 = master_by_id.get(pid)
            if parent_l1:
                clean_t = clean_chapter_title(parent_l1.get("original_heading") or "")
            else:
                clean_t = clean_chapter_title(m.get("parent_heading") or "")
            if clean_t in chapters:
                c_dict = chapters[clean_t]
                c_dict["master_ids"].add(m["example_id"])
                sh = m.get("normalized_heading") or m.get("original_heading")
                if sh and clean_chapter_title(sh) != clean_t:
                    c_dict["subheadings"][sh] += 1

    # Map model ready examples to chapters
    for mr in manifest_records:
        ph = mr.get("parent_heading") or ""
        clean_t = clean_chapter_title(ph)
        if clean_t in chapters:
            chapters[clean_t]["model_ready_count"] = chapters[clean_t].get("model_ready_count", 0) + 1

    file_a_rows = []
    for clean_t, data in sorted(chapters.items()):
        rule_id = data["rule_ids"].most_common(1)[0][0] if data["rule_ids"] else "N/A"
        cat = data["assigned_cats"].most_common(1)[0][0] if data["assigned_cats"] else "N/A"
        overlap = ", ".join(sorted(data["overlap_alts"])) if data["overlap_alts"] else "none"
        n_docs = len(data["docs"])
        n_master = len(data["master_ids"])
        n_mr = data.get("model_ready_count", 0)
        
        # top 3 subheadings
        top_sh = [s for s, _ in data["subheadings"].most_common(3)]
        sh1 = top_sh[0] if len(top_sh) > 0 else ""
        sh2 = top_sh[1] if len(top_sh) > 1 else ""
        sh3 = top_sh[2] if len(top_sh) > 2 else ""

        file_a_rows.append([
            clean_t,
            rule_id,
            cat,
            overlap,
            n_docs,
            n_master,
            n_mr,
            sh1,
            sh2,
            sh3,
            "",  # reviewer_label
            "",  # reviewer_confidence
            ""   # reviewer_comment
        ])

    headers_a = [
        "title", "rule_id", "assigned_category", "overlap_alternative",
        "n_documents", "n_master_records", "n_model_ready_examples",
        "example_subheading_1", "example_subheading_2", "example_subheading_3",
        "reviewer_label", "reviewer_confidence", "reviewer_comment"
    ]
    allowed_a = CATEGORIES_17 + ["mixed_chapter", "unsure"]
    write_workbook_and_csv(
        OUTPUT_DIR / "01_chapter_mapping_review.xlsx",
        OUTPUT_DIR / "01_chapter_mapping_review.csv",
        headers_a,
        file_a_rows,
        allowed_a,
        reviewer_label_col_idx=11,
        reviewer_conf_col_idx=12,
        col_widths=[32, 12, 22, 22, 12, 14, 18, 30, 30, 30, 22, 18, 35]
    )
    print(f"File A written: {len(file_a_rows)} distinct chapter titles.")

    # --------------------------------------------------------------------------
    # FILE B: 02_ambiguous_review
    # --------------------------------------------------------------------------
    print("\n--- Generating File B: 02_ambiguous_review ---")
    # The 60 L2 records with review_status=ambiguous
    ambiguous_l2 = [m for m in master_records if (m.get("level") == 2 or m.get("level") == "L2") and m.get("review_status") == "ambiguous"]
    ambiguous_l2.sort(key=lambda x: (x.get("document_id", ""), x.get("start_page", 0), x["example_id"]))

    file_b_rows = []
    for m in ambiguous_l2:
        rev_id = m["example_id"]
        doc = m.get("document_id", "")
        pages = f"{m.get('start_page', '')}-{m.get('end_page', '')}"
        heading = m.get("normalized_heading") or m.get("original_heading") or ""
        chapter = m.get("parent_heading") or ""
        body = m.get("section_text") or m.get("section_text_raw") or ""
        preview = normalize_text(body)[:600]
        prop_alts = ", ".join(m.get("category_proposed", []))

        file_b_rows.append([
            rev_id, doc, pages, heading, chapter, preview, prop_alts, "", "", ""
        ])

    headers_b = [
        "review_id", "document_id", "pdf_pages", "heading", "parent_chapter",
        "text_preview", "proposed_alternatives",
        "reviewer_label", "reviewer_confidence", "reviewer_comment"
    ]
    allowed_b = CATEGORIES_17 + ["segmentation_error", "unsure"]
    write_workbook_and_csv(
        OUTPUT_DIR / "02_ambiguous_review.xlsx",
        OUTPUT_DIR / "02_ambiguous_review.csv",
        headers_b,
        file_b_rows,
        allowed_b,
        reviewer_label_col_idx=8,
        reviewer_conf_col_idx=9,
        col_widths=[28, 20, 12, 30, 30, 65, 25, 22, 18, 35]
    )
    print(f"File B written: {len(file_b_rows)} ambiguous L2 records.")

    # --------------------------------------------------------------------------
    # FILE C: 03_queue_labeled
    # --------------------------------------------------------------------------
    print("\n--- Generating File C: 03_queue_labeled ---")
    # records with heading_keyword_overrides_chapter_label (48) or suspected_label_content_mismatch (29), de-duplicated
    queue_targeted = []
    seen_c_ids = set()
    for r in review_q_records:
        reasons = r.get("reasons", [])
        if "heading_keyword_overrides_chapter_label" in reasons or "suspected_label_content_mismatch" in reasons:
            eid = r["example_id"]
            if eid not in seen_c_ids:
                seen_c_ids.add(eid)
                queue_targeted.append(r)

    queue_targeted.sort(key=lambda x: (x.get("document_id", ""), x["example_id"]))

    file_c_rows = []
    for r in queue_targeted:
        eid = r["example_id"]
        doc = r.get("document_id", "")
        pages = r.get("pdf_pages", "")
        heading = r.get("heading", "")
        m_rec = master_by_id.get(eid, {})
        chapter = m_rec.get("parent_heading") or ""
        assigned = r.get("assigned_category") or m_rec.get("category") or ""
        alts = ", ".join(r.get("proposed_alternatives", []))
        preview = normalize_text(r.get("text_preview") or m_rec.get("section_text") or "")[:600]

        file_c_rows.append([
            eid, doc, pages, heading, chapter, assigned, alts, preview, "", "", ""
        ])

    headers_c = [
        "review_id", "document_id", "pdf_pages", "heading", "parent_chapter",
        "assigned_label", "proposed_alternatives", "text_preview",
        "reviewer_label", "reviewer_confidence", "reviewer_comment"
    ]
    allowed_c = CATEGORIES_17 + ["segmentation_error", "unsure"]
    write_workbook_and_csv(
        OUTPUT_DIR / "03_queue_labeled.xlsx",
        OUTPUT_DIR / "03_queue_labeled.csv",
        headers_c,
        file_c_rows,
        allowed_c,
        reviewer_label_col_idx=9,
        reviewer_conf_col_idx=10,
        col_widths=[28, 20, 12, 30, 30, 22, 25, 65, 22, 18, 35]
    )
    print(f"File C written: {len(file_c_rows)} records (48 overrides + 29 mismatches de-duplicated).")

    # --------------------------------------------------------------------------
    # FILE D: 04_segmentation_check
    # --------------------------------------------------------------------------
    print("\n--- Generating File D: 04_segmentation_check ---")
    # the 38 heading_is_paragraph_suspected plus 7 fragment_heading_suspected records
    seg_records = []
    seen_d_ids = set()
    for m in master_records:
        qf = m.get("quality_flags", [])
        if "heading_is_paragraph_suspected" in qf or "fragment_heading_suspected" in qf:
            eid = m["example_id"]
            if eid not in seen_d_ids:
                seen_d_ids.add(eid)
                seg_records.append(m)

    seg_records.sort(key=lambda x: (x.get("document_id", ""), x.get("start_page", 0), x["example_id"]))

    file_d_rows = []
    for m in seg_records:
        eid = m["example_id"]
        doc = m.get("document_id", "")
        pages = f"{m.get('start_page', '')}-{m.get('end_page', '')}"
        heading = (m.get("normalized_heading") or m.get("original_heading") or "")[:200]
        preview = normalize_text(m.get("section_text") or m.get("section_text_raw") or "")[:400]

        file_d_rows.append([
            eid, doc, pages, heading, preview, "", "", ""
        ])

    headers_d = [
        "review_id", "document_id", "pdf_pages", "heading", "text_preview",
        "reviewer_label", "reviewer_confidence", "reviewer_comment"
    ]
    allowed_d = ["ok", "segmentation_error", "unsure"]
    write_workbook_and_csv(
        OUTPUT_DIR / "04_segmentation_check.xlsx",
        OUTPUT_DIR / "04_segmentation_check.csv",
        headers_d,
        file_d_rows,
        allowed_d,
        reviewer_label_col_idx=6,
        reviewer_conf_col_idx=7,
        col_widths=[28, 20, 12, 35, 65, 20, 18, 35]
    )
    print(f"File D written: {len(file_d_rows)} segmentation check records (38 paragraph + 7 fragment).")

    # --------------------------------------------------------------------------
    # SAMPLING LOGIC FOR E, F, G, H
    # --------------------------------------------------------------------------
    print("\n--- Preparing Sampling Pool for Blind Files (E, F, G) ---")
    
    # Filter eligible examples:
    # not ambiguous, not extraction_error, not flagged heading_is_paragraph_suspected or fragment_heading_suspected,
    # and at most ONE example per duplicate cluster
    def get_eligible_examples(records, split_name):
        eligible = []
        cluster_seen = set()
        for r in records:
            eid = r["example_id"]
            m_rec = master_by_id.get(eid, {})
            # check review_status
            if m_rec.get("review_status") == "ambiguous":
                continue
            # check quality flags
            qf = set(m_rec.get("quality_flags", []))
            if "extraction_error" in qf:
                continue
            if "heading_is_paragraph_suspected" in qf or "fragment_heading_suspected" in qf:
                continue
            # check duplicate cluster
            cid = cluster_by_eid.get(eid)
            if cid:
                if cid in cluster_seen:
                    continue
                cluster_seen.add(cid)
            eligible.append(r)
        return eligible

    eligible_val = get_eligible_examples(val_records, "validation")
    eligible_test = get_eligible_examples(test_records, "test")
    eligible_train = get_eligible_examples(train_records, "train")

    print(f"Eligible records: Train={len(eligible_train)}/{len(train_records)}, Val={len(eligible_val)}/{len(val_records)}, Test={len(eligible_test)}/{len(test_records)}")

    # Stratified 100-sample allocator for E and G
    def sample_stratified_100(eligible_list, seed=42):
        random.seed(seed)
        by_cat = defaultdict(list)
        for ex in eligible_list:
            by_cat[ex["output"]].append(ex)

        # compute N_h
        N_h = {cat: len(items) for cat, items in by_cat.items()}
        
        # Determine target n_h:
        # 1. categories with < 5 eligible contribute ALL eligible examples
        # 2. categories with >= 5 eligible must have at least 3
        # 3. no category may exceed 25
        n_h = {}
        allocated = 0
        for cat, count in N_h.items():
            if count < 5:
                n_h[cat] = count
            else:
                n_h[cat] = 3
            allocated += n_h[cat]

        # Remaining budget to allocate up to 100
        budget = 100 - allocated
        # Distribute proportionally to remaining (N_h[cat] - n_h[cat]), capped at 25 total
        while budget > 0:
            candidates = [c for c in N_h if n_h[c] < min(25, N_h[c])]
            if not candidates:
                break
            # Pick candidate with largest remaining pool
            candidates.sort(key=lambda c: (N_h[c] - n_h[c]), reverse=True)
            for c in candidates:
                if budget == 0: break
                if n_h[c] < min(25, N_h[c]):
                    n_h[c] += 1
                    budget -= 1

        # Now sample n_h[cat] items deterministically
        sampled = []
        for cat in sorted(N_h.keys()):
            items = list(by_cat[cat])
            items.sort(key=lambda x: x["example_id"])
            sampled.extend(random.sample(items, n_h[cat]))

        random.shuffle(sampled)
        return sampled, N_h, n_h

    # Sample E (Validation)
    sample_e, N_h_val, n_h_val = sample_stratified_100(eligible_val, seed=42)
    # Sample G (Test)
    sample_g, N_h_test, n_h_test = sample_stratified_100(eligible_test, seed=42)

    # Sample F (Train Rule Audit)
    # for every label_rule_id with > 50 train examples: sample 4; for 20-50: sample 2
    by_rule = defaultdict(list)
    for ex in eligible_train:
        m_rec = master_by_id.get(ex["example_id"], {})
        rule = m_rec.get("label_rule_id", "none")
        by_rule[rule].append(ex)

    sample_f = []
    N_h_train_rules = {}
    n_h_train_rules = {}
    random.seed(42)
    for rule in sorted(by_rule.keys()):
        items = list(by_rule[rule])
        items.sort(key=lambda x: x["example_id"])
        cnt = len(items)
        N_h_train_rules[rule] = cnt
        if cnt > 50:
            target = 4
        elif cnt >= 20:
            target = 2
        else:
            target = 0
        n_h_train_rules[rule] = target
        if target > 0:
            sample_f.extend(random.sample(items, min(target, cnt)))

    random.shuffle(sample_f)

    # Verify disjointness across E, F, G
    e_ids = set(x["example_id"] for x in sample_e)
    f_ids = set(x["example_id"] for x in sample_f)
    g_ids = set(x["example_id"] for x in sample_g)
    assert len(e_ids & f_ids) == 0, "Overlap between E and F!"
    assert len(e_ids & g_ids) == 0, "Overlap between E and G!"
    assert len(f_ids & g_ids) == 0, "Overlap between F and G!"

    # Master Answer Key dict
    answer_key = {}

    # --------------------------------------------------------------------------
    # FILE E: 05_blind_validation
    # --------------------------------------------------------------------------
    print("\n--- Generating File E: 05_blind_validation ---")
    file_e_rows = []
    for idx, ex in enumerate(sample_e, start=1):
        blind_id = f"VAL-{idx:03d}"
        eid = ex["example_id"]
        inp = ex["input"]
        # extract heading and body
        if "Heading:" in inp and "Text:\n" in inp:
            parts = inp.split("Text:\n", 1)
            heading = parts[0].replace("Heading:", "").strip()
            body = parts[1].strip()
        else:
            heading = ""
            body = inp
        preview = normalize_text(body)[:800]
        file_e_rows.append([blind_id, heading, preview, "", "", ""])

        m_rec = master_by_id.get(eid, {})
        answer_key[blind_id] = {
            "example_id": eid,
            "assigned_label": ex["output"],
            "rule_id": m_rec.get("label_rule_id", "N/A"),
            "parent_chapter": m_rec.get("parent_heading", "N/A"),
            "document": m_rec.get("document_id", "century"),
            "split": "validation"
        }

    headers_blind = ["blind_id", "heading", "text_preview", "reviewer_label", "reviewer_confidence", "reviewer_comment"]
    allowed_blind = CATEGORIES_17 + ["segmentation_error", "unsure"]

    write_workbook_and_csv(
        OUTPUT_DIR / "05_blind_validation.xlsx",
        OUTPUT_DIR / "05_blind_validation.csv",
        headers_blind,
        file_e_rows,
        allowed_blind,
        reviewer_label_col_idx=4,
        reviewer_conf_col_idx=5,
        col_widths=[14, 35, 75, 22, 18, 35]
    )
    print(f"File E written: {len(file_e_rows)} blind validation records.")

    # --------------------------------------------------------------------------
    # FILE F: 06_blind_rule_audit
    # --------------------------------------------------------------------------
    print("\n--- Generating File F: 06_blind_rule_audit ---")
    file_f_rows = []
    for idx, ex in enumerate(sample_f, start=1):
        blind_id = f"RULE-{idx:03d}"
        eid = ex["example_id"]
        inp = ex["input"]
        if "Heading:" in inp and "Text:\n" in inp:
            parts = inp.split("Text:\n", 1)
            heading = parts[0].replace("Heading:", "").strip()
            body = parts[1].strip()
        else:
            heading = ""
            body = inp
        preview = normalize_text(body)[:800]
        file_f_rows.append([blind_id, heading, preview, "", "", ""])

        m_rec = master_by_id.get(eid, {})
        answer_key[blind_id] = {
            "example_id": eid,
            "assigned_label": ex["output"],
            "rule_id": m_rec.get("label_rule_id", "N/A"),
            "parent_chapter": m_rec.get("parent_heading", "N/A"),
            "document": m_rec.get("document_id", ""),
            "split": "train"
        }

    write_workbook_and_csv(
        OUTPUT_DIR / "06_blind_rule_audit.xlsx",
        OUTPUT_DIR / "06_blind_rule_audit.csv",
        headers_blind,
        file_f_rows,
        allowed_blind,
        reviewer_label_col_idx=4,
        reviewer_conf_col_idx=5,
        col_widths=[14, 35, 75, 22, 18, 35]
    )
    print(f"File F written: {len(file_f_rows)} blind rule audit records.")

    # --------------------------------------------------------------------------
    # FILE G: 07_blind_test_SEALED (Write ONLY to sealed/)
    # --------------------------------------------------------------------------
    print("\n--- Generating File G: 07_blind_test_SEALED ---")
    file_g_rows = []
    for idx, ex in enumerate(sample_g, start=1):
        blind_id = f"TEST-{idx:03d}"
        eid = ex["example_id"]
        inp = ex["input"]
        if "Heading:" in inp and "Text:\n" in inp:
            parts = inp.split("Text:\n", 1)
            heading = parts[0].replace("Heading:", "").strip()
            body = parts[1].strip()
        else:
            heading = ""
            body = inp
        preview = normalize_text(body)[:800]
        file_g_rows.append([blind_id, heading, preview, "", "", ""])

        m_rec = master_by_id.get(eid, {})
        answer_key[blind_id] = {
            "example_id": eid,
            "assigned_label": ex["output"],
            "rule_id": m_rec.get("label_rule_id", "N/A"),
            "parent_chapter": m_rec.get("parent_heading", "N/A"),
            "document": m_rec.get("document_id", "ideas_electricals"),
            "split": "test"
        }

    write_workbook_and_csv(
        SEALED_DIR / "07_blind_test_SEALED.xlsx",
        SEALED_DIR / "07_blind_test_SEALED.csv",
        headers_blind,
        file_g_rows,
        allowed_blind,
        reviewer_label_col_idx=4,
        reviewer_conf_col_idx=5,
        col_widths=[14, 35, 75, 22, 18, 35]
    )

    with open(SEALED_DIR / "SEALED_README.txt", "w", encoding="utf-8") as f:
        f.write("Do not open until rule changes have been approved in Phase 6; corrections here must NOT be used to change labeling rules.\n")

    print(f"File G written to {SEALED_DIR.relative_to(ROOT)}: {len(file_g_rows)} sealed test records.")

    # --------------------------------------------------------------------------
    # FILE H: 08_consistency_check (30 rows copied from E)
    # --------------------------------------------------------------------------
    print("\n--- Generating File H: 08_consistency_check ---")
    random.seed(42)
    # Sample 30 rows from file_e_rows
    h_source_indices = sorted(random.sample(range(len(file_e_rows)), 30))
    file_h_rows = []
    for new_idx, s_idx in enumerate(h_source_indices, start=1):
        src_row = file_e_rows[s_idx]
        old_blind_id = src_row[0]
        new_blind_id = f"CONSIST-{new_idx:03d}"
        file_h_rows.append([new_blind_id, src_row[1], src_row[2], "", "", ""])
        
        # Link in answer key
        src_ans = answer_key[old_blind_id]
        answer_key[new_blind_id] = {
            "replicated_from": old_blind_id,
            "example_id": src_ans["example_id"],
            "assigned_label": src_ans["assigned_label"],
            "rule_id": src_ans["rule_id"],
            "parent_chapter": src_ans["parent_chapter"],
            "document": src_ans["document"],
            "split": "consistency_validation"
        }

    write_workbook_and_csv(
        OUTPUT_DIR / "08_consistency_check.xlsx",
        OUTPUT_DIR / "08_consistency_check.csv",
        headers_blind,
        file_h_rows,
        allowed_blind,
        reviewer_label_col_idx=4,
        reviewer_conf_col_idx=5,
        col_widths=[14, 35, 75, 22, 18, 35]
    )
    print(f"File H written: {len(file_h_rows)} consistency check records.")

    # --------------------------------------------------------------------------
    # WRITE ANSWER KEY AND SAMPLING MANIFEST
    # --------------------------------------------------------------------------
    answer_key_path = PRIVATE_DIR / "answer_key.json"
    with open(answer_key_path, "w", encoding="utf-8") as f:
        json.dump(answer_key, f, indent=2)
    print(f"Answer key written strictly to private: {answer_key_path.relative_to(ROOT)} ({len(answer_key)} entries)")

    # Build sampling manifest
    sampling_manifest = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": 42,
            "script": "src/sme_clf/make_review_pack.py",
            "source_hashes": {
                "master_sections.jsonl": sha256_file(master_p),
                "train.jsonl": sha256_file(train_p),
                "validation.jsonl": sha256_file(val_p),
                "test.jsonl": sha256_file(test_p),
            }
        },
        "file_e_validation_strata": {
            cat: {
                "N_h_eligible": N_h_val[cat],
                "n_h_sampled": n_h_val[cat],
                "inclusion_probability": round(n_h_val[cat] / N_h_val[cat], 5) if N_h_val[cat] > 0 else 0
            }
            for cat in sorted(N_h_val.keys())
        },
        "file_g_test_strata": {
            cat: {
                "N_h_eligible": N_h_test[cat],
                "n_h_sampled": n_h_test[cat],
                "inclusion_probability": round(n_h_test[cat] / N_h_test[cat], 5) if N_h_test[cat] > 0 else 0
            }
            for cat in sorted(N_h_test.keys())
        },
        "file_f_train_rule_strata": {
            rule: {
                "N_h_eligible": N_h_train_rules[rule],
                "n_h_sampled": n_h_train_rules[rule],
                "inclusion_probability": round(n_h_train_rules[rule] / N_h_train_rules[rule], 5) if N_h_train_rules[rule] > 0 else 0
            }
            for rule in sorted(N_h_train_rules.keys())
            if n_h_train_rules[rule] > 0
        },
        "sampled_example_ids": {
            "file_e_validation": [ex["example_id"] for ex in sample_e],
            "file_f_rule_audit": [ex["example_id"] for ex in sample_f],
            "file_g_sealed_test": [ex["example_id"] for ex in sample_g],
            "file_h_consistency": [ex["example_id"] for ex in [sample_e[i] for i in h_source_indices]],
        }
    }

    manifest_path = OUTPUT_DIR / "sampling_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(sampling_manifest, f, indent=2)
    print(f"Sampling manifest written to: {manifest_path.relative_to(ROOT)}")
    print("=" * 70)
    print("MAKE_REVIEW_PACK COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
