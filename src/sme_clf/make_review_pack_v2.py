"""
make_review_pack_v2.py
Phase 2 Corrected Review Pack Generator
Seed: 42

Inputs strictly:
- data/claude_v4_extracted/data/rebuilt_v4/{master_sections.jsonl, model_ready_manifest.jsonl, train.jsonl, validation.jsonl, test.jsonl, review_queue.jsonl, dataset_statistics.json}
- reports/v4_forensics/duplicate_clusters.csv

Outputs strictly:
- reports/review_pack_v2/{01_chapter_mapping_review, 02_ambiguous_review, 03_queue_labeled, 04_segmentation_check, 05_blind_validation, 06_blind_rule_audit, 08_consistency_check}.{xlsx,csv}
- reports/review_pack_v2/sealed/07_blind_test_SEALED.{xlsx,csv} + SEALED_README.txt
- reports/review_pack_v2/private/answer_key.json
- reports/review_pack_v2/sampling_manifest.json
- reports/review_pack_v2/review_guide.md
- reports/review_pack_v2/review_plan.md
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
OUTPUT_DIR = ROOT / "reports" / "review_pack_v2"
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

def log_test_access(purpose, script_name="make_review_pack_v2.py"):
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
    ws_data.title = "Review"

    # Styling constants
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    text_font = Font(name="Calibri", size=10)
    center_align = Alignment(horizontal="center", vertical="top")
    left_align = Alignment(horizontal="left", vertical="top", wrap_text=True)
    thin_border_side = Side(style="thin", color="D9D9D9")
    cell_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)

    # Write headers
    for col_idx, header in enumerate(headers, start=1):
        cell = ws_data.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = cell_border

    # Write data rows
    for row_idx, row_data in enumerate(rows, start=2):
        for col_idx, val in enumerate(row_data, start=1):
            cell = ws_data.cell(row=row_idx, column=col_idx)
            if val is not None:
                cell.data_type = "s"
                cell.value = str(val)
            else:
                cell.value = ""
            cell.font = text_font
            cell.alignment = left_align
            cell.border = cell_border

    # Set column widths
    if col_widths:
        for idx, w in enumerate(col_widths, start=1):
            col_letter = get_column_letter(idx)
            ws_data.column_dimensions[col_letter].width = w

    # Freeze header row
    ws_data.freeze_panes = "A2"

    # Validation reference sheet (VISIBLE, NO HIDDEN SHEETS/ROWS/COLUMNS)
    ws_allowed = wb.create_sheet(title="allowed_values")
    ws_allowed.views.sheetView[0].hidden = False

    ws_allowed.cell(row=1, column=1, value="Allowed Reviewer Labels").font = Font(bold=True)
    for idx, l in enumerate(allowed_labels, start=2):
        cell = ws_allowed.cell(row=idx, column=1)
        cell.data_type = "s"
        cell.value = str(l)

    ws_allowed.cell(row=1, column=2, value="Allowed Confidence Values").font = Font(bold=True)
    for idx, c in enumerate(CONFIDENCE_VALUES, start=2):
        cell = ws_allowed.cell(row=idx, column=2)
        cell.data_type = "s"
        cell.value = str(c)

    ws_allowed.column_dimensions["A"].width = 28
    ws_allowed.column_dimensions["B"].width = 28

    n_labels = len(allowed_labels)
    n_confs = len(CONFIDENCE_VALUES)

    label_dv = DataValidation(
        type="list",
        formula1=f"=allowed_values!$A$2:$A${n_labels + 1}",
        allow_blank=True,
        showDropDown=True
    )
    conf_dv = DataValidation(
        type="list",
        formula1=f"=allowed_values!$B$2:$B${n_confs + 1}",
        allow_blank=True,
        showDropDown=True
    )

    ws_data.add_data_validation(label_dv)
    ws_data.add_data_validation(conf_dv)

    max_row = max(len(rows) + 1, 100)
    lbl_col_letter = get_column_letter(reviewer_label_col_idx)
    conf_col_letter = get_column_letter(reviewer_conf_col_idx)

    label_dv.add(f"{lbl_col_letter}2:{lbl_col_letter}{max_row}")
    conf_dv.add(f"{conf_col_letter}2:{conf_col_letter}{max_row}")

    wb.save(xlsx_path)

    # Write UTF-8-BOM CSV twin
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow([sanitize_csv_cell(c) for c in r])

def clean_chapter_title(title):
    t = re.sub(r'^SECTION\s+[IVXLCDM]+\s*[:\-–—]\s*', '', title, flags=re.IGNORECASE)
    t = re.sub(r'^SECTION\s+[IVXLCDM]+\b\s*', '', t, flags=re.IGNORECASE)
    return t.strip().upper()

def allocate_strata_proportional(N_h_dict, total_target=100, min_large=3, max_cap=25):
    """
    Proportional stratified sampling algorithm:
    1. Categories with N_h < 5 contribute ALL eligible examples (n_h = N_h).
    2. Categories with N_h >= 5 receive at least min_large (3).
    3. Remaining budget is allocated proportionally to N_h (Hamilton's largest remainder method).
    4. No category exceeds max_cap (25).
    """
    n_h = {}
    for c, N in N_h_dict.items():
        if N < 5:
            n_h[c] = N
        else:
            n_h[c] = min(min_large, N)

    current = sum(n_h.values())
    budget = total_target - current

    large_cats = [c for c, N in N_h_dict.items() if N >= 5]

    while budget > 0:
        eligible_for_more = [c for c in large_cats if n_h[c] < min(max_cap, N_h_dict[c])]
        if not eligible_for_more:
            break
        sum_N = sum(N_h_dict[c] for c in eligible_for_more)
        quotas = {c: budget * (N_h_dict[c] / sum_N) for c in eligible_for_more}
        floors = {c: int(quotas[c]) for c in eligible_for_more}
        added = sum(floors.values())
        if added > 0 and added <= budget:
            for c in eligible_for_more:
                add_c = min(floors[c], min(max_cap, N_h_dict[c]) - n_h[c])
                n_h[c] += add_c
                budget -= add_c
        else:
            # Distribute 1 by 1 by largest remainder
            remainders = sorted(eligible_for_more, key=lambda c: (quotas[c] - int(quotas[c]), N_h_dict[c]), reverse=True)
            for c in remainders:
                if budget == 0:
                    break
                if n_h[c] < min(max_cap, N_h_dict[c]):
                    n_h[c] += 1
                    budget -= 1

    return n_h

def main():
    print("=" * 70)
    print("STARTING MAKE_REVIEW_PACK_V2 (Seed: 42)")
    print("=" * 70)
    random.seed(42)

    # 1. Load inputs
    master_p = CANONICAL_V4 / "master_sections.jsonl"
    master_records = [json.loads(l) for l in open(master_p, encoding="utf-8") if l.strip()]
    master_by_id = {m["example_id"]: m for m in master_records}

    manifest_p = CANONICAL_V4 / "model_ready_manifest.jsonl"
    manifest_records = [json.loads(l) for l in open(manifest_p, encoding="utf-8") if l.strip()]
    manifest_by_id = {m["example_id"]: m for m in manifest_records}

    train_p = CANONICAL_V4 / "train.jsonl"
    val_p = CANONICAL_V4 / "validation.jsonl"
    test_p = CANONICAL_V4 / "test.jsonl"
    train_records = [json.loads(l) for l in open(train_p, encoding="utf-8") if l.strip()]
    val_records = [json.loads(l) for l in open(val_p, encoding="utf-8") if l.strip()]

    log_test_access("make_review_pack_v2.py: Loading test.jsonl for review pack v2 creation and sealed sampling")
    test_records = [json.loads(l) for l in open(test_p, encoding="utf-8") if l.strip()]

    review_q_p = CANONICAL_V4 / "review_queue.jsonl"
    review_q_records = [json.loads(l) for l in open(review_q_p, encoding="utf-8") if l.strip()]

    stats_p = CANONICAL_V4 / "dataset_statistics.json"
    dataset_stats = json.load(open(stats_p, encoding="utf-8"))

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
    l1_records = [m for m in master_records if m.get("level") == 1 or m.get("level") == "L1"]
    chapters = defaultdict(lambda: {
        "rule_ids": Counter(),
        "assigned_cats": Counter(),
        "overlap_alts": set(),
        "docs": set(),
        "master_ids": set(),
        "subheadings": Counter(),
        "model_ready_count": 0
    })

    for l1 in l1_records:
        raw_title = l1.get("original_heading") or l1.get("normalized_heading") or ""
        clean_t = clean_chapter_title(raw_title)
        c_dict = chapters[clean_t]
        c_dict["rule_ids"][l1.get("label_rule_id", "N/A")] += 1
        c_dict["assigned_cats"][l1.get("category", "N/A")] += 1
        c_dict["docs"].add(l1.get("document_id"))
        c_dict["master_ids"].add(l1["example_id"])
        prop = l1.get("category_proposed", [])
        for p_cat in prop:
            if p_cat != l1.get("category"):
                c_dict["overlap_alts"].add(p_cat)

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

    for mr in manifest_records:
        ph = mr.get("parent_heading") or ""
        clean_t = clean_chapter_title(ph)
        if clean_t in chapters:
            chapters[clean_t]["model_ready_count"] += 1

    file_a_rows = []
    for clean_t, data in sorted(chapters.items()):
        rule_id = data["rule_ids"].most_common(1)[0][0] if data["rule_ids"] else "N/A"
        cat = data["assigned_cats"].most_common(1)[0][0] if data["assigned_cats"] else "N/A"
        overlap = ", ".join(sorted(data["overlap_alts"])) if data["overlap_alts"] else "none"
        n_docs = len(data["docs"])
        n_master = len(data["master_ids"])
        n_mr = data["model_ready_count"]
        top_sh = [s for s, _ in data["subheadings"].most_common(3)]
        sh1 = top_sh[0] if len(top_sh) > 0 else ""
        sh2 = top_sh[1] if len(top_sh) > 1 else ""
        sh3 = top_sh[2] if len(top_sh) > 2 else ""

        file_a_rows.append([
            clean_t, rule_id, cat, overlap, n_docs, n_master, n_mr,
            sh1, sh2, sh3, "", "", ""
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
    print(f"File A written: {len(file_a_rows)} rows.")

    # --------------------------------------------------------------------------
    # FILE B: 02_ambiguous_review
    # --------------------------------------------------------------------------
    print("\n--- Generating File B: 02_ambiguous_review ---")
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
    print(f"File B written: {len(file_b_rows)} rows.")

    # --------------------------------------------------------------------------
    # FILE C: 03_queue_labeled
    # --------------------------------------------------------------------------
    print("\n--- Generating File C: 03_queue_labeled ---")
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
        m_rec = manifest_by_id.get(eid) or master_by_id.get(eid, {})
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
    print(f"File C written: {len(file_c_rows)} rows.")

    # --------------------------------------------------------------------------
    # FILE D: 04_segmentation_check
    # --------------------------------------------------------------------------
    print("\n--- Generating File D: 04_segmentation_check ---")
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
    print(f"File D written: {len(file_d_rows)} rows.")

    # --------------------------------------------------------------------------
    # SAMPLING LOGIC FOR E, F, G, H
    # --------------------------------------------------------------------------
    print("\n--- Preparing Sampling Pool for Blind Files (E, F, G) ---")
    
    def get_eligible_examples(records):
        eligible = []
        cluster_seen = set()
        for r in records:
            eid = r["example_id"]
            m_rec = manifest_by_id[eid]
            if m_rec.get("review_status") == "ambiguous":
                continue
            qf = set(m_rec.get("quality_flags", []))
            if "extraction_error" in qf:
                continue
            if "heading_is_paragraph_suspected" in qf or "fragment_heading_suspected" in qf:
                continue
            cid = cluster_by_eid.get(eid)
            if cid:
                if cid in cluster_seen:
                    continue
                cluster_seen.add(cid)
            eligible.append(r)
        return eligible

    eligible_val = get_eligible_examples(val_records)
    eligible_test = get_eligible_examples(test_records)
    eligible_train = get_eligible_examples(train_records)

    print(f"Eligible records: Train={len(eligible_train)}/{len(train_records)}, Val={len(eligible_val)}/{len(val_records)}, Test={len(eligible_test)}/{len(test_records)}")

    # Proportional sampling for E and G
    def sample_stratified_proportional(eligible_list, raw_counts_dict, seed=42):
        random.seed(seed)
        by_cat = defaultdict(list)
        for ex in eligible_list:
            by_cat[ex["output"]].append(ex)

        N_h_eligible = {cat: len(items) for cat, items in by_cat.items()}
        n_h_target = allocate_strata_proportional(N_h_eligible, total_target=100, min_large=3, max_cap=25)

        sampled = []
        for cat in sorted(N_h_eligible.keys()):
            items = list(by_cat[cat])
            items.sort(key=lambda x: x["example_id"])
            sampled.extend(random.sample(items, n_h_target[cat]))

        random.shuffle(sampled)
        return sampled, N_h_eligible, n_h_target

    val_raw_counts = dataset_stats["category_counts_per_split"]["validation"]
    test_raw_counts = dataset_stats["category_counts_per_split"]["test"]

    sample_e, N_h_val, n_h_val = sample_stratified_proportional(eligible_val, val_raw_counts, seed=42)
    sample_g, N_h_test, n_h_test = sample_stratified_proportional(eligible_test, test_raw_counts, seed=42)

    # Sample F (Train Rule Audit)
    by_rule = defaultdict(list)
    for ex in eligible_train:
        m_rec = manifest_by_id[ex["example_id"]]
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

    # Disjointness check
    e_ids = set(x["example_id"] for x in sample_e)
    f_ids = set(x["example_id"] for x in sample_f)
    g_ids = set(x["example_id"] for x in sample_g)
    assert len(e_ids & f_ids) == 0, "Overlap between E and F!"
    assert len(e_ids & g_ids) == 0, "Overlap between E and G!"
    assert len(f_ids & g_ids) == 0, "Overlap between F and G!"

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
        if "Heading:" in inp and "Text:\n" in inp:
            parts = inp.split("Text:\n", 1)
            heading = parts[0].replace("Heading:", "").strip()
            body = parts[1].strip()
        else:
            heading = ""
            body = inp
        preview = normalize_text(body)[:800]
        file_e_rows.append([blind_id, heading, preview, "", "", ""])

        m_rec = manifest_by_id[eid]
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
    print(f"File E written: {len(file_e_rows)} rows.")

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

        m_rec = manifest_by_id[eid]
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
    print(f"File F written: {len(file_f_rows)} rows.")

    # --------------------------------------------------------------------------
    # FILE G: sealed/07_blind_test_SEALED
    # --------------------------------------------------------------------------
    print("\n--- Generating File G: sealed/07_blind_test_SEALED ---")
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

        m_rec = manifest_by_id[eid]
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
    readme_content = (
        "WARNING: SEALED BENCHMARK FILE\n"
        "Do not open until rule changes have been approved in Phase 6;\n"
        "corrections here must NOT be used to change labeling rules.\n"
    )
    with open(SEALED_DIR / "SEALED_README.txt", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"File G written: {len(file_g_rows)} rows (SEALED).")

    # --------------------------------------------------------------------------
    # FILE H: 08_consistency_check
    # --------------------------------------------------------------------------
    print("\n--- Generating File H: 08_consistency_check ---")
    random.seed(42)
    h_indices = sorted(random.sample(range(len(sample_e)), 30))
    file_h_rows = []
    for new_idx, e_idx in enumerate(h_indices, start=1):
        orig_row = file_e_rows[e_idx]
        orig_blind_id = orig_row[0]
        orig_ex = sample_e[e_idx]
        new_blind_id = f"CONSIST-{new_idx:03d}"
        
        file_h_rows.append([
            new_blind_id, orig_row[1], orig_row[2], "", "", ""
        ])

        m_rec = manifest_by_id[orig_ex["example_id"]]
        answer_key[new_blind_id] = {
            "example_id": orig_ex["example_id"],
            "assigned_label": orig_ex["output"],
            "rule_id": m_rec.get("label_rule_id", "N/A"),
            "parent_chapter": m_rec.get("parent_heading", "N/A"),
            "document": m_rec.get("document_id", "century"),
            "split": "consistency_validation",
            "mirrors_blind_id": orig_blind_id
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
    print(f"File H written: {len(file_h_rows)} rows.")

    # --------------------------------------------------------------------------
    # ANSWER KEY (PRIVATE)
    # --------------------------------------------------------------------------
    with open(PRIVATE_DIR / "answer_key.json", "w", encoding="utf-8") as f:
        json.dump(answer_key, f, indent=2, ensure_ascii=False)
    print(f"Answer key written to private/answer_key.json: {len(answer_key)} entries.")

    # --------------------------------------------------------------------------
    # SAMPLING MANIFEST
    # --------------------------------------------------------------------------
    sampling_manifest = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": 42,
            "script": "src/sme_clf/make_review_pack_v2.py",
            "source_hashes": {
                "master_sections.jsonl": sha256_file(master_p),
                "model_ready_manifest.jsonl": sha256_file(manifest_p),
                "train.jsonl": sha256_file(train_p),
                "validation.jsonl": sha256_file(val_p),
                "test.jsonl": sha256_file(test_p),
                "review_queue.jsonl": sha256_file(review_q_p),
                "dataset_statistics.json": sha256_file(stats_p)
            }
        },
        "file_e_validation_strata": {},
        "file_g_test_strata": {},
        "file_f_train_rule_strata": {},
        "sampled_example_ids": {
            "file_e": [x["example_id"] for x in sample_e],
            "file_f": [x["example_id"] for x in sample_f],
            "file_g": [x["example_id"] for x in sample_g],
            "file_h": [sample_e[idx]["example_id"] for idx in h_indices]
        }
    }

    # Populate validation strata with N_h, N_h_before_filters, n_h, inclusion_prob
    for cat in sorted(CATEGORIES_17):
        N_before = val_raw_counts.get(cat, 0)
        N_after = N_h_val.get(cat, 0)
        n_sample = n_h_val.get(cat, 0)
        prob = round(n_sample / N_after, 5) if N_after > 0 else 0.0
        sampling_manifest["file_e_validation_strata"][cat] = {
            "N_h_before_filters": N_before,
            "N_h_eligible": N_after,
            "n_h_sampled": n_sample,
            "inclusion_probability": prob
        }

    # Populate test strata with N_h, N_h_before_filters, n_h, inclusion_prob
    for cat in sorted(CATEGORIES_17):
        N_before = test_raw_counts.get(cat, 0)
        N_after = N_h_test.get(cat, 0)
        n_sample = n_h_test.get(cat, 0)
        prob = round(n_sample / N_after, 5) if N_after > 0 else 0.0
        sampling_manifest["file_g_test_strata"][cat] = {
            "N_h_before_filters": N_before,
            "N_h_eligible": N_after,
            "n_h_sampled": n_sample,
            "inclusion_probability": prob
        }

    # Populate train rule strata
    for rule in sorted(N_h_train_rules.keys()):
        N_after = N_h_train_rules[rule]
        n_sample = n_h_train_rules[rule]
        prob = round(n_sample / N_after, 5) if N_after > 0 else 0.0
        sampling_manifest["file_f_train_rule_strata"][rule] = {
            "N_h_eligible": N_after,
            "n_h_sampled": n_sample,
            "inclusion_probability": prob
        }

    with open(OUTPUT_DIR / "sampling_manifest.json", "w", encoding="utf-8") as f:
        json.dump(sampling_manifest, f, indent=2)
    print("Sampling manifest written to reports/review_pack_v2/sampling_manifest.json.")

    # --------------------------------------------------------------------------
    # GUIDES (review_guide.md and review_plan.md)
    # --------------------------------------------------------------------------
    # Copy/write review_guide.md
    guide_src = ROOT / "reports" / "review_pack" / "review_guide.md"
    if guide_src.exists():
        guide_content = open(guide_src, encoding="utf-8").read()
    else:
        guide_content = "# Reviewer Guide\n"
    with open(OUTPUT_DIR / "review_guide.md", "w", encoding="utf-8") as f:
        f.write(guide_content)

    # Write corrected review_plan.md
    plan_content = """# SME IPO Section Classification — Review Plan (v2)

**Version:** 2.0 (Phase 2 Corrected Review Pack)  
**Date:** 2026-11-21  
**Protocol:** Tiered human review sequence with strict blinding and test set isolation  
**Unit Rate:** 45 seconds per row (0.75 minutes/row)

---

## 1. Review Tiers and Time Allocation

| Tier | Workbooks Included | Target Scope | Rows | Time (Min) | Est. Hours |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Tier 1 (High Priority)** | `01_chapter_mapping_review`<br>`02_ambiguous_review`<br>`05_blind_validation` | Taxonomy calibration, ambiguous master sections, and blind validation benchmark. | **247** | **185.25 min** | **~3.1 hrs** |
| **Tier 2 (Medium Priority)** | `03_queue_labeled`<br>`04_segmentation_check`<br>`06_blind_rule_audit` | Targeted override inspection, segmentation defect verification, and train rule audit. | **200** | **150.00 min** | **~2.5 hrs** |
| **Tier 3 (Consistency)** | `08_consistency_check` | Intra-rater agreement audit (30 disguised validation duplicates). | **30** | **22.50 min** | **~0.38 hrs** |
| **Total Active Review** | **Tiers 1, 2, and 3** | All unsealed reviewer materials. | **477** | **357.75 min** | **~6.0 hrs** |
| **Quarantined / Sealed** | `sealed/07_blind_test_SEALED` | Frozen test benchmark (**Sealed until Phase 6**). | **100** | **75.00 min** | **~1.25 hrs** |
| **Total Pack** | **All 8 Files** | Complete audit suite. | **577** | **432.75 min** | **~7.2 hrs** |

---

## 2. Execution Guidelines
- **Tier 1 must be completed first** to establish ground truth on chapter titles and baseline validation performance.
- **Tier 3 should be reviewed at least 24 hours after File E** to provide a valid intra-rater consistency measure.
- **File G remains sealed** under `sealed/` and must not be reviewed until model and rule changes are frozen in Phase 6.
"""
    with open(OUTPUT_DIR / "review_plan.md", "w", encoding="utf-8") as f:
        f.write(plan_content)

    print("Review guide and review plan written to reports/review_pack_v2/.")
    print("=" * 70)
    print("ALL V2 ARTIFACTS SUCCESSFULLY GENERATED")
    print("=" * 70)

if __name__ == "__main__":
    main()
