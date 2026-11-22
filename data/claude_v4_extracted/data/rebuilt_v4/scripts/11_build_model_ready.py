#!/usr/bin/env python3
"""
Phase 6/7: derive MODEL-READY section-classification data from master_sections.jsonl (master is never modified).

Unit of an example = one L2 section (heading-only L2 records are merged into the following sibling; provenance kept).
Labels are PROVISIONAL RULE LABELS (unverified). Splits are DOCUMENT-LEVEL. No tokenizer is used here (none available);
input is limited by a documented CHARACTER budget and the target label is a separate field, so truncation can never touch it.
"""
import json, re, csv, argparse, collections, statistics as st, itertools
from pathlib import Path
from datasketch import MinHash, MinHashLSH

ap = argparse.ArgumentParser(); ap.add_argument("--max-input-chars", type=int, default=2000); ap.add_argument("--max-heading-chars", type=int, default=300); args = ap.parse_args()
MAX_IN = args.max_input_chars; MAX_HEAD = args.max_heading_chars
OUT = Path("/mnt/user-data/outputs/data/rebuilt_v4")
M = [json.loads(l) for l in open(OUT / "master_sections.jsonl", encoding="utf-8")]
CATS = ["ipo_details", "risk_factors", "disclosures_and_definitions", "legal_and_regulatory", "promoters_and_management", "company_information",
        "financial_information", "financial_statements", "related_party_transactions", "capital_structure", "objects_of_issue",
        "qualitative_factors", "industry_information", "business_operations", "litigation", "declarations", "other"]
INSTRUCTION = ("Classify the following section of an Indian SME IPO offer document (DRHP / RHP / Prospectus) into exactly one of these categories: "
               + ", ".join(CATS) + ". Answer with the category name only.")
SPLIT = {"photonics_watertech": "train", "maxwell_engineering": "train", "axiom_gas": "train", "quanto_agroworld": "train",
         "century": "validation", "ideas_electricals": "test"}
by_id = {r["example_id"]: r for r in M}

def eligible(r):
    return r["level"] == 2 and r["category"] is not None and r["review_status"] == "needs_review" and not r["training_exclusion_reasons"]

excl = []   # (example_id, reason)
for r in M:
    if r["level"] != 2: continue
    if not eligible(r):
        reasons = list(r["training_exclusion_reasons"]) or ([f"review_status={r['review_status']}"] if r["review_status"] != "needs_review" else ["no_category"])
        for x in reasons: excl.append((r["example_id"], x))

# ---------- group L2 siblings, merging heading-only records forward (or backward at the end of a chapter)
sib = collections.defaultdict(list)
for r in M:
    if r["level"] == 2: sib[r["parent_section_id"]].append(r)
groups = []
for pid, lst in sib.items():
    lst.sort(key=lambda r: r["example_id"])
    pending, last = [], None
    for s in lst:
        if not eligible(s):
            for p in pending: excl.append((p["example_id"], "heading_only_not_merged_neighbour_excluded"))
            pending = []; last = None; continue
        if "heading_only" in s["quality_flags"]:
            pending.append(s); continue
        grp = [s]
        if pending:
            if all(p["category"] == s["category"] for p in pending): grp = pending + [s]
            else:
                for p in pending: excl.append((p["example_id"], "heading_only_category_conflict_with_next_section"))
        pending = []; groups.append(grp); last = grp
    if pending:
        if last is not None and all(p["category"] == last[0]["category"] for p in pending): last.extend(pending)
        else:
            for p in pending: excl.append((p["example_id"], "heading_only_no_mergeable_neighbour"))

def clean_input(t):
    t = re.sub(r"[ \t]{3,}", "  ", t); t = re.sub(r"\n{3,}", "\n\n", t); return t.strip()
def truncate(t, n):
    if len(t) <= n: return t, False
    cut = t[:n]
    if not t[n].isspace():
        k = cut.rfind(" "); cut = cut[:k] if k > n * 0.8 else cut
    return cut.rstrip(), True

examples, manifest = [], []
for grp in groups:
    first = grp[0]; cat = first["category"]
    heads = [g["original_heading"] for g in grp]
    heading_full = " > ".join(dict.fromkeys(heads))
    heading, heading_trunc = (heading_full, False) if len(heading_full) <= MAX_HEAD else (heading_full[:MAX_HEAD].rsplit(" ", 1)[0], True)
    full = clean_input("\n\n".join(g["section_text"] for g in grp))
    excerpt, trunc = truncate(full, MAX_IN)
    ex_id = first["example_id"] if len(grp) == 1 else first["example_id"] + "+" + str(len(grp) - 1)
    doc = first["document_id"]
    rec = {"example_id": ex_id, "instruction": INSTRUCTION, "input": f"Heading: {heading}\n\nText:\n{excerpt}", "output": cat,
           "label_status": "unverified_rule_label"}
    flags = sorted({f for g in grp for f in g["quality_flags"]})
    pages = [g["start_page"] for g in grp] + [g["end_page"] for g in grp]
    man = {"example_id": ex_id, "split": SPLIT[doc], "document_id": doc, "source_filename": first["source_filename"], "document_type": first["document_type"],
           "document_part": first["document_part"], "company_name": first["company_name"], "source_example_ids": [g["example_id"] for g in grp],
           "parent_section_id": first["parent_section_id"], "parent_heading": first["parent_heading"], "heading": heading,
           "heading_equals_chapter_title": heading.upper() == (first["parent_heading"] or "").upper(),
           "start_page": min(pages), "end_page": max(pages), "category": cat, "label_rule_id": first["label_rule_id"],
           "label_confidence_tier": first["label_confidence_tier"], "label_basis": first["label_basis"], "category_proposed": sorted({c for g in grp for c in g["category_proposed"]}),
           "review_status": "needs_review", "quality_flags": flags, "n_merged_heading_only": len(grp) - 1, "n_chars_full_input": len(full),
           "n_chars_input_excerpt": len(excerpt), "input_truncated": trunc, "heading_truncated_in_prompt": heading_trunc, "heading_chars_full": len(heading_full)}
    examples.append((SPLIT[doc], rec)); manifest.append(man)

for sp in ("train", "validation", "test"):
    with open(OUT / f"{sp}.jsonl", "w", encoding="utf-8") as fo:
        for s, rec in examples:
            if s == sp: fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
with open(OUT / "model_ready_manifest.jsonl", "w", encoding="utf-8") as fo:
    for m in manifest: fo.write(json.dumps(m, ensure_ascii=False) + "\n")
with open(OUT / "model_ready_exclusions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["master_example_id", "document_id", "pdf_page", "exclusion_reason"])
    for i, rs in sorted(set(excl)): w.writerow([i, by_id[i]["document_id"], by_id[i]["start_page"], rs])

# ---------- duplicates (exact + near) across the model-ready examples
def shingles(t, k=5):
    w = re.findall(r"[a-z0-9]+", t.lower()); return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}
sh, mh = {}, {}
lsh = MinHashLSH(threshold=0.8, num_perm=64)
for (sp, rec), man in zip(examples, manifest):
    body = rec["input"].split("Text:\n", 1)[1]
    s = shingles(body)
    if len(s) < 8: continue
    m = MinHash(num_perm=64)
    for x in s: m.update(x.encode())
    sh[rec["example_id"]] = s; mh[rec["example_id"]] = m; lsh.insert(rec["example_id"], m)
splitof = {m["example_id"]: m["split"] for m in manifest}; docof = {m["example_id"]: m["document_id"] for m in manifest}; catof = {m["example_id"]: m["category"] for m in manifest}
pairs = set()
for k, m in mh.items():
    for o in lsh.query(m):
        if o != k: pairs.add(tuple(sorted((k, o))))
dups = []
for a, b in pairs:
    j = len(sh[a] & sh[b]) / len(sh[a] | sh[b])
    if j >= 0.8:
        dups.append({"example_a": a, "example_b": b, "doc_a": docof[a], "doc_b": docof[b], "split_a": splitof[a], "split_b": splitof[b],
                     "category_a": catof[a], "category_b": catof[b], "jaccard_5word_shingles": round(j, 3), "exact_text": j == 1.0,
                     "same_document": docof[a] == docof[b], "cross_split": splitof[a] != splitof[b], "label_conflict": catof[a] != catof[b]})
with open(OUT / "duplicate_report.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(dups[0].keys())) if dups else None
    if w: w.writeheader(); w.writerows(sorted(dups, key=lambda d: -d["jaccard_5word_shingles"]))
cross = [d for d in dups if d["cross_split"]]
eval_with_dup = {sp: len({x for d in cross for x, s in ((d["example_a"], d["split_a"]), (d["example_b"], d["split_b"])) if s == sp and any(
    (d2["example_a"] == x and d2["split_b"] == "train") or (d2["example_b"] == x and d2["split_a"] == "train") for d2 in cross)}) for sp in ("validation", "test")}

# ---------- statistics / split manifest / review queue / label audit
cnt = {sp: collections.Counter() for sp in ("train", "validation", "test")}
for m in manifest: cnt[m["split"]][m["category"]] += 1
n_ex = {sp: sum(cnt[sp].values()) for sp in cnt}
docs_by_split = {sp: [d for d, s in SPLIT.items() if s == sp] for sp in cnt}
absent = {sp: sorted(set(CATS) - set(cnt[sp])) for sp in cnt}
low = {sp: sorted(c for c in CATS if 0 < cnt[sp][c] < 5) for sp in cnt}
per_doc = {d: sum(1 for m in manifest if m["document_id"] == d) for d in SPLIT}
chars = [m["n_chars_input_excerpt"] for m in manifest]; full = [m["n_chars_full_input"] for m in manifest]
stats = {
    "max_input_chars_setting": MAX_IN, "master_records": len(M), "master_L1": sum(1 for r in M if r["level"] == 1), "master_L2": sum(1 for r in M if r["level"] == 2),
    "model_ready_examples": len(manifest), "examples_per_split": n_ex, "examples_per_document": per_doc,
    "category_counts_per_split": {sp: {c: cnt[sp][c] for c in CATS if cnt[sp][c]} for sp in cnt},
    "categories_absent_per_split": absent, "categories_with_fewer_than_5_examples_per_split": low,
    "master_review_status_counts": dict(collections.Counter(r["review_status"] for r in M)),
    "master_category_counts_L2": dict(collections.Counter(r["category"] for r in M if r["level"] == 2)),
    "exclusion_reason_counts_master_L2_records": dict(collections.Counter(rs for _, rs in set(excl))),
    "n_master_L2_records_excluded_or_merged_away": len({i for i, _ in set(excl)}), "n_heading_only_records_merged": sum(m["n_merged_heading_only"] for m in manifest),
    "n_examples_with_merged_heading_only": sum(1 for m in manifest if m["n_merged_heading_only"]),
    "input_truncation": {"examples_truncated": sum(m["input_truncated"] for m in manifest), "share": round(sum(m["input_truncated"] for m in manifest) / len(manifest), 3),
                          "excerpt_chars_median": int(st.median(chars)), "excerpt_chars_max": max(chars), "full_chars_median": int(st.median(full)), "full_chars_p95": int(sorted(full)[int(.95 * len(full))]), "full_chars_max": max(full)},
    "headings_truncated_in_prompt": sum(m["heading_truncated_in_prompt"] for m in manifest), "max_heading_chars_setting": MAX_HEAD,
    "heading_equals_chapter_title_share": round(sum(m["heading_equals_chapter_title"] for m in manifest) / len(manifest), 3),
    "label_confidence_tier_counts": dict(collections.Counter(m["label_confidence_tier"] for m in manifest)),
    "label_basis_counts": dict(collections.Counter(m["label_basis"] for m in manifest)),
    "duplicates": {"near_or_exact_pairs_jaccard_ge_0.8": len(dups), "exact_text_pairs": sum(d["exact_text"] for d in dups), "same_document_pairs": sum(d["same_document"] for d in dups),
                   "cross_split_pairs": len(cross), "cross_split_pairs_with_label_conflict": sum(d["label_conflict"] for d in cross),
                   "validation_examples_with_a_near_duplicate_in_train": eval_with_dup["validation"], "test_examples_with_a_near_duplicate_in_train": eval_with_dup["test"],
                   "method": "MinHash LSH (64 perms) candidates, confirmed with exact Jaccard on 5-word shingles >= 0.8; sections with <8 shingles skipped"},
    "tokenizer_audit": {"status": "NOT_RUN", "reason": "target model/tokenizer not provided and not available offline; scripts/14_audit_tokenizer.py is provided for local use"},
    "labels": "PROVISIONAL rule-derived; zero records are human-verified",
}
(OUT / "dataset_statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
split_manifest = {
    "split_level": "document", "n_documents": 6,
    "assignment": {d: {"split": SPLIT[d], "document_type(s)": sorted({m["document_type"] for m in manifest if m["document_id"] == d}), "examples": per_doc[d]} for d in SPLIT},
    "documents_by_split": docs_by_split,
    "rationale": "test=Ideas (DRHP, text-clean, no scanned pages); validation=Century (RHP, text-clean); train=Photonics, Maxwell, AXIOM (abridged+RHP kept together), Quanto (fixed-price final prospectus). Chosen from document type/cleanliness and category coverage only, not from any model results.",
    "limitations": ["single-document validation and test sets: metrics are dominated by one company's filing and carry very wide uncertainty",
                    "Century (validation) is the only book-built RHP outside train; Quanto (fixed-price final prospectus) exists only in train, so the evaluation never tests the fixed-price document type",
                    "labels are rule-derived from chapter titles, so evaluation measures agreement with those rules, not with human-verified truth",
                    "the test document's extraction was inspected during extraction debugging (heading/segmentation checks), though no model has seen it and its labels were not tuned",
                    "boilerplate text recurs across SME prospectuses; see duplicate_report.csv for cross-split near-duplicates"],
    "leave_one_document_out_folds": [{"fold": i + 1, "held_out": d, "train": [x for x in SPLIT if x != d]} for i, d in enumerate(SPLIT)],
    "categories_absent_per_split": absent, "categories_under_5_examples_per_split": low,
    "test_set_policy": "frozen: do not change labels, rules or extraction for the test document based on test results; if rules change, re-derive and re-declare the test set as no longer untouched",
}
(OUT / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

QF = {"heading_is_paragraph_suspected", "suspected_label_content_mismatch", "heading_keyword_overrides_chapter_label", "fragment_heading_suspected", "multi_topic_chapter_no_single_label", "label_ambiguous_multi_topic_chapter", "no_chapter_rule_matched"}
with open(OUT / "review_queue.jsonl", "w", encoding="utf-8") as fo:
    n = 0
    for r in M:
        why = []
        if r["review_status"] in ("ambiguous", "extraction_error"): why.append(r["review_status"])
        why += sorted(QF & set(r["quality_flags"]))
        if r["category"] == "qualitative_factors": why.append("rare_category_single_rule")
        if why:
            n += 1
            fo.write(json.dumps({"example_id": r["example_id"], "document_id": r["document_id"], "level": r["level"], "pdf_pages": f"{r['start_page']}-{r['end_page']}",
                                 "heading": r["original_heading"], "assigned_category": r["category"], "proposed_alternatives": r["category_proposed"],
                                 "reasons": why, "label_rule_id": r["label_rule_id"], "text_preview": r["section_text"][:300]}, ensure_ascii=False) + "\n")
with open(OUT / "label_audit_report.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["example_id", "document_id", "level", "pdf_pages", "heading", "category", "label_rule_id", "confidence_tier", "label_basis", "review_status", "proposed_alternatives", "flags"])
    for r in M:
        if r["level"] == 1 or QF & set(r["quality_flags"]) or r["review_status"] in ("ambiguous", "extraction_error"):
            w.writerow([r["example_id"], r["document_id"], r["level"], f"{r['start_page']}-{r['end_page']}", r["original_heading"][:100], r["category"], r["label_rule_id"],
                        r["label_confidence_tier"], r["label_basis"], r["review_status"], "|".join(r["category_proposed"]), "|".join(r["quality_flags"])])
print("examples:", n_ex, "| per doc:", per_doc)
print("absent categories:", absent); print("<5 examples:", low)
print("truncated:", stats["input_truncation"]); print("duplicates:", stats["duplicates"]); print("review_queue rows:", n)
