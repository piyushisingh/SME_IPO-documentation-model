#!/usr/bin/env python3
"""Validation of extracted_sections.jsonl (Phase 2 outputs). Exits non-zero on any failure. Every check printed with its measured result."""
import json, sys, re, collections
from pathlib import Path
OUT=Path("/mnt/user-data/outputs/data/rebuilt_v4"); CFG=json.load(open(OUT/"scripts/doc_config.json"))
REQ=["example_id","document_id","source_filename","document_type","document_part","company_name","level","parent_section_id","start_page","end_page",
     "original_heading","heading_source","section_text_raw","section_text","pages","quality_flags","extraction_method","source_reference","category","review_status"]
fails=[]
def chk(name,ok,detail=""):
    print(("PASS " if ok else "FAIL ")+name+(f"  [{detail}]" if detail else "")); 
    if not ok: fails.append(name)
R=[]
for i,l in enumerate(open(OUT/"extracted_sections.jsonl",encoding="utf-8")):
    try: R.append(json.loads(l))
    except Exception as e: chk(f"jsonl line {i} parses",False,str(e))
chk("all lines parse as JSON",len(R)>0,f"{len(R)} records")
missing=[(r["example_id"],k) for r in R for k in REQ if k not in r]
chk("required fields present on every record",not missing,f"{len(missing)} missing")
ids=[r["example_id"] for r in R]; chk("example_id unique",len(ids)==len(set(ids)),f"{len(ids)} ids")
chk("no empty section_text_raw",all(r["section_text_raw"].strip() for r in R),f"{sum(1 for r in R if not r['section_text_raw'].strip())} empty")
chk("document_id set == configured 6 docs",{r["document_id"] for r in R}==set(CFG),str(sorted({r['document_id'] for r in R})))
chk("no excluded documents present (Project-Om / Shakti)",not any(re.search(r"project|shakti",r["source_filename"],re.I) for r in R))
chk("one company name and one source file per document",all(len({r['company_name'] for r in R if r['document_id']==d})==1 and len({r['source_filename'] for r in R if r['document_id']==d})==1 for d in CFG))
chk("source_filename matches config",all(r["source_filename"]==CFG[r["document_id"]]["source_filename"] for r in R))
chk("page ranges valid and inside document",all(1<=r["start_page"]<=r["end_page"] for r in R))
byd={d:max(p["pdf_end"] for p in CFG[d]["parts"]) for d in CFG}; chk("end_page <= pdf page count",all(r["end_page"]<=byd[r["document_id"]] for r in R))
chk("every L2 has an existing L1 parent in same document",all(r["parent_section_id"] in {x['example_id'] for x in R if x['document_id']==r['document_id'] and x['level']==1} for r in R if r["level"]==2))
chk("L2 parent page range contains child",all(True for _ in [0]) and all(
    next(p for p in R if p["example_id"]==r["parent_section_id"])["start_page"]<=r["start_page"] and r["end_page"]<=next(p for p in R if p["example_id"]==r["parent_section_id"])["end_page"] for r in R if r["level"]==2))
chk("document_type consistent with part config",all(any(p["part"]==r["document_part"] and p["document_type"]==r["document_type"] for p in CFG[r["document_id"]]["parts"]) for r in R))
chk("axiom abridged-part records confined to pdf pages 1-13, RHP part to 14-382",all((r["end_page"]<=13) if r["document_part"]=="abridged" else (r["start_page"]>=14) for r in R if r["document_id"]=="axiom_gas"))
chk("no label assigned yet (category is null, status unlabeled)",all(r["category"] is None and r["review_status"]=="unlabeled" for r in R))
chk("L2 raw text partitions L1 raw text (exact string equality)",all("\n".join(k["section_text_raw"] for k in R if k["parent_section_id"]==r["example_id"])==r["section_text_raw"] for r in R if r["level"]==1 and any(k["parent_section_id"]==r["example_id"] for k in R)))
st=json.load(open(OUT/"extraction_stats.json"))
chk("independent non-whitespace char conservation, L1 (all docs)",all(v["conservation"]["L1_equals_pages"] for v in st.values()))
chk("independent non-whitespace char conservation, L2+contents (all docs)",all(v["conservation"]["L2_equals_pages"] for v in st.values()))
chk("every PDF page belongs to some L1 section",all(not v["conservation"]["pages_not_in_any_L1"] for v in st.values()))
chk("no TOC listing leaked into body chapters",all(not v["boundary_integrity"]["toc_leak_sections"] for v in st.values()))
chk("all TOC entries resolved to a body location",all(v["toc"]["toc_entries_unresolved"]==0 for v in st.values()))
print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
