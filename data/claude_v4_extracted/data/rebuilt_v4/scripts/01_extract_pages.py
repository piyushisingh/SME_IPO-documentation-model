#!/usr/bin/env python3
"""
Phase 2, step 1: page-level extraction.

Reads the 6 source PDFs (read-only) and writes one JSONL per document to
data/rebuilt_v4/pages/<doc_id>.jsonl  (one record per PDF page).

Each page record keeps:
  - pdf_page            1-based physical page index in the PDF (source of truth for provenance)
  - text_layout         pdftotext -layout output (best for preserving table columns)
  - lines               PyMuPDF lines with font size / bold / bbox (used for heading detection)
  - n_chars, n_images, n_replacement_chars, etc. (extraction health signals)

Nothing is normalised here. Raw text is kept exactly as extracted.
"""
import json, subprocess, sys, hashlib, re
from pathlib import Path
import pymupdf

SRC = Path("/mnt/user-data/uploads")
OUT = Path("/mnt/user-data/outputs/data/rebuilt_v4")
PAGES = OUT / "pages"
PAGES.mkdir(parents=True, exist_ok=True)

DOCS = [
    ("century",            "Century.pdf"),
    ("photonics_watertech","DRHPPHOTONICSWATERTECHLIMITED.pdf"),
    ("maxwell_engineering","DRHP_Maxwell.pdf"),
    ("ideas_electricals",  "Ideas_Electricals__Engineers_DRHP_Hem_SME.pdf"),
    ("axiom_gas",          "Red_Herring_Prospectus_AXIOM.pdf"),
    ("quanto_agroworld",   "quanto-updated-prospectus-1646-pm-final.pdf"),
]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def layout_pages(pdf):
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         capture_output=True, check=True).stdout.decode("utf-8", errors="replace")
    pages = out.split("\f")
    if pages and pages[-1].strip() == "":
        pages = pages[:-1]
    return pages

def is_bold(span):
    return bool(span["flags"] & 16) or "bold" in span["font"].lower() or "black" in span["font"].lower()

def main():
    manifest = []
    for doc_id, fn in DOCS:
        pdf = SRC / fn
        lay = layout_pages(pdf)
        doc = pymupdf.open(pdf)
        assert len(lay) == doc.page_count, (doc_id, len(lay), doc.page_count)
        outp = PAGES / f"{doc_id}.jsonl"
        n_empty = 0
        with open(outp, "w", encoding="utf-8") as fo:
            for i, page in enumerate(doc):
                d = page.get_text("dict")
                lines = []
                for blk in d["blocks"]:
                    if blk.get("type") != 0:
                        continue
                    for ln in blk["lines"]:
                        spans = [s for s in ln["spans"] if s["text"].strip()]
                        if not spans:
                            continue
                        txt = "".join(s["text"] for s in ln["spans"]).strip()
                        size = max(round(s["size"], 1) for s in spans)
                        bold = all(is_bold(s) for s in spans)
                        x0, y0, x1, y1 = [round(v, 1) for v in ln["bbox"]]
                        lines.append({"t": txt, "sz": size, "b": bold, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
                text_layout = lay[i]
                n_chars = len(text_layout.strip())
                if n_chars == 0:
                    n_empty += 1
                rec = {
                    "doc_id": doc_id,
                    "source_filename": fn,
                    "pdf_page": i + 1,
                    "page_width": round(page.rect.width, 1),
                    "page_height": round(page.rect.height, 1),
                    "n_chars": n_chars,
                    "n_images": len(page.get_images()),
                    "n_replacement_chars": text_layout.count("\ufffd"),
                    "n_pua_chars": sum(1 for c in text_layout if 0xE000 <= ord(c) <= 0xF8FF),
                    "text_layout": text_layout,
                    "lines": lines,
                }
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest.append({
            "doc_id": doc_id, "source_filename": fn, "sha256": sha256(pdf),
            "size_bytes": pdf.stat().st_size, "n_pages": doc.page_count,
            "n_pages_no_text": n_empty,
        })
        print(f"{doc_id:22s} pages={doc.page_count:4d} empty_text_pages={n_empty}", flush=True)
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
