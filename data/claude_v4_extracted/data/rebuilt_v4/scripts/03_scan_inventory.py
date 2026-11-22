#!/usr/bin/env python3
"""Phase 2 step 3: find scanned / image-dominant pages and suspicious text layers. Read-only on PDFs.
Writes scan_inventory.json (per-page records only for pages with an image covering >=60% of the page, or low text)."""
import json, re, pymupdf
from pathlib import Path
SRC=Path("/mnt/user-data/uploads"); OUT=Path("/mnt/user-data/outputs/data/rebuilt_v4")
CFG=json.loads((OUT/"scripts/doc_config.json").read_text())
# characters typical of OCR confusion inside numbers / words:  digit-letter-digit  or letter-digit-letter
OCR_SUSPECT=re.compile(r"(?<![A-Za-z0-9])\d+[OolI]\d+|[A-Za-z][0-9][A-Za-z]|\b[A-Z]+[0-9][A-Z]+\b")
inv={}
for doc_id,c in CFG.items():
    doc=pymupdf.open(SRC/c["source_filename"]); rows=[]
    for i,pg in enumerate(doc):
        area=pg.rect.width*pg.rect.height
        cover=0.0
        for im in pg.get_image_info():
            x0,y0,x1,y1=im["bbox"]; cover=max(cover,max(0,(x1-x0))*max(0,(y1-y0))/area)
        txt=pg.get_text()
        n=len(txt.strip())
        susp=len(OCR_SUSPECT.findall(txt))
        if cover>=0.6 or n<150:
            rows.append({"pdf_page":i+1,"image_cover":round(cover,2),"n_chars":n,"ocr_like_errors":susp,
                         "kind":"image_only" if n<150 else "image_with_text_layer","first_text":re.sub(r"\s+"," ",txt.strip())[:70]})
    inv[doc_id]=rows
    io=[r["pdf_page"] for r in rows if r["kind"]=="image_only"]; it=[r["pdf_page"] for r in rows if r["kind"]=="image_with_text_layer"]
    print(f"{doc_id:22s} image_only={len(io):3d} {io[:25]}  | full-page image + text layer={len(it):3d} {it[:12]}{'...' if len(it)>12 else ''}")
(OUT/"scan_inventory.json").write_text(json.dumps(inv,indent=1,ensure_ascii=False),encoding="utf-8")
