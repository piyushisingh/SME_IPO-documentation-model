#!/usr/bin/env python3
"""Phase 2 step 4: OCR ONLY the pages flagged in scan_inventory.json. Original text layer is untouched;
OCR output is written separately to ocr/<doc_id>.jsonl with mean confidence and an agreement score vs. the embedded layer."""
import json, subprocess, tempfile, re, sys, csv, io, collections
from pathlib import Path
OUT=Path("/mnt/user-data/outputs/data/rebuilt_v4"); SRC=Path("/mnt/user-data/uploads")
CFG=json.loads((OUT/"scripts/doc_config.json").read_text()); INV=json.loads((OUT/"scan_inventory.json").read_text())
pages={d:{json.loads(l)["pdf_page"]:json.loads(l) for l in open(OUT/"pages"/f"{d}.jsonl",encoding="utf-8")} for d in INV if INV[d]}
def tok(s): return re.findall(r"[A-Za-z0-9]{2,}",s)
for doc_id,rows in INV.items():
    if not rows: continue
    outp=OUT/"ocr"/f"{doc_id}.jsonl"
    done={json.loads(l)["pdf_page"] for l in open(outp)} if outp.exists() else set()
    with open(outp,"a",encoding="utf-8") as fo:
        for r in rows:
            n=r["pdf_page"]
            if n in done: continue
            with tempfile.TemporaryDirectory() as td:
                subprocess.run(["pdftoppm","-r","300","-f",str(n),"-l",str(n),"-png",str(SRC/CFG[doc_id]["source_filename"]),f"{td}/p"],check=True)
                png=next(Path(td).glob("p*.png"))
                txt=subprocess.run(["tesseract",str(png),"-","--psm","6","-l","eng"],capture_output=True,text=True).stdout
                tsv=subprocess.run(["tesseract",str(png),"-","--psm","6","-l","eng","tsv"],capture_output=True,text=True).stdout
            confs=[float(x["conf"]) for x in csv.DictReader(io.StringIO(tsv),delimiter="\t",quoting=csv.QUOTE_NONE) if x.get("conf") not in (None,"-1") and (x.get("text") or "").strip()]
            emb=pages[doc_id][n]["text_layout"]
            et,ot=tok(emb),set(tok(txt))
            agree=(sum(1 for t in et if t in ot)/len(et)) if et else None
            rec={"doc_id":doc_id,"pdf_page":n,"kind":r["kind"],"ocr_engine":"tesseract --psm 6 -l eng @300dpi","ocr_text":txt,
                 "ocr_mean_word_conf":round(sum(confs)/len(confs),1) if confs else None,"ocr_n_words":len(confs),
                 "embedded_n_tokens":len(et),"embedded_token_agreement_with_ocr":round(agree,3) if agree is not None else None}
            fo.write(json.dumps(rec,ensure_ascii=False)+"\n"); fo.flush()
            print(doc_id,n,r["kind"],"conf",rec["ocr_mean_word_conf"],"words",rec["ocr_n_words"],"agree",rec["embedded_token_agreement_with_ocr"],flush=True)
