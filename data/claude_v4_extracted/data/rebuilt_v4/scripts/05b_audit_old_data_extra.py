#!/usr/bin/env python3
"""Phase 1 (data-only) supplementary audit; read-only. Writes old_data_audit_extra.json.  Only files that were provided are inspected."""
import json, re, collections, random
from pathlib import Path
UP=Path("/mnt/user-data/uploads"); OUT=Path("/mnt/user-data/outputs/data/rebuilt_v4")
M={"Century":"century","DRHP_Maxwell":"maxwell_engineering","DRHPPHOTONICSWATERTECHLIMITED":"photonics_watertech","Ideas_Electricals_&Engineers_DRHP_Hem_SME":"ideas_electricals","Red_Herring_Prospectus_AXIOM":"axiom_gas","quanto-updated-prospectus-1646-pm-final":"quanto_agroworld"}
old=json.load(open(UP/"sections.json",encoding="utf-8")); st=json.load(open(UP/"structured_ipo_data.json",encoding="utf-8"))
def an(s): return re.sub(r"[^a-z0-9]","",s.lower())
def toks(s): return re.findall(r"[a-z0-9]{3,}",s.lower())
X={}
# 1. 'other' label heading shape
oth=[s for k in M for s in old[k]["sections"] if s["category"]=="other"]
frag=[s for s in oth if len(s["heading"].split())<=3 or re.search(r"\b[A-Z] [A-Z]{2,}\b",s["heading"])]
random.seed(3)
X["other_label"]={"n_other_in_6_docs":len(oth),"n_sections_6_docs":sum(len(old[k]["sections"]) for k in M),"heading_le3_words_or_split_letter_pattern":len(frag),
                  "sample_headings":[s["heading"] for s in random.sample(oth,14)]}
# 2. artifact strings absent from the PDF text layer
arts={}
for pat in ["PRIV ATE","V ALID","QUANT AGRJ","TATA AIG"]:
    n_old=sum(s["heading"].count(pat)+s["text"].count(pat) for k in M for s in old[k]["sections"])
    n_new=sum(json.loads(l)["text_layout"].count(pat) for d in M.values() for l in open(OUT/"pages"/f"{d}.jsonl",encoding="utf-8"))
    arts[pat]={"occurrences_in_old_sections_json":n_old,"occurrences_in_pdf_text_layer":n_new}
X["artifact_strings"]=arts
# 3. token overlap for old text not found verbatim
res=collections.Counter()
for k,d in M.items():
    P={json.loads(l)["pdf_page"]:json.loads(l)["text_layout"] for l in open(OUT/"pages"/f"{d}.jsonl",encoding="utf-8")}
    allc="".join(an(t) for t in P.values())
    for s in old[k]["sections"]:
        key=an(s["text"])[:70]
        if len(key)<30 or key in allc: continue
        pg=int(s["page_start"]); ref=set()
        for q in (pg-1,pg,pg+1):
            if q in P: ref|=set(toks(P[q]))
        t=toks(s["text"])[:60]
        if t:
            ov=sum(w in ref for w in t)/len(t); res["token_overlap>=0.9" if ov>=.9 else ("0.6-0.9" if ov>=.6 else "<0.6")]+=1
X["old_text_not_found_verbatim_bag_of_words_overlap_with_claimed_page"]=dict(res)
# 4. risk factor lumping + size
c=[s for s in old["Century"]["sections"] if 23<=int(s["page_start"])<=46]
X["century_risk_factor_chapter_old_sections"]=[{"heading":s["heading"],"category":s["category"],"chars":len(s["text"]),"page_start":s["page_start"]} for s in c]
L=[len(s["text"]) for k in M for s in old[k]["sections"]]
X["old_section_size"]={"n":len(L),"gt_20k_chars":sum(l>20000 for l in L),"gt_50k_chars":sum(l>50000 for l in L)}
# 5. top headings per label
hc=collections.defaultdict(collections.Counter)
for k in M:
    for s in old[k]["sections"]: hc[s["category"]][re.sub(r"\s+"," ",s["heading"].strip().upper())[:50]]+=1
X["top_headings_for_labels"]={c:hc[c].most_common(8) for c in ["ipo_details","legal_and_regulatory","risk_factors","financial_information"]}
# 6. structured_ipo_data
sd={}
for k,d in M.items():
    items=[(c,x["text"]) for c,v in st[k].items() for x in v]; cnt=collections.Counter(t for _,t in items)
    sd[d]={"items":len(items),"unique_texts":len(cnt),"texts_repeated":sum(1 for t,n in cnt.items() if n>1),"has_heading_field":False,"item_keys":["page","source_file","text"]}
X["structured_ipo_data"]=sd
(OUT/"old_data_audit_extra.json").write_text(json.dumps(X,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps({k:X[k] for k in ["other_label","artifact_strings","old_text_not_found_verbatim_bag_of_words_overlap_with_claimed_page","old_section_size"]},indent=1)[:1800])
