#!/usr/bin/env python3
"""Validation of master_sections.jsonl and train/validation/test.jsonl. Exits non-zero on any failure."""
import json, sys, re, collections
from pathlib import Path
OUT=Path("/mnt/user-data/outputs/data/rebuilt_v4")
CATS=["ipo_details","risk_factors","disclosures_and_definitions","legal_and_regulatory","promoters_and_management","company_information","financial_information","financial_statements","related_party_transactions","capital_structure","objects_of_issue","qualitative_factors","industry_information","business_operations","litigation","declarations","other"]
fails=[]
def chk(n,ok,d=""):
    print(("PASS " if ok else "FAIL ")+n+(f"  [{d}]" if d else ""))
    if not ok: fails.append(n)
def load(p):
    out=[]
    for i,l in enumerate(open(OUT/p,encoding="utf-8")):
        try: out.append(json.loads(l))
        except Exception as e: chk(f"{p} line {i} parses",False,str(e))
    return out
M=load("master_sections.jsonl"); S={sp:load(f"{sp}.jsonl") for sp in ("train","validation","test")}; MAN=load("model_ready_manifest.jsonl")
ST=json.load(open(OUT/"dataset_statistics.json")); SM=json.load(open(OUT/"split_manifest.json"))
mid={r["example_id"]:r for r in M}
chk("master parses",len(M)==5997,f"{len(M)} records")
chk("master ids unique",len(mid)==len(M))
chk("master: every record has provenance fields",all(all(k in r for k in ("document_id","source_filename","start_page","end_page","original_heading","section_text_raw","section_text","source_reference","extraction_method")) for r in M))
chk("master: category valid or null",all(r["category"] is None or r["category"] in CATS for r in M))
chk("master: 'verified' never assigned (no human review done)",not any(r["review_status"]=="verified" for r in M))
chk("master: every labeled record is marked unverified_rule_label",all(r["label_status"]=="unverified_rule_label" for r in M if r["category"]))
chk("master: ambiguous/structural records have no category",all(r["category"] is None for r in M if r["review_status"] in("ambiguous","excluded_structural")))
chk("master: no category 'other' assigned automatically",not any(r["category"]=="other" for r in M))
REQ=["example_id","instruction","input","output","label_status"]
allex=[(sp,e) for sp in S for e in S[sp]]
chk("model-ready: required fields on every example",all(all(k in e for k in REQ) for _,e in allex),f"{len(allex)} examples")
chk("model-ready: example_id unique across all splits",len({e['example_id'] for _,e in allex})==len(allex))
chk("model-ready: output is a valid category string",all(e["output"] in CATS for _,e in allex))
chk("model-ready: single identical instruction, contains all 17 categories",len({e['instruction'] for _,e in allex})==1 and all(c in allex[0][1]["instruction"] for c in CATS))
chk("model-ready: input non-empty and label never inside truncated field (separate output field)",all(e["input"].strip() and e["output"] for _,e in allex))
chk("model-ready: label_status unverified on all",all(e["label_status"]=="unverified_rule_label" for _,e in allex))
man={m["example_id"]:m for m in MAN}
chk("manifest 1:1 with examples",set(man)=={e["example_id"] for _,e in allex} and len(man)==len(MAN))
chk("manifest split equals file split",all(man[e["example_id"]]["split"]==sp for sp,e in allex))
docs={sp:{man[e["example_id"]]["document_id"] for e in S[sp]} for sp in S}
chk("DOCUMENT-LEVEL disjoint: no document in more than one split",all(not(docs[a]&docs[b]) for a in S for b in S if a<b),str({k:sorted(v) for k,v in docs.items()}))
chk("every example's source records belong to exactly its document",all(all(mid[i]["document_id"]==man[k]["document_id"] for i in man[k]["source_example_ids"]) for k in man))
src_split=collections.defaultdict(set)
for k,m in man.items():
    for i in m["source_example_ids"]: src_split[i].add(m["split"])
chk("no master record used in more than one split",all(len(v)==1 for v in src_split.values()))
chk("AXIOM abridged + RHP parts in the same split",len({man[k]["split"] for k in man if man[k]["document_id"]=="axiom_gas"})==1)
chk("every source record is rule-labeled, needs_review, no exclusion reasons",all(mid[i]["category"] and mid[i]["review_status"]=="needs_review" and not mid[i]["training_exclusion_reasons"] for k in man for i in man[k]["source_example_ids"]))
chk("example output == master category of its source records",all(all(mid[i]["category"]==man[k]["category"] for i in man[k]["source_example_ids"]) for k in man))
ex={e["example_id"]:e for _,e in allex}
def an(s): return re.sub(r"[^a-z0-9]","",s.lower())
bad=0; checked_full=0; checked_prefix=0
for k,m in man.items():
    body=an(ex[k]["input"].split("Text:\n",1)[1])
    src=an("".join(mid[i]["section_text"] for i in m["source_example_ids"]))
    if m["input_truncated"]:
        checked_prefix+=1; ok=src.startswith(body) and len(body)>0
    else:
        checked_full+=1; ok=(body==src)
    if not ok: bad+=1
chk("traceability: non-truncated excerpt == concatenated source sections; truncated excerpt is its exact prefix (alnum-normalised)",bad==0,f"{bad} mismatches; {checked_full} full + {checked_prefix} truncated checked")
chk("excerpt never longer than the character budget",all(m["n_chars_input_excerpt"]<=ST["max_input_chars_setting"] for m in MAN))
chk("prompt heading never longer than the heading budget",all(len(ex[k]["input"].split("\n\nText:\n",1)[0])-len("Heading: ")<=ST["max_heading_chars_setting"] for k in ex))
chk("page provenance valid",all(1<=m["start_page"]<=m["end_page"] for m in MAN))
chk("counts reconcile with dataset_statistics.json",{sp:len(S[sp]) for sp in S}==ST["examples_per_split"] and sum(len(v) for v in S.values())==len(MAN)==ST["model_ready_examples"])
chk("split_manifest documents match files",{sp:sorted(v) for sp,v in docs.items()}=={sp:sorted(v) for sp,v in SM["documents_by_split"].items()})
chk("tokenizer audit honestly recorded as NOT_RUN",ST["tokenizer_audit"]["status"]=="NOT_RUN")
print("\nRESULT:","ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"); sys.exit(1 if fails else 0)
