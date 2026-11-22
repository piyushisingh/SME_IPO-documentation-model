#!/usr/bin/env python3
"""
LOCAL tokenizer / chat-template / target-preservation audit.  NOT RUN by the dataset builder (no tokenizer was available).
Usage:   python scripts/14_audit_tokenizer.py --model <hf-id-or-local-path> --max-seq-len 2048 [--data-dir data/rebuilt_v4]
Checks per example (all splits): renders the chat template with user=instruction+input, assistant=output; verifies the prompt is a prefix of the full
token sequence; verifies the assistant/target tokens decode to the category; counts examples whose full sequence exceeds --max-seq-len and, separately,
those where RIGHT-truncation to max_seq_len would remove or cut any target token.  Exit code 1 if any target would be lost.
--selftest runs the script logic with a toy character tokenizer to prove the script executes; it says NOTHING about your real tokenizer.
"""
import argparse, json, sys, collections
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--max-seq-len",type=int,default=2048)
ap.add_argument("--data-dir",default="data/rebuilt_v4"); ap.add_argument("--selftest",action="store_true"); a=ap.parse_args()
class Toy:
    chat_template="toy"
    def apply_chat_template(self,msgs,tokenize=True,add_generation_prompt=False):
        s="".join(f"<{m['role']}>{m['content']}</s>" for m in msgs)+("<assistant>" if add_generation_prompt else "")
        return [ord(c) for c in s] if tokenize else s
    def decode(self,ids,**k): return "".join(chr(i) for i in ids)
if a.selftest: tok=Toy(); name="SELFTEST-TOY-TOKENIZER (not a real model)"
else:
    if not a.model: sys.exit("--model required (or --selftest)")
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model); name=a.model
    if not getattr(tok,"chat_template",None): print("WARNING: tokenizer has no chat_template; configure the template you train with and edit render() below.")
def ids(x): return list(x["input_ids"]) if hasattr(x,"keys") else list(x)
def render(e):
    user=e["instruction"]+"\n\n"+e["input"]
    full=ids(tok.apply_chat_template([{"role":"user","content":user},{"role":"assistant","content":e["output"]}],tokenize=True))
    prompt=ids(tok.apply_chat_template([{"role":"user","content":user}],tokenize=True,add_generation_prompt=True))
    return full,prompt
R=collections.Counter(); worst=[]
for sp in ("train","validation","test"):
    for l in open(Path(a.data_dir)/f"{sp}.jsonl",encoding="utf-8"):
        e=json.loads(l); full,prompt=render(e); R[f"{sp}_examples"]+=1
        if full[:len(prompt)]!=prompt: R["prompt_not_prefix_of_full"]+=1; continue
        tgt=full[len(prompt):]; dec=tok.decode(tgt,skip_special_tokens=True) if not a.selftest else tok.decode(tgt)
        if e["output"] not in dec: R["target_not_decoded_to_category"]+=1
        n=len(full); worst.append((n,e["example_id"]))
        if n>a.max_seq_len: R["over_max_seq_len"]+=1
        if len(prompt)+len(tgt)>a.max_seq_len and len(prompt)>=a.max_seq_len-len(tgt)+0 and n>a.max_seq_len: R["right_truncation_would_cut_target"]+=1
        if len(prompt)>=a.max_seq_len: R["prompt_alone_fills_max_seq_len"]+=1
worst.sort(reverse=True)
rep={"tokenizer":name,"max_seq_len":a.max_seq_len,"counts":dict(R),"longest_5":worst[:5],"SELFTEST":a.selftest}
Path(a.data_dir,"tokenizer_audit_report.json").write_text(json.dumps(rep,indent=2),encoding="utf-8"); print(json.dumps(rep,indent=2))
sys.exit(1 if R["right_truncation_would_cut_target"] or R["target_not_decoded_to_category"] or R["prompt_not_prefix_of_full"] else 0)
