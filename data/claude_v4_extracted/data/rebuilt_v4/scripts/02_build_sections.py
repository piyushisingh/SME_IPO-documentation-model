#!/usr/bin/env python3
"""
Phase 2, step 2: build extracted sections (NO category labels yet).

Two levels:
  L1 = chapters / parts, anchored on the document's own Table of Contents (TOC) and located in the body text.
  L2 = subsections inside an L1 chapter, from heading detection (bold, left-margin, non-tabular lines).
Every L2 record has parent_section_id pointing to its L1 chapter. L2 records partition their L1 chapter exactly
(verified by assertion), so no text is dropped or duplicated.

Raw text = pdftotext -layout lines exactly as extracted (page by page).
Normalised text = raw minus repeated header/footer lines and footer page numbers, trailing whitespace stripped,
runs of blank lines collapsed. Both are stored.
"""
import json, re, collections, difflib
from pathlib import Path

OUT = Path("/mnt/user-data/outputs/data/rebuilt_v4")
CFG = json.loads((OUT / "scripts/doc_config.json").read_text(encoding="utf-8"))

def norm(s):
    return re.sub(r"\s+", " ", s.replace("\u00a0", " ")).strip()

def alnum(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())

def is_tabular(line):
    return len(re.findall(r"\S(?: {3,})\S", line.strip())) >= 1 and len(line.strip()) > 0 and len(re.split(r" {3,}", line.strip())) >= 3

# ---------------------------------------------------------------- load pages
def load_pages(doc_id):
    return [json.loads(l) for l in open(OUT / "pages" / f"{doc_id}.jsonl", encoding="utf-8")]

def printed_page_raw(p):
    H = p["page_height"]
    c = [l for l in p["lines"] if l["y0"] > H * 0.88 and re.fullmatch(r"\d{1,3}", l["t"].strip())]
    return int(c[-1]["t"]) if c else None

def printed_pages(P):
    """printed page numbers, kept only if (pdf_page - printed) agrees with a nearby page's offset (rejects stray digits)."""
    raw = [printed_page_raw(p) for p in P]
    off = [(i + 1 - r) if r is not None else None for i, r in enumerate(raw)]
    out = []
    for i, r in enumerate(raw):
        if r is None:
            out.append(None); continue
        near = [off[j] for j in range(max(0, i - 3), min(len(P), i + 4)) if j != i and off[j] is not None]
        out.append(r if off[i] in near else None)
    return out

def header_footer_set(pages):
    """normalised (digits masked) strings that recur in top/bottom margins on >=15% of pages."""
    cnt = collections.Counter()
    for p in pages:
        H = p["page_height"]
        seen = set()
        for l in p["lines"]:
            if l["y0"] < H * 0.07 or l["y0"] > H * 0.92:
                k = re.sub(r"\d+", "#", norm(l["t"]))
                if len(k) >= 3:
                    seen.add(k)
        cnt.update(seen)
    thr = max(5, int(0.15 * len(pages)))
    return {k for k, v in cnt.items() if v >= thr}

# ---------------------------------------------------------------- line model
def page_lines(p, hf):
    """layout lines of a page with bold / margin / boilerplate annotations."""
    py = p["lines"]
    idx = collections.defaultdict(list)
    for l in py:
        idx[norm(l["t"])].append(l)
    margin = None
    xs = [l["x0"] for l in py if len(l["t"]) > 30]
    if xs:
        margin = collections.Counter(round(x) for x in xs).most_common(1)[0][0]
    H = p["page_height"]
    out = []
    for raw in p["text_layout"].split("\n"):
        n = norm(raw)
        rec = {"raw": raw.rstrip(), "n": n, "bold": False, "x0": None, "tab": False, "hf": False, "pn": False}
        if n:
            m = idx.get(n)
            if m is None:
                first = re.split(r" {3,}", raw.strip())[0]
                m = idx.get(norm(first))
                if m is None:
                    # substring match (line merged with neighbouring column text)
                    best = None
                    for t, ls in idx.items():
                        if len(t) >= 5 and t in n and len(t) >= 0.6 * len(n):
                            if best is None or len(t) > len(best[0]):
                                best = (t, ls)
                    m = best[1] if best else None
            if m:
                rec["bold"] = all(x["b"] for x in m)
                rec["x0"] = m[0]["x0"]
                rec["y0"] = m[0]["y0"]
            rec["tab"] = is_tabular(raw)
            k = re.sub(r"\d+", "#", n)
            if k in hf and rec.get("y0") is not None and (rec["y0"] < H * 0.07 or rec["y0"] > H * 0.92):
                rec["hf"] = True
            if re.fullmatch(r"\d{1,3}", n) and rec.get("y0") is not None and rec["y0"] > H * 0.88:
                rec["pn"] = True
        out.append(rec)
    return out, margin

# ---------------------------------------------------------------- TOC parsing
ROMAN = r"(?:X{0,3}(?:IX|IV|V?I{0,3}))"
PART_RE = re.compile(rf"^(?:SECTION\s+)?({ROMAN})\s*[.:\-–—]*\s+(.*)$", re.I)
SKIP_TOC = re.compile(r"^(TABLE OF CONTENTS|SECTION\s+CONTENTS\s+PAGE NO\.?|CONTENTS|PAGE NO\.?)$", re.I)

def parse_toc(P, LN, toc_pdf_page, company_alnum=None):
    entries, pending, last_digit_final = [], "", False
    started = False
    stop = False
    stop_pos = None
    last_pos = None
    for pi in range(toc_pdf_page - 1, min(toc_pdf_page + 2, len(P))):
        for li_, r in enumerate(LN[pi][0]):
            n = r["n"]
            last_pos = (pi, li_)
            if not n:
                continue
            if not started:
                if n.upper().startswith("TABLE OF CONTENTS"):
                    started = True
                continue
            n_clean = re.sub(r"(\.{3,}|-{3,}|_{3,}|\u2026+)", " ", n)
            n_clean = norm(n_clean)
            if SKIP_TOC.match(n_clean):
                continue
            # stop when body begins (a title we already parsed appears again without a page number)
            if len(entries) >= 10:
                key = alnum(n_clean)
                if len(key) >= 10 and any(alnum(e["title"]) == key for e in entries) and not re.search(r"\d$", n_clean):
                    stop = True
                if re.match(r"^SECTION\s+I\b", n_clean, re.I) and len(entries) > 10:
                    stop = True
            if stop:
                stop_pos = (pi, li_)
                break
            m = re.match(r"^(.*?)[\s]*?(\d{1,3})$", n_clean)
            if m and re.search(r"[A-Za-z]{3}", m.group(1)):
                title = norm((pending + " " + m.group(1)).strip())
                entries.append({"title": title, "printed_page": int(m.group(2)), "via": "inline"})
                pending, last_digit_final = "", False
            elif re.fullmatch(r"\d{1,3}", n_clean) and pending:
                entries.append({"title": norm(pending), "printed_page": int(n_clean), "via": "digit_line"})
                pending, last_digit_final = "", True
            elif re.fullmatch(r"\d{1,3}", n_clean):
                continue  # footer page number
            elif PART_RE.match(n_clean) and PART_RE.match(n_clean).group(1) and (n_clean.upper().startswith("SECTION") or re.match(rf"^{ROMAN}\s*[.:]", n_clean, re.I)):
                entries.append({"title": n_clean, "printed_page": None, "via": "part_nopage"})
                pending, last_digit_final = "", False
            elif company_alnum and alnum(n_clean) == company_alnum:
                continue  # running footer showing the company name
            else:
                if last_digit_final and entries and not PART_RE.match(n_clean) and n_clean.upper() == n_clean:
                    entries[-1]["title"] = norm(entries[-1]["title"] + " " + n_clean)
                    entries[-1]["via"] += "+cont"
                else:
                    pending = (pending + " " + n_clean).strip()
                    last_digit_final = False
        if stop:
            break
    # parts without a page number ("I. GENERAL") remain in pending only when followed by a numbered line; handle parts
    return entries, pending, (stop_pos or last_pos)

def classify_entries(entries):
    out = []
    for e in entries:
        t = e["title"]
        m = PART_RE.match(t)
        is_part = bool(m and m.group(1) and (t.upper().startswith("SECTION") or re.match(rf"^{ROMAN}\s*[.:]", t, re.I)))
        out.append({**e, "kind": "part" if is_part else "chapter"})
    for i, e in enumerate(out):
        if e["kind"] == "part":
            nxt = out[i + 1] if i + 1 < len(out) else None
            own_pages = (nxt is not None and e.get("printed_page") is not None and nxt.get("printed_page") is not None
                         and nxt["printed_page"] > e["printed_page"])
            e["kind"] = "part_chapter" if (nxt is None or nxt["kind"] == "part" or own_pages) else "part_divider"
    return out

# ---------------------------------------------------------------- anchor location
def title_variants(title):
    t = title.strip()
    v = {t}
    m = PART_RE.match(t)
    if m and m.group(2):
        v.add(m.group(2))
        v.add("SECTION " + t)
    return [x for x in v if x]

def heading_match(rec, variants, loose=False):
    n = rec["n"]
    if not n or len(n) > 160:
        return False
    a = alnum(n)
    if len(a) < 4:
        return False
    for v in variants:
        av = alnum(v)
        if not av:
            continue
        if a == av:
            return True
        if len(a) >= 12 and av.startswith(a) and len(a) >= 0.45 * len(av):  # wrapped heading, first line only
            return True
        if len(av) >= 12 and a.startswith(av) and len(a) <= len(av) + 25:  # body heading extends the TOC title
            return True
        if difflib.SequenceMatcher(None, a, av).ratio() >= (0.80 if loose else 0.92):
            return True
    return False

def lookback_section_header(LN, found, floor, max_lines=6):
    """if a 'SECTION <roman> ...' divider line sits within a few non-blank lines above the anchor, start there."""
    pi, li = found
    steps = 0
    while steps < max_lines:
        li -= 1
        while li < 0:
            pi -= 1
            if pi < floor[0]:
                return found
            li = len(LN[pi][0]) - 1
        if (pi, li) <= floor:
            return found
        r = LN[pi][0][li]
        if not r["n"]:
            continue
        steps += 1
        if re.match(r"^SECTION\s+[IVX]+\b", r["n"], re.I):
            return (pi, li)
    return found

# ---------------------------------------------------------------- heading detection (L2)
SKIP_HEAD = re.compile(r"^(source\b|note\b|notes\b|total\b|particulars\b|sr\.?\s*no|s\.?\s*no|name\b|amount\b|\(?₹|\(?rs\.?|\(in\b|\(amount)", re.I)

BOILER_HEAD = re.compile(r"^(place\s*:|date\s*:|dated\s*:|signed\b|for and on behalf|for m/s|sd/-|\(?sd\)?\b|for\s+.+(&|and)\s+(co|associates)\b)", re.I)
CELLS2 = re.compile(r"\S {3,}\S")

def is_heading_line(rec, margin, prev=None, company_stem=None):
    n = rec["n"]
    if not rec["bold"] or rec["tab"] or rec["hf"] or rec["pn"]:
        return False
    if CELLS2.search(rec["raw"].strip()):          # >=2 cells separated by 3+ spaces: table header row
        return False
    if n[0].islower():                              # fragment of a bold phrase, not a heading
        return False
    if BOILER_HEAD.match(n):
        return False
    if company_stem and alnum(n).startswith(company_stem):   # repeated company-name page header
        return False
    if prev is not None and prev["n"] and not prev["bold"]:
        # mid-paragraph rule: previous non-bold line did not end a sentence, and this line is neither numbered nor ALL CAPS
        if not re.search(r"[.:;!?)\u201d\"']$", prev["n"]) and not re.match(r"^(\d+[.)]|[IVX]+[.)]|\([a-z0-9]+\))\s", n) and n.upper() != n:
            return False
    if len(n) < 4 or len(n) > 220:
        return False
    letters = sum(c.isalpha() for c in n)
    if letters < 0.55 * len(n):
        return False
    if SKIP_HEAD.match(n):
        return False
    if rec["x0"] is not None and margin is not None and rec["x0"] > margin + 30:
        return False
    return True

def prev_nonblank(lines, i):
    k = i - 1
    while k >= 0 and not lines[k]["n"]:
        k -= 1
    # a blank line between => paragraph break => no mid-paragraph restriction
    return lines[k] if k >= 0 and k == i - 1 else None

def heading_blocks(lines, margin, company_stem=None):
    """indices where a heading block starts; consecutive heading lines are merged into one block."""
    starts, i, n = [], 0, len(lines)
    while i < n:
        if is_heading_line(lines[i], margin, prev_nonblank(lines, i), company_stem):
            j = i
            while j + 1 < n and is_heading_line(lines[j + 1], margin, lines[j], company_stem) and lines[j + 1]["x0"] is not None and lines[j]["x0"] is not None \
                    and abs(lines[j + 1]["x0"] - lines[j]["x0"]) < 30 and (lines[j + 1].get("y0", 0) - lines[j].get("y0", 0)) < 20:
                j += 1
            starts.append((i, j))
            i = j + 1
        else:
            i += 1
    return starts

# ---------------------------------------------------------------- normalisation
def normalise(raw_lines):
    out, blank = [], 0
    removed = 0
    for r in raw_lines:
        if r["hf"] or r["pn"]:
            removed += 1
            continue
        s = r["raw"].rstrip()
        if not s.strip():
            blank += 1
            if blank <= 1:
                out.append("")
            continue
        blank = 0
        out.append(s)
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out), removed

# ---------------------------------------------------------------- main
def build_doc(doc_id):
    cfg = CFG[doc_id]
    P = load_pages(doc_id)
    assert cfg["cover_assert"] in P[0]["text_layout"].upper() or cfg["cover_assert"] in " ".join(P[0]["text_layout"].upper().split()), (doc_id, "cover assert failed")
    hf = header_footer_set(P)
    LN = [page_lines(p, hf) for p in P]
    printed = printed_pages(P)
    # flat line list: (page_idx, line_idx)
    flat = [(pi, li) for pi, (ls, _) in enumerate(LN) for li in range(len(ls))]
    pos = {k: i for i, k in enumerate(flat)}

    def part_of(pdf_page):
        for pt in cfg["parts"]:
            if pt["pdf_start"] <= pdf_page <= pt["pdf_end"]:
                return pt
    issues = []
    toc_end_info = {}
    L1 = []  # dicts with start (page_idx,line_idx)

    for pt in cfg["parts"]:
        s_idx, e_idx = pt["pdf_start"] - 1, pt["pdf_end"] - 1
        if pt["part"] == "abridged":
            L1.append({"title": "ABRIDGED PROSPECTUS (SALIENT FEATURES)", "kind": "synthetic", "start": (s_idx, 0), "part": pt,
                       "heading_source": "synthetic_part_header", "flags": ["scope_abridged_prospectus"], "toc": None})
            continue
        # TOC for this part
        toc_page = cfg["toc_pdf_page"] if pt["part"] in ("main", "rhp") else None
        entries, leftover, toc_end = parse_toc(P, LN, toc_page, alnum(cfg['company_name']))
        entries = classify_entries(entries)
        # front matter: cover (start .. TOC) and TOC page(s)
        toc_line = next((li for li, r in enumerate(LN[toc_page - 1][0]) if r["n"].upper().startswith("TABLE OF CONTENTS")), 0)
        L1.append({"title": "COVER PAGE AND FRONT MATTER", "kind": "synthetic", "start": (s_idx, 0), "part": pt,
                   "heading_source": "synthetic_front_matter", "flags": ["front_matter"], "toc": None})
        L1.append({"title": "TABLE OF CONTENTS", "kind": "synthetic", "start": (toc_page - 1, toc_line), "part": pt,
                   "heading_source": "toc_marker", "flags": ["contents_page"], "toc": None})
        # pdf pages by printed number (may repeat across numbering restarts)
        by_printed = collections.defaultdict(list)
        for i, pn in enumerate(printed):
            if pn is not None and s_idx <= i <= e_idx:
                by_printed[pn].append(i)
        cur_pi = toc_page - 1
        cur_li = toc_line
        # divider look-back must not reach into the TOC listing itself: floor = last TOC line before the body begins
        floor_first = (toc_end[0], toc_end[1] - 1) if toc_end else (cur_pi, cur_li)
        toc_end_info[pt["part"]] = {"toc_end_pdf_page": (toc_end[0] + 1) if toc_end else None, "toc_end_line": toc_end[1] if toc_end else None}
        cur_pi, cur_li = floor_first   # first body chapter must be searched AFTER the TOC listing
        first_resolved = False
        # skip entries whose heading ends up unresolved but keep a record
        current_part = None
        for e in entries:
            if e["kind"] == "part_divider":
                current_part = e["title"]
                continue
            if e["kind"] == "part_chapter":
                current_part = e["title"]
            variants = title_variants(e["title"])
            found, how = None, None
            exp = [i for i in by_printed.get(e["printed_page"], []) if i >= cur_pi]
            # expected page +/- 1, first matching heading line after the previous anchor
            for ep in exp[:3]:
                for pi in (ep - 1, ep, ep + 1):
                    if pi < cur_pi or pi > e_idx:
                        continue
                    for li, r in enumerate(LN[pi][0]):
                        if (pi, li) <= (cur_pi, cur_li) and not (pi == cur_pi and li > cur_li):
                            if (pi, li) <= (cur_pi, cur_li):
                                continue
                        if heading_match(r, variants, loose=True) and (r["bold"] or r["n"].upper() == r["n"]):
                            found, how = (pi, li), "toc_page_match"
                            break
                    if found:
                        break
                if found:
                    break
            if not found:  # forward scan, beyond the TOC page itself
                start_pi = cur_pi + (1 if cur_pi == toc_page - 1 else 0)
                for pi in range(start_pi, min(e_idx + 1, start_pi + 260)):
                    for li, r in enumerate(LN[pi][0]):
                        if (pi, li) <= (cur_pi, cur_li):
                            continue
                        if heading_match(r, variants) and (r["bold"] or r["n"].upper() == r["n"]) and not r["tab"]:
                            found, how = (pi, li), "forward_scan"
                            break
                    if found:
                        break
            if not found and exp:
                ep = exp[0]
                for li, r in enumerate(LN[ep][0]):
                    if r["n"] and not r["hf"] and not r["pn"] and alnum(r["n"]) != alnum(cfg["company_name"]) and (ep, li) > (cur_pi, cur_li):
                        found, how = (ep, li), "printed_page_only"
                        break
            if found:
                flags = [] if how == "toc_page_match" else (["anchor_by_forward_scan"] if how == "forward_scan" else ["anchor_by_printed_page_only_heading_not_found"])
                start = lookback_section_header(LN, found, floor_first if not any(x["kind"] in ("part_divider","part_chapter","chapter") and x.get("part") is pt for x in L1 if x["heading_source"] == "toc_anchored") else (cur_pi, cur_li))
                if start != found:
                    flags.append("start_moved_up_to_section_divider")
                L1.append({"title": e["title"], "kind": e["kind"], "start": start, "part": pt, "heading_source": "toc_anchored",
                           "section_part": current_part,
                           "flags": flags, "toc": {"title": e["title"], "printed_page": e["printed_page"], "parse_via": e["via"]}})
                cur_pi, cur_li = found
            else:
                issues.append({"doc_id": doc_id, "type": "toc_entry_unresolved", "detail": e})
        if leftover:
            issues.append({"doc_id": doc_id, "type": "toc_leftover_text", "detail": leftover[:200]})

    for sp in cfg.get("l1_splits", []):
        pt = part_of(sp["at_pdf_page"])
        L1.append({"title": sp["title"], "kind": "manual", "start": (sp["at_pdf_page"] - 1, 0), "part": pt,
                   "heading_source": sp["heading_source"], "flags": list(sp["flags"]), "toc": None, "evidence": sp["evidence"]})
    # order L1 by start; sort stable
    L1.sort(key=lambda d: (d["start"][0], d["start"][1]))
    # drop duplicate starts (e.g. a PART header and its single chapter on the same line) -> keep first, record as alias
    dedup = []
    for d in L1:
        if dedup and dedup[-1]["start"] == d["start"]:
            dedup[-1].setdefault("aliases", []).append(d["title"])
            if d["kind"] == "part" and dedup[-1]["kind"] == "chapter":
                pass
            continue
        dedup.append(d)
    L1 = dedup

    def line_slice(a, b):
        """flat lines from a (inclusive) to b (exclusive)."""
        ia = pos[a]
        ib = pos[b] if b in pos else len(flat)
        return flat[ia:ib]

    sections = []
    seq = 0
    for k, d in enumerate(L1):
        a = d["start"]
        nxt = L1[k + 1]["start"] if k + 1 < len(L1) else None
        # chapters must not cross a document-part boundary
        part_end = d["part"]["pdf_end"] - 1
        b = nxt
        if b is None or b[0] > part_end + 0 and b[0] != part_end + 1:
            b = None
        if nxt is not None and nxt[0] > part_end:
            b = (part_end + 1, 0) if part_end + 1 < len(P) else None
        sl = line_slice(a, b if b else (len(P), 0)) if b else flat[pos[a]:(pos[(part_end + 1, 0)] if (part_end + 1, 0) in pos else len(flat))]
        seq += 1
        sections.append({"d": d, "slice": sl, "seq": seq})
    issues.append({"doc_id": doc_id, "type": "info_toc_end", "detail": toc_end_info})
    return P, LN, printed, sections, issues, hf

def emit(doc_id, P, LN, printed, sections, issues):
    cfg = CFG[doc_id]
    recs = []
    margins = {pi: LN[pi][1] for pi in range(len(P))}
    company_stem = alnum(re.sub(r"\b(limited|ltd)\b\.?", "", cfg["company_name"], flags=re.I))[:18]
    l1n = 0
    for s in sections:
        d, sl = s["d"], s["slice"]
        if not sl:
            issues.append({"doc_id": doc_id, "type": "empty_l1", "detail": d["title"]})
            continue
        l1n += 1
        l1_id = f"{doc_id}-L1-{l1n:03d}"
        raw_lines = [LN[pi][0][li] for pi, li in sl]
        pages_idx = sorted({pi for pi, _ in sl})
        # heading text for chapter
        def mk(sec_id, level, parent, heading_src, heading, lines_flat, flags, toc=None, extra=None, part=None):
            rl = [LN[pi][0][li] for pi, li in lines_flat]
            segs = collections.OrderedDict()
            for (pi, li), r in zip(lines_flat, rl):
                segs.setdefault(pi, []).append(r["raw"])
            raw_text = "\n".join("\n".join(v) for v in segs.values())
            text, removed = normalise(rl)
            pg = sorted(segs)
            pp = [printed[i] for i in pg if printed[i] is not None]
            rec = {
                "example_id": sec_id, "document_id": doc_id, "source_filename": cfg["source_filename"],
                "document_type": part["document_type"], "document_part": part["part"], "doc_date": part["doc_date"],
                "company_name": cfg["company_name"], "level": level, "parent_section_id": parent,
                "start_page": pg[0] + 1, "end_page": pg[-1] + 1,
                "start_printed_page": printed[pg[0]], "end_printed_page": printed[pg[-1]],
                "original_heading": heading, "heading_source": heading_src,
                "section_text_raw": raw_text, "section_text": text,
                "n_chars_raw": len(raw_text), "n_chars": len(text), "n_lines_removed_hf_pn": removed,
                "pages": [{"pdf_page": i + 1, "printed_page": printed[i], "n_lines": len(segs[i])} for i in pg],
                "quality_flags": flags, "extraction_method": "pdftotext-layout + pymupdf-bold-lines",
                "source_reference": f"{cfg['source_filename']}#pdf_page={pg[0]+1}" + (f"-{pg[-1]+1}" if pg[-1] != pg[0] else ""),
                "category": None, "review_status": "unlabeled",
            }
            if toc:
                rec["toc"] = toc
            if extra:
                rec.update(extra)
            return rec
        part = d["part"]
        l1_rec = mk(l1_id, 1, None, d["heading_source"], d["title"], sl, list(d["flags"]), toc=d.get("toc"),
                    extra={"aliases": d.get("aliases", []), "section_part": d.get("section_part")}, part=part)
        recs.append(l1_rec)
        # ---- level 2
        if "contents_page" in d["flags"]:
            continue
        # headings within this chapter
        starts = []
        for pi in pages_idx:
            lines_p, margin = LN[pi]
            for (i, j) in heading_blocks(lines_p, margin, company_stem):
                if (pi, i) in set(sl):
                    starts.append((pi, i, j))
        # a chapter's own title line is the first line(s): do not start a subsection there
        first = sl[0]
        starts = [s_ for s_ in starts if (s_[0], s_[1]) > first or (s_[0], s_[1]) != first]
        # cut points in flat order
        cuts = [(pi, i) for pi, i, j in starts if (pi, i) != first]
        # drop headings that are the chapter title itself (same text as the L1 heading within first 3 lines)
        pos_in = {k_: n_ for n_, k_ in enumerate(sl)}
        cuts = [c for c in cuts if pos_in[c] >= 1]
        cuts = sorted(set(cuts), key=lambda c: pos_in[c])
        bounds = [0] + [pos_in[c] for c in cuts] + [len(sl)]
        l2n = 0
        for bi in range(len(bounds) - 1):
            seg = sl[bounds[bi]:bounds[bi + 1]]
            if not seg:
                continue
            l2n += 1
            if bi == 0:
                heading, src = d["title"], "chapter_preamble"
            else:
                hp, hi = seg[0]
                blk = next(x for x in starts if (x[0], x[1]) == (hp, hi))
                heading = norm(" ".join(LN[hp][0][k_]["n"] for k_ in range(blk[1], blk[2] + 1)))
                src = "detected_bold_heading"
            flags = []
            recs.append(mk(f"{doc_id}-L2-{l1n:03d}-{l2n:03d}", 2, l1_id, src, heading, seg, flags, part=part))
    return recs

def main():
    all_recs, all_issues = [], []
    meta = {}
    for doc_id in CFG:
        P, LN, printed, sections, issues, hf = build_doc(doc_id)
        recs = emit(doc_id, P, LN, printed, sections, issues)
        # ---- text conservation checks (hard assertions)
        l1 = [r for r in recs if r["level"] == 1]
        for r in l1:
            kids = [k for k in recs if k["parent_section_id"] == r["example_id"]]
            if kids:
                joined = "\n".join(k["section_text_raw"] for k in kids)
                assert joined == r["section_text_raw"], (doc_id, r["example_id"], "L2 does not partition L1")
        total_raw_lines = sum(len(L[0]) for L in LN)
        covered = sum(sum(p["n_lines"] for p in r["pages"]) for r in l1)
        # lines before first part start or after anchors are all covered by synthetic front matter -> expect equality
        if covered != total_raw_lines:
            issues.append({"doc_id": doc_id, "type": "line_coverage_mismatch", "detail": {"covered": covered, "total": total_raw_lines}})
        all_recs += recs
        all_issues += issues
        meta[doc_id] = {"n_pages": len(P), "n_header_footer_patterns": len(hf), "hf_patterns": sorted(hf)[:20],
                        "printed_page_found": sum(1 for x in printed if x is not None),
                        "line_coverage": {"covered": covered, "total": total_raw_lines}}
        print(f"{doc_id:22s} L1={len(l1):3d} L2={sum(1 for r in recs if r['level']==2):4d} issues={len(issues)} coverage={covered}/{total_raw_lines}", flush=True)
    with open(OUT / "extracted_sections.jsonl", "w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "structure_issues.json").write_text(json.dumps(all_issues, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "structure_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
