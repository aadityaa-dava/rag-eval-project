"""Build eval/eval_set.jsonl from reviewed + hand-written questions; verify every hand-written label."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "eval")
from handmade import HAND

PAGES = [json.loads(l) for l in open("data/processed/pages.jsonl")]
OUT = Path("eval/eval_set.jsonl")


def find(doc, anchor):
    a, hits = anchor.lower(), []
    for p in PAGES:
        if p["doc"] != doc:
            continue
        t = re.sub(r"\s+", " ", p["text"] + " " + p["footnotes"])
        i = t.lower().find(a)
        if i >= 0:
            page, basis = (p["printed_page"], "printed") if p["printed_page"] is not None else (p["pdf_page"], "pdf")
            hits.append({"doc": doc, "start_page": page, "end_page": page, "page_basis": basis,
                         "snippet": t[max(0, i - 300): i + len(anchor) + 300]})
    return hits


def present(pattern):
    rx = re.compile(pattern, re.I)
    return [(p["doc"][:26], p["printed_page"] or p["pdf_page"]) for p in PAGES
            if rx.search(p["text"] + " " + p["footnotes"])]


out, problems = [], []
for r in (json.loads(l) for l in open("eval/reviewed.jsonl")):
    if r["status"] == "rejected":
        continue
    out.append({"qid": r["qid"], "kind": "answerable", "question": r["question"],
                "gold_answer": r["gold_answer"], "gold_mode": "any", "source": "drafted_then_reviewed",
                "gold_docs": [r["gold_doc"]] + r.get("alt_docs", []),
                "gold_pages": [{"doc": r["gold_doc"], "start_page": r["gold_start_page"],
                                "end_page": r["gold_end_page"], "page_basis": r["page_basis"]}],
                "evidence": [r["evidence"]], "human_spot_checked": r["human_spot_checked"],
                "revised": bool(r.get("revised"))})

for h in HAND:
    rec = {"qid": h["qid"], "kind": h["kind"], "question": h["question"], "gold_answer": h["answer"],
           "gold_mode": h["mode"], "source": "handwritten", "gold_docs": [], "gold_pages": [],
           "evidence": [], "human_spot_checked": False, "revised": False}
    for doc, anchor in h.get("gold", []):
        hits = find(doc, anchor)
        if not hits:
            problems.append(f"{h['qid']}: anchor not found in {doc}: {anchor!r}")
        for x in hits:
            rec["evidence"].append(x.pop("snippet"))
            rec["gold_pages"].append(x)
        if doc not in rec["gold_docs"]:
            rec["gold_docs"].append(doc)
    for pat in h.get("absent", []):
        found = present(pat)
        if found:
            problems.append(f"{h['qid']}: {pat!r} should be absent but appears at {found[:3]}")
    where = ", ".join(f"{g['doc'][:22]} p{g['start_page']}" for g in rec["gold_pages"]) or "(absence checks passed)"
    print(f"{h['qid']} {h['kind']:13} {where}")
    out.append(rec)

if problems:
    print("\nPROBLEMS - nothing written:")
    print("\n".join("  " + p for p in problems))
    sys.exit(1)
OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
print(f"\nwrote {len(out)} questions to {OUT}")
print("by kind:", dict(Counter(r["kind"] for r in out)))
print("docs covered:", len({d for r in out for d in r["gold_docs"]}), "of 10")
