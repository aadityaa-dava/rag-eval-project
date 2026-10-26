"""Judge-free faithfulness proxies: token grounding in sources and cited-source word overlap."""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from rag.generate import CHUNKS
from rag.index import collection_name, get_client, load_model, search

EVAL, JUDGED, GEN = Path("eval/eval_set.jsonl"), Path("eval/results/judged_baseline.jsonl"), Path("eval/results/gen_baseline.jsonl")
STOP = set("the a an of to and or in on for with is are was were be by as at from that this it its their which what "
           "any all can may should must not no do does fda according guidance device devices".split())
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def special_tokens(ans):
    """Numbers, ids, and quoted phrases that must appear in the sources if the answer is grounded."""
    t = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", ans)
    toks = set(re.findall(r"\b(?:[A-Z]{2,}[A-Za-z]*\d+[A-Za-z]*|\d[\d,.\-/]*\d|\d)\b", t))
    quotes = re.findall(r"[\"“]([^\"”]{12,})[\"”]", t)
    return toks, quotes


def content_words(s):
    return {w for w in re.findall(r"[a-z0-9]{4,}", s.lower()) if w not in STOP}


def main():
    qs = {q["qid"]: q for q in (json.loads(l) for l in EVAL.open())}
    gens = {(g["qid"], g["condition"]): g for g in (json.loads(l) for l in GEN.open())}
    model = load_model()
    col = get_client().get_collection(collection_name(CHUNKS))
    out = []
    for r in (json.loads(l) for l in JUDGED.open()):
        if r["kind"] == "unanswerable" or r["refused"]:
            continue
        q = qs[r["qid"]]
        srcs = q["evidence"][:5] if r["condition"] == "B" else [h["text"] for h in search(model, col, q["question"], k=5)]
        blob = norm(" ".join(srcs))
        toks, quotes = special_tokens(r["answer"])
        ungrounded = [t for t in toks if norm(t) not in blob] + [qt for qt in quotes if norm(qt) not in blob]
        weak = []
        for sent in re.split(r"(?<=[.!?])\s+|\n", r["answer"]):
            for c in {int(x) for x in re.findall(r"\[(\d+)\]", sent)}:
                words = content_words(re.sub(r"\[\d+\]", "", sent))
                if 1 <= c <= len(srcs) and len(words) >= 4:
                    if len(words & content_words(srcs[c - 1])) / len(words) < 0.35:
                        weak.append(c)
        out.append({"qid": r["qid"], "condition": r["condition"], "ungrounded": ungrounded, "weak_cites": weak,
                    "grounded": not ungrounded, "cites_ok": not weak})
    Path("eval/results/grounding_baseline.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n")
    for c in ("A", "B"):
        s = [o for o in out if o["condition"] == c]
        print(f"condition {c} (n={len(s)}): all special tokens grounded {sum(o['grounded'] for o in s)}/{len(s)} "
              f"| every cited sentence overlaps its source {sum(o['cites_ok'] for o in s)}/{len(s)}")
    print("\nflagged ungrounded tokens (sample):")
    for o in [o for o in out if o["ungrounded"]][:8]:
        print(f"  {o['qid']} {o['condition']}: {o['ungrounded'][:4]}")
    print("\nflagged weak citations (sample):")
    for o in [o for o in out if o["weak_cites"]][:6]:
        print(f"  {o['qid']} {o['condition']}: cited sources {o['weak_cites']}")


if __name__ == "__main__":
    main()
