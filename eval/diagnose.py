"""Show, for each top-5 miss, what ranked first and what the gold passage looked like."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "eval")
from rag.index import collection_name, get_client, load_model, search
from retrieval_eval import CHUNKS, matches, targets_for

qs = {q["qid"]: q for q in (json.loads(l) for l in open("eval/eval_set.jsonl"))}
rows = [json.loads(l) for l in open("eval/results/retrieval_baseline.jsonl")]
miss = [r["qid"] for r in rows if r.get("hit@5") == 0.0]
model = load_model()
col = get_client().get_collection(collection_name(CHUNKS))
for qid in miss:
    q = qs[qid]
    hits = search(model, col, q["question"], k=10)
    ts = targets_for(q)
    g = q["gold_pages"][0]
    print("=" * 74)
    print(f"{qid}: {q['question']}")
    print(f"GOLD: {g['doc'][:30]} p{g['start_page']}-{g['end_page']}")
    for i, h in enumerate(hits[:3], 1):
        ok = "GOLD" if any(matches(h, t) for t in ts) else "    "
        print(f"  #{i} {ok} {h['distance']:.3f} {h['doc'][:28]} p{h['start_page']} | {h['text'][:110]!r}")
    print(f"  evidence starts: {q['evidence'][0][:110]!r}")
