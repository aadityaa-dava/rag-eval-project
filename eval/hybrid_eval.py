"""Hybrid retrieval (BM25 + dense via reciprocal rank fusion) scored against the same gold labels."""
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, "src"); sys.path.insert(0, "eval")
from rag import index as ix
from retrieval_eval import boot, score

CHUNKS = "data/processed/chunks_1000_150.jsonl"
MODEL = "BAAI/bge-small-en-v1.5"
OUT = Path("eval/results/hybrid.jsonl")
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())


def rrf(rank_lists, k=60, top=10):
    scores = {}
    for lst in rank_lists:
        for r, cid in enumerate(lst, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r)
    return sorted(scores, key=lambda c: -scores[c])[:top]


def main():
    chunks = [json.loads(l) for l in open(CHUNKS)]
    meta = {c["chunk_id"]: {"id": c["chunk_id"], "doc": c["doc"], "start_page": c["start_page"],
                            "end_page": c["end_page"], "page_basis": c["page_basis"]} for c in chunks}
    bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
    model = ix.load_model(MODEL)
    col = ix.get_client().get_collection(ix.collection_name(CHUNKS, MODEL))
    got = col.get(include=["embeddings"])
    E, ids = np.array(got["embeddings"]), got["ids"]
    qs = [q for q in (json.loads(l) for l in open("eval/eval_set.jsonl")) if q["kind"] != "unanswerable"]
    res = {"dense": [], "bm25": [], "hybrid": []}
    for q in qs:
        qv = ix.embed_query(model, q["question"], MODEL)
        dense = [ids[i] for i in np.argsort(-(E @ qv))[:30]]
        bscores = bm25.get_scores(tok(q["question"]))
        sparse = [chunks[i]["chunk_id"] for i in np.argsort(-bscores)[:30]]
        for name, lst in (("dense", dense[:10]), ("bm25", sparse[:10]), ("hybrid", rrf([dense, sparse]))):
            res[name].append(score(q, [meta[c] for c in lst]))
    rows = {}
    print(f"{'method':8} {'hit@1':>6} {'hit@5':>6} {'hit@10':>6} {'mrr':>6}  hit@5 CI")
    for name, r in res.items():
        row = {m: round(statistics.mean(x[m] for x in r), 3) for m in ("hit@1", "hit@5", "hit@10", "mrr")}
        row["hit@5_ci"] = [round(v, 3) for v in boot([x["hit@5"] for x in r])]
        row["per_question_first_rank"] = {q["qid"]: x["first_rank"] for q, x in zip(qs, r)}
        rows[name] = row
        print(f"{name:8} {row['hit@1']:6.3f} {row['hit@5']:6.3f} {row['hit@10']:6.3f} {row['mrr']:6.3f}  {row['hit@5_ci']}")
    OUT.write_text(json.dumps(rows))
    never = [q["qid"] for q, x in zip(qs, res["hybrid"]) if x["first_rank"] is None]
    print("\nstill outside top 10 with hybrid:", never)


if __name__ == "__main__":
    main()
