"""Hybrid retrieval (top 30) + cross-encoder rerank, scored on the same gold labels."""
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

sys.path.insert(0, "src"); sys.path.insert(0, "eval")
from rag import index as ix
from retrieval_eval import boot, score
from hybrid_eval import CHUNKS, MODEL, rrf, tok

RERANKERS = ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"]
OUT = Path("eval/results/rerank.jsonl")


def main():
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    chunks = [json.loads(l) for l in open(CHUNKS)]
    by_id = {c["chunk_id"]: c for c in chunks}
    meta = {c["chunk_id"]: {"id": c["chunk_id"], "doc": c["doc"], "start_page": c["start_page"],
                            "end_page": c["end_page"], "page_basis": c["page_basis"]} for c in chunks}
    bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
    model = ix.load_model(MODEL)
    col = ix.get_client().get_collection(ix.collection_name(CHUNKS, MODEL))
    got = col.get(include=["embeddings"])
    E, ids = np.array(got["embeddings"]), got["ids"]
    qs = [q for q in (json.loads(l) for l in open("eval/eval_set.jsonl")) if q["kind"] != "unanswerable"]

    cands = []
    for q in qs:
        qv = ix.embed_query(model, q["question"], MODEL)
        dense = [ids[i] for i in np.argsort(-(E @ qv))[:30]]
        sparse = [chunks[i]["chunk_id"] for i in np.argsort(-bm25.get_scores(tok(q["question"])))[:30]]
        cands.append(rrf([dense, sparse], top=30))

    rows = {"hybrid(no rerank)": [score(q, [meta[c] for c in cs[:10]]) for q, cs in zip(qs, cands)]}
    secs = {}
    for name in RERANKERS:
        ce = CrossEncoder(name, device=device, max_length=512)
        res, t0 = [], time.perf_counter()
        for q, cs in zip(qs, cands):
            s = ce.predict([(q["question"], by_id[c]["text"]) for c in cs], batch_size=16)
            res.append(score(q, [meta[cs[i]] for i in np.argsort(-s)[:10]]))
        rows[name], secs[name] = res, (time.perf_counter() - t0) / len(qs)

    out = {}
    print(f"{'method':42} {'hit@1':>6} {'hit@5':>6} {'hit@10':>6} {'mrr':>6} {'s/query':>8}")
    for name, r in rows.items():
        row = {m: round(statistics.mean(x[m] for x in r), 3) for m in ("hit@1", "hit@5", "hit@10", "mrr")}
        row["per_question_first_rank"] = {q["qid"]: x["first_rank"] for q, x in zip(qs, r)}
        row["rerank_s_per_query"] = round(secs.get(name, 0.0), 3)
        out[name] = row
        print(f"{name:42} {row['hit@1']:6.3f} {row['hit@5']:6.3f} {row['hit@10']:6.3f} {row['mrr']:6.3f} {row['rerank_s_per_query']:8.2f}")
    OUT.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
