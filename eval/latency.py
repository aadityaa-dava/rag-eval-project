"""Per-stage latency for the hybrid pipeline and token usage from saved generation runs."""
import json
import statistics
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, "src"); sys.path.insert(0, "eval")
from rag import index as ix
from hybrid_eval import CHUNKS, MODEL, rrf, tok

chunks = [json.loads(l) for l in open(CHUNKS)]
bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
emb = ix.load_model(MODEL)
col = ix.get_client().get_collection(ix.collection_name(CHUNKS, MODEL))
got = col.get(include=["embeddings"])
E, ids = np.array(got["embeddings"]), got["ids"]
qs = [q["question"] for q in (json.loads(l) for l in open("eval/eval_set.jsonl")) if q["kind"] != "unanswerable"][:20]

for q in qs[:3]:                                   # warm-up (model load, GPU kernels)
    ix.embed_query(emb, q, MODEL)

t = {"embed": [], "dense": [], "bm25": [], "fuse": []}
for q in qs:
    a = time.perf_counter(); qv = ix.embed_query(emb, q, MODEL)
    b = time.perf_counter(); dense = [ids[i] for i in np.argsort(-(E @ qv))[:30]]
    c = time.perf_counter(); sp = [chunks[i]["chunk_id"] for i in np.argsort(-bm25.get_scores(tok(q)))[:30]]
    d = time.perf_counter(); rrf([dense, sp], top=5)
    e = time.perf_counter()
    for k, v in zip(t, (b - a, c - b, d - c, e - d)):
        t[k].append(v * 1000)
print("retrieval stages, ms per query (n=20, after warm-up):")
for k, v in t.items():
    print(f"  {k:6} median {statistics.median(v):6.1f}  p95 {sorted(v)[int(0.95 * len(v)) - 1]:6.1f}")
tot = [sum(x) for x in zip(*t.values())]
print(f"  total  median {statistics.median(tot):6.1f}  p95 {sorted(tot)[int(0.95 * len(tot)) - 1]:6.1f}")

print("\ngeneration (llama3.1:8b, local), from saved runs:")
load = lambda p: [json.loads(l) for l in open(p)]
for name, rows in (("dense A", [r for r in load("eval/results/gen_baseline.jsonl") if r["condition"] == "A"]),
                   ("hybrid", load("eval/results/gen_hybrid.jsonl"))):
    g = [r["generate_s"] for r in rows]
    pt = [r["prompt_tokens"] for r in rows]; ot = [r["output_tokens"] for r in rows]
    print(f"  {name:8} n={len(rows):2d} | generate s: median {statistics.median(g):5.1f} p95 {sorted(g)[int(0.95 * len(g)) - 1]:5.1f}"
          f" | prompt tok median {statistics.median(pt):5.0f} | output tok median {statistics.median(ot):4.0f}")
    print(f"           total tokens for the run: {sum(pt):,} in / {sum(ot):,} out | truncated {sum(r['truncated'] for r in rows)}")
