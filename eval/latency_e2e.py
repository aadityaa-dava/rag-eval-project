"""Clean end-to-end latency: retrieval and generation timed separately, with Ollama's own load/eval timings."""
import json
import statistics
import sys
import time

import numpy as np
import requests
from rank_bm25 import BM25Okapi

sys.path.insert(0, "src"); sys.path.insert(0, "eval")
from rag import index as ix
from rag.generate import GEN_MODEL, OLLAMA_URL, SYSTEM, format_sources
from hybrid_eval import CHUNKS, MODEL, rrf, tok

chunks = [json.loads(l) for l in open(CHUNKS)]
by_id = {c["chunk_id"]: c for c in chunks}
bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
emb = ix.load_model(MODEL)
col = ix.get_client().get_collection(ix.collection_name(CHUNKS, MODEL))
got = col.get(include=["embeddings"])
E, ids = np.array(got["embeddings"]), got["ids"]
qs = [q["question"] for q in (json.loads(l) for l in open("eval/eval_set.jsonl")) if q["kind"] != "unanswerable"][:15]

# make the first call cold: unload the generator first
requests.post("http://localhost:11434/api/generate", json={"model": GEN_MODEL, "keep_alive": 0})
time.sleep(3)

rows = []
for i, q in enumerate(qs):
    t0 = time.perf_counter()
    qv = ix.embed_query(emb, q, MODEL)
    dense = [ids[j] for j in np.argsort(-(E @ qv))[:30]]
    sp = [chunks[j]["chunk_id"] for j in np.argsort(-bm25.get_scores(tok(q)))[:30]]
    hits = [by_id[c] for c in rrf([dense, sp], top=5)]
    t1 = time.perf_counter()
    r = requests.post(OLLAMA_URL, timeout=600, json={"model": GEN_MODEL, "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {q}"}]})
    t2 = time.perf_counter()
    d = r.json()
    rows.append({"retrieve_ms": (t1 - t0) * 1000, "generate_s": t2 - t1, "load_s": d.get("load_duration", 0) / 1e9,
                 "prefill_s": d.get("prompt_eval_duration", 0) / 1e9, "decode_s": d.get("eval_duration", 0) / 1e9,
                 "in_tok": d.get("prompt_eval_count"), "out_tok": d.get("eval_count")})
    print(f"q{i + 1:02d} retrieve {rows[-1]['retrieve_ms']:6.1f}ms | generate {rows[-1]['generate_s']:5.1f}s "
          f"(load {rows[-1]['load_s']:.1f}s, prefill {rows[-1]['prefill_s']:.1f}s, decode {rows[-1]['decode_s']:.1f}s) tokens {rows[-1]['in_tok']}/{rows[-1]['out_tok']}", flush=True)

warm = rows[1:]
tps = [r["out_tok"] / r["decode_s"] for r in warm if r["decode_s"] > 0]
print(f"\ncold first query: {rows[0]['generate_s']:.1f}s (model load {rows[0]['load_s']:.1f}s)")
print(f"warm queries (n={len(warm)}): generate median {statistics.median(r['generate_s'] for r in warm):.1f}s | "
      f"p95 {sorted(r['generate_s'] for r in warm)[int(0.95 * len(warm)) - 1]:.1f}s | decode {statistics.median(tps):.1f} tokens/s")
print(f"retrieval share of warm end-to-end time: {statistics.median(r['retrieve_ms'] / 1000 / (r['retrieve_ms'] / 1000 + r['generate_s']) for r in warm):.2%}")
json.dump(rows, open("eval/results/latency_e2e.json", "w"))
