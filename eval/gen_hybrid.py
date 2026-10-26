"""Generate + judge answers using hybrid retrieval (BM25 + dense via RRF), top 5. Resumable."""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
from rank_bm25 import BM25Okapi

sys.path.insert(0, "src"); sys.path.insert(0, "eval")
from rag import index as ix
from rag.generate import GEN_MODEL, IDK, OLLAMA_URL, SYSTEM, format_sources
from hybrid_eval import CHUNKS, MODEL, rrf, tok
from judge_eval import CORRECT, ask, fill

OUT = Path("eval/results/gen_hybrid.jsonl")


def main():
    chunks = [json.loads(l) for l in open(CHUNKS)]
    by_id = {c["chunk_id"]: c for c in chunks}
    bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
    model = ix.load_model(MODEL)
    col = ix.get_client().get_collection(ix.collection_name(CHUNKS, MODEL))
    got = col.get(include=["embeddings"])
    E, ids = np.array(got["embeddings"]), got["ids"]
    qs = [q for q in (json.loads(l) for l in open("eval/eval_set.jsonl")) if q["kind"] != "unanswerable"]
    done = {r["qid"] for r in (json.loads(l) for l in OUT.open())} if OUT.exists() else set()
    for q in qs:
        if q["qid"] in done:
            continue
        qv = ix.embed_query(model, q["question"], MODEL)
        dense = [ids[i] for i in np.argsort(-(E @ qv))[:30]]
        sparse = [chunks[i]["chunk_id"] for i in np.argsort(-bm25.get_scores(tok(q["question"])))[:30]]
        top = rrf([dense, sparse], top=5)
        hits = [{**by_id[c], "page_basis": by_id[c]["page_basis"]} for c in top]
        t = time.perf_counter()
        r = requests.post(OLLAMA_URL, timeout=600, json={
            "model": GEN_MODEL, "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {q['question']}"}]})
        r.raise_for_status()
        d = r.json()
        ans = d["message"]["content"].strip()
        refused = IDK.lower().rstrip(".") in ans.lower()
        verdict = ({"verdict": "incorrect", "reason": "refused to answer"} if refused
                   else ask(fill(CORRECT, q=q["question"], ref=q["gold_answer"], cand=ans)))
        rec = {"qid": q["qid"], "kind": q["kind"], "refused": refused, "answer": ans, "correct": verdict,
               "prompt_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count"),
               "generate_s": time.perf_counter() - t, "truncated": d.get("done_reason") == "length"}
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{q['qid']} {rec['generate_s']:5.1f}s refused={int(refused)} {verdict.get('verdict', '?'):9} | {ans[:55]!r}", flush=True)


if __name__ == "__main__":
    main()
