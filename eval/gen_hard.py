"""Run the hard unanswerable questions through hybrid retrieval + generator and save answers for review."""
import json
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

OUT = Path("eval/results/gen_hard.jsonl")


def main():
    chunks = [json.loads(l) for l in open(CHUNKS)]
    by_id = {c["chunk_id"]: c for c in chunks}
    bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
    model = ix.load_model(MODEL)
    col = ix.get_client().get_collection(ix.collection_name(CHUNKS, MODEL))
    got = col.get(include=["embeddings"])
    E, ids = np.array(got["embeddings"]), got["ids"]
    OUT.write_text("")
    for q in (json.loads(l) for l in open("eval/hard_unanswerable.jsonl")):
        qv = ix.embed_query(model, q["question"], MODEL)
        dense = [ids[i] for i in np.argsort(-(E @ qv))[:30]]
        sparse = [chunks[i]["chunk_id"] for i in np.argsort(-bm25.get_scores(tok(q["question"])))[:30]]
        top = rrf([dense, sparse], top=5)
        hits = [by_id[c] for c in top]
        r = requests.post(OLLAMA_URL, timeout=600, json={
            "model": GEN_MODEL, "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {q['question']}"}]})
        r.raise_for_status()
        ans = r.json()["message"]["content"].strip()
        rec = {"qid": q["qid"], "subtype": q["subtype"], "question": q["question"], "answer": ans,
               "exact_refusal": IDK.lower().rstrip(".") in ans.lower(),
               "top_sources": [f"{h['doc'][:26]} p{h['start_page']}" for h in hits]}
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n{q['qid']} [{q['subtype']}] exact_refusal={int(rec['exact_refusal'])}\nQ: {q['question']}\nA: {ans[:420]}", flush=True)


if __name__ == "__main__":
    main()
