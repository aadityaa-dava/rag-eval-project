"""Evidence-sufficiency gate: ask an LLM whether retrieved sources contain the answer, before generating."""
import json
import sys
from pathlib import Path

import numpy as np
import requests
from rank_bm25 import BM25Okapi

sys.path.insert(0, "src"); sys.path.insert(0, "eval")
from rag import index as ix
from rag.generate import OLLAMA_URL
from hybrid_eval import CHUNKS, MODEL, rrf, tok

GATE_MODELS = ["llama3.1:8b", "qwen2.5:7b"]
PROMPT = """You decide whether SOURCES contain the information needed to answer a QUESTION.
Answer "yes" only if the sources explicitly state the specific facts the question asks for. If the sources are only related to the topic, or the question assumes something the sources do not say, answer "no".
Reply as JSON: {"sufficient": "yes" or "no", "reason": "<one short sentence>"}

SOURCES:
<<src>>

QUESTION: <<q>>"""


def gate(model, question, hits):
    src = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    r = requests.post(OLLAMA_URL, timeout=600, json={
        "model": model, "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 80},
        "messages": [{"role": "user", "content": PROMPT.replace("<<src>>", src).replace("<<q>>", question)}]})
    r.raise_for_status()
    try:
        return json.loads(r.json()["message"]["content"]).get("sufficient", "").lower() == "yes"
    except Exception:
        return None


def main():
    chunks = [json.loads(l) for l in open(CHUNKS)]
    by_id = {c["chunk_id"]: c for c in chunks}
    bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
    emb = ix.load_model(MODEL)
    col = ix.get_client().get_collection(ix.collection_name(CHUNKS, MODEL))
    got = col.get(include=["embeddings"])
    E, ids = np.array(got["embeddings"]), got["ids"]
    qs = [json.loads(l) for l in open("eval/eval_set.jsonl")] + [json.loads(l) for l in open("eval/hard_unanswerable.jsonl")]
    rank = json.load(open("eval/results/hybrid.jsonl"))["hybrid"]["per_question_first_rank"]
    out = {m: [] for m in GATE_MODELS}
    for q in qs:
        qv = ix.embed_query(emb, q["question"], MODEL)
        dense = [ids[i] for i in np.argsort(-(E @ qv))[:30]]
        sparse = [chunks[i]["chunk_id"] for i in np.argsort(-bm25.get_scores(tok(q["question"])))[:30]]
        hits = [by_id[c] for c in rrf([dense, sparse], top=5)]
        if q["kind"] == "unanswerable":
            group = "unanswerable"
        else:
            group = "answerable, gold in top5" if rank[q["qid"]] is not None and rank[q["qid"]] <= 5 else "answerable, gold NOT in top5"
        for m in GATE_MODELS:
            out[m].append({"qid": q["qid"], "group": group, "pass": gate(m, q["question"], hits)})
        print(q["qid"], end=" ", flush=True)
    Path("eval/results/gate.json").write_text(json.dumps(out))
    print()
    for m in GATE_MODELS:
        print(f"\ngate model: {m}")
        for g in ("answerable, gold in top5", "answerable, gold NOT in top5", "unanswerable"):
            rows = [r for r in out[m] if r["group"] == g]
            passed = sum(r["pass"] is True for r in rows)
            err = sum(r["pass"] is None for r in rows)
            print(f"  {g:30} n={len(rows):2d} passed gate {passed:2d} ({passed / len(rows):.0%}) blocked {len(rows) - passed - err:2d} errors {err}")


if __name__ == "__main__":
    main()
