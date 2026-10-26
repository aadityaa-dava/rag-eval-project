"""Generate answers for each eval question: A = retrieved context, B = oracle (gold) context. Resumable."""
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, "src")
from rag.generate import CHUNKS, GEN_MODEL, OLLAMA_URL, SYSTEM, format_sources
from rag.index import collection_name, get_client, load_model, search

EVAL = Path("eval/eval_set.jsonl")
OUT = Path("eval/results/gen_baseline.jsonl")


def generate(question, hits):
    t = time.perf_counter()
    r = requests.post(OLLAMA_URL, timeout=600, json={
        "model": GEN_MODEL, "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"}]})
    r.raise_for_status()
    d = r.json()
    return {"answer": d["message"]["content"].strip(), "truncated": d.get("done_reason") == "length",
            "prompt_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count"),
            "generate_s": time.perf_counter() - t}


def oracle_hits(q):
    out = []
    for g, ev in list(zip(q["gold_pages"], q["evidence"]))[:5]:
        out.append({"doc": g["doc"], "page_basis": g["page_basis"], "start_page": g["start_page"],
                    "end_page": g["end_page"], "section": "", "text": ev})
    return out


def main(only=None):
    qs = [json.loads(l) for l in EVAL.open()]
    if only:
        qs = [q for q in qs if q["qid"] in only]
    done = {(r["qid"], r["condition"]) for r in (json.loads(l) for l in OUT.open())} if OUT.exists() else set()
    model = load_model()
    col = get_client().get_collection(collection_name(CHUNKS))
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    for q in qs:
        for cond in ("A", "B"):
            if cond == "B" and q["kind"] == "unanswerable":
                continue
            if (q["qid"], cond) in done:
                continue
            t0 = time.perf_counter()
            if cond == "A":
                hits = search(model, col, q["question"], k=5)
                retrieve_s = time.perf_counter() - t0
            else:
                hits, retrieve_s = oracle_hits(q), 0.0
            rec = {"qid": q["qid"], "kind": q["kind"], "condition": cond, "retrieve_s": retrieve_s,
                   "sources": [{"doc": h["doc"], "start_page": h["start_page"], "end_page": h["end_page"]} for h in hits]}
            rec.update(generate(q["question"], hits))
            with OUT.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            print(f"{q['qid']} {cond} {rec['generate_s']:5.1f}s tokens {rec['prompt_tokens']}/{rec['output_tokens']}"
                  f"{' TRUNCATED' if rec['truncated'] else ''} | {rec['answer'][:70]!r}", flush=True)
    print(f"\ngenerated {n} new answers -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
