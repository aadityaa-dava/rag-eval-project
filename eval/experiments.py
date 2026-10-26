"""Run retrieval experiments: (chunk size, overlap, embedding model) -> hit@k / MRR with CIs. Retrieval only."""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "eval")
from rag import chunk as chunker
from rag import index as ix
from retrieval_eval import boot, score

EVAL, OUT = Path("eval/eval_set.jsonl"), Path("eval/results/experiments.jsonl")


def run_config(size, overlap, model_name="BAAI/bge-small-en-v1.5", repeat_check=True):
    ix.MODEL_NAME = model_name
    chunks_path = Path(f"data/processed/chunks_{size}_{overlap}.jsonl")
    if not chunks_path.exists():
        chunker.main(size, overlap)
    name = ix.build(str(chunks_path))
    model = ix.load_model(model_name)
    col = ix.get_client().get_collection(name)
    qs = [q for q in (json.loads(l) for l in EVAL.open()) if q["kind"] != "unanswerable"]
    passes = []
    for _ in range(2 if repeat_check else 1):
        t = time.perf_counter()
        res = [score(q, ix.search(model, col, q["question"], k=10)) for q in qs]
        passes.append((res, time.perf_counter() - t))
    same = repeat_check and [r["first_rank"] for r in passes[0][0]] == [r["first_rank"] for r in passes[1][0]]
    res, secs = passes[0]
    row = {"size": size, "overlap": overlap, "model": model_name, "n_chunks": col.count(),
           "deterministic": same, "query_s": secs / len(qs)}
    for m in ("hit@1", "hit@5", "hit@10", "mrr"):
        v = [r[m] for r in res]
        row[m] = round(statistics.mean(v), 3)
        row[m + "_ci"] = [round(x, 3) for x in boot(v)]
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


if __name__ == "__main__":
    configs = [(1000, 150), (500, 75), (2000, 300), (1000, 0), (1000, 300)]
    print(f"{'size':>5} {'ovl':>4} {'chunks':>6} {'hit@1':>6} {'hit@5':>6} {'hit@10':>6} {'mrr':>6} {'same':>5}  hit@5 CI")
    for size, ovl in configs:
        r = run_config(size, ovl)
        print(f"{size:5d} {ovl:4d} {r['n_chunks']:6d} {r['hit@1']:6.3f} {r['hit@5']:6.3f} {r['hit@10']:6.3f} "
              f"{r['mrr']:6.3f} {str(r['deterministic']):>5}  {r['hit@5_ci']}", flush=True)
