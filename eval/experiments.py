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


def evaluate_config(size, overlap, model_name="BAAI/bge-small-en-v1.5"):
    """Runs in a FRESH process: reopen the persisted index, score with Chroma and with exact search."""
    import numpy as np
    chunks_path = f"data/processed/chunks_{size}_{overlap}.jsonl"
    col = ix.get_client().get_collection(ix.collection_name(chunks_path, model_name))
    model = ix.load_model(model_name)
    got = col.get(include=["embeddings", "metadatas"])
    E, ids, metas = np.array(got["embeddings"]), got["ids"], got["metadatas"]
    qs = [q for q in (json.loads(l) for l in EVAL.open()) if q["kind"] != "unanswerable"]
    per_q = {"chroma": [], "exact": []}
    for q in qs:
        per_q["chroma"].append(score(q, ix.search(model, col, q["question"], k=10, model_name=model_name)))
        qv = ix.embed_query(model, q["question"], model_name)
        order = np.argsort(-(E @ qv))[:10]
        per_q["exact"].append(score(q, [{**metas[i], "id": ids[i]} for i in order]))
    assert E.shape[1] == model.get_sentence_embedding_dimension(), (E.shape, model.get_sentence_embedding_dimension())
    row = {"size": size, "overlap": overlap, "model": model_name, "n_chunks": col.count(), "dim": int(E.shape[1])}
    for mode, res in per_q.items():
        for m in ("hit@1", "hit@5", "hit@10", "mrr"):
            v = [r[m] for r in res]
            row[f"{mode}_{m}"] = round(statistics.mean(v), 3)
    row["exact_hit@5_ci"] = [round(x, 3) for x in boot([r["hit@5"] for r in per_q["exact"]])]
    row["chroma_equals_exact"] = [r["first_rank"] for r in per_q["chroma"]] == [r["first_rank"] for r in per_q["exact"]]
    row["per_question_first_rank"] = {q["qid"]: r["first_rank"] for q, r in zip(qs, per_q["exact"])}
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def build_config(size, overlap, model_name="BAAI/bge-small-en-v1.5"):
    """Runs in its own process: chunk + embed + persist. Never queries."""
    chunks_path = Path(f"data/processed/chunks_{size}_{overlap}.jsonl")
    if not chunks_path.exists():
        chunker.main(size, overlap)
    ix.build(str(chunks_path), model_name)


if __name__ == "__main__":
    import subprocess
    if len(sys.argv) > 1 and sys.argv[1] in ("build", "eval"):
        size, ovl = int(sys.argv[2]), int(sys.argv[3])
        model_name = sys.argv[4] if len(sys.argv) > 4 else "BAAI/bge-small-en-v1.5"
        (build_config if sys.argv[1] == "build" else evaluate_config)(size, ovl, model_name)
        sys.exit(0)
    configs = [(1000, 150), (500, 75), (2000, 300), (1000, 0), (1000, 300)]
    print(f"{'size':>5} {'ovl':>4} {'chunks':>6} {'hit@1':>6} {'hit@5':>6} {'hit@10':>6} {'mrr':>6}  exact hit@5 CI   chroma==exact")
    for size, ovl in configs:
        for stage in ("build", "eval"):
            subprocess.run([sys.executable, __file__, stage, str(size), str(ovl)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**__import__("os").environ, "PYTHONPATH": "src"})
        r = [json.loads(l) for l in OUT.open()][-1]
        print(f"{size:5d} {ovl:4d} {r['n_chunks']:6d} {r['exact_hit@1']:6.3f} {r['exact_hit@5']:6.3f} "
              f"{r['exact_hit@10']:6.3f} {r['exact_mrr']:6.3f}  {r['exact_hit@5_ci']}  {r['chroma_equals_exact']}", flush=True)
