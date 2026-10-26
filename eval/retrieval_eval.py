"""Retrieval metrics (hit@k, MRR, recall@k) against page-level gold labels."""
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from rag.index import collection_name, get_client, load_model, search

EVAL = Path("eval/eval_set.jsonl")
CHUNKS = "data/processed/chunks_1000_150.jsonl"
KS = (1, 3, 5, 10)


def targets_for(q):
    """'any' mode = one target (any gold page); 'all' mode = one target per gold document."""
    gp = q["gold_pages"]
    if q["gold_mode"] == "all":
        by = defaultdict(list)
        for g in gp:
            by[g["doc"]].append(g)
        return [{"pages": v, "docs": set()} for v in by.values()]
    return [{"pages": gp, "docs": set(q["gold_docs"]) - {g["doc"] for g in gp}}]


def matches(hit, t):
    if hit["doc"] in t["docs"]:
        return True
    return any(hit["doc"] == g["doc"] and hit["page_basis"] == g["page_basis"]
               and hit["start_page"] <= g["end_page"] and g["start_page"] <= hit["end_page"]
               for g in t["pages"])


def score(q, hits):
    ts = targets_for(q)
    first = next((i for i, h in enumerate(hits, 1) if any(matches(h, t) for t in ts)), None)
    out = {"mrr": 1 / first if first else 0.0, "first_rank": first}
    for k in KS:
        out[f"hit@{k}"] = float(first is not None and first <= k)
        out[f"recall@{k}"] = sum(any(matches(h, t) for h in hits[:k]) for t in ts) / len(ts)
    return out


def boot(vals, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def run(chunks_path=CHUNKS, k=10):
    qs = [json.loads(l) for l in EVAL.open()]
    model = load_model()
    col = get_client().get_collection(collection_name(chunks_path))
    rows = []
    for q in qs:
        hits = search(model, col, q["question"], k=k)
        r = {"qid": q["qid"], "kind": q["kind"], "top1_distance": hits[0]["distance"],
             "top1": f"{hits[0]['doc'][:26]} p{hits[0]['start_page']}"}
        if q["kind"] != "unanswerable":
            r.update(score(q, hits))
        rows.append(r)
    return qs, rows


if __name__ == "__main__":
    qs, rows = run()
    scored = [r for r in rows if "mrr" in r]
    print(f"\nscored {len(scored)} answerable-type questions (unanswerable excluded)\n")
    print(f"{'metric':10} {'value':>6}   95% CI")
    for m in [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["mrr"]:
        v = [r[m] for r in scored]
        lo, hi = boot(v)
        print(f"{m:10} {statistics.mean(v):6.3f}   [{lo:.3f}, {hi:.3f}]")
    print("\nby kind (hit@5 / mrr / n):")
    for kind in sorted({r["kind"] for r in scored}):
        sub = [r for r in scored if r["kind"] == kind]
        print(f"  {kind:14} {statistics.mean(r['hit@5'] for r in sub):.2f} / "
              f"{statistics.mean(r['mrr'] for r in sub):.2f} / {len(sub)}")
    ans = [r["top1_distance"] for r in scored]
    una = [r["top1_distance"] for r in rows if r["kind"] == "unanswerable"]
    print(f"\ntop-1 distance, answerable: median {statistics.median(ans):.3f} "
          f"(max {max(ans):.3f}) | unanswerable: median {statistics.median(una):.3f} "
          f"(min {min(una):.3f}, n={len(una)})")
    miss = [r for r in scored if r["hit@5"] < 1]
    print(f"\nmisses at k=5: {len(miss)}")
    texts = {q["qid"]: q["question"] for q in qs}
    for r in miss[:10]:
        print(f"  {r['qid']} [{r['kind'][:6]}] first_rank={r['first_rank']} top1={r['top1']} | {texts[r['qid']][:62]}")
    Path("eval/results/retrieval_baseline.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
