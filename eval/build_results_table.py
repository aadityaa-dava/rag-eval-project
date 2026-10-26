"""Assemble every retrieval experiment into one markdown table with paired comparisons vs the baseline."""
import json
import random
import statistics
from pathlib import Path

R = Path("eval/results")
rr = lambda x: 0.0 if x is None else 1.0 / x
h = lambda x, k: x is not None and x <= k


def boot_ci(d, n=5000, seed=0):
    rng = random.Random(seed)
    m = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


exp = {}
for l in open(R / "experiments.jsonl"):
    r = json.loads(l)
    exp[(r["size"], r["overlap"], r["model"])] = r
hy = json.load(open(R / "hybrid.jsonl"))
rk = json.load(open(R / "rerank.jsonl"))

BASE = exp[(1000, 150, "BAAI/bge-small-en-v1.5")]["per_question_first_rank"]
qids = list(BASE)

entries = [("Baseline: dense, bge-small, 1000/150", BASE, "0")]
for (s, o, m), r in exp.items():
    if (s, o, m) == (1000, 150, "BAAI/bge-small-en-v1.5"):
        continue
    label = f"Dense, {m.split('/')[-1]}, {s}/{o}"
    entries.append((label, r["per_question_first_rank"], "0"))
entries += [("BM25 only, 1000/150", hy["bm25"]["per_question_first_rank"], "0"),
            ("Hybrid BM25 + dense (RRF), 1000/150", hy["hybrid"]["per_question_first_rank"], "0")]
for name, r in rk.items():
    if name.startswith("hybrid"):
        continue
    entries.append((f"Hybrid + rerank {name.split('/')[-1]}", r["per_question_first_rank"], str(r["rerank_s_per_query"])))

lines = ["| Configuration | hit@1 | hit@5 | hit@10 | MRR | dMRR vs baseline (95% paired CI) | better/worse | rerank s/q |",
         "|---|---|---|---|---|---|---|---|"]
for name, fr, secs in entries:
    f = lambda k: statistics.mean(h(fr[q], k) for q in qids)
    mrr = statistics.mean(rr(fr[q]) for q in qids)
    d = [rr(fr[q]) - rr(BASE[q]) for q in qids]
    if name.startswith("Baseline"):
        ci, bw = "-", "-"
    else:
        lo, hi = boot_ci(d)
        ci = f"{statistics.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" *" if lo > 0 or hi < 0 else "")
        bw = f"{sum(x > 0 for x in d)}/{sum(x < 0 for x in d)}"
    lines.append(f"| {name} | {f(1):.3f} | {f(5):.3f} | {f(10):.3f} | {mrr:.3f} | {ci} | {bw} | {secs} |")
text = "\n".join(lines) + "\n\n`*` = paired 95% interval excludes 0. n=49 answerable-type questions; 10 comparisons against the baseline, so one starred row can appear by chance.\n"
(R / "results_table.md").write_text(text)
print(text)
