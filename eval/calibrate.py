"""Blind human grading of a stratified sample of judged answers. Resumable. Judge verdicts are hidden."""
import json
import random
import sys
from pathlib import Path

JUDGED, EVAL = Path("eval/results/judged_baseline.jsonl"), Path("eval/eval_set.jsonl")
GEN, LOG = Path("eval/results/gen_baseline.jsonl"), Path("eval/results/human_grades.jsonl")
sys.path.insert(0, "src")


def load(p):
    return [json.loads(l) for l in p.open()] if p.exists() else []


def pick():
    rows = [r for r in load(JUDGED) if r["kind"] != "unanswerable" and not r["refused"]]
    rng = random.Random(11)
    sus = [r for r in rows if r["condition"] == "B" and r["faithful"]["supported"] == "none"]
    oth = [r for r in rows if r not in sus]
    sample = rng.sample(sus, min(6, len(sus))) + rng.sample(oth, 9)
    rng.shuffle(sample)
    return sample


def main():
    qs = {q["qid"]: q for q in load(EVAL)}
    gens = {(g["qid"], g["condition"]): g for g in load(GEN)}
    done = {(r["qid"], r["condition"]) for r in load(LOG)}
    from rag.generate import CHUNKS
    from rag.index import collection_name, get_client, load_model, search
    model = load_model()
    col = get_client().get_collection(collection_name(CHUNKS))
    sample = pick()
    for n, r in enumerate(sample, 1):
        if (r["qid"], r["condition"]) in done:
            continue
        q = qs[r["qid"]]
        srcs = q["evidence"][:5] if r["condition"] == "B" else [h["text"] for h in search(model, col, q["question"], k=5)]
        print("\n" + "=" * 78)
        print(f"[{n}/{len(sample)}]  QUESTION: {q['question']}\n")
        print("SOURCES the answer was based on:")
        for i, t in enumerate(srcs, 1):
            print(f"  [{i}] {t[:700]}{' ...' if len(t) > 700 else ''}\n")
        print(f"REFERENCE ANSWER: {q['gold_answer']}\n\nANSWER TO GRADE:  {r['answer']}\n")
        c = f = ""
        while c not in ("c", "p", "i", "q"):
            c = input("correctness vs reference: c=correct p=partial i=incorrect  (q=quit) > ").strip().lower()
        if c == "q":
            break
        while f not in ("a", "m", "s", "n"):
            f = input("faithfulness to SOURCES: a=all m=most s=some n=none > ").strip().lower()
        with LOG.open("a") as fh:
            fh.write(json.dumps({"qid": r["qid"], "condition": r["condition"],
                                 "correct": {"c": "correct", "p": "partial", "i": "incorrect"}[c],
                                 "faithful": {"a": "all", "m": "most", "s": "some", "n": "none"}[f]}) + "\n")

    human = {(h["qid"], h["condition"]): h for h in load(LOG)}
    judged = {(r["qid"], r["condition"]): r for r in load(JUDGED)}
    if human:
        both = [(judged[k], h) for k, h in human.items()]
        ca = sum(j["correct"]["verdict"] == h["correct"] for j, h in both)
        fa = sum(j["faithful"]["supported"] == h["faithful"] for j, h in both)
        print(f"\ngraded {len(both)} | judge agrees on correctness {ca}/{len(both)} | on faithfulness {fa}/{len(both)}")
        for j, h in both:
            if j["correct"]["verdict"] != h["correct"] or j["faithful"]["supported"] != h["faithful"]:
                print(f"  {j['qid']} {j['condition']}: correct judge={j['correct']['verdict']} you={h['correct']} | "
                      f"faithful judge={j['faithful']['supported']} you={h['faithful']}")


if __name__ == "__main__":
    main()
