"""Score generated answers: deterministic checks + a local LLM judge from a different model family."""
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, "src")
from rag.generate import CHUNKS, IDK, OLLAMA_URL
from rag.index import collection_name, get_client, load_model, search

JUDGE = "qwen2.5:7b"
EVAL, GEN = Path("eval/eval_set.jsonl"), Path("eval/results/gen_baseline.jsonl")
OUT = Path("eval/results/judged_baseline.jsonl")

CORRECT = """You are grading an answer to a question about FDA guidance documents.
Compare the CANDIDATE answer to the REFERENCE answer. Judge only whether the candidate conveys the same facts as the reference. Ignore style, length, and citation markers like [1].
- "correct": all key facts of the reference are present and nothing contradicts it.
- "partial": some key facts are present, but important ones are missing or something is wrong.
- "incorrect": key facts are missing or contradicted.
Reply as JSON: {"verdict": "correct" or "partial" or "incorrect", "reason": "<one sentence>"}

QUESTION: <<q>>
REFERENCE: <<ref>>
CANDIDATE: <<cand>>"""

FAITH = """Decide whether the factual statements in the ANSWER are supported by the SOURCES. Use only the SOURCES, not outside knowledge. Ignore citation markers like [1].
- "all": every factual statement is supported.
- "most": nearly all are supported, with minor unsupported detail.
- "some": a meaningful part is unsupported.
- "none": the answer is unsupported or contradicts the sources.
Reply as JSON: {"supported": "all" or "most" or "some" or "none", "reason": "<one sentence>"}

SOURCES:
<<src>>

ANSWER: <<ans>>"""


def fill(t, **kw):
    for k, v in kw.items():
        t = t.replace(f"<<{k}>>", v)
    return t


def ask(prompt):
    r = requests.post(OLLAMA_URL, timeout=600, json={
        "model": JUDGE, "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 200},
        "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    text = r.json()["message"]["content"]
    try:
        return json.loads(text)
    except Exception:
        return {"error": text[:200]}


def main(only=None):
    qs = {q["qid"]: q for q in (json.loads(l) for l in EVAL.open())}
    gens = [json.loads(l) for l in GEN.open()]
    if only:
        gens = [g for g in gens if g["qid"] in only]
    done = {(r["qid"], r["condition"]) for r in (json.loads(l) for l in OUT.open())} if OUT.exists() else set()
    model = load_model()
    col = get_client().get_collection(collection_name(CHUNKS))
    mismatches, n = [], 0
    for g in gens:
        if (g["qid"], g["condition"]) in done:
            continue
        q, ans = qs[g["qid"]], g["answer"]
        if g["condition"] == "A":
            hits = search(model, col, q["question"], k=5)
            if [(h["doc"], h["start_page"], h["end_page"]) for h in hits] != \
               [(s["doc"], s["start_page"], s["end_page"]) for s in g["sources"]]:
                mismatches.append(g["qid"])
            srcs = [h["text"] for h in hits]
        else:
            srcs = q["evidence"][:5]
        refused = IDK.lower().rstrip(".") in ans.lower()
        cites = [int(c) for c in re.findall(r"\[(\d+)\]", ans)]
        rec = {"qid": g["qid"], "kind": q["kind"], "condition": g["condition"], "refused": refused,
               "n_sources": len(srcs), "has_cite": bool(cites),
               "bad_cite": any(c < 1 or c > len(srcs) for c in cites), "answer": ans}
        if q["kind"] == "unanswerable":
            rec["correct"] = rec["faithful"] = None
        elif refused:
            rec.update(correct={"verdict": "incorrect", "reason": "refused to answer"},
                       faithful={"supported": "all", "reason": "refusal makes no claims"})
        else:
            rec["correct"] = ask(fill(CORRECT, q=q["question"], ref=q["gold_answer"], cand=ans))
            rec["faithful"] = ask(fill(FAITH, src="\n\n".join(f"[{i}] {t}" for i, t in enumerate(srcs, 1)), ans=ans))
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        c = (rec["correct"] or {}).get("verdict", "-")
        s = (rec["faithful"] or {}).get("supported", "-")
        print(f"{g['qid']} {g['condition']} src={len(srcs)} refused={int(refused)} cite={int(rec['has_cite'])} "
              f"badcite={int(rec['bad_cite'])} correct={c:9} faithful={s:5} | {ans[:60]!r}", flush=True)
    print(f"\njudged {n} new answers -> {OUT}")
    print(f"condition-A retrieval mismatches vs generation time: {len(mismatches)} {mismatches}")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
