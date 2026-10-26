"""Interactive review of drafted eval questions. Resumable: each decision is saved immediately."""
import json
from collections import Counter
from pathlib import Path

DRAFTS = Path("eval/drafts.jsonl")
DONE = Path("eval/reviewed.jsonl")


def load(path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


def main():
    drafts = load(DRAFTS)
    done = {r["qid"] for r in load(DONE)}
    pending = sorted((d for d in drafts if d["qid"] not in done),
                     key=lambda d: (bool(d["flags"]), d["qid"]))      # clean drafts first
    print(f"{len(done)} reviewed, {len(pending)} pending.  Keys: k=keep  e=edit  r=reject  q=quit\n")
    for n, d in enumerate(pending, 1):
        print("=" * 78)
        print(f"[{n}/{len(pending)}] {d['qid']}  {d['gold_doc']}  pages {d['gold_start_page']}-{d['gold_end_page']}  flags={d['flags']}")
        print(f"\nEVIDENCE:\n{d['evidence']}\n")
        print(f"QUESTION:     {d['question']}\nDRAFT ANSWER: {d['draft_answer']}\n")
        key = ""
        while key not in ("k", "e", "r", "q"):
            key = input("k/e/r/q > ").strip().lower()
        if key == "q":
            break
        rec = dict(d)
        if key == "r":
            rec["status"] = "rejected"
        else:
            if key == "e":
                q = input("new question (Enter = keep current): ").strip()
                a = input("new answer   (Enter = keep current): ").strip()
                rec["question"] = q or d["question"]
                rec["draft_answer"] = a or d["draft_answer"]
            rec["gold_answer"] = rec["draft_answer"]
            rec["status"] = "edited" if key == "e" else "approved"
        with DONE.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    rows = load(DONE)
    print("\nstatus so far:", dict(Counter(r["status"] for r in rows)),
          "| still pending:", len(drafts) - len(rows))
    print("kept per document:", dict(Counter(r["gold_doc"][:30] for r in rows if r["status"] != "rejected")))


if __name__ == "__main__":
    main()
