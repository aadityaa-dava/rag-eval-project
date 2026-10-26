"""Human spot-check of Claude-reviewed eval questions. Resumable; each verdict is saved immediately."""
import json
import random
from pathlib import Path

REV = Path("eval/reviewed.jsonl")
LOG = Path("eval/spot_checks.jsonl")


def load(p):
    return [json.loads(l) for l in p.open()] if p.exists() else []


def main():
    rows = load(REV)
    pool = [r for r in rows if r["status"] in ("approved", "edited")]
    sample = random.Random(7).sample(pool, 10)
    done = {r["qid"] for r in load(LOG)}
    for n, r in enumerate(sample, 1):
        if r["qid"] in done:
            continue
        print("=" * 78)
        print(f"[{n}/10] {r['qid']}  {r['gold_doc']}  pages {r['gold_start_page']}-{r['gold_end_page']}  ({r['status']})")
        print(f"\nEVIDENCE:\n{r['evidence']}\n")
        print(f"QUESTION:    {r['question']}\nGOLD ANSWER: {r['gold_answer']}\n")
        key = ""
        while key not in ("y", "n", "q"):
            key = input("Is the gold answer fully supported by the evidence, and is the question clear? y/n/q > ").strip().lower()
        if key == "q":
            break
        note = input("What is wrong? > ").strip() if key == "n" else ""
        with LOG.open("a") as f:
            f.write(json.dumps({"qid": r["qid"], "ok": key == "y", "note": note}) + "\n")

    log = {r["qid"]: r for r in load(LOG)}
    for r in rows:
        r["human_spot_checked"] = r["qid"] in log
    REV.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    bad = [(q, v["note"]) for q, v in log.items() if not v["ok"]]
    print(f"\nchecked {len(log)}/10 | ok {len(log) - len(bad)} | problems {len(bad)}")
    for q, note in bad:
        print(f"  {q}: {note}")


if __name__ == "__main__":
    main()
