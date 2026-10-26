"""Draft eval questions from sampled chunks with the local LLM. A human reviews every draft."""
import json
import random
import re
from collections import Counter
from pathlib import Path

import requests

CHUNKS = Path("data/processed/chunks_1000_150.jsonl")
OUT = Path("eval/drafts.jsonl")
QUOTA = {"fda_ai_lifecycle_draft_2025": 14, "fda_cybersecurity_premarket_2026": 10,
         "fda_premarket_software_content_2023": 10, "fda_human_factors_2016": 8,
         "fda_cybersecurity_webinar_transcript_2023": 6, "fr_pccp_final_notice_2024": 3,
         "fr_ai_lifecycle_draft_notice_2025": 3, "fr_cybersecurity_notice_2023": 2,
         "fr_cybersecurity_notice_2025": 2, "fr_human_factors_notice_2016": 2}
PROMPT = ("Read the passage and write ONE question that a regulatory affairs professional might ask, "
          "which is fully answerable from the passage alone.\n"
          "Rules: the question must make sense without seeing the passage (never say 'the passage', "
          "'the text', 'this document' or 'above'); use your own wording, not long phrases copied from "
          "the passage; the answer must be 1-3 sentences and use only facts in the passage.\n"
          'Reply as JSON: {"question": "...", "answer": "..."}\n\nPassage:\n')


def longest_run(q, passage):
    """Longest run of consecutive words shared by the question and the passage."""
    qw, pw = re.findall(r"\w+", q.lower()), re.findall(r"\w+", passage.lower())
    best = 0
    for i in range(len(qw)):
        n = best + 1
        while i + n <= len(qw) and any(pw[j:j + n] == qw[i:i + n] for j in range(len(pw) - n + 1)):
            best, n = n, n + 1
    return best


def draft(passage):
    r = requests.post("http://localhost:11434/api/chat", timeout=300, json={
        "model": "llama3.1:8b", "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 300},
        "messages": [{"role": "user", "content": PROMPT + passage}]})
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def main():
    chunks = [json.loads(l) for l in CHUNKS.open()]
    pool = [c for c in chunks if c["n_chars"] >= 400 and "Disease X" not in c["text"]]
    rng, rows = random.Random(42), []
    for doc, n in QUOTA.items():
        for c in rng.sample([c for c in pool if c["doc"] == doc], n):
            try:
                d = draft(c["text"])
                q, a = d["question"].strip(), d["answer"].strip()
            except Exception as e:
                print("skip", c["chunk_id"], e)
                continue
            flags = []
            if re.search(r"\b(passage|the text|this document|above)\b", q, re.I):
                flags.append("not_self_contained")
            if longest_run(q, c["text"]) > 5:
                flags.append("copies_wording")
            rows.append({"qid": f"q{len(rows) + 1:03d}", "kind": "answerable", "question": q,
                         "draft_answer": a, "gold_doc": doc, "gold_start_page": c["start_page"],
                         "gold_end_page": c["end_page"], "page_basis": c["page_basis"],
                         "evidence": c["text"], "source_chunk": c["chunk_id"],
                         "flags": flags, "status": "draft"})
            if len(rows) % 10 == 0:
                print(f"  {len(rows)} drafted...", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"\nwrote {len(rows)} drafts to {OUT}")
    print("flagged:", Counter(f for r in rows for f in r["flags"]), "| clean:", sum(not r["flags"] for r in rows))
    for r in rows[::20]:
        print(f"\n{r['qid']} [{r['gold_doc'][:28]} p{r['gold_start_page']}] flags={r['flags']}\n  Q: {r['question']}\n  A: {r['draft_answer'][:200]}")


if __name__ == "__main__":
    main()
