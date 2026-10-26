"""Harder unanswerable questions: adjacent-topic, false-premise, out-of-scope. Each is checked against the corpus."""
import json
import re
from pathlib import Path

PAGES = [json.loads(l) for l in open("data/processed/pages.jsonl")]
corpus = re.sub(r"\s+", " ", " ".join(p["text"] + " " + p["footnotes"] for p in PAGES)).lower()

# (qid, type, question, regexes that must NOT match anywhere in the corpus)
HARD = [
 ("u01", "adjacent", "Which specific file format must a Software Bill of Materials be submitted in for a cyber device?", [r"spdx", r"cyclonedx"]),
 ("u02", "adjacent", "How many days does FDA take to review a PCCP submitted with a marketing submission?", [r"pccp.{0,80}\d+ days", r"review (clock|timeline)"]),
 ("u03", "adjacent", "What is the maximum number of days FDA allows a sponsor to respond to an additional information request on a 510(k)?", [r"additional information (request|hold)", r"\d+ days to respond", r"respond within \d+"]),
 ("u04", "adjacent", "What specific encryption algorithm and key length does FDA require for data in transit from a connected medical device?", [r"aes-?\s?\d", r"\b256-bit\b", r"tls 1\.\d"]),
 ("u05", "false_premise", "Why does FDA's draft guidance on AI-enabled devices require all models to be retrained every six months?", [r"retrain.{0,60}every (six|6) months", r"every (six|6) months.{0,60}retrain"]),
 ("u06", "false_premise", "Why does FDA's cybersecurity guidance require manufacturers to submit a penetration test report from an FDA-approved third-party laboratory?", [r"fda-approved (third|laborator)", r"(accredited|approved) (third-party )?(testing )?laborator", r"third-party (testing )?laborator"]),
 ("u07", "false_premise", "Under which section does FDA's 2023 software guidance ban the use of off-the-shelf software in all device software functions?", [r"off-the-shelf.{0,80}(prohibit|ban)", r"(prohibit|ban).{0,80}off-the-shelf"]),
 ("u08", "out_of_scope", "What are FDA's labeling requirements for dietary supplements?", [r"dietary supplement"]),
 ("u09", "out_of_scope", "What is the FDA's process for approving a new generic drug application?", [r"abbreviated new drug", r"generic drug"]),
 ("u10", "out_of_scope", "What does FDA say about the safety of tattoo inks?", [r"tattoo"]),
]

bad = []
for qid, kind, q, pats in HARD:
    hits = [p for p in pats if re.search(p, corpus)]
    if hits:
        bad.append((qid, hits))
print("absence checks failed:", bad if bad else "none")
if not bad:
    recs = [{"qid": qid, "kind": "unanswerable", "subtype": kind, "question": q, "gold_answer": "I don't know based on the provided documents.",
             "gold_docs": [], "gold_pages": [], "evidence": [], "gold_mode": "any", "source": "handwritten_hard"} for qid, kind, q, _ in HARD]
    Path("eval/hard_unanswerable.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    print(f"wrote {len(recs)} questions to eval/hard_unanswerable.jsonl")
