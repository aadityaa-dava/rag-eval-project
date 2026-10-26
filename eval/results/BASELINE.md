# Baseline (chunk 1000/150, bge-small-en-v1.5, k=5, llama3.1:8b generator, qwen2.5:7b judge)

Eval set: 55 questions (41 reviewed from drafts, 6 unanswerable, 2 cross-doc, 5 version, 1 inconsistency).
Retrieval metrics use the 49 answerable-type questions; 95% bootstrap CIs in brackets.

| Metric | Value |
|---|---|
| hit@1 | 0.388 [0.245, 0.531] |
| hit@5 | 0.694 [0.571, 0.816] |
| hit@10 | 0.857 [0.755, 0.939] |
| MRR | 0.510 [0.397, 0.625] |

Generation (qwen2.5:7b judge; 13/15 correctness agreement with Claude-graded sample):

| Context | correct | partial | incorrect | answered |
|---|---|---|---|---|
| retrieved (A) | 24/49 | 13 | 12 | 41/49 |
| oracle gold passages (B) | 47/49 | 2 | 0 | 49/49 |

Unanswerable: 6/6 refused (small, easy set).
Faithfulness: judge agreement 7/15 (supported vs not) -> not reported. Heuristic proxies tested and dropped.

Known issues:
- Labels were drafted by llama3.1:8b and reviewed by Claude; 10 spot-checked (8 fine, 2 needed fixes, fixed).
- Calibration graded by Claude, not by a human.
- One baseline retrieval run disagreed with a later rerun on one question (q027); cause not found.
