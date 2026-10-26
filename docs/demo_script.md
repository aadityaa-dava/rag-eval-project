# Demo video script (about 3 minutes)

Before recording: start Ollama, start the API (uvicorn) and Streamlit, and ask any question once so the model is loaded (the first query after idle takes about 4 s longer). Record only what you actually see. If the app behaves differently from this script, say what it did.

0:00 - Problem (20 s)
- "Companies deploying AI in regulated settings need answers they can check. This answers questions over 10 FDA documents, and I built it around measuring where it fails."
- Show the README headline table.

0:20 - The corpus (20 s)
- Show data/raw or the manifest. Mention: real PDFs and Federal Register pages, margin line numbers, three-column layouts, a spoken-language transcript, and three overlapping versions of the cybersecurity guidance.

0:40 - A good answer (40 s)
- In the app, click "What is data drift?". Point at the answer, then at the CITED source and the page number.
- Expand the cited source and read the sentence that supports the answer. This is the point of the app: the claim can be checked.
- Point at the timing line: retrieval in milliseconds, generation in seconds, all local.

1:20 - A refusal (20 s)
- Click the 510(k) user fee question. Show the "I don't know" warning. Say: this is a topic the documents do not cover.

1:40 - A known failure (30 s)
- Ask the most-recent-cybersecurity-guidance question. If the answer is wrong or overconfident, say so plainly and point at the eval: this is the one confidently wrong answer found among 49 questions.
- If the app answers it correctly, say that too, and describe the failure from the README instead. Do not stage a failure.

2:10 - What the evaluation found (40 s)
- Show the README results table. Say: hybrid BM25 plus vector search raised fully correct answers from 49% to 69%; chunk size, embedding model, two rerankers and an LLM evidence gate did not help; with the right passage supplied, the model was right 47 of 49 times, so retrieval was the bottleneck.

2:50 - Honest limits (20 s)
- "49 questions, written from the same documents, reviewed by Claude and spot-checked by me, so the intervals are wide. The next step is questions written by someone who has not seen the passages."
