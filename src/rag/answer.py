"""Generate a cited answer from retrieved chunks with a local Ollama model."""
import time

import requests

from rag.generate import GEN_MODEL, IDK, OLLAMA_URL, SYSTEM, format_sources  # noqa: F401  (IDK re-exported)


def generate_answer(question, hits, model=GEN_MODEL, timeout=300):
    t = time.perf_counter()
    r = requests.post(OLLAMA_URL, timeout=timeout, json={
        "model": model, "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"}]})
    r.raise_for_status()
    d = r.json()
    return {"answer": d["message"]["content"].strip(), "truncated": d.get("done_reason") == "length",
            "prompt_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count"),
            "generate_s": time.perf_counter() - t}
