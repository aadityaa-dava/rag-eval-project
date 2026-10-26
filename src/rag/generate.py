"""Baseline RAG: retrieve top-k chunks, then generate a cited answer with a local Ollama model."""
import time

import requests

from rag.index import collection_name, get_client, load_model, search

import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
GEN_MODEL = "llama3.1:8b"
CHUNKS = "data/processed/chunks_1000_150.jsonl"
IDK = "I don't know based on the provided documents."

SYSTEM = (
    "You answer questions about FDA guidance documents using ONLY the numbered sources provided. "
    "Cite the sources you use inline, like [1] or [2][3]. "
    f"If the sources do not contain the answer, reply exactly: {IDK}"
)


def page_label(h):
    unit = "p." if h["page_basis"] == "printed" else "PDF p."
    pages = h["start_page"] if h["start_page"] == h["end_page"] else f"{h['start_page']}-{h['end_page']}"
    return f"{unit}{pages}"


def format_sources(hits):
    return "\n\n".join(f"[{i}] ({h['doc']}, {page_label(h)}, {h['section']})\n{h['text']}"
                       for i, h in enumerate(hits, 1))


class RAG:
    def __init__(self, chunks_path=CHUNKS, k=5, gen_model=GEN_MODEL):
        self.k, self.gen_model = k, gen_model
        self.embedder = load_model()
        self.col = get_client().get_collection(collection_name(chunks_path))

    def ask(self, question):
        t0 = time.perf_counter()
        hits = search(self.embedder, self.col, question, k=self.k)
        t_retrieve = time.perf_counter() - t0

        t1 = time.perf_counter()
        r = requests.post(OLLAMA_URL, timeout=300, json={
            "model": self.gen_model, "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"},
            ],
        })
        r.raise_for_status()
        d = r.json()
        return {
            "question": question,
            "answer": d["message"]["content"].strip(),
            "sources": hits,
            "retrieve_s": t_retrieve,
            "generate_s": time.perf_counter() - t1,
            "prompt_tokens": d.get("prompt_eval_count"),
            "output_tokens": d.get("eval_count"),
            "ollama_eval_s": d.get("eval_duration", 0) / 1e9,
            "truncated": d.get("done_reason") == "length",
        }
