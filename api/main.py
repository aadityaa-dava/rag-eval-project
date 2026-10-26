"""FastAPI service: /health, /search (retrieval only), /ask (retrieval + cited answer)."""
import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag.answer import IDK, generate_answer
from rag.generate import page_label
from rag.retriever import HybridRetriever

LOG = Path("logs/requests.jsonl")


@asynccontextmanager
async def lifespan(app):
    app.state.retriever = HybridRetriever()
    yield


app = FastAPI(title="FDA AI-device guidance RAG", version="0.1.0", lifespan=lifespan)


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(5, ge=1, le=10)


def source_view(n, h, cited):
    return {"n": n, "chunk_id": h["chunk_id"], "doc": h["doc"], "pages": page_label(h),
            "section": h["section"], "cited": n in cited, "snippet": h["text"][:300]}


def log_request(rec):
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass                                   # logging must never break a request


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: Query, request: Request):
    t = time.perf_counter()
    hits = request.app.state.retriever.search(req.question, k=req.k)
    return {"retrieve_ms": round((time.perf_counter() - t) * 1000, 1),
            "results": [source_view(i, h, set()) for i, h in enumerate(hits, 1)]}


@app.post("/ask")
def ask(req: Query, request: Request):
    t0 = time.perf_counter()
    hits = request.app.state.retriever.search(req.question, k=req.k)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    try:
        gen = generate_answer(req.question, hits)
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=503, detail="generator unavailable: is Ollama running on localhost:11434?")
    answer = gen["answer"]
    refused = IDK.lower().rstrip(".") in answer.lower()
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    out = {"answer": answer, "refused": refused, "truncated": gen["truncated"],
           "sources": [source_view(i, h, cited) for i, h in enumerate(hits, 1)],
           "tokens": {"prompt": gen["prompt_tokens"], "output": gen["output_tokens"]},
           "timing": {"retrieve_ms": round(retrieve_ms, 1), "generate_s": round(gen["generate_s"], 2)}}
    log_request({"ts": time.time(), "q_chars": len(req.question), "k": req.k, "refused": refused,
                 "truncated": gen["truncated"], **out["tokens"], **out["timing"]})
    return out
