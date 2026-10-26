import pytest
import requests
from fastapi.testclient import TestClient

import api.main as m
from rag.answer import IDK
from rag.retriever import rrf, tok


def hit(i):
    return {"chunk_id": f"d::{i:04d}", "doc": "doc", "section": "II. Scope", "start_page": i,
            "end_page": i, "page_basis": "printed", "text": f"passage {i} text"}


class StubRetriever:
    def search(self, query, k=5):
        return [hit(i) for i in range(1, k + 1)]


def fake(answer):
    def _gen(question, hits, **kw):
        return {"answer": answer, "truncated": False, "prompt_tokens": 100, "output_tokens": 10, "generate_s": 0.01}
    return _gen


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "LOG", tmp_path / "requests.jsonl")
    m.app.state.retriever = StubRetriever()
    return TestClient(m.app)          # not used as a context manager, so the model-loading lifespan is skipped


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_search_returns_k_results_with_pages(client):
    r = client.post("/search", json={"question": "what is a pccp", "k": 3}).json()
    assert len(r["results"]) == 3 and r["results"][0]["pages"] == "p.1"
    assert r["results"][0]["text"] == "passage 1 text"


def test_ask_marks_only_cited_sources(client, monkeypatch):
    monkeypatch.setattr(m, "generate_answer", fake("The answer is X [2]."))
    r = client.post("/ask", json={"question": "what is x", "k": 3}).json()
    assert r["refused"] is False
    assert [s["cited"] for s in r["sources"]] == [False, True, False]
    assert r["tokens"] == {"prompt": 100, "output": 10}


def test_ask_flags_refusal(client, monkeypatch):
    monkeypatch.setattr(m, "generate_answer", fake(IDK))
    assert client.post("/ask", json={"question": "what is x"}).json()["refused"] is True


def test_ask_rejects_short_question(client):
    assert client.post("/ask", json={"question": "a"}).status_code == 422


def test_ask_returns_503_when_generator_down(client, monkeypatch):
    def down(question, hits, **kw):
        raise requests.exceptions.ConnectionError()
    monkeypatch.setattr(m, "generate_answer", down)
    assert client.post("/ask", json={"question": "what is x"}).status_code == 503


def test_rrf_puts_doubly_ranked_item_first():
    out = rrf([["a", "b"], ["a", "c"]], top=3)
    assert out[0] == "a" and set(out) == {"a", "b", "c"}


def test_tok():
    assert tok("PCCP, Section 524B!") == ["pccp", "section", "524b"]
