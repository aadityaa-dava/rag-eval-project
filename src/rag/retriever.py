"""Hybrid retriever: BM25 + dense (bge-small) fused with reciprocal rank fusion."""
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from rag import index as ix

CHUNKS = "data/processed/chunks_1000_150.jsonl"
MODEL = "BAAI/bge-small-en-v1.5"


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def rrf(rank_lists, k=60, top=10):
    scores = {}
    for lst in rank_lists:
        for r, cid in enumerate(lst, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r)
    return sorted(scores, key=lambda c: -scores[c])[:top]


class HybridRetriever:
    def __init__(self, chunks_path=CHUNKS, model_name=MODEL):
        self.chunks = [json.loads(l) for l in open(chunks_path)]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tok(c["text"]) for c in self.chunks])
        self.model_name = model_name
        self.model = ix.load_model(model_name)
        col = ix.get_client().get_collection(ix.collection_name(chunks_path, model_name))
        got = col.get(include=["embeddings"])
        self.E, self.ids = np.array(got["embeddings"]), got["ids"]

    def search(self, query, k=5, depth=30):
        qv = ix.embed_query(self.model, query, self.model_name)
        dense = [self.ids[i] for i in np.argsort(-(self.E @ qv))[:depth]]
        sparse = [self.chunks[i]["chunk_id"] for i in np.argsort(-self.bm25.get_scores(tok(query)))[:depth]]
        return [self.by_id[c] for c in rrf([dense, sparse], top=k)]
