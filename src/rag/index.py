"""Embed chunks with a local model and build a Chroma vector index."""
import json
import sys
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # BGE: queries only
CHROMA_DIR = "chroma_db"


def load_model(name=MODEL_NAME):
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {name} on {device}")
    return SentenceTransformer(name, device=device)


def embed_passages(model, texts):
    return model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)


def embed_query(model, query, model_name=MODEL_NAME):
    prefix = QUERY_PREFIX if "bge" in model_name.lower() else ""
    return model.encode(prefix + query, normalize_embeddings=True)


def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))


def collection_name(chunks_path, model_name=MODEL_NAME):
    return f"{Path(chunks_path).stem}__{model_name.split('/')[-1]}"


def build(chunks_path, model_name=MODEL_NAME):
    chunks = [json.loads(l) for l in open(chunks_path)]
    model = load_model(model_name)
    t0 = time.time()
    vectors = embed_passages(model, [c["text"] for c in chunks])
    print(f"embedded {len(chunks)} chunks in {time.time() - t0:.1f}s, dim={vectors.shape[1]}")

    client = get_client()
    name = collection_name(chunks_path, model_name)
    try:
        client.delete_collection(name)           # rebuild from scratch each run
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 500):
        batch = chunks[i:i + 500]
        col.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=vectors[i:i + 500].tolist(),
            metadatas=[{k: c[k] for k in ("doc", "section", "start_page", "end_page", "page_basis", "n_chars")}
                       for c in batch],
        )
    print(f"collection {name!r}: {col.count()} vectors")
    return name


def search(model, col, query, k=5, model_name=MODEL_NAME):
    res = col.query(query_embeddings=[embed_query(model, query, model_name).tolist()], n_results=k)
    return [{"id": i, "distance": d, "text": t, **m}
            for i, d, t, m in zip(res["ids"][0], res["distances"][0], res["documents"][0], res["metadatas"][0])]


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/processed/chunks_1000_150.jsonl",
          sys.argv[2] if len(sys.argv) > 2 else MODEL_NAME)
