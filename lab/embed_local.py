"""A local (self-hosted) embedding model, BAAI/bge-m3, via
sentence-transformers.

Unlike the bothub client embed.py — works offline, ~2.3 GB, inference on
CPU. bge-m3 dense: 1024-dim, multilingual. The query needs NO
instruction prefix (unlike qwen3) — the model is trained symmetric for
retrieval.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
_ST = None


def _st():
    """Lazy model loading (once per process)."""
    global _ST
    if _ST is None:
        print(f"  загрузка {MODEL_NAME} (первый раз качает ~2.3 ГБ)...", flush=True)
        _ST = SentenceTransformer(MODEL_NAME)
    return _ST


def embed_local(texts, batch=32):
    """texts -> (list[list[float]], usage). usage=None: local, no billing."""
    vecs = _st().encode(
        list(texts),
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32).tolist(), None
