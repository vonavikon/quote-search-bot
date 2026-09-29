"""Локальная (self-host) эмбеддинг-модель BAAI/bge-m3 через sentence-transformers.

В отличие от bothub-клиента embed.py — работает офлайн, весит ~2.3 ГБ, инференс
на CPU. bge-m3 dense: 1024-dim, мультиязычный. Инструкционный префикс запросу
НЕ нужен (в отличие от qwen3) — модель натренирована симметричной для retrieval.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
_ST = None


def _st():
    """Ленивая загрузка модели (один раз за процесс)."""
    global _ST
    if _ST is None:
        print(f"  загрузка {MODEL_NAME} (первый раз качает ~2.3 ГБ)...", flush=True)
        _ST = SentenceTransformer(MODEL_NAME)
    return _ST


def embed_local(texts, batch=32):
    """texts -> (list[list[float]], usage). usage=None: локально, без тарификации."""
    vecs = _st().encode(
        list(texts),
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32).tolist(), None
