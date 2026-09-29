"""Мини-эвал 5 эмбеддинг-моделей на своём корпусе.

Метрики (vs строгий ground truth — канонические посты по теме):
  recall@5  — доля GT-постов, попавших в топ-5
  hit@5     — доля запросов, где хоть один GT в топ-5
  MRR       — среднее 1/ранг первого GT-поста (по полной выдаче, ранг с 1)

Запуск:  python -X utf8 eval.py
Корпус каждой модели кэшируется в eval_cache_<slug>.npz.
"""
import os
import re

import numpy as np

from embed import embed
from evalset_draft import EVAL
from load_corpus import load_posts

MODELS = [
    ("text-embedding-3-small", 1536),
    ("text-embedding-3-large", 3072),
    ("text-embedding-ada-002", 1536),
    ("qwen3-embedding-8b", 4096),
    ("qwen3-embedding-4b", 2560),
    ("bge-m3", 1024),
]

# self-host модели: идут локально через embed_local, не через bothub.
LOCAL = {"bge-m3"}

# Qwen3-Embedding — асимметричная instruction-tuned: запросу нужен префикс,
# иначе эмбеддинг запроса деградирует (см. отчёт: recall 0.07 -> 0.56).
_QWEN_INST = ("Instruct: Given a user topic query, "
              "retrieve relevant quotes/passages on that theme.\nQuery: ")


def format_query(model, query):
    return _QWEN_INST + query if model.startswith("qwen3-embedding") else query


def embed_texts(texts, model):
    """Диспетчер: локальные модели -> embed_local, остальные -> bothub."""
    if model in LOCAL:
        from lab.embed_local import embed_local
        return embed_local(texts)
    return embed(texts, model=model)


HERE = os.path.dirname(os.path.abspath(__file__))


def _slug(model):
    return re.sub(r"[^a-z0-9]+", "_", model)


def ensure_corpus(model, posts, batch=32):
    """Векторы всех постов для модели. Кэш на диске (рядом со скриптом)."""
    path = os.path.join(HERE, f"eval_cache_{_slug(model)}.npz")
    try:
        return np.load(path)["V"]
    except FileNotFoundError:
        pass
    texts = [p["text"] for p in posts]
    out = []
    for i in range(0, len(texts), batch):
        vecs, _ = embed_texts(texts[i:i + batch], model)
        out.extend(vecs)
        print(f"  [{model}] embed {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    V = np.asarray(out, dtype=np.float32)
    np.savez(path, V=V)
    return V


def rank_metrics(V, q, posts, gt):
    """Косинус ко всем постам -> recall@5, hit@5, MRR против gt-множества id."""
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Vn = V / norms
    qn = np.linalg.norm(q) or 1.0
    sims = Vn @ (q / qn)
    order = np.argsort(sims)[::-1]
    gt = set(gt)
    top5 = set(posts[int(i)]["id"] for i in order[:5])
    recall5 = len(top5 & gt) / len(gt)
    hit5 = 1.0 if (top5 & gt) else 0.0
    mrr = 0.0
    for rank, i in enumerate(order, 1):
        if posts[int(i)]["id"] in gt:
            mrr = 1.0 / rank
            break
    return recall5, hit5, mrr


def main():
    posts = load_posts()
    queries = list(EVAL.keys())
    rows = []

    for model, dim in MODELS:
        print(f"\n=== {model} (dim={dim}) ===")
        try:
            V = ensure_corpus(model, posts)
        except Exception as e:
            print(f"  SKIP: {str(e)[:120]}")
            rows.append((model, dim, None, None, None))
            continue

        # векторы запросов одним батчем (qwen — с instruction-префиксом)
        qv, _ = embed_texts([format_query(model, q) for q in queries], model)
        qv = [np.asarray(v, dtype=np.float32) for v in qv]

        rec, hit, mrr = [], [], []
        for q, vec in zip(queries, qv):
            r5, h5, m = rank_metrics(V, vec, posts, EVAL[q])
            rec.append(r5)
            hit.append(h5)
            mrr.append(m)
        rows.append((model, dim, np.mean(rec), np.mean(hit), np.mean(mrr)))

    # --- таблица ---
    print("\n" + "=" * 78)
    print(f"{'model':<26}{'dim':>6}{'recall@5':>10}{'hit@5':>8}{'MRR':>8}")
    print("-" * 78)
    for model, dim, r5, h5, m in sorted(rows, key=lambda x: -(x[4] or 0)):
        if r5 is None:
            print(f"{model:<26}{dim:>6}   (ошибка)")
            continue
        print(f"{model:<26}{dim:>6}{r5:>10.3f}{h5:>8.3f}{m:>8.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
