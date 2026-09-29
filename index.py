"""Строит и кэширует индекс: шьёт посты корпуса один раз, сохраняет в corpus.npz.

Инкрементален: при новом экспорте result.json дошивает только НОВЫЕ и
ИЗМЕНЁННЫЕ посты (сопоставление по post_id + тексту), старые векторы берёт
из кэша. Удалённые из корпуса посты просто выпадают из матрицы — перешив
не нужен. Полный перешив — флагом --rebuild. Прототип будущей /resync.

Параллельные поиски и ингест пишут кэш под одним локом, запись атомарна
(tmp-файл + os.replace): читатель не видит полупустой corpus_meta.json.
"""
import json
import os
import sys
import threading

import numpy as np

from load_corpus import DATA_DIR, load_posts
from embed import MODEL, embed

CACHE = os.path.join(DATA_DIR, "corpus.npz")
META = os.path.join(DATA_DIR, "corpus_meta.json")
BATCH = 32

_LOCK = threading.RLock()  # build держит лок и зовёт save — реентерабельность нужна
_MEM = None  # (V, posts) последнего успешного build в этом процессе


def _load_cached():
    """(vectors, cached_posts, cached_model) или None, если кэша нет."""
    if not os.path.exists(CACHE):
        return None
    d = np.load(CACHE)
    meta = json.load(open(META, encoding="utf-8"))
    # старый формат — просто список постов; новый — {"model": ..., "posts": [...]}
    cached_posts = meta["posts"] if isinstance(meta, dict) else meta
    cached_model = meta.get("model") if isinstance(meta, dict) else None
    return d["vectors"], cached_posts, cached_model


def _embed_all(posts):
    vecs = []
    for i in range(0, len(posts), BATCH):
        v, _ = embed([p["text"] for p in posts[i:i + BATCH]])
        vecs.extend(v)
        print(f"  {len(vecs)}/{len(posts)}")
    return vecs


def save(V, posts, model=MODEL):
    """Атомарная запись кэша: оба файла через tmp + os.replace под локом."""
    global _MEM
    with _LOCK:
        tmp_c, tmp_m = os.fspath(CACHE) + ".tmp", os.fspath(META) + ".tmp"
        with open(tmp_c, "wb") as f:  # file-объект: np.savez не дописывает .npz
            np.savez(f, vectors=V)
        with open(tmp_m, "w", encoding="utf-8") as f:
            json.dump({"model": model, "posts": posts}, f, ensure_ascii=False)
        os.replace(tmp_c, CACHE)
        os.replace(tmp_m, META)
        _MEM = (V, posts)


def build(force=False):
    global _MEM
    if not force and _MEM is not None:
        return _MEM
    with _LOCK:
        if not force and _MEM is not None:  # соседний поток уже собрал
            return _MEM
        posts = load_posts()

        if not force:
            cached = _load_cached()
            if cached is not None:
                V_old, cached_posts, cached_model = cached
                if cached_model != MODEL:
                    print(f"!! кэш corpus.npz от модели «{cached_model or 'старый формат'}»,"
                          f" а embed.MODEL = {MODEL}. Пересобираю с нуля...")
                elif len(cached_posts) != len(V_old):
                    print("!! кэш рассинхронизирован (векторов != постов), пересборка...")
                else:
                    vec_by_id = {p["id"]: V_old[i] for i, p in enumerate(cached_posts)}
                    text_by_id = {p["id"]: p["text"] for p in cached_posts}
                    fresh = [p for p in posts
                             if p["id"] not in vec_by_id or text_by_id[p["id"]] != p["text"]]
                    if not fresh and len(posts) == len(cached_posts):
                        print(f"кэш актуален: {len(posts)} векторов.")
                        _MEM = (V_old, posts)
                        return _MEM
                    if fresh:
                        print(f"инкремент: новых/изменённых постов {len(fresh)}, шьём...")
                        for p, v in zip(fresh, _embed_all(fresh)):
                            vec_by_id[p["id"]] = v
                    else:
                        print("посты исчезли из корпуса: векторы выпадают без перешива")
                    # исчезнувшие id не попадают в матрицу — их векторы не нужны
                    V = np.array([vec_by_id[p["id"]] for p in posts], dtype=np.float32)
                    save(V, posts)
                    print(f"готово: {V.shape} (дошито {len(fresh)}).")
                    return V, posts

        print(f"полная пересборка: шьём {len(posts)} постов батчами по {BATCH}...")
        V = np.array(_embed_all(posts), dtype=np.float32)
        save(V, posts)
        print(f"готово: {V.shape}, кэш {CACHE}")
        return V, posts


if __name__ == "__main__":
    build(force="--rebuild" in sys.argv)
