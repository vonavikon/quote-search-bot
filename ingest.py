"""Live ingest: channel_post / edited_channel_post and /resync.

The master before the move to SQLite: corpus.npz + corpus_meta.json
(posts and vectors) and normalization.json (metadata). Everything is
idempotent by post_id + text: re-editing a post re-embeds only its
vector, normalization is called only for unlabeled ids.
"""
import numpy as np

import index
from load_corpus import load_ocr, load_posts, load_splits, split_child_ids
import normalize


def add_posts(new_posts):
    """Add posts to the corpus and normalize the new ones.
    -> {"added", "normalized", "skipped"}.

    new_posts: [{id, date, author, text}] in load_corpus format.
    skipped — [(id, reason)]: posts that must not be ingested (the parent
    of a glue or a collision with a child id — both curated in
    splits.json; the text of an OCR post — in ocr.json, a channel edit
    there would be masked by merge_ocr).
    """
    cached = index._load_cached()
    if cached is None:
        raise RuntimeError("corpus.npz не найден — сначала полный python index.py")
    V_old, cached_posts, cached_model = cached
    parents = {int(k) for k in load_splits()}
    children = {int(c) for c in split_child_ids()}
    ocr_ids = set(load_ocr())
    kept, skipped = [], []
    for p in new_posts:
        if p["id"] in parents:
            # otherwise a live edit of the parent would put it back into the corpus next to its children
            skipped.append((p["id"], "родитель склейки: правки курируются в splits.json"))
        elif p["id"] in ocr_ids:
            skipped.append((p["id"], "OCR-пост: текст курируется в ocr.json, "
                                     "live-правку перенеси туда"))
        elif p["id"] in children:
            skipped.append((p["id"], "id коллидирует с дочерним id склейки — "
                                     "обнови схему в splits.json"))
        else:
            kept.append(p)
    new_posts = kept
    vec_by_id = {p["id"]: V_old[i] for i, p in enumerate(cached_posts)}
    text_by_id = {p["id"]: p["text"] for p in cached_posts}
    by_id_new = {p["id"]: p for p in new_posts}
    fresh = [p for p in new_posts
             if p["id"] not in vec_by_id or text_by_id[p["id"]] != p["text"]]
    if not fresh:
        return {"added": 0, "normalized": 0, "skipped": skipped}

    for p, v in zip(fresh, index._embed_all(fresh)):
        vec_by_id[p["id"]] = v
    # the old order is preserved (texts updated in place), new ones go to the end
    posts, seen = [], set()
    for p in cached_posts:
        posts.append(by_id_new.get(p["id"], p))
        seen.add(p["id"])
    posts += [p for p in new_posts if p["id"] not in seen]

    V = np.array([vec_by_id[p["id"]] for p in posts], dtype=np.float32)
    index.save(V, posts, cached_model)

    # normalize.run() only picks up the new ids (its posts come from meta, already with fresh)
    cache, stats, _ = normalize.run()
    # SQLite is the source of truth: every ingest record is duplicated
    # into the base; items — with the split-children merge, so the canon
    # keeps their metadata
    import cards
    import store
    store.upsert(store.connect(), posts, cards.load_norm())
    return {"added": len(fresh), "normalized": stats["n"], "skipped": skipped}


def resync(export_path):
    """A result.json export from the chat -> pick up what is missing.
    -> same shape as add_posts."""
    return add_posts(load_posts(export_path))


def stats():
    """Summary for /stats."""
    cached = index._load_cached()
    n_vec = int(cached[0].shape[0]) if cached else 0
    posts = load_posts()
    import cards
    items = cards.load_norm()
    kinds = {}
    for it in items.values():
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    signed = sum(1 for it in items.values() if it.get("author"))
    out = {
        "постов": len(posts),
        "векторов": n_vec,
        "размечено": len(items),
        "kind": kinds,
        "автор подписан": signed,
    }
    try:
        import store
        out["SQLite"] = store.counts()
    except Exception:  # noqa: BLE001 — the base is optional until the first migration
        pass
    return out
