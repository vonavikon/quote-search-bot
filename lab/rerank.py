"""Reranking: a cross-encoder over the top-N candidates from the hybrid.

A bi-encoder (embeddings, stages 1-5 in hybrid.py) embeds the query and
the document SEPARATELY — each text is "blind" to the other. Fast, but
it catches only coarse thematic similarity. A cross-encoder reads
query+document TOGETHER as one pair → a single relevance number. It sees
nuance ("loneliness as suffering" vs "loneliness as solitude"), and
tells a theme from a random mention of a word. Expensive: one model pass
per pair. Hence two-stage RAG: cheap recall (hybrid → top-50) →
expensive precision (cross-encoder → top-5).

Model: BAAI/bge-reranker-v2-m3 (multilingual, ~2.2 GB, separate from
bge-m3).
Run:  python -X utf8 rerank.py одиночество
"""
import sys

from sentence_transformers import CrossEncoder

from hybrid import retrieve

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_CE = None


def _ce():
    """Lazy cross-encoder loading (the first run downloads ~2.2 GB)."""
    global _CE
    if _CE is None:
        print(f"загрузка {MODEL_NAME}...", flush=True)
        _CE = CrossEncoder(MODEL_NAME)
    return _CE


def rerank(query, candidates, top=5):
    """Rerank candidates with the cross-encoder.

    candidates: [(idx, text), ...] — what the hybrid returned (top-50).
    Returns: [(idx, score), ...] — sorted by score descending, [:top].
    """
    pairs = [[query, text] for _, text in candidates]
    scores = _ce().predict(pairs).tolist()
    ranked = sorted(zip([i for i, _ in candidates], scores), key=lambda x: -x[1])
    return ranked[:top]


def _snippet(text, n=90):
    clean = " ".join(text.split())
    return clean[:n] + ("…" if len(clean) > n else "")


def run(query, pool=50, top=5):
    cand_idx, posts, dense, bm25, _path = retrieve(query, n=pool)
    candidates = [(i, posts[i]["text"]) for i in cand_idx]

    # "before" — the hybrid order (RRF) inside the top-{top}
    before = cand_idx[:top]
    # "after" — the cross-encoder
    after = rerank(query, candidates, top=top)
    after_idx = [i for i, _ in after]
    after_scores = dict(after)

    print(f"\nзапрос: «{query}»   (пул: топ-{pool} от гибрида → реранк до {top})\n")
    print(f"{'#':<3}{'гибрид (RRF)':<58}{'rerank (cross-enc)':<22}")
    print("-" * 83)
    for r in range(top):
        bi = before[r]
        ai = after_idx[r]
        b_line = f"id{posts[bi]['id']:<4} {_snippet(posts[bi]['text'], 48)}"
        a_mark = "↑ вытащил" if ai not in before[:r] else "—"
        a_line = f"id{posts[ai]['id']:<4} {after_scores[ai]:>6.3f} {a_mark}"
        print(f"{r+1:<3}{b_line:<58}{a_line:<22}")

    # whom the cross-encoder rescued from deep in the pool (not in the hybrid top-5)
    rescued = [i for i in after_idx if i not in before]
    if rescued:
        print(f"\nвытащил из глубины пула (вне гибрид-топ-{top}):")
        for i in rescued:
            depth = cand_idx.index(i) + 1
            print(f"  id{posts[i]['id']} (был #{depth} у гибрида, "
                  f"теперь #{after_idx.index(i)+1}) — {_snippet(posts[i]['text'], 70)}")
    else:
        print(f"\ncross-encoder не изменил топ-{top}.")

    print(f"\n--- rerank топ-{top}, полным текстом ---")
    for r, (i, s) in enumerate(after, 1):
        p = posts[i]
        print(f"\n{r}. {p['author'] or '—'} · {p['date']}  (rerank {s:.3f})")
        print(_snippet(p["text"], 300))


if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "одиночество")
