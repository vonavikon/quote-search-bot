"""Retrieval: dense (cosine) by default + BM25 as a fallback via RRF.

A naive hybrid made recall@5 worse (0.431 vs 0.556 for dense-only): BM25
drags in noise on abstract topics. So the base path is dense-only, and
the lexical leg is attached selectively (wants_lexical).
Run:
  python -X utf8 hybrid.py одиночество
"""
import re
import sys
from collections import Counter

import numpy as np

from embed import embed_query
from index import build


# ---------- stage 3: tokenization ----------
_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+")


def tokenize(text):
    """Text -> a list of tokens (lowercase, letters/digits only)."""
    return _TOKEN_RE.findall(text.lower())


# ---------- stage 4: BM25 ----------
class BM25:
    """BM25 over a tokenized corpus. Parameters k1=1.5, b=0.75 (standard)."""

    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.avgdl = sum(len(d) for d in tokenized_docs) / max(1, self.N)
        self.k1, self.b = k1, b
        self.df = Counter()          # how many documents contain the term
        self.tf = []                 # tf[i] = Counter of terms in document i
        for d in tokenized_docs:
            c = Counter(d)
            self.tf.append(c)
            for t in c:
                self.df[t] += 1

    def idf(self, term):
        df = self.df.get(term, 0)
        return math_log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens, i):
        s, dl = 0.0, len(self.docs[i])
        for t in query_tokens:
            f = self.tf[i].get(t, 0)
            if f == 0:
                continue
            idf = self.idf(t)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * (f * (self.k1 + 1)) / denom
        return s

    def scores(self, query_tokens):
        return [self.score(query_tokens, i) for i in range(self.N)]


def math_log(x):
    import math
    return math.log(x)


# ---------- stage 5: reciprocal rank fusion ----------
def rrf_fusion(dense_order, bm25_order, k=60):
    """Fuse two ranked lists. score(doc) = Σ 1/(k + rank)."""
    score = Counter()
    for pos, doc in enumerate(dense_order):
        score[doc] += 1 / (k + pos + 1)
    for pos, doc in enumerate(bm25_order):
        score[doc] += 1 / (k + pos + 1)
    return sorted(score, key=lambda d: -score[d]), score


# ---------- lexical fallback policy ----------

WEAK_TOP1 = 0.45  # below this dense found nothing confident (garbage/OOV query)
RARE_DF = 3       # df of a name in the corpus at which lexical search helps

_NAME_RE = re.compile(r"[А-ЯЁA-Z][а-яёa-z-]+")


def wants_lexical(query, bm, dense_top1):
    """Decides whether to attach BM25. Returns (bool, reason).

    Calibrated on qwen3-4b: thematic queries top-1 0.56–0.67, names/books
    0.50–0.55 (df 0–3), garbage 0.39. Cosine alone is not enough (the
    bands overlap), hence three rules: a weak top-1, a rare proper name,
    quotes.
    """
    if dense_top1 < WEAK_TOP1:
        return True, f"dense top-1 {dense_top1:.3f} < {WEAK_TOP1}"
    for tok in _NAME_RE.findall(query):
        df = bm.df.get(tok.lower(), 0)
        if 0 < df <= RARE_DF:
            return True, f"редкое имя «{tok}» (df={df})"
    if '"' in query or "«" in query:
        return True, "дословный фрагмент в кавычках"
    return False, ""


def retrieve(query, n=50):
    """Search: dense-only by default, BM25 as a fallback → top-n posts.

    Returns: (order, posts, dense_scores, bm25_scores, path).
    order — post indexes (0..N-1); path = "dense" or "hybrid (reason)".
    """
    V, posts = build()

    # dense: cosine of the query (with the instruction prefix) against all posts
    q = np.array(embed_query(query)[0][0], dtype=np.float32)
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    dense = Vn @ (q / np.linalg.norm(q))
    dense_order = np.argsort(dense)[::-1].tolist()

    # sparse: BM25 over tokens
    tokens_all = [tokenize(p["text"]) for p in posts]
    bm = BM25(tokens_all)
    bm_scores = bm.scores(tokenize(query))

    fallback, why = wants_lexical(query, bm, float(dense[dense_order[0]]))
    if not fallback:
        return dense_order[:n], posts, dense, bm_scores, "dense"

    bm25_order = sorted(range(len(posts)), key=lambda i: -bm_scores[i])
    fused, _ = rrf_fusion(dense_order, bm25_order)
    return fused[:n], posts, dense, bm_scores, f"hybrid ({why})"


def run(query, k=5):
    order, posts, dense, bm_scores, path = retrieve(query, n=k)
    print(f"\nзапрос: «{query}»  ·  путь: {path}\n")
    for r, i in enumerate(order[:k], 1):
        p = posts[i]
        print(f"{r}. [dense {dense[i]:.3f} | bm25 {bm_scores[i]:.2f}] "
              f"{p['author'] or 'аноним'} · {p['date']}")
        print(f"   {p['text'][:280]}\n")


if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "одиночество")
