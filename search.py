"""Brute-force поиск по корпусу: запрос -> косинус ко всем постам -> top-k.

  python -X utf8 search.py одиночество
  python -X utf8 search.py "про дружбу"
"""
import sys

import numpy as np

from embed import embed_query
from index import build


def search(query, k=5):
    V, posts = build()
    q = np.array(embed_query(query)[0][0], dtype=np.float32)

    # косинус ко всем одним матричным умножением: нормируем и (V @ q)
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    sims = Vn @ (q / np.linalg.norm(q))
    idx = np.argsort(sims)[::-1][:k]

    print(f"\nзапрос: «{query}»  (dense top-{k})\n")
    for rank, i in enumerate(idx, 1):
        p = posts[i]
        who = p["author"] or "—"
        print(f"{rank}. [{sims[i]:.3f}] {who}  ·  {p['date']}")
        print(f"   {p['text'][:280]}")
        print()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "одиночество"
    search(q)
