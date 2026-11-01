"""A rerank eval: hybrid-only vs hybrid+rerank on our own eval set.

The lab question: does a cross-encoder over the top-50 improve
precision@5? Metrics inside the final top-5:
  recall@5 — share of GT posts in the top-5
  hit@5    — share of queries with at least one GT in the top-5
  mrr@5    — 1/rank of the first GT in the top-5 (0 if none)
Plus recall@50 — how much GT is in the pool at all (the rerank ceiling:
what is missing from the top-50, the rerank cannot rescue).

Run:  python -X utf8 eval_rerank.py
"""
import numpy as np

from evalset_draft import EVAL
from hybrid import retrieve
from lab.rerank import rerank


def metrics5(idx_order, posts, gt):
    """idx_order — a list of post indexes; take [:5].
    → (recall5, hit5, mrr5)."""
    gt = set(gt)
    top5_ids = set(posts[i]["id"] for i in idx_order[:5])
    recall5 = len(top5_ids & gt) / len(gt)
    hit5 = 1.0 if (top5_ids & gt) else 0.0
    mrr5 = 0.0
    for rank, i in enumerate(idx_order[:5], 1):
        if posts[i]["id"] in gt:
            mrr5 = 1.0 / rank
            break
    return recall5, hit5, mrr5


def main():
    queries = list(EVAL.keys())

    # one retrieve per query → the top-50 pool; reused for baseline and rerank
    pools = {}
    for q in queries:
        cand_idx, posts, _, _, _ = retrieve(q, n=50)
        pools[q] = (cand_idx, posts)

    # baseline: the hybrid order (RRF) inside the top-5
    base = np.array([metrics5(pools[q][0], pools[q][1], EVAL[q]) for q in queries])

    # the ceiling: how much GT is in the top-50 at all
    pool_rec = []
    for q in queries:
        cand_idx, posts = pools[q]
        ids = set(posts[i]["id"] for i in cand_idx)
        pool_rec.append(len(ids & set(EVAL[q])) / len(EVAL[q]))

    # rerank: cross-encoder over the pool → top-5
    rr = []
    for q in queries:
        cand_idx, posts = pools[q]
        cands = [(i, posts[i]["text"]) for i in cand_idx]
        after = rerank(q, cands, top=5)
        rr.append(metrics5([i for i, _ in after], posts, EVAL[q]))
    rr = np.array(rr)

    bm = base.mean(0)
    rm = rr.mean(0)
    print("\n" + "=" * 56)
    print(f"{'подход':<24}{'recall@5':>10}{'hit@5':>8}{'mrr@5':>8}")
    print("-" * 56)
    print(f"{'hybrid (RRF)':<24}{bm[0]:>10.3f}{bm[1]:>8.3f}{bm[2]:>8.3f}")
    print(f"{'+ rerank (cross-enc)':<24}{rm[0]:>10.3f}{rm[1]:>8.3f}{rm[2]:>8.3f}")
    print("-" * 56)
    d = rm - bm
    print(f"{'дельта':<24}{d[0]:>+10.3f}{d[1]:>+8.3f}{d[2]:>+8.3f}")
    print(f"\nrecall@50 (потолок пула): {np.mean(pool_rec):.3f}")
    print("=" * 56)

    # per query: where the rerank helped / hurt
    print(f"\n{'запрос':<26}{'hyb r@5':>8}{'rr r@5':>8}{'Δ':>6}   что изменилось")
    print("-" * 70)
    for j, q in enumerate(queries):
        h5 = base[j, 0]
        r5 = rr[j, 0]
        cand_idx, posts = pools[q]
        cands = [(i, posts[i]["text"]) for i in cand_idx]
        after_idx = [i for i, _ in rerank(q, cands, top=5)]
        before_set = set(posts[i]["id"] for i in cand_idx[:5])
        rescued = [posts[i]["id"] for i in after_idx if posts[i]["id"] not in before_set]
        note = ("вытащил " + ",".join(map(str, rescued))) if rescued else "—"
        print(f"{q[:26]:<26}{h5:>8.3f}{r5:>8.3f}{r5-h5:>+6.2f}   {note}")


if __name__ == "__main__":
    main()
