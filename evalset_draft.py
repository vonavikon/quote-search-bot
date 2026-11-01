"""Eval set: queries + ground truth (canonical posts of your corpus).

  python -X utf8 evalset_draft.py        # the canonical set + GT (print)
  python -X utf8 evalset_draft.py --auto # candidates: GT = the top-3 of
                                         # the current search (for the
                                         # owner's review)

The GT is labeled by the corpus owner: relevant posts are marked,
irrelevant ones dropped from the top. The GT is limited to the top shown
— the metrics measure the consistency of the result head, not full
recall over a theme.
"""
import sys

from load_corpus import load_posts

# query (as the user types it) -> [relevant post ids; your labeling]
# Fill in for your corpus: post ids come from result.json or from the
# --auto output. Example:
# EVAL = {
#     "одиночество": [397, 231],
#     "любовь": [55, 54],
# }
EVAL = {}

# Candidates: frequent themes + query formats of live users.
# The GT is picked by the current search (--auto) and approved by the owner.
CANDIDATE_QUERIES = [
    # theme core
    "одиночество", "любовь", "смерть", "вера", "дорога", "ночь", "море", "свет",
    "тоска", "память", "надежда", "тишина", "покой", "судьба", "прощение",
    "молитва", "страх", "гордость", "смирение", "мудрость", "свобода",
    "счастье", "совесть", "дружба", "красота", "труд", "борьба", "путь",
    # feelings and states
    "печаль", "радость", "тревога", "отчаяние", "усталость", "вина",
    "благодарность", "ревность", "обида", "вдохновение",
    # author facets
    "пастернак о любви", "чехов о людях", "достоевский о страдании",
    "кьеркегор о вере", "ницше о свободе", "юнг о душе",
    # exact phrases
    "жалость к себе", "внутренний человек", "ничего не меняется",
    # everyday phrasings
    "про осень", "про зиму и снег", "про утро", "про вечер",
    "как пережить потерю", "о смысле жизни", "про старость",
    "стих про дорогу", "цитата про книги", "про молчание",
]


def main():
    posts = load_posts()
    by_id = {p["id"]: p for p in posts}
    items = __import__("cards").load_norm()
    if "--auto" not in sys.argv:
        total = sum(len(v) for v in EVAL.values())
        print(f"# Эвал-сет: {len(EVAL)} запросов, {total} GT-постов\n")
        for n, (q, gt) in enumerate(EVAL.items(), 1):
            print(f"{'='*70}\n{n}. «{q}»  GT={gt}")
            for pid in gt:
                p = by_id.get(pid)
                if not p:
                    print(f"   !! нет поста {pid}")
                    continue
                # the card's author comes from normalization; p["author"]
                # is the forwarding channel's name, never shown by the bot
                who = items.get(str(pid), {}).get("author") or "аноним"
                print(f"   [{pid}] {who} · {p['date']}")
                print(f"   {p['text'][:380]}\n")
        return

    # --auto mode: candidates + the top-3 of the current search as a draft GT
    from router import route, search_cards
    print(f"# Кандидаты: {len(CANDIDATE_QUERIES)} запросов, "
          f"GT = топ текущего поиска (требует ревью)\n")
    for n, q in enumerate(CANDIDATE_QUERIES, 1):
        r = route(q)
        if r["intent"] != "search":
            print(f"{n}. «{q}»: роутер не распознал как поиск — исключить?\n")
            continue
        found = search_cards(r, 3)
        ids = [p["id"] for p, _, _ in found]
        print(f"{'='*70}\n{n}. «{q}»  GT-черновик={ids}")
        for p, score, _ in found:
            who = items.get(str(p["id"]), {}).get("author") or "аноним"
            print(f"   [{p['id']}] {who} · {score:.3f}")
            print(f"   {p['text'][:300]}\n")


if __name__ == "__main__":
    main()
