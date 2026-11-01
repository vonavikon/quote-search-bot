"""Bot keyboards and decks: pure functions without Telegram."""
import bot


def _deck(n, query="q"):
    cards_ = [{"id": i, "text": f"текст {i}", "author": None, "source": None,
               "kind": "quote"} for i in range(n)]
    return bot.new_deck(cards_, query)


def _rows(markup):
    return [len(r) for r in markup.inline_keyboard]


def test_kb_single_card():
    key = _deck(1)
    assert _rows(bot.kb(key, 0)) == [2, 1]  # "Об авторе"+"О картине" / "Sic"
    labels = [b.text for r in bot.kb(key, 0).inline_keyboard for b in r]
    assert "Об авторе" in labels and "О картине" in labels


def test_kb_similar_button():
    """A random card gets a "Похожая цитата" button."""
    key = _deck(1)
    assert _rows(bot.kb(key, 0, similar=True)) == [1, 2, 1]
    labels = [b.text for r in bot.kb(key, 0,
                                     similar=True).inline_keyboard for b in r]
    assert "Похожая цитата" in labels


def test_kb_more_button_rows():
    """Instead of a ◀▶ carousel — a single "Ещё цитата" button."""
    key = _deck(3)
    assert _rows(bot.kb(key, 1)) == [1, 2, 1]  # "Ещё" / reference / "Sic"


def test_kb_truncated_adds_full_button():
    key = _deck(2)
    assert _rows(bot.kb(key, 0, truncated=True)) == [1, 1, 2, 1]


def test_diversify_round_robin_authors():
    """Cards by the same author never come in a row."""
    a = {"id": 1, "author": "А", "source": None, "kind": "quote", "text": "1"}
    b = {"id": 2, "author": "Б", "source": None, "kind": "quote", "text": "2"}
    cs = [dict(a, id=i) for i in (1, 2, 3)] + [dict(b, id=4)]
    out = bot.diversify(cs)
    assert [c["author"] for c in out][:2] == ["А", "Б"]  # not "А, А"
    assert sorted(c["id"] for c in out) == [1, 2, 3, 4]  # nobody lost


def test_diversify_single_author_untouched():
    a = {"id": 1, "author": None, "source": None, "kind": "quote", "text": "1"}
    out = bot.diversify([dict(a, id=i) for i in (1, 2)])
    assert [c["id"] for c in out] == [1, 2]


def test_new_deck_cap_fifo(monkeypatch):
    monkeypatch.setattr(bot, "DECK_CAP", 2)
    bot.DECKS.clear()
    k1 = _deck(1)
    k2 = _deck(1)
    k3 = _deck(1)
    assert k1 not in bot.DECKS and k2 in bot.DECKS and k3 in bot.DECKS
    bot.DECKS.clear()


def test_caption_threshold():
    """The photo caption threshold: Telegram's 1024 limit minus headroom."""
    assert bot.CAPTION_MAX < 1024


def test_caption_painting_prefix():
    """The painting caption is marked with the word «Картина:» — so the
    reader tells it apart from the quote's author signature."""
    c = {"id": 1, "text": "цитата", "author": "Автор", "source": None,
         "kind": "quote"}
    painting = {"id": "p", "author": "Моне", "title": "Впечатление", "year": 1872}
    cap = bot._caption(c, painting)
    assert "Картина: Моне, «Впечатление», 1872" in cap
    assert cap.index("— Автор") < cap.index("Картина:")  # the author above the painting line


def test_caption_painting_no_year():
    painting = {"id": "p", "author": "Моне", "title": "Впечатление", "year": None}
    c = {"id": 1, "text": "цитата", "author": None, "source": None, "kind": "quote"}
    assert "Картина: Моне, «Впечатление»" in bot._caption(c, painting)
    assert "None" not in bot._caption(c, painting)


def test_search_status_by_path():
    """Quotes in the query — the "дословно" status, everything else —
    "по смыслу"."""
    assert "дословно" in bot.search_status("direct-quoted", "жалость к себе")
    assert "по смыслу" in bot.search_status("direct", "одиночество")
    assert "«жалость»" in bot.search_status("direct-quoted", "жалость")


def test_channel_gap():
    """A gap between post ids: an empty list — continuous or the first
    post."""
    assert bot.channel_gap(0, 500) == []          # the first message after startup
    assert bot.channel_gap(100, 101) == []        # consecutive
    assert bot.channel_gap(100, 100) == []        # a post edit, not a new one
    assert bot.channel_gap(100, 103) == [101, 102]


def test_rate_ok_limit_and_admin(tmp_data, monkeypatch):
    """The limit: 100 actions pass, the 101st doesn't; the admin and an
    empty uid — always yes."""
    calls = []

    def fake_incr(uid):
        calls.append(uid)
        return len(calls)
    monkeypatch.setattr(bot.store, "rate_incr", fake_incr)
    assert bot.rate_ok(7) is True    # counter 1
    monkeypatch.setattr(bot, "DAY_LIMIT", 2)
    assert bot.rate_ok(7) is True    # 2
    assert bot.rate_ok(7) is False   # 3 > 2
    assert bot.rate_ok(None) is True  # uid unknown — not counted
    monkeypatch.setattr(bot, "ADMIN", 7)
    assert bot.rate_ok(7) is True    # the admin — no limit
    assert calls == [7, 7, 7]        # the admin and None never touched the counter


def _post(pid):
    return {"id": pid, "text": f"текст {pid}", "author": "", "date": "2024-01-01"}


def test_similar_branch_skips_seen(tmp_data, monkeypatch):
    """The similar branch skips the current post and everything shown,
    filling up to 5."""
    posts = [_post(i) for i in (1, 2, 3, 4, 5, 6, 7)]
    monkeypatch.setattr(bot, "search_cards",
                        lambda r, k: [(p, 0.5, "dense") for p in posts])
    out = bot.similar_branch("текст", 1, seen={3})
    assert [p["id"] for p in out] == [2, 4, 5, 6, 7]


def test_similar_branch_empty_when_all_seen(tmp_data, monkeypatch):
    """The nearest are exhausted — the branch is empty, the bot falls
    back to random."""
    posts = [_post(i) for i in (1, 2, 3)]
    monkeypatch.setattr(bot, "search_cards",
                        lambda r, k: [(p, 0.5, "dense") for p in posts])
    assert bot.similar_branch("текст", 1, seen={2, 3}) == []


def test_random_branch_prefers_unseen(tmp_data, norm_cache, monkeypatch):
    """The random branch seeds with an unshown post and takes no shown
    neighbors."""
    pool = [_post(i) for i in (1, 2, 3)]
    monkeypatch.setattr(bot, "load_posts", lambda: pool)
    monkeypatch.setattr(bot.store, "shown_posts", lambda uid: {1, 2})
    monkeypatch.setattr(bot, "search_cards",
                        lambda r, k: [(pool[0], 0.9, "dense")])
    monkeypatch.setattr(bot.random, "choice", lambda seq: seq[0])
    out = bot.random_branch(7)
    # seed = the only unshown post (3); neighbor 1 is shown — not included
    assert [p["id"] for p in out] == [3]


def test_random_branch_all_seen_falls_back_to_pool(tmp_data, norm_cache,
                                                   monkeypatch):
    """Everything read — seed from the whole pool, not an empty branch."""
    pool = [_post(5)]
    monkeypatch.setattr(bot, "load_posts", lambda: pool)
    monkeypatch.setattr(bot.store, "shown_posts", lambda uid: {5})
    monkeypatch.setattr(bot, "search_cards", lambda r, k: [])
    assert [p["id"] for p in bot.random_branch(7)] == [5]
