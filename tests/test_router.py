"""Router: intent rules, the author facet, stopwords, paths without the
LLM."""
import router


def test_intro_dictionary():
    for t in ("привет", "Привет!", "спасибо", "что ты умеешь", "помощь"):
        assert router.route(t)["intent"] == "intro", t


def test_admin_commands():
    r = router.route("/stats")
    assert r["intent"] == "admin"
    assert router.route("/resync остатки")["intent"] == "admin"


def test_strip_stopwords():
    assert router.strip_stopwords("подскажи цитату про одиночество") == "одиночество"
    assert router.strip_stopwords("хочу найти стихи о море") == "найти море"


def test_match_author_declension():
    idx = {"пастернак": "Борис Пастернак", "грасиан": "Бальтасар Грасиан"}
    author, rest = router.match_author("пастернака о любви", idx)
    assert author == "Борис Пастернак"
    assert "любви" in rest and "пастернак" not in rest
    # a surname shorter than the token doesn't match by accident in either direction
    author2, _ = router.match_author("море и солнце", idx)
    assert author2 is None


def test_route_direct_short(norm_cache):
    """<= 2 meaningful words: no LLM, the direct path."""
    r = router.route("про одиночество")
    assert r["intent"] == "search" and r["path"] == "direct"
    assert r["query"] == "одиночество" and r["author"] is None


def test_route_author_facet(norm_cache):
    r = router.route("пастернак о любви")
    assert r["author"] == "Борис Пастернак"
    assert r["path"] == "direct"
    assert "любви" in r["query"]


def test_route_quoted_no_llm(norm_cache, monkeypatch):
    """Quotes — the literal path, the LLM reformulation is not called."""
    def boom(text, model=None):
        raise AssertionError("reformulate не должен зваться")
    monkeypatch.setattr(router, "reformulate", boom)
    r = router.route("«жалость к себе» в большой разговорной фразе про жизнь вообще")
    assert r["path"] == "direct-quoted"


def test_route_long_text_no_llm(norm_cache, monkeypatch):
    """A wall of text without quotes (> 12 words) — the literal path, no
    LLM: a pasted quote fragment is not reformulated and no tokens are
    burned."""
    def boom(text, model=None):
        raise AssertionError("reformulate не должен зваться")
    monkeypatch.setattr(router, "reformulate", boom)
    r = router.route("я помню только кусок который звучал примерно так "
                     "человек всю жизнь бежит от самого себя и никак "
                     "не может догнать и понять кто он вообще такой")
    assert r["intent"] == "search" and r["path"] == "direct-quoted"
    assert len(r["query"]) > 0 and r["author"] is None


def test_route_llm_reformulation(norm_cache, monkeypatch):
    monkeypatch.setattr(router, "reformulate", lambda t, model=None: "дружба после ссоры")
    r = router.route("подскажи что-нибудь мудрое про дружбу после ссоры")
    assert r["path"] == "llm" and r["query"] == "дружба после ссоры"


def test_route_llm_fallback(norm_cache, monkeypatch):
    """bothub is down — the router lives on the raw text with stopwords
    cut."""
    monkeypatch.setattr(router, "reformulate", lambda t, model=None: None)
    r = router.route("подскажи что-нибудь мудрое про дружбу после ссоры")
    assert r["path"] == "direct-fallback"
    assert "дружбу" in r["query"]


def test_route_no_llm_when_limit_spent(norm_cache, monkeypatch):
    """allow_llm=False (the daily limit spent) — the reformulation is
    not called, a conversational query takes the raw path; intro is
    detected anyway, before the LLM."""
    def boom(t, model=None):
        raise AssertionError("reformulate не должен зваться без остатка лимита")
    monkeypatch.setattr(router, "reformulate", boom)
    r = router.route("подскажи что-нибудь мудрое про дружбу после ссоры",
                     allow_llm=False)
    assert r["intent"] == "search" and r["path"] == "direct-fallback"
    assert router.route("привет", allow_llm=False)["intent"] == "intro"


def test_search_cards_filters(tmp_data, norm_cache, monkeypatch):
    """The kind filter (reflection is not served) and the author surname
    facet."""
    import hybrid

    posts = [
        {"id": 101, "date": "2024-01-01", "author": "", "text": "про любовь"},
        {"id": 102, "date": "2024-01-02", "author": "", "text": "размышление"},
        {"id": 103, "date": "2024-01-03", "author": "", "text": "стих"},
        {"id": 104, "date": "2024-01-04", "author": "Чужой Автор", "text": "другое"},
    ]
    order = [0, 1, 2, 3]
    dense = [0.9, 0.8, 0.7, 0.6]
    monkeypatch.setattr(hybrid, "retrieve",
                        lambda q, n=50: (order, posts, dense, [0.0] * 4, "dense"))
    out = router.search_cards({"query": "любовь", "author": None}, k=10)
    # 102 is a reflection, filtered out; 104 has no normalization — public by default
    assert [p["id"] for p, _, _ in out] == [101, 103, 104]

    out2 = router.search_cards({"query": "любовь", "author": "Борис Пастернак"}, k=10)
    assert [p["id"] for p, _, _ in out2] == [101]  # 104: the author is not Pasternak
