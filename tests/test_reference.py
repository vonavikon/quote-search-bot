"""Справки по Википедии: кэш, дизамбиг-фильтр, заглушки, стрим — без сети."""
import json

import reference


def test_no_author_stub():
    """Без автора — статичная заглушка, LLM не участвует."""
    assert "не нашли" in reference.about_author(None)
    assert "не нашли" in reference.about_author("")


def test_no_wiki_stub(monkeypatch):
    monkeypatch.setattr(reference, "resolve", lambda t: None)
    out = reference.about_author("Малоизвестный Автор")
    assert "не нашли" in out and "открытых источниках" in out


def test_author_cached_no_llm(monkeypatch):
    """Готовая справка в кэше — LLM не зовётся, сеть не трогаем."""
    def boom(*a, **k):
        raise AssertionError("compose не должен зваться")
    monkeypatch.setattr(reference, "_compose", boom)
    monkeypatch.setattr(reference, "_cache_get",
                        lambda key, max_age=0: "Готовая справка." if key.startswith("ref:") else None)
    assert reference.about_author("Монтень") == "Готовая справка."


def test_disambig_extract_rejected(monkeypatch):
    """Страница-дизамбиг («может означать») — не статья, справке не годится."""
    def fake_http(params):
        if params.get("action") == "opensearch":
            return ["", ["Никто"]]
        page = {"extract": "Никто — многозначное слово.\n\nНикто (альбом) и прочее"}
        return {"query": {"pages": {"1": page}}}
    monkeypatch.setattr(reference, "_http", fake_http)
    monkeypatch.setattr(reference, "_cache_get", lambda key, max_age=0: None)
    monkeypatch.setattr(reference, "_cache_put", lambda key, v: None)
    assert reference.extract("Никто") is None
    assert "не нашли" in reference.about_author("Никто")


def test_disambig_structural_rejected(monkeypatch):
    """Дизамбиг без маркера «многозначн» (живой кейс с прода: «Нарцисс»)
    ловится структурно — >= 3 строк «Нарцисс — значение» с ударением
    в титуле первой строки. Иначе текст страницы значений уходит в LLM."""
    page_text = ("Нарци́сс (лат. Narcissus от др.-греч. Νάρκισσος):\n\n"
                 "Нарцисс, точнее Нарки́сс — персонаж древнегреческой мифологии.\n"
                 "Нарцисс — род однодольных растений.\n"
                 "Нарцисс — человек, страдающий нарциссизмом.\n"
                 "«Нарцисс» — бригантина Черноморского флота.")
    monkeypatch.setattr(reference, "_http",
                        lambda p: {"query": {"pages":
                                    {"1": {"extract": page_text}}}})
    monkeypatch.setattr(reference, "_cache_get", lambda key, max_age=0: None)
    monkeypatch.setattr(reference, "_cache_put", lambda key, v: None)
    assert reference.extract("Нарцисс") is None


def test_meta_refusal_never_cached(monkeypatch):
    """LLM выдала мету про вход («в предоставленном тексте нет статьи...») —
    пользователь видит статичную заглушку, мусор не кэшируется на год.
    Живой кейс: дизамбиг проскочил старый фильтр, справка закэшировалась
    с разбором страницы значений и жила до ручной чистки."""
    def fake_compose(subj, texts, on_chunk=None):
        return ("В предоставленном тексте нет статьи о картине «Нарцисс» "
                "Караваджо. Текст представляет собой страницу значений "
                "слова «Нарцисс»: мифологический персонаж, растение, термин.")

    cached = []
    monkeypatch.setattr(reference, "resolve", lambda t: "Нарцисс")
    monkeypatch.setattr(reference, "extract",
                        lambda t: "Нарцисс — миф про отражение." * 3)
    monkeypatch.setattr(reference, "_compose", fake_compose)
    monkeypatch.setattr(reference, "_cache_get", lambda key, max_age=0: None)
    monkeypatch.setattr(reference, "_cache_put",
                        lambda key, v: cached.append(key))
    p = {"id": "x", "author": "Караваджо", "title": "Нарцисс", "year": 1599}
    out = reference.about_painting(p)
    assert "не нашли" in out and "открытых источниках" in out
    assert "предоставленном" not in out
    assert cached == []  # мета в кэш не попала


def test_cached_meta_selfheals(monkeypatch):
    """Мета-справка, закэшированная на год старой версией, при чтении
    мимоходом вычищается: кэш игнорируется, справка пересобирается."""
    bad = ("В предоставленном тексте нет статьи о картине. "
           "Текст представляет собой страницу значений.")
    monkeypatch.setattr(reference, "_cache_get",
                        lambda key, max_age=0: bad if key.startswith("ref:paint")
                        else None)
    monkeypatch.setattr(reference, "resolve", lambda t: "Нарцисс (Караваджо)")
    monkeypatch.setattr(reference, "extract",
                        lambda t: "Картина Караваджо 1599 года.")
    monkeypatch.setattr(reference, "_compose",
                        lambda subj, texts, on_chunk=None: "СПРАВКА")
    monkeypatch.setattr(reference, "_cache_put", lambda key, v: None)
    p = {"id": "x", "author": "Караваджо", "title": "Нарцисс", "year": 1599}
    out = reference.about_painting(p)
    assert "СПРАВКА" in out and "предоставленном" not in out


def test_cached_disambig_extract_selfheals(monkeypatch):
    """Extract с дизамбигом, закэшированный до структурного фильтра,
    при чтении перехватывается и перезаписывается пустым."""
    page_text = ("Нарцисс — персонаж мифологии.\n"
                 "Нарцисс — род растений.\n"
                 "Нарцисс — термин психологии.")
    monkeypatch.setattr(reference, "_cache_get", lambda key, max_age=0: page_text)
    saved = []
    monkeypatch.setattr(reference, "_cache_put",
                        lambda key, v: saved.append((key, v)))
    assert reference.extract("Нарцисс") is None
    assert ("extract:Нарцисс", "") in saved  # кэш вылечен


def test_painting_no_wiki_stub(monkeypatch):
    monkeypatch.setattr(reference, "resolve", lambda t: None)
    monkeypatch.setattr(reference, "_search_article", lambda what, who: None)
    monkeypatch.setattr(reference, "_cache_get", lambda key, max_age=0: None)
    p = {"id": "x", "author": "Художник", "title": "Картина", "year": 1850}
    out = reference.about_painting(p)
    assert "не нашли" in out and "открытых источниках" in out


def test_painting_search_fallback(monkeypatch):
    """opensearch промахнулся, короткое название — дизамбиг: запасной
    list=search находит статью «Нарцисс (Караваджо)» (живой кейс с прода).
    Статья обязана звать художника в тексте — миф о Нарцисс не пройдёт."""
    def fake_resolve(t):
        return "Нарцисс" if t == "Нарцисс" else None

    def fake_extract(t):
        if t == "Нарцисс (Караваджо)":
            return "Картина Караваджо 1599 года, хранится в Риме."
        return None  # дизамбиг и нерелевантные — пусто

    def fake_http(params):
        assert params["action"] == "query" and params["list"] == "search"
        return {"query": {"search": [{"title": "Нарцисс (мифология)"},
                                     {"title": "Нарцисс (Караваджо)"}]}}

    monkeypatch.setattr(reference, "resolve", fake_resolve)
    monkeypatch.setattr(reference, "extract", fake_extract)
    monkeypatch.setattr(reference, "_http", fake_http)
    monkeypatch.setattr(reference, "_compose",
                        lambda subj, texts, on_chunk=None: "СПРАВКА")
    monkeypatch.setattr(reference, "_cache_get", lambda key, max_age=0: None)
    monkeypatch.setattr(reference, "_cache_put", lambda key, v: None)
    p = {"id": "x", "author": "Караваджо", "title": "Нарцисс", "year": 1599}
    out = reference.about_painting(p)
    assert "СПРАВКА" in out and "открытых источников" in out


def test_stream_pieces_and_usage(monkeypatch):
    """SSE-стрим: копятся только content-куски (reasoning мимо),
    usage приходит отдельным чанком, [DONE] завершает."""
    import urllib.request
    lines = [
        'data: {"choices":[{"delta":{"reasoning_content":"думаю"}}]}'.encode(),
        'data: {"choices":[{"delta":{"content":"Монтень"}}]}'.encode(),
        'data: {"choices":[{"delta":{"content":" — эссеист."}}]}'.encode(),
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
        b'data: [DONE]',
        ': keep-alive ignore'.encode(),
    ]

    class FakeResp:
        def __init__(self):
            self._it = iter(lines)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._it)

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=0: FakeResp())
    monkeypatch.setattr(reference, "_key", lambda: "test-key")  # CI без .env
    chunks = []
    text, usage = reference._stream(b"{}", chunks.append)
    assert text == "Монтень — эссеист."
    assert chunks == ["Монтень", " — эссеист."]
    assert usage["completion_tokens"] == 5


def test_compose_falls_back_when_stream_dies(monkeypatch):
    """Стрим упал посередине — один неразрывный повтор, текст собирается."""
    import urllib.request

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Готово."}}],
                               "usage": {}}).encode()

    def fake_urlopen(req, timeout=0):
        if b'"stream": true' in req.data:
            raise OSError("сеть моргнула на стриме")
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(reference, "_key", lambda: "test-key")  # CI без .env
    monkeypatch.setattr("spend.log_call", lambda *a, **k: None)
    assert reference._compose("тест", ["текст"]) == "Готово."


def test_about_author_source_tail(monkeypatch):
    """source из подписи: статья произведения в сборке + строка о подписи —
    даже когда вики-статьи о произведении нет."""
    seen = {}

    def fake_compose(subj, texts, on_chunk=None):
        seen["subj"], seen["n"] = subj, len(texts)
        return "СПРАВКА"

    monkeypatch.setattr(reference, "resolve", lambda t: "Автор, Имя")
    monkeypatch.setattr(reference, "extract", lambda t: "текст автора" if "Автор" in t
                         else None)
    monkeypatch.setattr(reference, "_compose", fake_compose)
    monkeypatch.setattr(reference, "_cache_get", lambda key, max_age=0: None)
    monkeypatch.setattr(reference, "_cache_put", lambda key, v: None)
    out = reference.about_author("Автор", "Произведение")
    assert seen["n"] == 1 and "и о произведении" not in seen["subj"]
    assert "СПРАВКА" in out and "подписи канала" in out
    assert "Произведение" in out and "открытых источников" in out
