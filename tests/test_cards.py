"""Cards: assembly from a post + normalization, the signature, cutting."""
import cards


def _post(text, pid=1):
    return {"id": pid, "date": "2024-01-01", "author": "", "text": text}


def test_card_defaults():
    c = cards.card(_post("текст"))
    assert c == {"id": 1, "text": "текст", "author": None, "source": None,
                 "kind": "quote"}


def test_card_from_norm_item():
    c = cards.card(_post("текст", 7), {"author": "Автор", "source": "Книга",
                                       "kind": "poem"})
    assert c["author"] == "Автор" and c["source"] == "Книга" and c["kind"] == "poem"


def test_card_text_signature():
    c = {"id": 1, "text": "цитата", "author": "Автор", "source": "Книга",
         "kind": "quote"}
    assert cards.card_text(c) == "«цитата»\n\n— Автор\n(Книга)"
    c["source"] = None
    assert cards.card_text(c) == "«цитата»\n\n— Автор"


def test_poem_not_quoted():
    """Poems are not wrapped in quotes — only quotes are (kind=quote)."""
    c = {"id": 1, "text": "строчка стиха", "author": None, "source": None,
         "kind": "poem"}
    assert cards.card_text(c) == "строчка стиха\n\n— Какой-то автор, в каком-то году"


def test_own_quotes_not_doubled():
    """The channel adds quotes itself: its own pair + the wrapper = ««double»» — we strip the outer one."""
    c = {"id": 1, "text": "«Уже в кавычках»", "author": None, "source": None,
         "kind": "quote"}
    assert cards.card_text(c) == "«Уже в кавычках»\n\n— Какой-то автор, в каком-то году"


def test_own_quotes_with_period():
    """Its own pair with a period after: strip the pair, keep the period,
    one wrapper."""
    c = {"id": 1, "text": "«Цитата».", "author": None, "source": None,
         "kind": "quote"}
    assert cards.card_text(c).startswith("«Цитата.»")


def test_nested_quotes_balance():
    """Its own outer pair with a nested quote inside: the quote balance is
    preserved."""
    c = {"id": 1, "text": "«Он сказал «да» и ушёл»", "author": None,
         "source": None, "kind": "quote"}
    t = cards.card_text(c)
    assert t.count("«") == t.count("»") == 2


def test_inner_quotes_kept():
    """Inner quotes of the quote stay, the wrapper is added once."""
    c = {"id": 1, "text": "он сказал «да» и всё", "author": None, "source": None,
         "kind": "quote"}
    assert cards.card_text(c).startswith("«он сказал «да» и всё»")


def test_unknown_author_fallback():
    """A public card without an author is signed with an honest stub."""
    c = {"id": 1, "text": "цитата", "author": None, "source": None,
         "kind": "quote"}
    assert cards.card_text(c) == "«цитата»\n\n— Какой-то автор, в каком-то году"


def test_card_text_truncation():
    c = {"id": 1, "text": "а" * 4000, "author": "Автор", "source": None,
         "kind": "quote"}
    t = cards.card_text(c)
    assert t.startswith("«") and "…" in t  # the cut happens inside the quotes
    assert cards.truncated(c)
    assert len(cards.card_text(c, full=True)) == 4000 + 2 + len("\n\n— Автор")


def test_card_text_short_not_truncated():
    c = {"id": 1, "text": "короткий", "author": None, "source": None,
         "kind": "quote"}
    assert not cards.truncated(c)
    assert cards.card_text(c) == "«короткий»\n\n— Какой-то автор, в каком-то году"


def test_load_norm_empty(tmp_data):
    assert cards.load_norm() == {}  # no file — empty, no crash


def test_immutable_span():
    """The card stores the source verbatim: display only changes the
    typography."""
    raw = 'он сказал - "да"\nвторая строка'
    c = cards.card(_post(raw))
    assert c["text"] == raw  # the source is not rewritten
    assert cards.card_text(c, full=True).startswith("«он сказал — «да»")


# ---------- stripping the trailing author signature ----------

def test_strip_author_tail_removes_signature():
    """The author signature at the post tail is cut: the author is not
    duplicated and never lands inside the quote's «quotes»."""
    c = {"id": 1, "text": "мудрая мысль\n\n— Достоевский", "author": "Достоевский",
         "source": None, "kind": "quote"}
    assert cards.card_text(c, full=True) == "«мудрая мысль»\n\n— Достоевский"


def test_strip_author_tail_two_lines():
    """A two-line signature (name + dash with surname) is cut whole."""
    c = {"id": 1, "text": "мысль\nФёдор Достоевский\n— Достоевский",
         "author": "Фёдор Достоевский", "source": None, "kind": "quote"}
    assert cards.card_text(c, full=True) == "«мысль»\n\n— Фёдор Достоевский"


def test_strip_author_tail_no_signature_keeps_text():
    """A post without an in-text signature is unchanged (the last line
    doesn't match)."""
    c = {"id": 1, "text": "мысль про Достоевского в тексте",
         "author": "Достоевский", "source": None, "kind": "quote"}
    assert cards.card_text(c, full=True) == \
        "«мысль про Достоевского в тексте»\n\n— Достоевский"


def test_strip_author_tail_no_author():
    """No author — nothing to cut: the stub, the text whole."""
    c = {"id": 1, "text": "цитата\n— Кто-то", "author": None,
         "source": None, "kind": "quote"}
    assert cards.card_text(c, full=True) == \
        "«цитата\n— Кто-то»\n\n— Какой-то автор, в каком-то году"


def test_strip_author_tail_guard_all_matched():
    """If the cut would eat the whole text — return the original (the
    emptiness guard)."""
    c = {"id": 1, "text": "Достоевский", "author": "Достоевский",
         "source": None, "kind": "quote"}
    assert cards.strip_author_tail(c) == "Достоевский"


def test_strip_author_tail_stops_after_limit():
    """No more than 3 lines per cut — a fourth matching line stays."""
    c = {"id": 1, "text": "мысль\n" + "Достоевский\n" * 4,
         "author": "Достоевский", "source": None, "kind": "quote"}
    assert cards.strip_author_tail(c).count("Достоевский") == 1


def test_strip_author_tail_with_source_line():
    """A tail of "author + book": both lines are cut (the source is
    rendered separately)."""
    c = {"id": 1, "text": "мысль\nАртур Шопенгауэр\n«Афоризмы житейской мудрости»",
         "author": "Артур Шопенгауэр", "source": "Афоризмы житейской мудрости",
         "kind": "quote"}
    assert cards.strip_author_tail(c) == "мысль"


def test_strip_author_tail_short_word_in_body():
    """A short word of the author's name inside the quote («как Петр
    Великий научил...») does not trigger the cut: only the surname — the
    longest word — matches."""
    c = {"id": 1, "text": "Я люблю моё Отечество, как Петр Великий научил "
                          "меня любить его.\n\nПётр Чаадаев",
         "author": "Пётр Чаадаев", "source": None, "kind": "quote"}
    assert cards.strip_author_tail(c).endswith("любить его.")


def test_strip_author_tail_long_line_guard():
    """A match in a long line (200+ chars) — that is content, not a
    signature."""
    long_quote = ("Артур Шопенгауэр: «Больше всего толпа ненавидит того, кто "
                  "думает иначе; дело не столько в самом мнении, сколько "
                  "в дерзости желания думать самостоятельно».")
    c = {"id": 1, "text": long_quote, "author": "Артур Шопенгауэр",
         "source": None, "kind": "quote"}
    assert cards.strip_author_tail(c) == long_quote
