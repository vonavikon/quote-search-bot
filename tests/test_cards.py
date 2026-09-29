"""Карточки: сборка из поста + нормализации, подпись, обрезка."""
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
    """Стихи в кавычки не берутся — только цитаты (kind=quote)."""
    c = {"id": 1, "text": "строчка стиха", "author": None, "source": None,
         "kind": "poem"}
    assert cards.card_text(c) == "строчка стиха\n\n— Какой-то автор, в каком-то году"


def test_own_quotes_not_doubled():
    """Канал сам ставит кавычки: своя пара + обёртка = ««двойные»» — снимаем внешнюю."""
    c = {"id": 1, "text": "«Уже в кавычках»", "author": None, "source": None,
         "kind": "quote"}
    assert cards.card_text(c) == "«Уже в кавычках»\n\n— Какой-то автор, в каком-то году"


def test_own_quotes_with_period():
    """Своя пара с точкой после: снимаем пару, точка остаётся, обёртка одна."""
    c = {"id": 1, "text": "«Цитата».", "author": None, "source": None,
         "kind": "quote"}
    assert cards.card_text(c).startswith("«Цитата.»")


def test_nested_quotes_balance():
    """Своя внешняя пара с вложенной цитатой: баланс кавычек сохраняется."""
    c = {"id": 1, "text": "«Он сказал «да» и ушёл»", "author": None,
         "source": None, "kind": "quote"}
    t = cards.card_text(c)
    assert t.count("«") == t.count("»") == 2


def test_inner_quotes_kept():
    """Внутренние кавычки цитаты остаются, обёртка добавляется одна."""
    c = {"id": 1, "text": "он сказал «да» и всё", "author": None, "source": None,
         "kind": "quote"}
    assert cards.card_text(c).startswith("«он сказал «да» и всё»")


def test_unknown_author_fallback():
    """Публичная карточка без автора подписывается честную заглушку."""
    c = {"id": 1, "text": "цитата", "author": None, "source": None,
         "kind": "quote"}
    assert cards.card_text(c) == "«цитата»\n\n— Какой-то автор, в каком-то году"


def test_card_text_truncation():
    c = {"id": 1, "text": "а" * 4000, "author": "Автор", "source": None,
         "kind": "quote"}
    t = cards.card_text(c)
    assert t.startswith("«") and "…" in t  # обрезка внутри кавычек
    assert cards.truncated(c)
    assert len(cards.card_text(c, full=True)) == 4000 + 2 + len("\n\n— Автор")


def test_card_text_short_not_truncated():
    c = {"id": 1, "text": "короткий", "author": None, "source": None,
         "kind": "quote"}
    assert not cards.truncated(c)
    assert cards.card_text(c) == "«короткий»\n\n— Какой-то автор, в каком-то году"


def test_load_norm_empty(tmp_data):
    assert cards.load_norm() == {}  # файла нет — пусто, не падает


def test_immutable_span():
    """Карточка хранит исходник дословно: показ меняет только типографику."""
    raw = 'он сказал - "да"\nвторая строка'
    c = cards.card(_post(raw))
    assert c["text"] == raw  # исходник не переписан
    assert cards.card_text(c, full=True).startswith("«он сказал — «да»")


# ---------- срез хвостовой подписи автора ----------

def test_strip_author_tail_removes_signature():
    """Подпись автора в хвосте поста срезается: автор не дублируется и
    не попадает внутрь «кавычек» цитаты."""
    c = {"id": 1, "text": "мудрая мысль\n\n— Достоевский", "author": "Достоевский",
         "source": None, "kind": "quote"}
    assert cards.card_text(c, full=True) == "«мудрая мысль»\n\n— Достоевский"


def test_strip_author_tail_two_lines():
    """Подпись из двух строк (имя + тире с фамилией) срезается целиком."""
    c = {"id": 1, "text": "мысль\nФёдор Достоевский\n— Достоевский",
         "author": "Фёдор Достоевский", "source": None, "kind": "quote"}
    assert cards.card_text(c, full=True) == "«мысль»\n\n— Фёдор Достоевский"


def test_strip_author_tail_no_signature_keeps_text():
    """Пост без подписи в тексте не меняется (последняя строка не матчится)."""
    c = {"id": 1, "text": "мысль про Достоевского в тексте",
         "author": "Достоевский", "source": None, "kind": "quote"}
    assert cards.card_text(c, full=True) == \
        "«мысль про Достоевского в тексте»\n\n— Достоевский"


def test_strip_author_tail_no_author():
    """Без автора резать нечего — заглушка, текст целиком."""
    c = {"id": 1, "text": "цитата\n— Кто-то", "author": None,
         "source": None, "kind": "quote"}
    assert cards.card_text(c, full=True) == \
        "«цитата\n— Кто-то»\n\n— Какой-то автор, в каком-то году"


def test_strip_author_tail_guard_all_matched():
    """Если срез съел бы весь текст — возвращаем исходник (guard на пустоту)."""
    c = {"id": 1, "text": "Достоевский", "author": "Достоевский",
         "source": None, "kind": "quote"}
    assert cards.strip_author_tail(c) == "Достоевский"


def test_strip_author_tail_stops_after_limit():
    """Не больше 3 строк за один срез — четвёртая строка-совпадение остаётся."""
    c = {"id": 1, "text": "мысль\n" + "Достоевский\n" * 4,
         "author": "Достоевский", "source": None, "kind": "quote"}
    assert cards.strip_author_tail(c).count("Достоевский") == 1


def test_strip_author_tail_with_source_line():
    """Хвост «автор + книга»: срезаются обе строки (источник выводится отдельно)."""
    c = {"id": 1, "text": "мысль\nАртур Шопенгауэр\n«Афоризмы житейской мудрости»",
         "author": "Артур Шопенгауэр", "source": "Афоризмы житейской мудрости",
         "kind": "quote"}
    assert cards.strip_author_tail(c) == "мысль"


def test_strip_author_tail_short_word_in_body():
    """Короткое слово автора внутри цитаты («как Петр Великий научил...») не
    провоцирует срез: матчится только фамилия — самое длинное слово."""
    c = {"id": 1, "text": "Я люблю моё Отечество, как Петр Великий научил "
                          "меня любить его.\n\nПётр Чаадаев",
         "author": "Пётр Чаадаев", "source": None, "kind": "quote"}
    assert cards.strip_author_tail(c).endswith("любить его.")


def test_strip_author_tail_long_line_guard():
    """Совпадение в длинной строке (200+ знаков) — это контент, не подпись."""
    long_quote = ("Артур Шопенгауэр: «Больше всего толпа ненавидит того, кто "
                  "думает иначе; дело не столько в самом мнении, сколько "
                  "в дерзости желания думать самостоятельно».")
    c = {"id": 1, "text": long_quote, "author": "Артур Шопенгауэр",
         "source": None, "kind": "quote"}
    assert cards.strip_author_tail(c) == long_quote
