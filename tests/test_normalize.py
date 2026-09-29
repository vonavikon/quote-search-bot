"""Нормализация: clip, детерминированное правило доверия."""
import normalize


def _post(pid, text):
    return {"id": pid, "date": "2024-01-01", "author": "", "text": text}


def test_clip_short_passthrough():
    assert normalize.clip("короткий") == "короткий"


def test_clip_long_head_tail():
    text = "а" * 1500
    clipped = normalize.clip(text)
    assert len(clipped) < len(text)
    assert "[...сокращено...]" in clipped
    assert clipped.startswith("а" * normalize.HEAD)
    assert clipped.endswith("а" * normalize.TAIL)


def test_enforce_trust_signed_keeps():
    items = {"1": {"author": "Бальтасар Грасиан", "author_evidence": "signed"}}
    posts = [_post(1, "цитата\n— Бальтасар Грасиан")]
    assert normalize.enforce_trust(items, posts) == 0
    assert items["1"]["author"] == "Бальтасар Грасиан"


def test_enforce_trust_demotes_unsupported():
    """Автор не в последних 2 строках — понижение в guess (мисатрибуция)."""
    items = {"2": {"author": "Флоренский", "author_evidence": "signed",
                   "author_guess": None}}
    posts = [_post(2, "цитата без подписи автора")]
    assert normalize.enforce_trust(items, posts) == 1
    assert items["2"]["author"] is None
    assert items["2"]["author_evidence"] == "none"
    assert items["2"]["author_guess"] == "Флоренский"


def test_enforce_trust_yo_e_insensitive():
    items = {"3": {"author": "Пётр Терёшин", "author_evidence": "signed"}}
    posts = [_post(3, "текст\n- Петр Терешин")]  # е вместо ё в подписи
    assert normalize.enforce_trust(items, posts) == 0


def test_enforce_trust_short_words_guard():
    """Все слова автора короче 4 букв («Ян Ли») — значимых слов нет, проверка
    бессильна: автор остаётся (проверка не опровергла подпись, а не доказала)."""
    items = {"4": {"author": "Ян Ли", "author_evidence": "signed"}}
    posts = [_post(4, "путь в пустоте")]
    assert normalize.enforce_trust(items, posts) == 0
    assert items["4"]["author"] == "Ян Ли"


def test_enforce_trust_missing_post_ignored():
    items = {"99": {"author": "Кто-то", "author_evidence": "signed"}}
    assert normalize.enforce_trust(items, [_post(1, "текст")]) == 0
