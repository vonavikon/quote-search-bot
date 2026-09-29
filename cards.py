"""Карточка: текст + автор + источник.

Автор и источник - только из нормализации (правило доверия); не подписан -
строка опускается, ничего не выдумывается. Пороги длинных цитат: лимит
сообщения Telegram 4096, запас на подпись - обрезаем на 3900 + кнопка
«Читать полностью»; картина добавила порог подписи 1024.

Immutable span: карточка хранит исходный текст поста дословно; при показе
поверх работают только проекции - срез хвостовой подписи автора и
детерминированная типографика text_clean.
"""
import json
import os
import re

import text_clean
from load_corpus import DATA_DIR, load_splits

NORM_PATH = os.path.join(DATA_DIR, "normalization.json")
CUT = 3900


def _norm(s):
    return s.lower().replace("ё", "е")


def load_norm():
    """{post_id: item} или {} (нормализации ещё нет).

    Дети склеек (splits.json) входят с курируемыми метаданными манифеста —
    они перекрывают кэш нормализации, если LLM его уже писал для этих id.
    """
    items = {}
    if os.path.exists(NORM_PATH):
        with open(NORM_PATH, encoding="utf-8") as f:
            items = json.load(f)["items"]
    for pid, blocks in load_splits().items():
        for i, b in enumerate(blocks, 1):
            items[f"{int(pid) * 100 + i}"] = {
                "kind": b.get("kind") or "quote",
                "author": b.get("author"),
                "source": b.get("source"),
                "author_evidence": "split"}
    return items


def card(post, item=None):
    """Пост + запись нормализации -> карточка для колоды."""
    item = item or {}
    return {
        "id": post["id"],
        "text": post["text"],
        "author": item.get("author") or None,
        "source": item.get("source") or None,
        "kind": item.get("kind") or "quote",
    }


SIG_LINE_MAX = 80  # подпись — короткая строка; длинное совпадение — это контент


def _sig_word(s):
    """Самое длинное значимое слово подписи (у автора — обычно фамилия).
    Короткие слова («Петр») ловят ложные совпадения в теле цитат."""
    ws = [w for w in re.split(r"[^\w]+", _norm(s)) if len(w) >= 4]
    return max(ws, key=len) if ws else None


def strip_author_tail(c):
    """Текст поста без хвостовых подписей автора и источника. Канал подписывает
    посты последними строками («— Достоевский», «Книга»); карточка выводит
    автора и источник отдельными строками, значит в теле (и внутри «кавычек»
    цитаты) подписи быть не должно. Режем снизу до 3 коротких строк (<= 80
    знаков), совпадающих с фамилией автора или названием источника
    (е/ё-нечувствительно). Если после среза пусто — возвращаем исходник."""
    text = c["text"]
    keys = {w for w in (_sig_word(c.get("author") or ""),
                        _sig_word(c.get("source") or "")) if w}
    if not keys:
        return text
    lines = text.strip().splitlines()
    removed = 0
    while lines and removed < 3:
        line = lines[-1]
        if not line.strip():  # пустая строка после срезанной подписи — тоже долой
            lines.pop()
            continue
        if len(line) <= SIG_LINE_MAX and any(w in _norm(line) for w in keys):
            lines.pop()
            removed += 1
        else:
            break
    stripped = "\n".join(lines)
    return stripped or text


def card_text(c, full=False):
    """Текст карточки. full=True - без обрезки (кнопка «Читать полностью»).
    Цитаты (kind=quote) показываются в «кавычках»; стихи - как есть."""
    body = text_clean.apply(strip_author_tail(c))
    if not full and len(body) > CUT:
        body = body[:CUT].rstrip() + "…"
    if c.get("kind") == "quote":
        # канал часто ставит кавычки сам: своя пара + обёртка карточки
        # давали ««двойные»» кавычки. Снимаем крайнюю пару целиком —
        # открывающую « и её парную » по глубине вложенности: обёртка
        # добавит ровно такую же пару заново, парность не меняется
        core = body.strip()
        if core.startswith("«"):
            depth = 0
            for i, ch in enumerate(core):
                if ch == "«":
                    depth += 1
                elif ch in ("»", "“"):  # „…“ тоже встречается в исходниках
                    depth -= 1
                    if depth == 0:
                        core = core[1:i] + core[i + 1:]
                        break
        body = f"«{core}»"
    lines = [body]
    if c["author"]:
        lines += ["", f"— {c['author']}"]
    elif c.get("kind") in ("quote", "poem"):
        lines += ["", "— Какой-то автор, в каком-то году"]
    if c["source"]:
        lines += [f"({c['source']})"]
    return "\n".join(lines)


def truncated(c):
    return len(strip_author_tail(c)) > CUT
