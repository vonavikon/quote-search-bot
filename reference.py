"""Справки об авторах и картинах по Википедии (без слопа и ложных фактов).

Метод — три слоя честности:
  1. данные корпуса (автор, произведение) — подписи в постах канала;
  2. статьи Википедии дословно — человеческий текст, не генерация;
  3. сборка нарратива — LLM ТОЛЬКО по поданным вики-текстам: извлечение,
     не вспоминание. Промпт запрещает факты и числа вне текстов.

Нет статьи в Википедии — статичная заглушка, LLM не зовётся вовсе
(беспроигрышно: нечему врать). Кэш в DATA_DIR/wiki_cache.json: сырые
extracts и готовые справки, повторные клики бесплатны и мгновенны.

Запуск (прототип на постах):
  python -X utf8 reference.py автор Монтень
  python -X utf8 reference.py картина 42
"""
import json
import logging
import os
import time
import unicodedata

from embed import BASE, _key
from load_corpus import DATA_DIR

log = logging.getLogger("quote-search.reference")

UA = {"User-Agent": "quote-search-bot/1.0 "
    "(https://github.com/vonavikon/quote-search-bot)"}
EXTRACT_MAX = 8000   # знаков статьи в контекст LLM (история создания — в начале)
CACHE_TTL = 30 * 24 * 3600  # переопросить Википедию через месяц

REF_PROMPT = """Ты пишешь краткую справку для карточки цитаты в боте. Ниже — тексты статей Википедии{о_чём}.
Напиши 3-6 предложений: кто автор и чем известен; при каких обстоятельствах и в какой период написано произведение — если статья об этом говорит.
ЖЕЛЕЗНЫЕ ПРАВИЛА:
- каждый факт — только из текстов ниже; ничего от себя, никаких внешних знаний
- даты, числа, названия — только дословно из текста
- обстоятельств создания в текстах нет — не выдумывай их, просто скажи, кто автор
- без оценочных слов («великий», «интересно, что»), без обращений к читателю"""


# ---------- кэш ----------

def _cache_path():
    return os.path.join(DATA_DIR, "wiki_cache.json")


def _cache_load():
    try:
        with open(_cache_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 — нет файла или битый: начинаем пустой
        return {}


def _cache_put(key, value):
    c = _cache_load()
    c[key] = {"ts": int(time.time()), "v": value}
    try:
        with open(_cache_path(), "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False)
    except Exception:  # noqa: BLE001 — кэш не критичен
        pass


def _cache_get(key, max_age=CACHE_TTL):
    e = _cache_load().get(key)
    if not e:
        return None
    if time.time() - e["ts"] > max_age:
        return None
    return e["v"]


# ---------- Википедия ----------

def _http(url):
    import urllib.parse
    import urllib.request
    url = "https://ru.wikipedia.org/w/api.php?" + urllib.parse.urlencode(url)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def resolve(title):
    """Точное название статьи (opensearch): «Шваб» -> «Шваб, Клаус».
    None — статьи нет."""
    cached = _cache_get(f"resolve:{title}")
    if cached is not None:
        return cached or None
    try:
        names = _http({"action": "opensearch", "format": "json",
                       "limit": 1, "search": title})[1]
        hit = names[0] if names else ""
    except Exception:  # noqa: BLE001 — сеть моргнула: без кэша этой записи
        return None
    _cache_put(f"resolve:{title}", hit)
    return hit or None


def _plain_word(s):
    """Строка без надстрочных знаков: «Нарци́сс» -> «Нарцисс»."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def _is_disambig(text, title):
    """Страница значений? Два шаблона. Явный: «многозначн»/«может означать»
    в первых строках. Структурный (живой кейс «Нарцисс»): страница без этих
    слов, но устроена списком — >= 3 строк вида «Нарцисс — род...» /
    ««Нарцисс» — бригантина...» начинаются с титула и идут с тире.
    Обычная статья так не устроена: титул встречается в первой строке
    один раз, а не строкой на каждое значение слова."""
    head = text[:250].lower()
    if "многозначн" in head or "может означать" in head:
        return True
    stem = _plain_word(title.split("(")[0].strip())
    if len(stem) < 4:
        return False
    hits = 0
    for line in text[:2000].splitlines():
        s = _plain_word(line.strip().lstrip("«"))
        if s.startswith(stem) and " — " in line[:80]:
            hits += 1
    return hits >= 3


def extract(title):
    """Плоский текст статьи (explaintext), обрезан до EXTRACT_MAX.
    None — статьи нет, сеть или страница-дизамбиг. Отрицательный
    результат кэшируется."""
    key = f"extract:{title}"
    cached = _cache_get(key)
    if cached is not None:
        if cached and _is_disambig(cached, title):
            _cache_put(key, "")  # в кэше дизамбиг со старого фильтра: чиним
            return None
        return cached or None
    try:
        d = _http({"action": "query", "prop": "extracts", "explaintext": 1,
                   "exlimit": 1, "redirects": 1, "format": "json",
                   "titles": title})
        page = next(iter(d["query"]["pages"].values()))
        text = (page.get("extract") or "").strip()
    except Exception:  # noqa: BLE001
        return None
    if text and _is_disambig(text, title):
        text = ""  # страница-дизамбиг: не статья, справке не годится
    text = text[:EXTRACT_MAX] if text else ""
    _cache_put(key, text)
    return text or None


# ---------- сборка справки (единственный LLM-слой) ----------

def _messages(subject, texts):
    return [
        {"role": "system",
         "content": REF_PROMPT.format(о_чём=f" — {subject}")},
        {"role": "user",
         "content": "\n\n---\n\n".join(texts)},
    ]


def _stream(body, on_chunk):
    """SSE-стрим chat/completions -> (текст, usage | None).
    reasoning_content игнорируем: копим только куски content."""
    import urllib.request
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_key()}", **UA},
        method="POST")
    pieces, usage = [], None
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            if not d.get("choices"):
                continue  # чанк только с usage (include_usage)
            piece = (d["choices"][0].get("delta") or {}).get("content")
            if piece:
                pieces.append(piece)
                if on_chunk:
                    on_chunk(piece)
    return "".join(pieces), usage


def _plain(body, model):
    """Неразрывный вызов (фолбэк стрима). -> str | None."""
    import urllib.request
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_key()}", **UA},
        method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    import spend
    spend.log_call(model, "reference", resp.get("usage"))
    return (resp["choices"][0]["message"]["content"] or "").strip() or None


def _compose(subject, texts, model="deepseek-v4-flash", on_chunk=None):
    """LLM-сборка по поданным текстам. -> str | None.
    Со стримом при on_chunk (живой текст в кнопке); сбой или пустой стрим —
    один неразрывный повтор; его сбой — None (не падляем)."""
    base = {"model": model, "temperature": 0,
            "messages": _messages(subject, texts)}
    try:
        text, usage = _stream(
            json.dumps({**base, "stream": True,
                        "stream_options": {"include_usage": True}}).encode(),
            on_chunk)
        if text:
            import spend
            spend.log_call(model, "reference", usage)
            return text
        log.warning("стрим справки пуст (%s) — повтор без стрима", subject)
    except Exception:  # noqa: BLE001
        log.warning("стрим справки упал (%s) — повтор без стрима",
                    subject, exc_info=True)
    try:
        return _plain(json.dumps(base).encode(), model)
    except Exception:  # noqa: BLE001
        log.warning("неразрывный вызов справки упал (%s)", subject,
                    exc_info=True)
        return None


NO_AUTHOR = "К сожалению, мы не нашли каких-то данных по этому автору " \
            "в открытых источниках."
NO_PAINTING = "К сожалению, мы не нашли данных по этой картине " \
              "в открытых источниках."

# LLM иногда вместо справки рассказывает про сам вход («в предоставленном
# тексте нет статьи...», разбор страницы значений). Это служебный мусор,
# а не справка: показывать нельзя, кэшировать тем более.
_META = ("предоставленн", "нет статьи", "страница значений",
         "текст представляет собой", "не содержится")


def _is_meta_refusal(out):
    head = out[:400].lower()
    return any(m in head for m in _META)


def about_author(author, source=None, on_chunk=None):
    """Справка об авторе (и произведении, если подписано). -> str.
    Источник из подписи виден в справке строкой «Произведение из подписи
    канала» — даже когда вики-статьи о нём нет. Заглушки — статичные,
    без LLM: нечему соврать. on_chunk — колбек стрима (кэш-хит и
    заглушки его не зовут)."""
    if not author:
        return NO_AUTHOR
    key = f"ref:author:{author}|{source or ''}"
    cached = _cache_get(key, max_age=365 * 24 * 3600)  # справка живёт год
    if cached and not _is_meta_refusal(cached):
        return cached  # метасправка со старой версии — мимо кэша, пересборка
    title = resolve(author)
    if not title:
        return NO_AUTHOR
    texts, subj = [], f"об авторе «{author}»"
    a = extract(title)
    if a:
        texts.append(a)
    tail = ""
    if source:
        s = extract(source)
        if s:
            texts.append(f"Статья о произведении «{source}»:\n{s}")
            subj += " и о произведении"
        tail = f"Произведение из подписи канала: «{source}»."
    if not texts:
        return NO_AUTHOR
    out = _compose(subj, texts, on_chunk=on_chunk)
    if out and _is_meta_refusal(out):
        log.warning("справка об авторе %r ушла в мету про вход — заглушка",
                    author)
        return NO_AUTHOR
    if out:
        if tail:
            out += f"\n\n{tail}"
        _cache_put(key, out)
        return out + "\n\n(из открытых источников)"
    return "Справка не собралась, попробуй ещё раз."


def _search_article(what, who):
    """Запасной полнотекстовый поиск статьи о картине -> (title, text) | None.

    opensearch — автодополнение по префиксу: «Караваджо Нарцисс» он не
    находит, а короткое название («Нарцисс») уходит в дизамбиг и отвергается
    extract-ом. list=search ищет по всему тексту статей; берём первую, где
    в названии есть слово из названия картины, а в тексте — фамилия автора:
    статья именно об этой картине, не об одноимённом мифе или цветке."""
    word = max(what.split(), key=len) if what else ""
    surname = max(who.split(), key=len) if who else ""
    try:
        d = _http({"action": "query", "list": "search", "format": "json",
                   "srsearch": f"{what} {who}".strip(), "srlimit": 5})
        titles = [x["title"] for x in d["query"]["search"]]
    except Exception:  # noqa: BLE001
        return None
    for t in titles:
        text = extract(t)
        if text and word.lower() in t.lower() and surname.lower() in text.lower():
            return t, text
    return None


def about_painting(painting, on_chunk=None):
    """Справка о картине: «Художник, Название» -> статья -> сборка."""
    who, what = painting.get("author", ""), painting.get("title", "")
    key = f"ref:paint:{who}|{what}"
    cached = _cache_get(key, max_age=365 * 24 * 3600)
    if cached and not _is_meta_refusal(cached):
        return cached  # метасправка со старой версии — мимо кэша, пересборка
    title = resolve(f"{who} {what}".strip()) or resolve(what)
    text = extract(title) if title else None
    if not text:  # opensearch промахнулся или название — дизамбиг («Нарцисс»)
        found = _search_article(what, who)
        if found:
            title, text = found
    if not text:
        return NO_PAINTING
    out = _compose(f"о картине «{what}» ({who})", [text], on_chunk=on_chunk)
    if out and _is_meta_refusal(out):
        log.warning("справка о картине %r ушла в мету про вход — заглушка",
                    key)
        return NO_PAINTING
    if out:
        _cache_put(key, out)
        return out + "\n\n(из открытых источников)"
    return "Справка не собралась, попробуй ещё раз."


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "автор":
        print(about_author(" ".join(sys.argv[2:])))
    elif len(sys.argv) > 2 and sys.argv[1] == "картина" and sys.argv[2].isdigit():
        import images
        pid = sys.argv[2]
        p = next((x for x in images._manifest() if str(x["id"]) == pid), None)
        print(about_painting(p) if p else "нет такой картины")
    else:
        print(__doc__)
