"""Reference cards about authors and paintings from Wikipedia (no slop
and no false facts).

The method is three layers of honesty:
  1. corpus data (author, work) — signatures in the channel posts;
  2. Wikipedia articles verbatim — human text, not generation;
  3. narrative assembly — an LLM working ONLY from the supplied wiki
     texts: extraction, not recall. The prompt forbids facts and numbers
     outside the texts.

No Wikipedia article — a static stub, the LLM is not called at all
(fail-safe: there is nothing to lie about). Cache in
DATA_DIR/wiki_cache.json: raw extracts and ready reference cards,
repeated clicks are free and instant.

Run (a prototype over posts; the CLI keywords are Russian — автор =
author, картина = painting):
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
EXTRACT_MAX = 8000   # chars of article into the LLM context (creation history is at the top)
CACHE_TTL = 30 * 24 * 3600  # re-ask Wikipedia after a month

REF_PROMPT = """Ты пишешь краткую справку для карточки цитаты в боте. Ниже — тексты статей Википедии{о_чём}.
Напиши 3-6 предложений: кто автор и чем известен; при каких обстоятельствах и в какой период написано произведение — если статья об этом говорит.
ЖЕЛЕЗНЫЕ ПРАВИЛА:
- каждый факт — только из текстов ниже; ничего от себя, никаких внешних знаний
- даты, числа, названия — только дословно из текста
- обстоятельств создания в текстах нет — не выдумывай их, просто скажи, кто автор
- без оценочных слов («великий», «интересно, что»), без обращений к читателю"""


# ---------- cache ----------

def _cache_path():
    return os.path.join(DATA_DIR, "wiki_cache.json")


def _cache_load():
    try:
        with open(_cache_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 — no file or corrupt: start empty
        return {}


def _cache_put(key, value):
    c = _cache_load()
    c[key] = {"ts": int(time.time()), "v": value}
    try:
        with open(_cache_path(), "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False)
    except Exception:  # noqa: BLE001 — the cache is not critical
        pass


def _cache_get(key, max_age=CACHE_TTL):
    e = _cache_load().get(key)
    if not e:
        return None
    if time.time() - e["ts"] > max_age:
        return None
    return e["v"]


# ---------- Wikipedia ----------

def _http(url):
    import urllib.parse
    import urllib.request
    url = "https://ru.wikipedia.org/w/api.php?" + urllib.parse.urlencode(url)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def resolve(title):
    """The exact article title (opensearch): «Шваб» -> «Шваб, Клаус».
    None — no article."""
    cached = _cache_get(f"resolve:{title}")
    if cached is not None:
        return cached or None
    try:
        names = _http({"action": "opensearch", "format": "json",
                       "limit": 1, "search": title})[1]
        hit = names[0] if names else ""
    except Exception:  # noqa: BLE001 — the network blinked: skip caching this record
        return None
    _cache_put(f"resolve:{title}", hit)
    return hit or None


def _plain_word(s):
    """A string without diacritics: «Нарци́сс» -> «Нарцисс»."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def _is_disambig(text, title):
    """A disambiguation page? Two patterns. Explicit: «многозначн» /
    «может означать» in the first lines. Structural (a live case,
    «Нарцисс»): a page without those words but laid out as a list —
    >= 3 lines like «Нарцисс — род...» / ««Нарцисс» — бригантина...»
    starting with the title and going with a dash. A regular article is
    not laid out like that: the title appears once in the first line, not
    a line per meaning of the word."""
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
    """Flat article text (explaintext), cut at EXTRACT_MAX.
    None — no article, network, or a disambiguation page. A negative
    result is cached."""
    key = f"extract:{title}"
    cached = _cache_get(key)
    if cached is not None:
        if cached and _is_disambig(cached, title):
            _cache_put(key, "")  # disambig cached by the old filter: fix it
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
        text = ""  # a disambiguation page: not an article, useless for a reference card
    text = text[:EXTRACT_MAX] if text else ""
    _cache_put(key, text)
    return text or None


# ---------- reference assembly (the only LLM layer) ----------

def _messages(subject, texts):
    return [
        {"role": "system",
         "content": REF_PROMPT.format(о_чём=f" — {subject}")},
        {"role": "user",
         "content": "\n\n---\n\n".join(texts)},
    ]


def _stream(body, on_chunk):
    """SSE stream of chat/completions -> (text, usage | None).
    reasoning_content is ignored: we collect content chunks only."""
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
                continue  # a chunk with usage only (include_usage)
            piece = (d["choices"][0].get("delta") or {}).get("content")
            if piece:
                pieces.append(piece)
                if on_chunk:
                    on_chunk(piece)
    return "".join(pieces), usage


def _plain(body, model):
    """A non-streaming call (the stream fallback). -> str | None."""
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
    """LLM assembly over the supplied texts. -> str | None.
    Streams when on_chunk is set (live text on the button); a failure or
    an empty stream means one non-streaming retry; its failure — None
    (no crash)."""
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

# The LLM sometimes talks about its own input instead of giving a
# reference card («в предоставленном тексте нет статьи...», a
# disambiguation-page breakdown). That is service garbage, not a
# reference card: it must not be shown, let alone cached.
_META = ("предоставленн", "нет статьи", "страница значений",
         "текст представляет собой", "не содержится")


def _is_meta_refusal(out):
    head = out[:400].lower()
    return any(m in head for m in _META)


def about_author(author, source=None, on_chunk=None):
    """Reference card about the author (and the work, if signed).
    -> str. A source from the signature shows up in the card as the line
    «Произведение из подписи канала» — even when there is no wiki article
    about it. Stubs are static, no LLM: nothing to lie. on_chunk is the
    stream callback (a cache hit and the stubs never call it)."""
    if not author:
        return NO_AUTHOR
    key = f"ref:author:{author}|{source or ''}"
    cached = _cache_get(key, max_age=365 * 24 * 3600)  # a reference card lives a year
    if cached and not _is_meta_refusal(cached):
        return cached  # meta-card from an old version — bypass the cache, rebuild
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
    """Backup full-text search for the painting article -> (title, text) | None.

    opensearch is prefix autocompletion: it does not find «Караваджо
    Нарцисс», while the short title («Нарцисс») goes into a disambiguation
    and gets rejected by extract. list=search searches full article text;
    we take the first one whose title contains a word from the painting
    title and whose text contains the author's surname: an article about
    this exact painting, not the same-named myth or flower."""
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
    """Reference card about a painting: «Художник, Название» -> article ->
    assembly."""
    who, what = painting.get("author", ""), painting.get("title", "")
    key = f"ref:paint:{who}|{what}"
    cached = _cache_get(key, max_age=365 * 24 * 3600)
    if cached and not _is_meta_refusal(cached):
        return cached  # meta-card from an old version — bypass the cache, rebuild
    title = resolve(f"{who} {what}".strip()) or resolve(what)
    text = extract(title) if title else None
    if not text:  # opensearch missed or the title is a disambiguation («Нарцисс»)
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
