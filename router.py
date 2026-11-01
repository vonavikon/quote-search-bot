"""A rule-based router: no LLM classifier.

One domain — any textual query leads to search; the LLM is only involved
in reformulating a conversational query (longer than 2 words). The bot's
buttons (Sic erat in fatis, "explain") are separate intents and never get
here.

Rules in order:
  1. "/..." -> admin (/resync /stats; anything else — a hint)
  2. greeting/meta from a dictionary -> intro (a canned reply, not search)
  3. an author name from the base (normalization.json, declensions by
     prefix) -> a facet: the rest of the query is the topic, the author
     himself is a filter on the results
  4. <= 2 meaningful words -> to search as is, no LLM called
  5. longer than 12 words -> the literal path without an LLM (a pasted
     quote fragment)
  6. everything else -> an LLM reformulation into 2-5 words (a failure —
     the raw text)

Results are filtered by kind in quote/poem: the channel author's personal
reflections never reach a subscriber as a "quote".

Run:
  python -X utf8 router.py подскажи что-нибудь мудрое про дружбу
"""
import json
import os
import re
import sys

from embed import BASE, _key
from normalize import CACHE as NORM_CACHE

PUBLIC_KINDS = ("quote", "poem")

INTRO_RE = re.compile(
    r"^(привет\w*|здравств\w*|добр(ый|ое) (день|вечер|утро)|хай|ку|спасибо|"
    r"благодар\w*|что ты умеешь|помощь|\/?start|help)[!,. ]*$", re.I)

# packaging words: they carry no topic, cut them before search
_STOP = ("цитата цитаты цитату цитат стих стихи стихотворение найди подскажи "
         "покажи хочу ищи пожалуйста дай посмотри про о об на тему мне мне-бы "
         "хочу-бы какое какая какие").split()
STOPWORDS = set(_STOP)

MAX_WORDS_DIRECT = 2  # <= 2 meaningful words: to search as is, no LLM
# longer than 12 words without quotes — the literal path, no LLM: it is
# almost always a pasted quote fragment ("помню только фрагмент..."), not
# a conversational query. Also anti-abuse: a 4000-char wall of text never
# goes into a reformulation
MAX_WORDS_QUOTED = 12

REFORM_PROMPT = """Ты готовишь поисковый запрос для базы философских цитат и стихов.
Перепиши разговорный запрос пользователя компактным поисковым запросом из 2-5 слов:
голая суть темы. Убери вежливость и просьбы («подскажи», «хочу найти», «пожалуйста»).
Имя автора, если есть, сохрани. Ничего не выдумывай и не расширяй тему.
Верни СТРОГО JSON: {"query": "..."}"""


def _norm(s):
    return s.lower().replace("ё", "е")


# ---------- author facet ----------

def author_index(cache_path=None):
    """{normalized surname: canonical name} from the signed authors."""
    if cache_path is None:  # the global at call time, not an import-time default
        cache_path = NORM_CACHE
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, encoding="utf-8") as f:
        items = json.load(f)["items"]
    idx = {}
    for it in items.values():
        if not it.get("author"):
            continue
        surname = _norm(it["author"]).split()[-1]
        if len(surname) >= 4:  # short ones (Гёте, Кант) dense handles fine without a facet
            idx.setdefault(surname, it["author"])
    return idx


def match_author(query, idx):
    """Looks for an author surname in the query (declensions by prefix:
    «пастернака» starts with «пастернак»). Returns (name|None, the query
    without the author's words)."""
    tokens = re.findall(r"[а-яёa-z0-9-]+", _norm(query))
    keep, hit = [], None
    for tok in tokens:
        m = next((a for s, a in idx.items()
                  if len(tok) >= 4 and (tok.startswith(s) or s.startswith(tok))),
                 None)
        if m:
            hit = hit or m
        else:
            keep.append(tok)
    return hit, " ".join(keep)


def strip_stopwords(query):
    toks = [t for t in re.findall(r"[а-яёa-z0-9-]+", _norm(query))
            if t not in STOPWORDS]
    return " ".join(toks)


# ---------- reformulation (the router's only LLM layer) ----------

def reformulate(text, model="deepseek-v4-flash"):
    """A conversational query -> a compact search query. A failure — None
    (no crash)."""
    import urllib.request
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": REFORM_PROMPT},
            {"role": "user", "content": text},
        ],
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_key()}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
        import spend
        spend.log_call(model, "reformulate", resp.get("usage"))
        raw = resp["choices"][0]["message"]["content"]
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
        return json.loads(raw).get("query") or None
    except Exception:  # noqa: BLE001 — the router lives on without the LLM too
        return None


# ---------- routing ----------

def route(text, allow_llm=True):
    """The user's text -> a route for the bot.

    {"intent": "search", "query": ..., "author": name|None, "path": "direct|llm"}
    intent: search | admin | intro.
    allow_llm=False — don't call the reformulation (the raw path, same as
    on an LLM failure): the bot checks the daily limit BEFORE route, so a
    user who spent it doesn't burn flash on reformulation (intro/admin go
    before the LLM anyway).
    """
    t = text.strip()

    first_line = t.split("\n", 1)[0].strip()
    if first_line.startswith("/"):
        # a command is exactly the first line; "/resync\ngarbage" is text;
        # multi-line pastes must not fall into the admin intent
        return {"intent": "admin", "query": ""}

    if INTRO_RE.match(t):
        return {"intent": "intro", "query": ""}

    idx = author_index()
    author, rest = match_author(t, idx)
    rest = strip_stopwords(rest)

    # a literal fragment in quotes: no reformulation (the exact phrase
    # beats compactness) — wants_lexical in hybrid picks up the quotes
    if '"' in t or "«" in t:
        return {"intent": "search", "query": rest or t.strip('"«» '),
                "author": author, "path": "direct-quoted"}

    if len(rest.split()) > MAX_WORDS_QUOTED:
        # a wall of text — a literal fragment without quotes: no
        # reformulation needed (the exact phrase beats compactness), no
        # LLM — otherwise pasting big texts would burn tokens for nothing
        return {"intent": "search", "query": rest or t,
                "author": author, "path": "direct-quoted"}

    if len(rest.split()) <= MAX_WORDS_DIRECT:
        # short: as is; just an author — search for the author himself
        return {"intent": "search", "query": rest or (author or ""),
                "author": author, "path": "direct"}

    q = reformulate(t) if allow_llm else None
    if q:
        # the reformulation could add or drop the author — match the facet once more
        author2, rest2 = match_author(q, idx)
        if author2 and not author:
            author, rest = author2, strip_stopwords(rest2)
            q = rest or q
        return {"intent": "search", "query": q, "author": author, "path": "llm"}
    return {"intent": "search", "query": rest, "author": author, "path": "direct-fallback"}


# ---------- search with product filters ----------

def search_cards(route_, k=5):
    """Search by route + kind and author filters. -> [(post, score, path)]"""
    import hybrid
    import cards

    order, posts, dense, bm_scores, path = hybrid.retrieve(route_["query"], n=max(200, k * 4))
    items = cards.load_norm()  # with the curated metadata of split children
    # the facet surname: a post passes if the author is signed OR the channel forwarded it
    surname = _norm(route_.get("author") or "").split()[-1] if route_.get("author") else ""

    out = []
    for i in order:
        p = posts[i]
        it = items.get(str(p["id"]), {})
        if it.get("kind", "quote") not in PUBLIC_KINDS:
            continue
        if surname and surname not in _norm(it.get("author") or "") \
                and surname not in _norm(p["author"]):
            continue
        out.append((p, float(dense[i]), path))
        if len(out) >= k:
            break
    return out


if __name__ == "__main__":
    demo = [
        "про одиночество",
        "пастернак о любви",
        "«жалость к себе»",
        "привет",
        "подскажи что-нибудь мудрое про дружбу после ссоры",
        "/stats",
    ]
    for q in ([" ".join(sys.argv[1:])] if len(sys.argv) > 1 else demo):
        r = route(q)
        print(f"\n=== «{q}»\n    {r}")
        if r["intent"] == "search":
            for p, s, path in search_cards(r, k=3):
                who = p["author"] or "аноним"
                print(f"  [{s:.3f} | {path}] {who}: {p['text'][:90]}...")
