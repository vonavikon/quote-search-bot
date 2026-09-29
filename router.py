"""Роутер на правилах: без LLM-классификатора.

Один домен — любой текстовый запрос ведёт к поиску; LLM подключается только
для переформулировки разговорного запроса (длиннее 2 слов). Кнопки бота
(Sic erat in fatis, «поясни») — отдельные интенты, сюда не попадают.

Правила по порядку:
  1. «/...» -> admin (/resync /stats; прочее — подсказка)
  2. приветствие/мета по словарю -> intro (заготовленный ответ, не поиск)
  3. имя автора из базы (normalization.json, склонения префиксом) -> фасет:
     остаток запроса — тема, сам автор — фильтр выдачи
  4. <= 2 значимых слов -> в поиск как есть, LLM не зовём
  5. длиннее 12 слов -> дословный путь без LLM (вставленный фрагмент цитаты)
  6. остальное -> LLM-переформулировка в 2-5 слов (сбой — сырой текст)

Выдача фильтруется kind in quote/poem: личные размышления
автора канала не прилетают подписчику как «цитата».

Запуск:
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

# слова-упаковка: не несут темы, срезаем до поиска
_STOP = ("цитата цитаты цитату цитат стих стихи стихотворение найди подскажи "
         "покажи хочу ищи пожалуйста дай посмотри про о об на тему мне мне-бы "
         "хочу-бы какое какая какие").split()
STOPWORDS = set(_STOP)

MAX_WORDS_DIRECT = 2  # <= 2 значимых слов: в поиск как есть, без LLM
# длиннее 12 слов без кавычек — дословный путь, без LLM: это почти всегда
# вставленный кусок цитаты («помню только фрагмент...»), а не разговорный
# запрос. Заодно анти-абьюз: простыня в 4000 знаков не уходит в переформулировку
MAX_WORDS_QUOTED = 12

REFORM_PROMPT = """Ты готовишь поисковый запрос для базы философских цитат и стихов.
Перепиши разговорный запрос пользователя компактным поисковым запросом из 2-5 слов:
голая суть темы. Убери вежливость и просьбы («подскажи», «хочу найти», «пожалуйста»).
Имя автора, если есть, сохрани. Ничего не выдумывай и не расширяй тему.
Верни СТРОГО JSON: {"query": "..."}"""


def _norm(s):
    return s.lower().replace("ё", "е")


# ---------- фасет автора ----------

def author_index(cache_path=None):
    """{нормализованная фамилия: каноническое имя} из подписанных авторов."""
    if cache_path is None:  # глобаль в момент вызова, не дефолт при импорте
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
        if len(surname) >= 4:  # короткие (Гёте, Кант) ловит dense и без фасета
            idx.setdefault(surname, it["author"])
    return idx


def match_author(query, idx):
    """Ищет фамилию автора в запросе (склонения — префиксом: «пастернака»
    начинается с «пастернак»). Возвращает (имя|None, запрос без слов автора)."""
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


# ---------- переформулировка (единственный LLM-слой роутера) ----------

def reformulate(text, model="deepseek-v4-flash"):
    """Разговорный запрос -> компактный поисковый. Сбой — None (не падляем)."""
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
    except Exception:  # noqa: BLE001 — роутер живёт и без LLM
        return None


# ---------- маршрутизация ----------

def route(text, allow_llm=True):
    """Текст пользователя -> маршрут для бота.

    {"intent": "search", "query": ..., "author": имя|None, "path": "direct|llm"}
    intent: search | admin | intro.
    allow_llm=False — переформулировку не зовём (сырой путь, как при сбое LLM):
    бот проверяет дневной лимит ДО route, чтобы исчерпавший его пользователь
    не жёг flash на переформулировке (intro/admin и без того идут до LLM).
    """
    t = text.strip()

    first_line = t.split("\n", 1)[0].strip()
    if first_line.startswith("/"):
        # команда — ровно первая строка; «/resync\nмусор» — это текст,
        # многострочные вставки не должны попадать в админский интент
        return {"intent": "admin", "query": ""}

    if INTRO_RE.match(t):
        return {"intent": "intro", "query": ""}

    idx = author_index()
    author, rest = match_author(t, idx)
    rest = strip_stopwords(rest)

    # дословный фрагмент в кавычках: без переформулировки (точная фраза
    # дороже компактности) — wants_lexical в hybrid подхватит кавычки
    if '"' in t or "«" in t:
        return {"intent": "search", "query": rest or t.strip('"«» '),
                "author": author, "path": "direct-quoted"}

    if len(rest.split()) > MAX_WORDS_QUOTED:
        # простыня текста — дословный фрагмент без кавычек: реформулировка
        # не нужна (точная фраза дороже компактности), LLM не зовём —
        # иначе кидание больших текстов жгло бы токены на ровном месте
        return {"intent": "search", "query": rest or t,
                "author": author, "path": "direct-quoted"}

    if len(rest.split()) <= MAX_WORDS_DIRECT:
        # короткий: как есть; только автор — ищем его самого
        return {"intent": "search", "query": rest or (author or ""),
                "author": author, "path": "direct"}

    q = reformulate(t) if allow_llm else None
    if q:
        # переформулировка могла занести/потерять автора — сверяем фасет ещё раз
        author2, rest2 = match_author(q, idx)
        if author2 and not author:
            author, rest = author2, strip_stopwords(rest2)
            q = rest or q
        return {"intent": "search", "query": q, "author": author, "path": "llm"}
    return {"intent": "search", "query": rest, "author": author, "path": "direct-fallback"}


# ---------- выдача с фильтрами продукта ----------

def search_cards(route_, k=5):
    """Поиск по маршруту + фильтры kind и автора. -> [(пост, скор, путь)]"""
    import hybrid
    import cards

    order, posts, dense, bm_scores, path = hybrid.retrieve(route_["query"], n=max(200, k * 4))
    items = cards.load_norm()  # с курируемыми метаданными детей склеек
    # фамилия фасета: пост проходит, если автор подписан ИЛИ канал пересылки
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
