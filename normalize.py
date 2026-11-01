"""LLM normalization of the corpus with a minimal cut: kind, author, source.
The trust rule: the author and the source come ONLY from an explicit
signature in the post text. Guessing by style/recognition is forbidden —
a misattributed quote is worse than a missing author. Enforced in code:
author/source are wiped unless evidence == "signed".

Public visibility is NOT an LLM field: it is a product rule, kind in
(quote, poem) is checked in router.PUBLIC_KINDS at serving time. origin
(forwarded channel) comes from the export without an LLM.

The normalization.json cache is idempotent by post_id: a re-run only
picks up new posts, the cost does not accumulate.
"""
import json
import os
import re
import sys
import urllib.request

from embed import BASE, _key
from load_corpus import DATA_DIR, load_posts, split_child_ids

MODEL = "deepseek-v4-flash"  # picked by a mini eval (agreement with pro at a third of the price)
PM, CM = 0.27, 1.10          # $/M tokens (bothub; re-check when the model changes)
CACHE = os.path.join(DATA_DIR, "normalization.json")
BATCH = 15
HEAD, TAIL, HARD_CAP = 700, 500, 1200  # the signature is at the end: the tail is mandatory

KINDS = ("quote", "poem", "reflection")

PROMPT = """Ты нормализуешь посты Telegram-канала с цитатами (философия, психология, этика, поэзия). Для каждого поста верни метаданные.

kind — одно из:
- "quote": чужая мысль — цитата, афоризм, отрывок прозы (любая чужая публикация, не стихи)
- "poem": стихотворение или стихотворный фрагмент (строфы, ритм, рифма)
- "reflection": авторское размышление автора канала, НЕ чужая цитата. Признаки: разговорный стиль от первого лица, нет подписи другого автора, это совет/мысль/наблюдение самого автора

author — имя автора, ТОЛЬКО если оно явно подписано в тексте поста (последняя строка с именем, «— Имя», «© Имя»). Подпись «кто-то умный» и подобные шутливые — верни как написано, это честная подпись источника. НЕ УГАДЫВАЙ автора по содержанию, стилю или узнаваемости текста: если подписи нет — null, даже если ты уверен, кто автор.

source — произведение, ТОЛЬКО если названо явно в подписи или тексте («Джейн Эйр», «Опыты, или наставления нравственные и политические»). Может быть подписано отдельно от автора, последней строкой («Искусство Войны», «Волшебный голос Джельсомино») — тогда author null, а source заполнен. Иначе null.

author_evidence — "signed", если author взят из явной подписи в тексте, иначе "none".

author_guess — если автор НЕ подписан, но ты узнал текст с высокой уверенностью (знаменитая цитата, узнаваемый стиль), укажи предполагаемого автора здесь. Это гипотеза для внутренней проверки, НЕ факт: в author её не переносить.

Верни СТРОГО JSON: {"posts": [{"id": <id>, "kind": ..., "author": ...|null, "source": ...|null, "author_evidence": "signed"|"none", "author_guess": ...|null}]}. Никакого текста вне JSON."""


def clip(text):
    """Long posts: head + tail (the author signature is always at the end)."""
    if len(text) <= HARD_CAP:
        return text
    return text[:HEAD] + "\n[...сокращено...]\n" + text[-TAIL:]


def call(model, posts):
    """posts: [{id, text}] -> [{"id", "kind", "author", "source", "author_evidence"}]."""
    payload = [{"id": p["id"], "text": clip(p["text"])} for p in posts]
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps({"posts": payload}, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_key()}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    raw = data["choices"][0]["message"]["content"]
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
    items = json.loads(raw)["posts"]
    # a hard guarantee of the trust rule: no signature — no author (the
    # source lives by its own signature — a work can be named without an
    # author)
    for it in items:
        if it.get("author_evidence") != "signed":
            it["author"] = None
            it["author_evidence"] = "none"
        if it.get("kind") not in KINDS:
            raise ValueError(f"post {it.get('id')}: неизвестный kind {it.get('kind')!r}")
        it["author"] = (it.get("author") or None)
        it["source"] = (it.get("source") or None)
    usage = data.get("usage", {})
    return items, usage


def _norm(s):
    return s.lower().replace("ё", "е")


def enforce_trust(items, posts):
    """A deterministic check of the trust rule on top of the LLM's words:
    the claimed author must occur in the last 2 lines of the post — by at
    least one meaningful word (>= 4 letters, е/ё-insensitive; covers the
    declension «Грасиана» and a surname-only signature «пастернак, 1956»).
    Otherwise the author is not signed: demoted into author_guess.
    Returns the number of demotions."""
    by_id = {p["id"]: p for p in posts}
    demoted = 0
    for pid, it in items.items():
        if not it.get("author"):
            continue
        p = by_id.get(int(pid))
        if not p:
            continue
        tail = _norm("\n".join(p["text"].strip().splitlines()[-2:]))
        words = [w for w in re.split(r"[^\w]+", _norm(it["author"])) if len(w) >= 4]
        if words and not any(w in tail for w in words):
            it["author_guess"] = it.get("author_guess") or it["author"]
            it["author"] = None
            it["author_evidence"] = "none"
            demoted += 1
    return demoted


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {"model": MODEL, "items": {}}


def run(sample=None, model=MODEL):
    """sample=N — run only N random posts (for eyeballing).
    Returns (cache, stats)."""
    posts = load_posts()
    if sample:
        import random
        random.seed(42)
        posts = random.sample(posts, min(sample, len(posts)))
    cache = load_cache()
    all_posts = load_posts()  # the trust check runs over the full corpus
    demoted = enforce_trust(cache["items"], all_posts)
    if demoted:
        print(f"правило доверия: понижено в guess {demoted} авторов", file=sys.stderr)
    # split children are not normalized by the LLM — their metadata is curated by splits.json
    curated = split_child_ids()
    todo = [p for p in posts
            if str(p["id"]) not in cache["items"]
            and str(p["id"]) not in curated]
    total_cost = 0.0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        items, usage = run_call(model, chunk)
        for it in items:
            cache["items"][str(it["id"])] = it
        cost = (usage.get("prompt_tokens", 0) / 1e6 * PM + usage.get("completion_tokens", 0) / 1e6 * CM) if usage else 0
        total_cost += cost
        import spend
        # we append cost ourselves: an OpenAI-compatible usage field is
        # optional, without it the /stats telemetry would silently show $0
        spend.log_call(model, "normalize", {**(usage or {}), "cost": cost})
        print(f"  батч {i // BATCH + 1}: {len(items)} постов, ${cost:.5f}", file=sys.stderr)
        with open(CACHE, "w", encoding="utf-8") as f:  # write after every batch
            json.dump(cache, f, ensure_ascii=False, indent=1)
    if demoted and not todo:  # a cache repair with no new posts — persist it too
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
    stats = summarize(cache, posts)
    return cache, stats, total_cost


def run_call(model, chunk):
    last = None
    for _ in range(3):  # retries on broken JSON/timeout
        try:
            return call(model, chunk)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"батч не прошёл после 3 попыток: {last}")


def summarize(cache, posts):
    items = [cache["items"][str(p["id"])] for p in posts if str(p["id"]) in cache["items"]]
    kinds = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    signed = sum(1 for it in items if it["author"])
    return {"n": len(items), "kinds": kinds, "signed": signed}


if __name__ == "__main__":
    sample = None
    if "--sample" in sys.argv:
        sample = int(sys.argv[sys.argv.index("--sample") + 1])
    cache, stats, cost = run(sample=sample)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"суммарно ${cost:.5f}")
