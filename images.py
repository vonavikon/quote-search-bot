"""Картинки карточек: пул PD-art с Wikimedia Commons.

Подбор картины детерминирован, без LLM:
1. совпадение темы запроса с тегами картины — префиксное сравнение с запасом
   на окончание (одиночество/одиночестве, свеча/свечи);
2. семантическая карта painting_map.json — заранее посчитанное соответствие
   «настроение текста поста -> картина» (косинус эмбеддинга текста против
   эмбеддингов LLM-описаний картин). В рантайме это чтение json;
3. нет совпадения — та же проба по словам текста самой цитаты: «упражняй
   себя трудами» получает тематическую картину, а не случайную по хэшу;
4. и это не совпало — стабильный хэш post_id: карточка всегда с одной
   картиной;
5. занятые в колоде картины пропускаются (перебор вперёд от хэша).

Манифест paintings/paintings.json: [{id, file, title, author, year, themes[]}].
painting_vecs.npz рядом с манифестом — матрица эмбеддингов описаний,
выровнена по порядку манифеста (для досшивки карты новым постам).
Кэш file_id Telegram живёт в DATA_DIR (volume), файлы пула докачиваются
fetch_paintings.py.
"""
import hashlib
import json
import os
import re
from functools import lru_cache

from load_corpus import DATA_DIR

POOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paintings")
MANIFEST_PATH = os.path.join(POOL_DIR, "paintings.json")
FILE_IDS_PATH = os.path.join(DATA_DIR, "file_ids.json")

# семантическая карта: сид в репо (только чтение), рабочая копия в volume
MAP_SEED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "painting_map.json")
MAP_PATH = os.path.join(DATA_DIR, "painting_map.json")
VECS_PATH = os.path.join(POOL_DIR, "painting_vecs.npz")
TOP_K = 5

# сколько первых слов текста цитаты участвует в подборе темы
MAX_TOKENS = 80


def _norm(s):
    return s.lower().replace("ё", "е")


def _prefix_match(tok, theme):
    """Совпадение с запасом на окончание: общий префикс покрывает короткую
    сторону почти целиком (минус 1) и не короче 4 букв."""
    cp = os.path.commonprefix([tok, theme])
    return len(cp) >= 4 and len(cp) >= min(len(tok), len(theme)) - 1


@lru_cache(maxsize=1)
def _manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _theme_index(manifest):
    """[(норм. тег, [картины в порядке манифеста])] в порядке первого
    появления тега. Уникальных тегов ~90, пар картина-тег ~1400: слово
    сравнивается с уникальными тегами, а не с каждой парой — на длинных
    текстах это на порядок дешевле."""
    order, by_theme = [], {}
    for p in manifest:
        for th in p["themes"]:
            t = _norm(th)
            if t not in by_theme:
                by_theme[t] = []
                order.append(t)
            by_theme[t].append(p)
    return [(t, by_theme[t]) for t in order]


@lru_cache(maxsize=1)
def _cached_theme_index():
    return _theme_index(_manifest())


@lru_cache(maxsize=1)
def _painting_map():
    """Карта «post_id -> [id картин top-5]»: сид из репо, поверх — рабочая
    копия из volume (её дошивает extend_painting_map). Слияние, а не выбор
    одной: обновлённый деплоем сид виден и при устаревшей копии в volume."""
    out = {}
    for path in (MAP_SEED_PATH, MAP_PATH):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out.update(json.load(f))
    return out


@lru_cache(maxsize=1)
def _by_id():
    return {p["id"]: p for p in _manifest()}


def semantic_candidates(post_id):
    """Кандидаты семантической карты: топ-K картин по настроению текста
    поста (посчитано офлайн). Поста нет в карте — []."""
    out = []
    for pid in _painting_map().get(str(post_id), []):
        p = _by_id().get(pid)
        if p is not None:
            out.append(p)
    return out


def extend_painting_map(texts):
    """Дошивает карту новым постам {post_id: текст} — вызывается на старте
    бота после index.build. Один embed_query на пост (батчами по 32) против
    матрицы описаний painting_vecs.npz; результат пишется в DATA_DIR, сид в
    репо не трогается. Сеть/матрица недоступны или размер не сходится —
    тихий возврат 0: посты остаются на темах текста и хэше."""
    todo = {str(pid): t[:1000] for pid, t in texts.items()
            if str(pid) not in _painting_map() and t}
    if not todo:
        return 0
    if not os.path.exists(VECS_PATH):
        return 0
    import numpy as np  # lazy: рантайну без досшивки numpy не нужен
    V = np.load(VECS_PATH)["v"]
    manifest = _manifest()
    if V.shape[0] != len(manifest):
        return 0
    from embed import embed_query
    items = sorted(todo.items())
    pmap = dict(_painting_map())
    try:
        for i in range(0, len(items), 32):
            batch = items[i:i + 32]
            Q, _ = embed_query([t for _, t in batch])
            Q = np.array(Q, dtype=np.float32)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            sims = Q @ V.T
            for j, (pid, _) in enumerate(batch):
                top = sims[j].argsort()[::-1][:TOP_K]
                pmap[pid] = [manifest[k]["id"] for k in top]
    except Exception as e:  # сеть bothub — не повод ронять старт
        print(f"painting_map: досшивка прервана ({e}), {len(pmap)} готово")
    added = sum(1 for pid in todo if pid in pmap)
    if added:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(pmap, f, ensure_ascii=False, indent=1)
        _painting_map.cache_clear()
    return added


def theme_candidates(query, manifest=None):
    """Все картины, чей тег совпадает со словом запроса, в порядке слов
    запроса и порядка манифеста. [] — совпадений нет.

    Слова дедуплицируются, берётся не больше MAX_TOKENS первых — текст
    цитаты бывает длинным, а хвост всё равно не выигрывает у головы."""
    index = _cached_theme_index() if manifest is None else _theme_index(manifest)
    tokens, seen = [], set()
    for tok in re.findall(r"[а-яёa-z0-9-]+", _norm(query)):
        if len(tok) >= 4 and tok not in seen:
            seen.add(tok)
            tokens.append(tok)
            if len(tokens) >= MAX_TOKENS:
                break
    out, out_ids = [], set()
    for tok in tokens:
        for th, ps in index:
            if _prefix_match(tok, th):
                for p in ps:
                    if p["id"] not in out_ids:
                        out_ids.add(p["id"])
                        out.append(p)
    return out


def pick(post_id, query="", exclude=None, text=""):
    """Картина для карточки. Порядок попыток:
    1. тема запроса — первая свободная из тематических кандидатов
       (намерение пользователя главнее заготовки по настроению);
    2. семантическая карта — топ-K картин поста по настроению текста;
    3. тема текста самой цитаты — та же проба по словам цитаты (фолбэк
       до хэша: стоическая цитата о труде получает «Сеятеля», а не
       случайную картину по хэшу post_id);
    4. стабильный хэш post_id с перебором вперёд — карточка без запроса
       всегда с одной картиной, а exclude (занятые в этой колоде) пропускается.
    Если свободных не осталось (колода длиннее пула) — последний резерв."""
    manifest = _manifest()
    exclude = {e["id"] if isinstance(e, dict) else e for e in (exclude or ())}
    if query:
        for p in theme_candidates(query, manifest):
            if p["id"] not in exclude:
                return p
    for p in semantic_candidates(post_id):
        if p["id"] not in exclude:
            return p
    if text:
        for p in theme_candidates(text, manifest):
            if p["id"] not in exclude:
                return p
    h = int(hashlib.sha1(str(post_id).encode()).hexdigest(), 16)
    for k in range(len(manifest)):
        p = manifest[(h + k) % len(manifest)]
        if p["id"] not in exclude:
            return p
    return manifest[h % len(manifest)]


def painting_path(painting):
    return os.path.join(POOL_DIR, painting["file"])


# ---------- кэш file_id Telegram ----------

def load_file_ids():
    if not os.path.exists(FILE_IDS_PATH):
        return {}
    with open(FILE_IDS_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_file_id(painting_id):
    return load_file_ids().get(painting_id)


def save_file_id(painting_id, file_id):
    ids = load_file_ids()
    if ids.get(painting_id) == file_id:
        return
    ids[painting_id] = file_id
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FILE_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=1)
