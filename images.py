"""Card images: a PD-art pool from Wikimedia Commons.

Painting selection is deterministic, no LLM:
1. the query topic matching a painting's tags — a prefix comparison with
   slack for the ending (одиночество/одиночестве, свеча/свечи);
2. the semantic map painting_map.json — a precomputed correspondence
   "mood of the post text -> painting" (cosine of the text embedding
   against embeddings of LLM painting descriptions). At runtime it is a
   json read;
3. no match — the same probe over the words of the quote text itself:
   «упражняй себя трудами» gets a thematic painting, not a random one by
   hash;
4. that missed too — a stable hash of post_id: the card always comes
   with the same painting;
5. paintings already used in the deck are skipped (a forward walk from
   the hash).

The paintings/paintings.json manifest: [{id, file, title, author, year,
themes[]}]. painting_vecs.npz next to the manifest is the description
embedding matrix, aligned with the manifest order (for catching the map
up on new posts). The Telegram file_id cache lives in DATA_DIR (a
volume); pool files are downloaded by fetch_paintings.py.
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

# semantic map: the seed lives in the repo (read-only), the working copy in the volume
MAP_SEED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "painting_map.json")
MAP_PATH = os.path.join(DATA_DIR, "painting_map.json")
VECS_PATH = os.path.join(POOL_DIR, "painting_vecs.npz")
TOP_K = 5

# how many leading words of the quote text take part in theme matching
MAX_TOKENS = 80


def _norm(s):
    return s.lower().replace("ё", "е")


def _prefix_match(tok, theme):
    """A match with slack for the ending: the common prefix covers the
    short side almost entirely (minus 1) and is at least 4 letters."""
    cp = os.path.commonprefix([tok, theme])
    return len(cp) >= 4 and len(cp) >= min(len(tok), len(theme)) - 1


@lru_cache(maxsize=1)
def _manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _theme_index(manifest):
    """[(normalized tag, [paintings in manifest order])] in the order of
    the tag's first appearance. ~90 unique tags, ~1400 painting-tag
    pairs: a word is compared against unique tags, not against every
    pair — an order of magnitude cheaper on long texts."""
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
    """The map "post_id -> [top-5 painting ids]": the seed from the repo,
    with the working copy from the volume layered on top (extend_painting_map
    adds to it). A merge, not a choice of one: a seed updated by a deploy
    is visible even when the volume copy is stale."""
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
    """Semantic map candidates: the top-K paintings for the post text's
    mood (computed offline). A post missing from the map — []."""
    out = []
    for pid in _painting_map().get(str(post_id), []):
        p = _by_id().get(pid)
        if p is not None:
            out.append(p)
    return out


def extend_painting_map(texts):
    """Catches the map up on new posts {post_id: text} — called at bot
    startup after index.build. One embed_query per post (batches of 32)
    against the painting_vecs.npz description matrix; the result goes to
    DATA_DIR, the repo seed is not touched. Network/matrix unavailable or
    the size doesn't add up — a silent return of 0: the posts stay on
    text themes and the hash."""
    todo = {str(pid): t[:1000] for pid, t in texts.items()
            if str(pid) not in _painting_map() and t}
    if not todo:
        return 0
    if not os.path.exists(VECS_PATH):
        return 0
    import numpy as np  # lazy: runtime without the catch-up doesn't need numpy
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
    except Exception as e:  # the bothub network is no reason to kill startup
        print(f"painting_map: досшивка прервана ({e}), {len(pmap)} готово")
    added = sum(1 for pid in todo if pid in pmap)
    if added:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(pmap, f, ensure_ascii=False, indent=1)
        _painting_map.cache_clear()
    return added


def theme_candidates(query, manifest=None):
    """All paintings whose tag matches a word of the query, in the order
    of query words and manifest order. [] — no matches.

    Words are deduplicated and only the first MAX_TOKENS are taken —
    quote texts run long, and the tail never beats the head anyway."""
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
    """A painting for the card. Order of attempts:
    1. the query theme — the first free one among the thematic candidates
       (the user's intent beats a precomputed mood pick);
    2. the semantic map — the top-K paintings for the post by the mood of
       its text;
    3. the theme of the quote text itself — the same probe over the
       quote's words (a fallback before the hash: a Stoic quote about
       labor gets "The Sower", not a random painting by post_id hash);
    4. a stable hash of post_id with a forward walk — a card without a
       query always has the same painting, and exclude (the ones taken
       in this deck) is skipped.
    Nothing free left (a deck longer than the pool) — the last resort."""
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


# ---------- Telegram file_id cache ----------

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
