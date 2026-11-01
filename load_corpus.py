"""Corpus loader from a Telegram Desktop export (result.json).

Returns posts with non-empty text. One post = one chunk (short quotes,
nothing to split).

Server mode: when the export file is missing (we don't keep one on the
server), posts come from corpus_meta.json — the master lives in the
data, the export arrives only via /resync.
"""
import json
import os
import sys

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "result.json")

# data folder: corpus.npz / corpus_meta.json / normalization.json.
# In docker a volume is mounted and BOT_DATA=/data is set.
DATA_DIR = os.environ.get("BOT_DATA", os.path.dirname(os.path.abspath(__file__)))
META_PATH = os.path.join(DATA_DIR, "corpus_meta.json")

# the glue manifest travels with the code (data/ is not in git, but
# splits must be known both to the server and to the local export)
SPLITS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json")

# texts of image-only posts recognized by OCR (gpt-4.1-mini, a one-off
# backfill): in the export they have no text field, and the server
# corpus does not know them at all
OCR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr.json")


def load_ocr():
    """{post_id: {"date", "author", "text"}} of recognized image-only
    posts.

    A corrupt file does not kill the bot: degrade to a corpus without
    the OCR texts.
    """
    if not os.path.exists(OCR_PATH):
        return {}
    try:
        raw = json.load(open(OCR_PATH, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — the file is edited by hand/script
        print(f"!! ocr.json не читается ({e}); OCR-посты не добавляются",
              file=sys.stderr)
        return {}
    out = {}
    for k, v in raw.items():
        try:
            if str(int(k)) == k and (v.get("text") or "").strip():
                out[int(k)] = {"date": v.get("date") or "",
                               "author": v.get("author") or "",
                               "text": v["text"].strip()}
        except (AttributeError, TypeError, ValueError):
            print(f"!! ocr.json: запись {k!r} не по схеме, пропущена",
                  file=sys.stderr)
    return out


def load_splits():
    """{post id: [blocks]} or {}. Keys with "_" are service keys, not
    splits.

    A corrupt file (a hand edit, a truncation) does not kill the bot:
    degrade to uncut glued posts with a log record.
    """
    if not os.path.exists(SPLITS_PATH):
        return {}
    try:
        raw = json.load(open(SPLITS_PATH, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — the manifest is hand-edited, a crash is unacceptable
        print(f"!! splits.json не читается ({e}); склейки не режутся",
              file=sys.stderr)
        return {}
    out = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        try:
            ok = str(int(k)) == k and isinstance(v, list)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            print(f"!! splits.json: запись {k!r} не по схеме (int-ключ, list-блоки), "
                  f"пропущена", file=sys.stderr)
            continue
        out[k] = v
    return out


def split_child_ids():
    """Child ids of all glued posts as strings (the normalization key format)."""
    return {f"{int(pid) * 100 + i}"
            for pid, blocks in load_splits().items()
            for i in range(1, len(blocks) + 1)}


def apply_splits(posts):
    """Posts with glued entries cut into children (id*100 + No).

    The children's texts live in the manifest (curated content):
    corpus_meta.json after index.save already contains the children, and
    reapplying is not just normal — it is the delivery mechanism for an
    edit: a child's text is refreshed from the manifest, and index
    incrementality (post_id + text) re-embeds the vector.
    """
    splits = load_splits()
    if not splits:
        return posts
    child_text = {int(pid) * 100 + i: b["text"].strip()
                  for pid, blocks in splits.items()
                  for i, b in enumerate(blocks, 1)}
    out, seen = [], set()
    for p in posts:
        seen.add(p["id"])
        blocks = splits.get(str(p["id"]))
        if blocks:
            out += [{"id": p["id"] * 100 + i, "date": p["date"],
                     "author": p["author"], "text": b["text"].strip()}
                    for i, b in enumerate(blocks, 1)]
        elif p["id"] in child_text:
            out.append({**p, "text": child_text[p["id"]]})
        else:
            out.append(p)
    # keep the warning for a real curator typo only: the parent is
    # missing AND the children did not appear
    missing = [k for k in splits
               if int(k) not in seen and int(k) * 100 + 1 not in seen]
    if missing:
        print(f"!! splits.json: посты {missing} не найдены в корпусе, "
              f"сплиты по ним не применены", file=sys.stderr)
    return out


def flatten(t):
    """text in the export is sometimes a string, sometimes a list of {'text': ...} segments."""
    if t is None:
        return ""
    if isinstance(t, str):
        return t
    out = []
    for seg in t:
        out.append(seg.get("text", "") if isinstance(seg, dict) else str(seg))
    return "".join(out)


def _parse_export(data):
    ocr = load_ocr()
    posts = []
    for m in data.get("messages", []):
        if m.get("type") != "message":
            continue
        text = flatten(m.get("text")).strip()
        if not text:
            o = ocr.get(m.get("id"))  # an image-only post recognized by OCR
            if o:
                text = o["text"]
        if not text:
            continue
        posts.append({
            "id": m.get("id"),
            "date": (m.get("date") or "")[:10],
            "author": m.get("forwarded_from") or "",
            "text": text,
        })
    return posts


def merge_ocr(posts):
    """Posts + image-only OCR records missing from the corpus.

    The server corpus (corpus_meta.json) contains no image-only posts:
    after a deploy with ocr.json they are added here, and index
    incrementality (post_id + text) re-embeds the vectors on the very
    first start. The text of an existing OCR post is also taken from
    ocr.json: OCR fixes (recognition garbage in the text) reach the
    server corpus without hand-editing corpus_meta.json — like the
    children's texts in splits.json.
    """
    ocr = load_ocr()
    if not ocr:
        return posts
    out, seen = [], set()
    for p in posts:
        o = ocr.get(p["id"])
        if o and o["text"] != p.get("text"):
            p = {**p, "text": o["text"]}
        out.append(p)
        seen.add(p["id"])
    for pid in sorted(ocr):
        if pid not in seen:
            out.append({"id": pid, **ocr[pid]})
    return out


def load_posts(path=None):
    if path is None:
        path = DEFAULT_PATH
        if not os.path.exists(path):
            with open(META_PATH, encoding="utf-8") as f:
                meta = json.load(f)
            posts = meta["posts"] if isinstance(meta, dict) else meta
            return merge_ocr(apply_splits(posts))
    with open(path, encoding="utf-8") as f:
        return apply_splits(_parse_export(json.load(f)))


if __name__ == "__main__":
    posts = load_posts()
    print(f"постов с текстом: {len(posts)}")
    for p in posts[:3]:
        head = f"[{p['id']}] {p['date']}"
        if p["author"]:
            head += f" · {p['author']}"
        print(f"\n{head}\n{p['text'][:200]}")
