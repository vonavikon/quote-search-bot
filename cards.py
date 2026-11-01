"""A card: text + author + source.

The author and the source come only from normalization (the trust rule);
unsigned — the line is dropped, nothing is invented. Long-quote
thresholds: the Telegram message limit is 4096, headroom for the
signature — cut at 3900 + a "Читать полностью" button; the painting
added the caption threshold of 1024.

Immutable span: the card stores the original post text verbatim; at
display time only projections run on top — the author tail strip and the
deterministic typography of text_clean.
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
    """{post_id: item} or {} (no normalization yet).

    Children of glued posts (splits.json) come in with the manifest's
    curated metadata — it overrides the normalization cache if the LLM
    has already written one for these ids.
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
    """A post + its normalization record -> a card for the deck."""
    item = item or {}
    return {
        "id": post["id"],
        "text": post["text"],
        "author": item.get("author") or None,
        "source": item.get("source") or None,
        "kind": item.get("kind") or "quote",
    }


SIG_LINE_MAX = 80  # a signature is a short line; a long match is content


def _sig_word(s):
    """The longest meaningful word of the signature (for an author it is
    usually the surname). Short words («Петр») catch false matches inside
    quote bodies."""
    ws = [w for w in re.split(r"[^\w]+", _norm(s)) if len(w) >= 4]
    return max(ws, key=len) if ws else None


def strip_author_tail(c):
    """The post text without the trailing author and source signatures.
    The channel signs posts with the last lines («— Достоевский»,
    «Книга»); the card renders the author and the source on separate
    lines, so the body (and the quote's «quotes» too) must not contain
    the signatures. We cut from the bottom, up to 3 short lines
    (<= 80 chars) matching the author's surname or the source title
    (е/ё-insensitive). Empty after the cut — return the original."""
    text = c["text"]
    keys = {w for w in (_sig_word(c.get("author") or ""),
                        _sig_word(c.get("source") or "")) if w}
    if not keys:
        return text
    lines = text.strip().splitlines()
    removed = 0
    while lines and removed < 3:
        line = lines[-1]
        if not line.strip():  # a blank line after a cut signature — out too
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
    """The card text. full=True — no cutting (the "Читать полностью"
    button). Quotes (kind=quote) are shown in «quotes»; poems — as is."""
    body = text_clean.apply(strip_author_tail(c))
    if not full and len(body) > CUT:
        body = body[:CUT].rstrip() + "…"
    if c.get("kind") == "quote":
        # the channel often adds quotes itself: its own pair + the card
        # wrapper produced ««double»» quotes. We remove the outermost
        # pair whole — the opening « and its matching » by nesting depth:
        # the wrapper adds exactly the same pair back, so parity is
        # unchanged
        core = body.strip()
        if core.startswith("«"):
            depth = 0
            for i, ch in enumerate(core):
                if ch == "«":
                    depth += 1
                elif ch in ("»", "“"):  # „…“ also occurs in the sources
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
