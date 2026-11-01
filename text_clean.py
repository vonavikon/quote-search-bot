"""Display typography.

The corpus source is unchanged (immutable span): text_clean is applied
only when rendering a card, as a pure deterministic function. The rules
are conservative (length never grows, cut thresholds stay honest):

  1. Straight double quotes "..." -> «...» (paired, alternating over the
     whole text: a multi-line quote opens on one line and closes on
     another — per-line pairing broke such posts).
  2. Inter-word dash « - » -> « — » (a hyphen inside words is untouched).
  3. Trailing spaces on lines are cut (invisible in Telegram, 40% of
     the corpus).

Apostrophes, ellipses and the rest are untouched: minimalism beats
perfection.
"""
import re

_DASH = re.compile(r"(?<=\S) - (?=\S)")


def apply(text):
    """A typographic projection of the text for display. Pure, no network
    and no LLM."""
    out = []
    open_quote = True  # pairing over the whole text: multi-line quotes
    for ch in text:
        if ch == '"':
            out.append("«" if open_quote else "»")
            open_quote = not open_quote
        else:
            out.append(ch)
    lines = [l.rstrip() for l in "".join(out).split("\n")]
    return "\n".join(_DASH.sub(" — ", l) for l in lines)
