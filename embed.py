"""bothub /embeddings client. An OpenAI-compatible gateway,
https://openai.bothub.chat/v1.

The key comes from BOTHUB_API_KEY (env) or a .env next to the code.
Never print it or commit it.
"""
import json
import os
import re
import urllib.request

BASE = "https://openai.bothub.chat/v1"
MODEL = "qwen3-embedding-4b"  # picked by a recall@5 measurement, dim 2560

# $/token as measured on 2026-09-25 (28 query tokens = $0.0000006); the
# embeddings usage from bothub carries no cost — we append it ourselves
# for /stats
PM = 0.022 / 1e6

# Qwen3-Embedding is an asymmetric instruction-tuned model: documents are
# embedded as is, the query — with a task prefix. Without the prefix the
# query embedding loses its search meaning: recall@5 drops 0.556 -> 0.069.
QUERY_PREFIX = ("Instruct: Given a user topic query, "
                "retrieve relevant quotes/passages on that theme.\nQuery: ")


def _key() -> str:
    k = os.environ.get("BOTHUB_API_KEY")
    if k:
        return k.strip()
    candidates = [
        os.path.join(os.path.dirname(__file__), ".env"),  # the project's local .env
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"^\s*BOTHUB_API_KEY\s*=\s*(.*)$", line)
                    if m:
                        return m.group(1).strip().strip('"').strip("'")
    raise RuntimeError("BOTHUB_API_KEY не найден ни в env, ни в .env")


def embed(text, model=MODEL):
    """text: str | list[str] -> (list[list[float]], usage).

    Always returns a list of vectors (even for a single text) + usage
    for cost tracking.
    """
    inp = [text] if isinstance(text, str) else text
    body = json.dumps({"model": model, "input": inp}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/embeddings",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_key()}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    vecs = sorted(data["data"], key=lambda d: d["index"])
    usage = data.get("usage") or {}
    if "cost" not in usage and usage.get("prompt_tokens") is not None:
        usage = {**usage, "cost": usage["prompt_tokens"] * PM}
    import spend
    spend.log_call(model, "embedding", usage)
    return [v["embedding"] for v in vecs], usage


def embed_query(query, model=MODEL):
    """The embedding of a search query (str or list[str]). For
    qwen3-embedding it prepends the instruction prefix (documents are
    embedded without it, see QUERY_PREFIX)."""
    if model.startswith("qwen3-embedding"):
        pre = lambda s: s if s.startswith("Instruct:") else QUERY_PREFIX + s  # noqa: E731
        query = pre(query) if isinstance(query, str) else [pre(s) for s in query]
    return embed(query, model=model)
