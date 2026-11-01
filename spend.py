"""LLM spend tracking. Telemetry must never break the request path: any
write error is swallowed with a log, the call lives on."""
import logging

log = logging.getLogger("quote-search.spend")


def log_call(model, action, usage):
    """usage — a dict from bothub (tokens + cost); None/empty — not written."""
    if not usage:
        return
    try:
        import store
        store.log_llm(model, action, usage)
    except Exception:  # noqa: BLE001 — spend must not kill the search
        log.exception("расход не записан: %s/%s", model, action)
