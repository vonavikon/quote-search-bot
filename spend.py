"""Учёт расходов на LLM. Телеметрия не должна ломать путь запроса:
любая ошибка записи глотается с логом, вызов живёт дальше."""
import logging

log = logging.getLogger("quote-search.spend")


def log_call(model, action, usage):
    """usage — словарь от bothub (токены + cost); None/пустой — не пишем."""
    if not usage:
        return
    try:
        import store
        store.log_llm(model, action, usage)
    except Exception:  # noqa: BLE001 — расходы не роняют поиск
        log.exception("расход не записан: %s/%s", model, action)
