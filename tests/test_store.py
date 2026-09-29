"""SQLite-хранилище: миграция из json-мастера, идемпотентность, счётчики."""
import json

import load_corpus
import normalize
import store


def _write_master(tmp_path, posts, items=None):
    # META_PATH — только через атрибут: импорт значения захватывает путь при
    # загрузке модуля, monkeypatch load_corpus.META_PATH тогда не действует,
    # и тест молча пишет в реальный corpus_meta.json (случилось 2026-09-24).
    with open(load_corpus.META_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": "test", "posts": posts}, f, ensure_ascii=False)
    with open(normalize.CACHE, "w", encoding="utf-8") as f:
        json.dump({"model": "test", "items": items or {}}, f, ensure_ascii=False)


def test_migrate_seeds_from_json(tmp_data):
    _write_master(tmp_data, [
        {"id": 101, "date": "2024-01-01", "author": "", "text": "про любовь"},
        {"id": 102, "date": "2024-01-02", "author": "Канал", "text": "про смерть"},
    ], {"101": {"kind": "quote", "author": "Пастернак", "source": None}})
    r = store.migrate()
    assert r == {"migrated": True, "posts": 2}
    c = store.counts()
    assert c["всего"] == 2 and c["автор подписан"] == 1 \
        and c["публичных (quote/poem)"] == 1
    con = store.connect()
    row = con.execute("SELECT text, author, kind FROM posts WHERE post_id=101").fetchone()
    assert row == ("про любовь", "Пастернак", "quote")
    con.close()


def test_migrate_idempotent(tmp_data):
    _write_master(tmp_data, [{"id": 1, "date": "", "author": "", "text": "a"}])
    assert store.migrate()["migrated"] is True
    assert store.migrate()["migrated"] is False  # база непуста — мастер не читаем


def test_upsert_updates_only_changed(tmp_data):
    _write_master(tmp_data, [{"id": 1, "date": "", "author": "", "text": "было"}])
    store.migrate()
    con = store.connect()
    n = store.upsert(con, [{"id": 1, "date": "", "author": "", "text": "было"}])
    assert n == 0  # текст тот же — строки нет
    n = store.upsert(con, [{"id": 1, "date": "", "author": "", "text": "стало"}])
    assert n == 1
    row = con.execute("SELECT text FROM posts WHERE post_id=1").fetchone()
    assert row == ("стало",)
    con.close()


def test_migrate_without_master(tmp_data):
    assert store.migrate() == {"migrated": False, "posts": 0}


def test_usage_telemetry(tmp_data):
    """Выдачи карточек пишутся в events: уникальные пользователи и счётчик."""
    store.log_event(1, 101)
    store.log_event(1, 102)
    store.log_event(2, 101)
    u = store.usage()
    assert u == {"уникальных пользователей": 2, "карточек выдано": 3}


def test_events_painting_migration(tmp_data):
    """База, созданная до колонки painting_id, расширяется при connect,
    старые записи живут дальше с NULL."""
    con = store.connect()  # уже с миграцией из connect
    con.execute("DROP TABLE events")
    con.execute("CREATE TABLE events(ts TEXT, user_id INTEGER, post_id INTEGER)")
    con.commit()
    con.close()
    con = store.connect()  # миграция докатывает колонку
    cols = [r[1] for r in con.execute("PRAGMA table_info(events)")]
    assert "painting_id" in cols
    con.execute("INSERT INTO events(ts, user_id, post_id) VALUES('t', 1, 9)")
    con.commit()
    con.close()
    store.log_event(1, 10, "bosch-haywain")  # не упало на старой строке
    assert store.shown_paintings(1) == {"bosch-haywain"}


def test_shown_posts_and_paintings(tmp_data):
    """Память показов: посты все, картины только записанные с id."""
    store.log_event(1, 101, "bosch-haywain")
    store.log_event(1, 102, "caravaggio-narcissus")
    store.log_event(1, 103)  # картина не записалась — NULL
    store.log_event(2, 101, "ge-truth")
    assert store.shown_posts(1) == {101, 102, 103}
    assert store.shown_paintings(1) == {"bosch-haywain", "caravaggio-narcissus"}
    assert store.shown_paintings(2) == {"ge-truth"}
    assert store.shown_posts(3) == set()
    assert store.shown_paintings(3) == set()


def test_shown_paintings_limit_keeps_freshest(tmp_data):
    """Лимит берёт самые свежие показы, а не первые."""
    for k in range(5):
        store.log_event(1, 100 + k, f"p{k}")
    assert store.shown_paintings(1, limit=2) == {"p3", "p4"}


def test_state_roundtrip(tmp_data):
    """Ключ-значение для служебного состояния (last_channel_id)."""
    assert store.get_state("last_channel_id") is None
    store.set_state("last_channel_id", "100")
    assert store.get_state("last_channel_id") == "100"
    store.set_state("last_channel_id", "105")  # перезапись
    assert store.get_state("last_channel_id") == "105"


def test_rate_incr_counts_and_resets_by_day(tmp_data):
    """Дневной счётчик: растёт в течение дня, новый день — с нуля."""
    assert store.rate_incr(1, day="2026-09-25") == 1
    assert store.rate_incr(1, day="2026-09-25") == 2
    assert store.rate_incr(2, day="2026-09-25") == 1  # счётчики раздельные
    assert store.rate_incr(1, day="2026-09-26") == 1  # новый день — с нуля
    assert store.rate_incr(1, day="2026-09-25") == 3  # вчерашний продолжает счёт


def test_msk_day_format(tmp_data):
    import re
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", store.msk_day())


def test_log_llm_and_spend(tmp_data):
    """LLM-вызовы пишутся с токенами и ценой; spend суммирует по действию."""
    store.log_llm("qwen3-embedding-4b", "embedding",
                  {"prompt_tokens": 28, "completion_tokens": 0, "cost": 6e-7})
    store.log_llm("deepseek-v4-flash", "explain",
                  {"prompt_tokens": 900, "completion_tokens": 500, "cost": 0.001})
    store.log_llm("deepseek-v4-flash", "explain",
                  {"prompt_tokens": 800, "completion_tokens": 400, "cost": 0.0008})
    s = store.spend()
    assert s["embedding"] == {"вызовов": 1, "$": 0.0}  # 6e-7 округляется в 0.0
    assert s["explain"]["вызовов"] == 2
    assert abs(s["explain"]["$"] - 0.0018) < 1e-9


def test_log_llm_empty_usage(tmp_data):
    """Пустой usage не падает: вызов считается, цена 0."""
    store.log_llm("m", "a", None)
    assert store.spend() == {"a": {"вызовов": 1, "$": 0}}


def test_spend_other_month_excluded(tmp_data):
    """Расходы прошлого месяца в сводку текущего не попадают."""
    store.log_llm("m", "a", {"cost": 1.0})
    con = store.connect()
    con.execute("UPDATE llm_calls SET day='2020-01-15'")
    con.commit()
    con.close()
    assert store.spend() == {}


def test_spend_log_call_swallows_failure(tmp_data, monkeypatch):
    """spend.log_call глотает сбой записи — телеметрия не роняет запрос."""
    import spend
    monkeypatch.setattr(store, "log_llm", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("db locked")))
    spend.log_call("m", "a", {"cost": 1})  # не поднялся
    spend.log_call("m", "a", None)  # пустой usage — вообще без записи
