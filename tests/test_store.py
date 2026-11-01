"""The SQLite store: migration from the json master, idempotency,
counters."""
import json

import load_corpus
import normalize
import store


def _write_master(tmp_path, posts, items=None):
    # META_PATH — only through the attribute: importing the value
    # captures the path at module load, monkeypatching
    # load_corpus.META_PATH then has no effect, and the test silently
    # writes into the real corpus_meta.json (happened 2026-09-24).
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
    assert store.migrate()["migrated"] is False  # the base is nonempty — the master is not read


def test_upsert_updates_only_changed(tmp_data):
    _write_master(tmp_data, [{"id": 1, "date": "", "author": "", "text": "было"}])
    store.migrate()
    con = store.connect()
    n = store.upsert(con, [{"id": 1, "date": "", "author": "", "text": "было"}])
    assert n == 0  # same text — no row
    n = store.upsert(con, [{"id": 1, "date": "", "author": "", "text": "стало"}])
    assert n == 1
    row = con.execute("SELECT text FROM posts WHERE post_id=1").fetchone()
    assert row == ("стало",)
    con.close()


def test_migrate_without_master(tmp_data):
    assert store.migrate() == {"migrated": False, "posts": 0}


def test_usage_telemetry(tmp_data):
    """Dealt cards go into events: unique users and a counter."""
    store.log_event(1, 101)
    store.log_event(1, 102)
    store.log_event(2, 101)
    u = store.usage()
    assert u == {"уникальных пользователей": 2, "карточек выдано": 3}


def test_events_painting_migration(tmp_data):
    """A base created before the painting_id column is extended on
    connect; old rows live on with NULL."""
    con = store.connect()  # already with the connect-time migration
    con.execute("DROP TABLE events")
    con.execute("CREATE TABLE events(ts TEXT, user_id INTEGER, post_id INTEGER)")
    con.commit()
    con.close()
    con = store.connect()  # the migration adds the column
    cols = [r[1] for r in con.execute("PRAGMA table_info(events)")]
    assert "painting_id" in cols
    con.execute("INSERT INTO events(ts, user_id, post_id) VALUES('t', 1, 9)")
    con.commit()
    con.close()
    store.log_event(1, 10, "bosch-haywain")  # doesn't break on the old row
    assert store.shown_paintings(1) == {"bosch-haywain"}


def test_shown_posts_and_paintings(tmp_data):
    """The shown-memory: all posts, but only paintings recorded with an
    id."""
    store.log_event(1, 101, "bosch-haywain")
    store.log_event(1, 102, "caravaggio-narcissus")
    store.log_event(1, 103)  # no painting recorded — NULL
    store.log_event(2, 101, "ge-truth")
    assert store.shown_posts(1) == {101, 102, 103}
    assert store.shown_paintings(1) == {"bosch-haywain", "caravaggio-narcissus"}
    assert store.shown_paintings(2) == {"ge-truth"}
    assert store.shown_posts(3) == set()
    assert store.shown_paintings(3) == set()


def test_shown_paintings_limit_keeps_freshest(tmp_data):
    """The limit takes the freshest shows, not the first ones."""
    for k in range(5):
        store.log_event(1, 100 + k, f"p{k}")
    assert store.shown_paintings(1, limit=2) == {"p3", "p4"}


def test_state_roundtrip(tmp_data):
    """Key-value for service state (last_channel_id)."""
    assert store.get_state("last_channel_id") is None
    store.set_state("last_channel_id", "100")
    assert store.get_state("last_channel_id") == "100"
    store.set_state("last_channel_id", "105")  # overwrite
    assert store.get_state("last_channel_id") == "105"


def test_rate_incr_counts_and_resets_by_day(tmp_data):
    """The daily counter: grows during the day, a new day starts from
    zero."""
    assert store.rate_incr(1, day="2026-09-25") == 1
    assert store.rate_incr(1, day="2026-09-25") == 2
    assert store.rate_incr(2, day="2026-09-25") == 1  # counters are separate
    assert store.rate_incr(1, day="2026-09-26") == 1  # a new day — from zero
    assert store.rate_incr(1, day="2026-09-25") == 3  # yesterday's keeps counting


def test_msk_day_format(tmp_data):
    import re
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", store.msk_day())


def test_log_llm_and_spend(tmp_data):
    """LLM calls are written with tokens and cost; spend sums per
    action."""
    store.log_llm("qwen3-embedding-4b", "embedding",
                  {"prompt_tokens": 28, "completion_tokens": 0, "cost": 6e-7})
    store.log_llm("deepseek-v4-flash", "explain",
                  {"prompt_tokens": 900, "completion_tokens": 500, "cost": 0.001})
    store.log_llm("deepseek-v4-flash", "explain",
                  {"prompt_tokens": 800, "completion_tokens": 400, "cost": 0.0008})
    s = store.spend()
    assert s["embedding"] == {"вызовов": 1, "$": 0.0}  # 6e-7 rounds to 0.0
    assert s["explain"]["вызовов"] == 2
    assert abs(s["explain"]["$"] - 0.0018) < 1e-9


def test_log_llm_empty_usage(tmp_data):
    """An empty usage doesn't crash: the call counts, the cost is 0."""
    store.log_llm("m", "a", None)
    assert store.spend() == {"a": {"вызовов": 1, "$": 0}}


def test_spend_other_month_excluded(tmp_data):
    """Last month's spend doesn't get into the current month's
    summary."""
    store.log_llm("m", "a", {"cost": 1.0})
    con = store.connect()
    con.execute("UPDATE llm_calls SET day='2020-01-15'")
    con.commit()
    con.close()
    assert store.spend() == {}


def test_spend_log_call_swallows_failure(tmp_data, monkeypatch):
    """spend.log_call swallows a write failure — telemetry never kills
    the request."""
    import spend
    monkeypatch.setattr(store, "log_llm", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("db locked")))
    spend.log_call("m", "a", {"cost": 1})  # didn't raise
    spend.log_call("m", "a", None)  # empty usage — no write at all
