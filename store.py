"""SQLite — источник истины для корпуса и метаданных.

Таблица posts идемпотентна по post_id: повторная миграция/ингест не меняет
строку, если текст и нормализация не изменились. Поиск по-прежнему идёт по
векторам (corpus.npz) и corpus_meta.json — SQLite хранит канон: что есть
в базе и с какими метками, включая историю обновлений (updated_at).

Путь к базе берётся из load_corpus.DATA_DIR в момент вызова (не при импорте:
иначе дефолт аргумента фиксирует путь до патча теста).
"""
import os
import sqlite3
from datetime import datetime, timezone, timedelta

import load_corpus

MSK = timezone(timedelta(hours=3))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts(
  post_id        INTEGER PRIMARY KEY,
  date           TEXT,
  fwd_author     TEXT,
  text           TEXT,
  kind           TEXT,
  author         TEXT,
  source         TEXT,
  author_evidence TEXT,
  updated_at     TEXT
);
CREATE TABLE IF NOT EXISTS events(
  ts     TEXT,
  user_id INTEGER,
  post_id INTEGER,
  painting_id TEXT
);
CREATE TABLE IF NOT EXISTS state(
  k TEXT PRIMARY KEY,
  v TEXT
);
CREATE TABLE IF NOT EXISTS rate(
  user_id INTEGER,
  day     TEXT,
  n       INTEGER,
  PRIMARY KEY(user_id, day)
);
CREATE TABLE IF NOT EXISTS llm_calls(
  day     TEXT,
  ts      TEXT,
  model   TEXT,
  action  TEXT,
  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  cost    REAL
)
"""


def db_path():
    return os.path.join(load_corpus.DATA_DIR, "quotes.db")


def connect():
    con = sqlite3.connect(db_path())
    con.executescript(_SCHEMA)
    # CREATE IF NOT EXISTS не расширяет существующую таблицу — у базы,
    # живущей с первой версии, нет painting_id (добавлен 2026-09-28 для
    # неповтора картин между колодами)
    cols = [r[1] for r in con.execute("PRAGMA table_info(events)")]
    if "painting_id" not in cols:
        con.execute("ALTER TABLE events ADD COLUMN painting_id TEXT")
        con.commit()
    return con


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert(con, posts, items=None):
    """Посты (+ записи нормализации по str(post_id)) -> SQLite. -> число изменённых."""
    items = items or {}
    changed = 0
    for p in posts:
        it = items.get(str(p["id"]), {})
        row = (p["id"], p.get("date") or "", p.get("author") or "", p["text"],
               it.get("kind"), it.get("author"), it.get("source"),
               it.get("author_evidence"))
        cur = con.execute(
            """INSERT INTO posts(post_id, date, fwd_author, text, kind, author,
                                 source, author_evidence, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(post_id) DO UPDATE SET
                 date=excluded.date, fwd_author=excluded.fwd_author,
                 text=excluded.text, kind=excluded.kind, author=excluded.author,
                 source=excluded.source, author_evidence=excluded.author_evidence,
                 updated_at=excluded.updated_at
               WHERE posts.text IS NOT excluded.text
                  OR posts.date IS NOT excluded.date
                  OR posts.fwd_author IS NOT excluded.fwd_author
                  OR posts.kind IS NOT excluded.kind
                  OR posts.author IS NOT excluded.author
                  OR posts.source IS NOT excluded.source
                  OR posts.author_evidence IS NOT excluded.author_evidence""",
            row + (_now(),))
        changed += cur.rowcount if cur.rowcount > 0 else 0
    con.commit()
    return changed


def migrate(con=None):
    """Синхронизирует базу с corpus_meta.json + normalization.json.

    Идемпотентный зеркальный синк на каждом старте: дошил посты (включая
    детей склеек — их метаданные из cards.load_norm()), удалил исчезнувшие
    id (разобранные склейки-родители). -> {"migrated": bool, "posts": n},
    migrated=True, если база была пуста (первый посев).
    """
    own = con is None
    con = con or connect()
    try:
        (n,) = con.execute("SELECT count(*) FROM posts").fetchone()
        if not os.path.exists(load_corpus.META_PATH):
            return {"migrated": False, "posts": n}
        import cards
        posts = load_corpus.load_posts()
        upsert(con, posts, cards.load_norm())
        keep = [p["id"] for p in posts]
        if keep:
            ph = ",".join("?" * len(keep))
            con.execute(f"DELETE FROM posts WHERE post_id NOT IN ({ph})", keep)
        else:  # пустой корпус: NOT IN () — синтаксическая ошибка SQL
            con.execute("DELETE FROM posts")
        con.commit()
        return {"migrated": not n, "posts": len(posts)}
    finally:
        if own:
            con.close()


def counts(con=None):
    """Сводка для /stats: всего, подписанных авторов, литературных kind."""
    own = con is None
    con = con or connect()
    try:
        total, = con.execute("SELECT count(*) FROM posts").fetchone()
        signed, = con.execute(
            "SELECT count(*) FROM posts WHERE author IS NOT NULL").fetchone()
        public, = con.execute(
            "SELECT count(*) FROM posts WHERE kind IN ('quote','poem')").fetchone()
        return {"всего": total, "автор подписан": signed,
                "публичных (quote/poem)": public}
    finally:
        if own:
            con.close()


def log_event(user_id, post_id, painting_id=None, con=None):
    """Выдана карточка пользователю (телеметрия /stats + память показов:
    пост и картина). Append-only."""
    own = con is None
    con = con or connect()
    try:
        con.execute(
            "INSERT INTO events(ts, user_id, post_id, painting_id) "
            "VALUES(?,?,?,?)", (_now(), user_id, post_id, painting_id))
        con.commit()
    finally:
        if own:
            con.close()


# сколько последних картин пользователя исключается из подбора: достаточно,
# чтобы развернуть концентрацию семантической карты (топ-5 картин закрывают
# четверть слотов), и меньше пула — перебор вперёд всегда находит свободную
RECENT_PAINTINGS = 150


def shown_posts(user_id, con=None):
    """Посты, уже выданные пользователю. -> set (память неповтора для
    «похожей» и Sic; у новичка пуст)."""
    own = con is None
    con = con or connect()
    try:
        return {p for (p,) in con.execute(
            "SELECT DISTINCT post_id FROM events WHERE user_id=?", (user_id,))}
    finally:
        if own:
            con.close()


def shown_paintings(user_id, limit=RECENT_PAINTINGS, con=None):
    """Последние картины, выданные пользователю (limit штук по свежести).
    -> set id картин — exclude для images.pick."""
    own = con is None
    con = con or connect()
    try:
        return {p for (p,) in con.execute(
            "SELECT painting_id FROM events WHERE user_id=? "
            "AND painting_id IS NOT NULL ORDER BY rowid DESC LIMIT ?",
            (user_id, limit))}
    finally:
        if own:
            con.close()


def usage(con=None):
    """Аудитория: уникальные пользователи и число выданных карточек."""
    own = con is None
    con = con or connect()
    try:
        users, = con.execute("SELECT count(DISTINCT user_id) FROM events").fetchone()
        shown, = con.execute("SELECT count(*) FROM events").fetchone()
        return {"уникальных пользователей": users, "карточек выдано": shown}
    finally:
        if own:
            con.close()


def get_state(k, con=None):
    """Значение ключа из state (last_channel_id и прочее). |None."""
    own = con is None
    con = con or connect()
    try:
        row = con.execute("SELECT v FROM state WHERE k=?", (k,)).fetchone()
        return row[0] if row else None
    finally:
        if own:
            con.close()


def set_state(k, v, con=None):
    own = con is None
    con = con or connect()
    try:
        con.execute("INSERT INTO state(k, v) VALUES(?,?) "
                    "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
        con.commit()
    finally:
        if own:
            con.close()


def msk_day():
    """Текущий день по Москве (дневные лимиты живут по часовой зоне аудитории)."""
    return datetime.now(MSK).strftime("%Y-%m-%d")


def rate_count(user_id, day=None, con=None):
    """Сколько платящих действий у пользователя за день. Только чтение,
    счётчик не инкрементирует — для проверки ДО вызова route: LLM-переформулировка
    не должна звестись у исчерпавшего лимит (сам лимит знает бот, не стор)."""
    day = day or msk_day()
    own = con is None
    con = con or connect()
    try:
        row = con.execute("SELECT n FROM rate WHERE user_id=? AND day=?",
                          (user_id, day)).fetchone()
        return row[0] if row else 0
    finally:
        if own:
            con.close()


def rate_incr(user_id, day=None, con=None):
    """+1 к дневному счётчику действий пользователя. -> новое значение счётчика."""
    day = day or msk_day()
    own = con is None
    con = con or connect()
    try:
        con.execute(
            "INSERT INTO rate(user_id, day, n) VALUES(?,?,1) "
            "ON CONFLICT(user_id, day) DO UPDATE SET n = n + 1",
            (user_id, day))
        con.commit()
        n, = con.execute("SELECT n FROM rate WHERE user_id=? AND day=?",
                         (user_id, day)).fetchone()
        return n
    finally:
        if own:
            con.close()


def log_llm(model, action, usage, con=None):
    """LLM-вызов в копилку расходов (usage от bothub: токены + cost). Append-only."""
    own = con is None
    con = con or connect()
    try:
        con.execute(
            "INSERT INTO llm_calls(day, ts, model, action, prompt_tokens, "
            "completion_tokens, cost) VALUES(?,?,?,?,?,?,?)",
            (msk_day(), _now(), model, action,
             (usage or {}).get("prompt_tokens"),
             (usage or {}).get("completion_tokens"),
             (usage or {}).get("cost")))
        con.commit()
    finally:
        if own:
            con.close()


def spend(con=None):
    """Расходы за текущий месяц по МСК: {action: {"вызовов": n, "$": сумма}}."""
    own = con is None
    con = con or connect()
    try:
        month = datetime.now(MSK).strftime("%Y-%m")
        out = {}
        for action, n, cost in con.execute(
                "SELECT action, count(*), sum(cost) FROM llm_calls "
                "WHERE day LIKE ? GROUP BY action", (month + "%",)):
            out[action] = {"вызовов": n, "$": round(cost or 0, 4)}
        return out
    finally:
        if own:
            con.close()


if __name__ == "__main__":
    print(migrate())
    print(counts())
