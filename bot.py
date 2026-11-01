"""Telegram bot that searches quotes in its own corpus: cards with a
painting, random deals, reference cards about authors, live ingest from
the channel. Long polling (a webhook needs a domain and a TLS route, not
needed here). Run: python -X utf8 bot.py.

Env (or a .env next to this file): TG_BOT_TOKEN, ADMIN_TG_ID.
"""
import asyncio
import html
import json
import logging
import os
import random
import re
import sys

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import FSInputFile, CallbackQuery, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

import cards
import images
import ingest
import reference
import store
from load_corpus import DATA_DIR, load_posts
from router import PUBLIC_KINDS, route, search_cards

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("quote-search")


def _env(name):
    v = os.environ.get(name)
    if v:
        return v.strip()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            m = re.search(rf"^\s*{name}\s*=\s*(.+)$", f.read(), re.M)
            if m:
                return m.group(1).strip()
    return None


TOKEN = _env("TG_BOT_TOKEN")
ADMIN = int(_env("ADMIN_TG_ID") or 0)
DECKS = {}          # key -> {"cards": [...], "query": str}
DECK_CAP = 300
CAPTION_MAX = 1000  # photo caption limit is 1024; headroom for markup
DAY_LIMIT = 100     # safety daily cap of paid (LLM) actions per user

RATE_LIMIT_TEXT = ("На сегодня всё: лимит 100 запросов исчерпан. "
                   "Возвращайся завтра — цитаты никуда не денутся.")

INTRO = (
    "Привет! Этот бот ищет по коллекции цитат, стихов и афоризмов: "
    "напиши тему, имя автора или фразу в кавычках.\n\n"
    "Поиск работает по ближайшим ассоциациям и смыслам. "
    "Пример: «мысль о дружбе»."
)
INTRO_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "intro.jpg")


async def send_intro(m: Message, bot: Bot):
    """/start and intro: a picture with the greeting (intro.jpg if present)
    + the Sic button. We cache file_id — after the first send only the id
    travels, no upload. No file — plain text."""
    fid = images.get_file_id("intro")
    if fid or os.path.exists(INTRO_IMG):
        media = fid or FSInputFile(INTRO_IMG)
        sent = await m.answer_photo(media, caption=INTRO, parse_mode="HTML",
                                    reply_markup=sic_kb())
        if not fid and getattr(sent, "photo", None):
            images.save_file_id("intro", sent.photo[-1].file_id)
        return
    await m.answer(INTRO, parse_mode="HTML", reply_markup=sic_kb(),
                   disable_web_page_preview=True)

# Clarification for one-word queries: when the result is empty we offer
# theme buttons. Facets are tags from the painting manifest (deterministic,
# no LLM).
THEMES = ("одиночество", "любовь", "смерть", "вера", "дорога", "ночь", "море", "свет")


def clarify_kb():
    b = InlineKeyboardBuilder()
    for t in THEMES:
        b.button(text=t.capitalize(), callback_data=f"t:{t}")
    b.button(text="Sic erat in fatis", callback_data="sic")  # always at hand
    b.adjust(4, 4, 1)
    return b.as_markup()


def sic_kb():
    """Random-card button — in /start and the intro, as text."""
    b = InlineKeyboardBuilder()
    b.button(text="Sic erat in fatis", callback_data="sic")
    return b.as_markup()


# ---------- feedback: the "typing..." indicator for the whole search ----------

async def _keep_typing(bot: Bot, chat_id):
    """A chat action lives ~5 seconds — keep renewing it while the search runs."""
    try:
        while True:
            await bot.send_chat_action(chat_id, "typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


def search_status(path, query):
    """Live search status text. The router has already picked the path; the
    only fib is the word "searching": the embed request inside
    hybrid.retrieve is not shown."""
    how = "дословно" if path == "direct-quoted" else "по смыслу"
    return f"Ищу {how}: «{query}»..."


async def _send_later(m: Message, text: str, box: list, delay: float = 0.9):
    """The search status appears only if the search drags on: fast replies
    don't flash a ghost message. Whatever was sent goes into box."""
    await asyncio.sleep(delay)
    box.append(await m.answer(text))


async def _drop(box: list):
    """Remove the search status if it managed to appear."""
    if box:
        try:
            await box[0].delete()
        except Exception:  # noqa: BLE001 — the status is not critical
            pass


def diversify(cs):
    """Interleaves authors: two cards by the same author never come in a row.

    Groups by author (None — "unsigned" is a group too) and deals them
    round-robin; within a group the relevance order is preserved.
    """
    groups = {}
    for c in cs:
        groups.setdefault(c["author"], []).append(c)
    queues = list(groups.values())
    out = []
    while queues:
        for q in queues[:]:
            out.append(q.pop(0))
            if not q:
                queues.remove(q)
    return out


# ---------- decks and keyboard ----------

def new_deck(cs, query):
    if len(DECKS) >= DECK_CAP:  # FIFO: old decks get evicted
        DECKS.pop(next(iter(DECKS)))
    key = f"{random.getrandbits(48):012x}"
    DECKS[key] = {"cards": cs, "query": query, "used": set()}
    return key


def kb(key, i, truncated=False, similar=False):
    deck = DECKS[key]
    b = InlineKeyboardBuilder()
    rows = []
    if len(deck["cards"]) > 1:
        b.button(text="Ещё цитата", callback_data=f"n:{key}:{i}")
        rows.append(1)
    if truncated:
        b.button(text="Читать полностью", callback_data=f"f:{key}:{i}")
        rows.append(1)
    if similar:
        b.button(text="Похожая цитата", callback_data=f"l:{key}:{i}")
        rows.append(1)
    # reference cards: Wikipedia + corpus data, no free generation (reference)
    b.button(text="Об авторе", callback_data=f"a:{key}:{i}")
    b.button(text="О картине", callback_data=f"p:{key}:{i}")
    rows.append(2)
    b.button(text="Sic erat in fatis", callback_data="sic")
    rows.append(1)
    b.adjust(*rows)
    return b.as_markup()


# ---------- card rendering: painting + text (three length thresholds) ----------

async def card_media(bot: Bot, painting):
    """file_id from the cache (no upload) or the pool file from disk."""
    fid = images.get_file_id(painting["id"])
    return fid or FSInputFile(images.painting_path(painting))


def _remember_file_id(painting, msg):
    """After a send Telegram assigns the file a file_id — we cache it, from
    then on only the id travels."""
    if getattr(msg, "photo", None):
        images.save_file_id(painting["id"], msg.photo[-1].file_id)


def _caption(c, painting, full=False):
    """Message caption: the card text (HTML-escaped) + author + source +
    the painting line in italics. As a separate paragraph with a blank line —
    so the author signature never sticks to the painting caption."""
    body = html.escape(cards.card_text(c, full=full))
    label = f"Картина: {painting['author']}, «{painting['title']}»"
    if painting.get("year"):
        label += f", {painting['year']}"
    return f"{body}\n\n<i>{html.escape(label)}</i>"


async def send_card(m: Message, bot: Bot, c, key, i, similar=False, user_id=None):
    """The card in full. <=1000 chars: painting + caption in one message;
    longer: painting separately, text with buttons in its own message.
    A painting never repeats within a deck or among the ones recently shown
    to the user (images.pick with exclude); the chosen one is remembered in
    the deck — the "About the painting" button looks at it, and both the
    card and the painting go into events."""
    deck = DECKS[key]
    exclude = set(deck["used"])
    if user_id:
        try:
            exclude |= store.shown_paintings(user_id)
        except Exception:  # noqa: BLE001 — the memory is not critical
            log.exception("память картин не прочиталась")
    painting = images.pick(c["id"], deck.get("query", ""), exclude=exclude,
                           text=c.get("text", ""))
    deck["used"].add(painting["id"])
    deck.setdefault("paintings", {})[i] = painting
    caption = _caption(c, painting)
    markup = kb(key, i, truncated=cards.truncated(c), similar=similar)
    media = await card_media(bot, painting)
    if len(caption) <= CAPTION_MAX:
        sent = await m.answer_photo(media, caption=caption, parse_mode="HTML",
                                    reply_markup=markup)
        _remember_file_id(painting, sent)
        sent_card = sent
    else:
        await m.answer_photo(media)
        sent_card = await m.answer(caption, parse_mode="HTML", reply_markup=markup)
    if user_id:
        try:
            store.log_event(user_id, c["id"], painting["id"])
        except Exception:  # noqa: BLE001
            log.exception("телеметрия карточки не записалась")
    return sent_card


def similar_branch(text, not_id, seen):
    """Similar-quotes branch: dense search over the current card's text.
    Skips itself and everything the user has seen (seen); when the nearest
    are exhausted it just walks further down the list — up to 5 unread.
    -> [posts], possibly empty."""
    found = search_cards({"query": text[:400], "author": None}, 8)
    out = []
    for p, _, _ in found:
        if p["id"] == not_id or p["id"] in seen:
            continue
        out.append(p)
        if len(out) >= 5:
            break
    return out


def random_branch(user_id):
    """Random branch: an unshown random post + its nearest neighbors (also
    unshown) — a deck for Sic and the "similar" fallback. Everything read —
    pick from the whole pool. Synchronous (search_cards), called in an
    executor."""
    items = cards.load_norm()
    pool = [p for p in load_posts()
            if items.get(str(p["id"]), {}).get("kind", "quote") in PUBLIC_KINDS]
    try:
        seen = store.shown_posts(user_id) if user_id else set()
    except Exception:  # noqa: BLE001
        seen = set()
    fresh = [p for p in pool if p["id"] not in seen] or pool
    seed = random.choice(fresh)
    out = [seed]
    try:
        found = search_cards({"query": seed["text"][:400], "author": None}, 8)
    except Exception:  # noqa: BLE001 — no neighbors, live on a single post
        found = []
    for p, _, _ in found:
        if p["id"] != seed["id"] and p["id"] not in seen:
            out.append(p)
        if len(out) >= 5:
            break
    return out


# ---------- author and painting reference cards (on button, reference.py) ----------

STREAM_THROTTLE = 0.9  # edit a message no more than ~1/s (edit_text flood limit)


async def send_reference(m: Message, fn, args):
    """Reference card with streaming: assembly runs in a thread pool (a
    synchronous call would freeze the whole bot), the text is appended to
    the message by edits once per STREAM_THROTTLE. Cache hits and stubs
    return instantly — a single final edit, no empty blinking."""
    loop = asyncio.get_running_loop()
    msg = await m.answer("Собираю справку...")
    buf: list = []  # the stream appends chunks, we read them: append is atomic
    task = loop.run_in_executor(None, lambda: fn(*args, on_chunk=buf.append))
    shown = ""
    while not task.done():
        await asyncio.sleep(STREAM_THROTTLE)
        text = "".join(buf)
        if text and text != shown:
            shown = text
            try:
                await msg.edit_text(text[:4000])
            except Exception:  # noqa: BLE001 — not modified or flood
                pass
    try:
        await msg.edit_text((await task)[:4000])
    except Exception:  # noqa: BLE001
        pass


# ---------- handlers ----------

async def on_start(m: Message, bot: Bot):
    await send_intro(m, bot)


def is_admin(m: Message) -> bool:
    return bool(ADMIN) and m.from_user and m.from_user.id == ADMIN


def rate_ok(uid) -> bool:
    """Daily cap on "paid" actions (search / explanation / similar /
    reference cards). Only model calls count; "Ещё цитата", Sic, "Читать
    полностью" are local. The admin has no cap, so debugging doesn't hit
    it."""
    if not uid or uid == ADMIN:
        return True
    return store.rate_incr(uid) <= DAY_LIMIT


def rate_left(uid) -> bool:
    """Limit remainder without an increment — for the check BEFORE route:
    a user who spent the limit must not burn an LLM reformulation.
    intro/admin are detected by the router before the LLM and spend
    nothing; the counter grows only on a paid action (rate_ok inside
    do_search)."""
    if not uid or uid == ADMIN:
        return True
    return store.rate_count(uid) < DAY_LIMIT


async def on_stats(m: Message):
    if not is_admin(m):
        return
    await m.answer(json.dumps({**ingest.stats(), **store.usage(),
                               "расходы за месяц $": store.spend()},
                              ensure_ascii=False, indent=1))


async def on_resync(m: Message):
    if not is_admin(m):
        return
    await m.answer("Пришли свежий result.json (Telegram Desktop: Settings > Advanced > "
                   "Export Telegram data) файлом в этот чат — дошью недостающее.")


async def on_document(m: Message, bot: Bot):
    if not is_admin(m) or not m.document or m.document.file_name != "result.json":
        return
    export = os.path.join(DATA_DIR, "result.json")
    await bot.download(m.document, destination=export)
    await m.answer("читаю экспорт, дошиваю...")
    try:
        r = await asyncio.get_running_loop().run_in_executor(
            None, ingest.resync, export)
        await m.answer(f"готово: добавлено/обновлено {r['added']}, "
                       f"размечено всего {r['normalized']}")
    except Exception as e:  # noqa: BLE001
        log.exception("resync упал")
        await m.answer(f"не получилось: {e.__class__.__name__}: {e}")


def channel_gap(last_id, new_id):
    """Channel post ids missed between the last one seen and the current.
    [] — nothing is missing (or the first message after startup: last is
    unknown)."""
    if not last_id or new_id <= last_id + 1:
        return []
    return list(range(last_id + 1, new_id))


async def on_channel_post(m: Message, bot: Bot):
    """Live ingest: new channel posts and edits (the bot is a channel
    admin). Registered without a text filter: a textless post does not get
    ingested, but its message_id is counted — otherwise it would look like
    a gap."""
    last = int(store.get_state("last_channel_id") or 0)
    if m.message_id > last:
        store.set_state("last_channel_id", str(m.message_id))
    missing = channel_gap(last, m.message_id)
    if missing and ADMIN:
        rng = (f"{missing[0]}" if len(missing) == 1
               else f"с {missing[0]} по {missing[-1]} ({len(missing)} шт.)")
        try:
            await bot.send_message(
                ADMIN, f"Возможен пропуск постов канала: не видел {rng}. "
                       f"Если там был контент, досинхронизируй /resync "
                       f"(посты без текста в индекс и так не попадают).")
        except Exception:  # noqa: BLE001
            log.exception("алерт о пропуске не ушёл")
    text = (m.text or m.caption or "").strip()
    if not text:
        return
    post = {"id": m.message_id, "date": m.date.strftime("%Y-%m-%d"),
            "author": "", "text": text}
    try:  # ingest takes a minute (embed + LLM normalization) — thread pool
        # only; a synchronous call would freeze the event loop and the
        # whole bot with it
        loop = asyncio.get_running_loop()
        r = await loop.run_in_executor(None, ingest.add_posts, [post])
        log.info("channel_post %s: %s", m.message_id, r)
        if r.get("skipped") and ADMIN:
            reasons = "; ".join(f"{pid}: {why}" for pid, why in r["skipped"])
            await bot.send_message(
                ADMIN, f"Пост {m.message_id} не проингестирован ({reasons}).")
    except Exception:  # noqa: BLE001
        log.exception("ингест channel_post %s упал", m.message_id)


async def do_search(m: Message, bot: Bot, text, r=None):
    """Search over any text (the user's query or a theme button).
    While the embed request runs: "typing..." plus (after 0.9 s) the live
    status "Ищу по смыслу/дословно: query" — it disappears once the card is
    ready. r is a ready route from route(): on_text has already computed
    it; a second call would mean another LLM reformulation of the same
    query."""
    loop = asyncio.get_running_loop()
    if r is None:  # theme button: run route in a pool (it reads the author index file)
        r = await loop.run_in_executor(
            None, lambda: route(text, allow_llm=rate_left(
                m.from_user.id if m.from_user else None)))
    if r["intent"] != "search":
        return
    if not r["query"].strip():
        # empty after stopwords and the facet (emoji, "найди цитату") — this
        # is not a query: clarify with theme buttons right away, not 5 random
        # cards
        await m.answer("Не понял, о чём искать. Напиши тему («одиночество»), "
                       "имя автора или точную фразу в кавычках.",
                       reply_markup=clarify_kb())
        return
    if not rate_ok(m.from_user.id if m.from_user else None):
        return await m.answer(RATE_LIMIT_TEXT)
    typing = asyncio.create_task(_keep_typing(bot, m.chat.id))
    prog: list = []
    delayed = asyncio.create_task(
        _send_later(m, search_status(r["path"], r["query"]), prog))
    try:
        found = await loop.run_in_executor(None, search_cards, r, 5)
    except Exception:  # noqa: BLE001
        log.exception("поиск упал")
        delayed.cancel()
        await _drop(prog)
        await m.answer("Поиск сломался, попробуй ещё раз.")
        return
    finally:
        typing.cancel()
        delayed.cancel()
    if not found:
        empty = f"По «{r['query']}» ничего не нашёл. Попробуй короче: " \
                f"тема («одиночество») или имя автора."
        if prog:  # the status is already up — turn it into the answer instead of blinking
            try:
                await prog[0].edit_text(empty, reply_markup=clarify_kb())
                return
            except Exception:  # noqa: BLE001 — the edit failed, send a new one
                pass
        await m.answer(empty, reply_markup=clarify_kb())
        return
    items = cards.load_norm()
    cs = diversify([cards.card(p, items.get(str(p["id"]))) for p, _, _ in found])
    key = new_deck(cs, r["query"])
    await send_card(m, bot, cs[0], key, 0,
                    user_id=m.from_user.id if m.from_user else None)
    await _drop(prog)


async def on_text(m: Message, bot: Bot):
    # route in a pool: a long query triggers an LLM reformulation (urllib,
    # up to 30 s) — a synchronous call would freeze the whole bot. For a
    # user who spent the daily limit the reformulation is disabled: check
    # first, then route
    loop = asyncio.get_running_loop()
    r = await loop.run_in_executor(
        None, lambda: route(m.text, allow_llm=rate_left(
            m.from_user.id if m.from_user else None)))
    if r["intent"] == "intro":
        await send_intro(m, bot)
        return
    if r["intent"] == "admin":
        await m.answer("Команды: /start. Админские: /stats, /resync.")
        return
    await do_search(m, bot, m.text, r=r)


async def on_callback(q: CallbackQuery, bot: Bot):
    data = q.data or ""
    if data == "sic":
        # a random branch of unshown posts: a deck of ~5 cards so that
        # "Ещё цитата" survives Sic too (the deck used to hold a single card)
        branch = await asyncio.get_running_loop().run_in_executor(
            None, random_branch, q.from_user.id)
        items = cards.load_norm()
        cs = [cards.card(p, items.get(str(p["id"]))) for p in branch]
        key = new_deck(cs, "")
        await send_card(q.message, bot, cs[0], key, 0,
                        similar=True, user_id=q.from_user.id)
        return await q.answer()
    if data.startswith("t:"):  # theme button from the clarification prompt
        await q.answer()
        return await do_search(q.message, bot, data[2:])
    _, key, idx = (data.split(":") + ["", ""])[:3]
    deck = DECKS.get(key)
    if not deck:
        return await q.answer("Колода устарела, поищи заново", show_alert=True)
    cs = deck["cards"]
    # isdigit() passes unicode "digits" (²) — int() on them raises;
    # ASCII digits only
    i = int(idx) % len(cs) if idx.isascii() and idx.isdigit() and cs else 0
    if data[0] == "n":  # next card in the deck
        i = (i + 1) % len(cs)
        await send_card(q.message, bot, cs[i], key, i, user_id=q.from_user.id)
        return await q.answer()
    if data[0] == "l":  # similar quote: a branch of nearest neighbors without repeats
        if not rate_ok(q.from_user.id):
            return await q.answer(RATE_LIMIT_TEXT, show_alert=True)
        await q.answer("ищу похожую...")
        loop = asyncio.get_running_loop()
        typing = asyncio.create_task(_keep_typing(bot, q.message.chat.id))
        try:
            seen = store.shown_posts(q.from_user.id)
        except Exception:  # noqa: BLE001 — the memory is not critical
            seen = set()
        try:
            branch = await loop.run_in_executor(
                None, similar_branch, cs[i]["text"], cs[i]["id"], seen)
        except Exception:  # noqa: BLE001
            log.exception("похожая цитата упала")
            branch = []
        finally:
            typing.cancel()
        if not branch:  # nearest exhausted — a random unshown branch
            try:
                branch = await loop.run_in_executor(
                    None, random_branch, q.from_user.id)
            except Exception:  # noqa: BLE001
                log.exception("случайная ветка упала")
        if not branch:
            return await q.message.answer("Похожей цитаты не нашлось.")
        items = cards.load_norm()
        cs2 = [cards.card(p, items.get(str(p["id"]))) for p in branch]
        key2 = new_deck(cs2, "")
        await send_card(q.message, bot, cs2[0], key2, 0,
                        similar=True, user_id=q.from_user.id)
        return
    if data[0] == "f":  # read in full
        await q.message.answer(cards.card_text(cs[i], full=True))
        return await q.answer()
    if data[0] in "ap":  # reference: about the author / the painting (reference.py)
        if not rate_ok(q.from_user.id):
            return await q.answer(RATE_LIMIT_TEXT, show_alert=True)
        await q.answer()
        if data[0] == "a":
            c = cs[i]
            fn, args = reference.about_author, (c["author"], c["source"])
        else:
            painting = deck.get("paintings", {}).get(i)
            if not painting:
                return await q.message.answer("Картина этой карточки уже не помнится.")
            fn, args = reference.about_painting, (painting,)
        try:
            await send_reference(q.message, fn, args)
        except Exception:  # noqa: BLE001
            log.exception("справка не собралась")
            await q.message.answer("Справка не собралась, попробуй ещё раз.")
        return
    await q.answer()


async def main():
    if not TOKEN:
        raise SystemExit("TG_BOT_TOKEN не найден (env или .env)")
    try:  # catch up vectors for new posts right away; the embedding API
        # sometimes drops the connection on big batch uploads — hence several
        # attempts; after the last one we don't kill startup, the first
        # search finishes the job (the index is incremental)
        import index
        for attempt in range(3):
            try:
                V, _ = index.build()
                log.info("индекс: %s векторов", V.shape[0])
                break
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    raise
                log.warning("стартовая досшивка не удалась (попытка %s), "
                            "жду 15 с", attempt + 1)
                await asyncio.sleep(15)
    except Exception:  # noqa: BLE001
        log.exception("стартовая досшивка индекса не удалась")
    try:  # semantic painting map: catch up new posts (no vectors stored
        # at runtime — one embed_query per new post, then json only).
        # Card kinds only — a reflection never becomes a card, sewing it
        # into the map is pointless
        norm = cards.load_norm()
        added = images.extend_painting_map(
            {p["id"]: p["text"] for p in load_posts()
             if norm.get(str(p["id"]), {}).get("kind", "quote") in PUBLIC_KINDS})
        if added:
            log.info("painting_map: дошито %s постов", added)
    except Exception:  # noqa: BLE001
        log.exception("досшивка painting_map не удалась")
    mig = store.migrate()  # SQLite is a mirror of the corpus, synced on every start
    log.info("SQLite: %s", mig)
    bot = Bot(TOKEN)
    dp = Dispatcher()
    dp.message.register(on_start, Command("start"))
    dp.message.register(on_stats, Command("stats"))
    dp.message.register(on_resync, Command("resync"))
    dp.message.register(on_document, F.document)
    dp.channel_post.register(on_channel_post)
    dp.edited_channel_post.register(on_channel_post)
    dp.message.register(on_text, F.text)
    dp.callback_query.register(on_callback)
    log.info("старт long polling, admin=%s", ADMIN or "не задан")
    await dp.start_polling(bot)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
