"""TG-бот поиска цитат по своему корпусу: карточки с картиной, случайная
выдача, справки об авторах, живой ингест из канала. Long polling (webhook
требует домен и TLS-роут, здесь не нужен). Запуск: python -X utf8 bot.py.

Env (или .env рядом): TG_BOT_TOKEN, ADMIN_TG_ID.
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
CAPTION_MAX = 1000  # лимит подписи к фото 1024, запас на разметку
DAY_LIMIT = 100     # защитный дневной лимит платящих действий на пользователя

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
    """/start и интро: картинка с приветствием (intro.jpg, если есть)
    + кнопка Sic. file_id кэшируем — после первой отправки летает id
    без загрузки. Файла нет — просто текст."""
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

# Уточнение однословников: при пустом результате предлагаем темы кнопками.
# Фасеты — теги манифеста картин (детерминированно, без LLM).
THEMES = ("одиночество", "любовь", "смерть", "вера", "дорога", "ночь", "море", "свет")


def clarify_kb():
    b = InlineKeyboardBuilder()
    for t in THEMES:
        b.button(text=t.capitalize(), callback_data=f"t:{t}")
    b.button(text="Sic erat in fatis", callback_data="sic")  # всегда под рукой
    b.adjust(4, 4, 1)
    return b.as_markup()


def sic_kb():
    """Кнопка случайной карточки — в /start и интро текстом."""
    b = InlineKeyboardBuilder()
    b.button(text="Sic erat in fatis", callback_data="sic")
    return b.as_markup()


# ---------- отклик: индикатор «печатает…» на всё время поиска ----------

async def _keep_typing(bot: Bot, chat_id):
    """Chat action живёт ~5 секунд — продлеваем, пока идёт поиск."""
    try:
        while True:
            await bot.send_chat_action(chat_id, "typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


def search_status(path, query):
    """Текст живого статуса поиска. Путь уже известен роутеру, врём только
    словом «ищу»: сам embed-запрос внутри hybrid.retrieve не виден."""
    how = "дословно" if path == "direct-quoted" else "по смыслу"
    return f"Ищу {how}: «{query}»..."


async def _send_later(m: Message, text: str, box: list, delay: float = 0.9):
    """Статус поиска появляется, только если поиск затянулся: быстрые ответы
    не мигают сообщением-призраком. Отправленное кладётся в box."""
    await asyncio.sleep(delay)
    box.append(await m.answer(text))


async def _drop(box: list):
    """Убрать статус поиска, если он успел появиться."""
    if box:
        try:
            await box[0].delete()
        except Exception:  # noqa: BLE001 — статус не критичен
            pass


def diversify(cs):
    """Разводит авторов по очереди: две карточки одного автора не идут подряд.

    Группирует по автору (None — «не подписан» тоже группа) и раздаёт
    round-robin, внутри группы порядок релевантности сохраняется.
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


# ---------- колоды и клавиатура ----------

def new_deck(cs, query):
    if len(DECKS) >= DECK_CAP:  # FIFO: старые колоды вылетают
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
    # справки: Википедия + данные корпуса, без свободной генерации (reference)
    b.button(text="Об авторе", callback_data=f"a:{key}:{i}")
    b.button(text="О картине", callback_data=f"p:{key}:{i}")
    rows.append(2)
    b.button(text="Sic erat in fatis", callback_data="sic")
    rows.append(1)
    b.adjust(*rows)
    return b.as_markup()


# ---------- отрисовка карточки: картина + текст (три порога длины) ----------

async def card_media(bot: Bot, painting):
    """file_id из кэша (без загрузки) либо файл пула с диска."""
    fid = images.get_file_id(painting["id"])
    return fid or FSInputFile(images.painting_path(painting))


def _remember_file_id(painting, msg):
    """После отправки Telegram даёт файлу file_id — кэшируем, дальше летает id."""
    if getattr(msg, "photo", None):
        images.save_file_id(painting["id"], msg.photo[-1].file_id)


def _caption(c, painting, full=False):
    """Подпись сообщения: текст карточки (HTML-экранированный) + автор +
    источник + строка картины курсивом. Отдельным абзацем, с пустой строкой —
    чтобы подпись автора не склеивалась с подписью картины."""
    body = html.escape(cards.card_text(c, full=full))
    label = f"Картина: {painting['author']}, «{painting['title']}»"
    if painting.get("year"):
        label += f", {painting['year']}"
    return f"{body}\n\n<i>{html.escape(label)}</i>"


async def send_card(m: Message, bot: Bot, c, key, i, similar=False, user_id=None):
    """Карточка целиком. <=1000 знаков: картина + подпись одним сообщением;
    длиннее: картина отдельно, текст с кнопками отдельным сообщением.
    Картина не повторяется ни внутри колоды, ни с последними показанными
    пользователю (images.pick с exclude); выбранная запоминается в колоде —
    кнопка «О картине» смотрит на неё, а в events уходит и картина."""
    deck = DECKS[key]
    exclude = set(deck["used"])
    if user_id:
        try:
            exclude |= store.shown_paintings(user_id)
        except Exception:  # noqa: BLE001 — память не критична
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
    """Ветка похожих цитат: dense-поиск по тексту текущей карточки.
    Себя и всё показанное пользователю (seen) пропускает; когда ближайшие
    исчерпаны, просто идёт дальше по списку — до 5 непрочитанных.
    -> [posts], может быть пустым."""
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
    """Случайная ветка: непоказанный случайный пост + его ближайшие соседи
    (тоже непоказанные) — колода для Sic и фолбэка «похожей». Всё прочитано —
    берём из всего пула. Синхронная (search_cards), зовётся в executor."""
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
    except Exception:  # noqa: BLE001 — соседей нет, живём одним постом
        found = []
    for p, _, _ in found:
        if p["id"] != seed["id"] and p["id"] not in seen:
            out.append(p)
        if len(out) >= 5:
            break
    return out


# ---------- справки об авторе и картине (по кнопке, reference.py) ----------

STREAM_THROTTLE = 0.9  # правка сообщения не чаще ~1/с (флуд-лимит edit_text)


async def send_reference(m: Message, fn, args):
    """Справка со стримингом: сборка идёт в пуле потоков (синхронный вызов
    заморозил бы весь бот), текст дописывается в сообщение правками раз в
    STREAM_THROTTLE. Кэш-хиты и заглушки возвращаются мгновенно — одна
    финальная правка, без пустого мигания."""
    loop = asyncio.get_running_loop()
    msg = await m.answer("Собираю справку...")
    buf: list = []  # поток дописывает куски, мы читаем: append атомарен
    task = loop.run_in_executor(None, lambda: fn(*args, on_chunk=buf.append))
    shown = ""
    while not task.done():
        await asyncio.sleep(STREAM_THROTTLE)
        text = "".join(buf)
        if text and text != shown:
            shown = text
            try:
                await msg.edit_text(text[:4000])
            except Exception:  # noqa: BLE001 — not modified или флуд
                pass
    try:
        await msg.edit_text((await task)[:4000])
    except Exception:  # noqa: BLE001
        pass


# ---------- хендлеры ----------

async def on_start(m: Message, bot: Bot):
    await send_intro(m, bot)


def is_admin(m: Message) -> bool:
    return bool(ADMIN) and m.from_user and m.from_user.id == ADMIN


def rate_ok(uid) -> bool:
    """Дневной лимит «платящих» действий (поиск / пояснение / похожая / справки).
    Считаются только вызовы модели; «Ещё цитата», Sic, «Читать полностью» —
    локальные. Админ — без лимита, чтобы отладка не упиралась."""
    if not uid or uid == ADMIN:
        return True
    return store.rate_incr(uid) <= DAY_LIMIT


def rate_left(uid) -> bool:
    """Остаток лимита без инкремента — для проверки ДО route: исчерпавший
    лимит пользователь не должен жечь LLM-переформулировку. intro/admin
    определяются роутером до LLM и лимит не тратят, счётчик растёт только
    на платящем действии (rate_ok внутри do_search)."""
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
    """Пропущенные id постов канала между последним увиденным и текущим.
    [] — пропусков нет (или первое сообщение после старта: last неизвестен)."""
    if not last_id or new_id <= last_id + 1:
        return []
    return list(range(last_id + 1, new_id))


async def on_channel_post(m: Message, bot: Bot):
    """Живой ингест: новые посты и правки канала (бот — админ канала).
    Зарегистрирован без фильтра на текст: пост без текста в ингест не
    попадает, но его message_id учитывается — иначе он выглядел бы пропуском."""
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
    try:  # ингест минутный (embed + LLM-нормализация) — только в пуле потоков,
        # синхронный вызов заморозил бы event loop и весь бот целиком
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
    """Поиск по любому тексту (запрос пользователя или кнопка-тема).
    Пока идёт embed-запрос: «печатает…» плюс (дольше 0.9 с) живой статус
    «Ищу по смыслу/дословно: запрос» — исчезает, когда карточка готова.
    r — готовый маршрут из route(): on_text уже вычислил его, второй вызов
    означал бы повторную LLM-переформулировку того же запроса."""
    loop = asyncio.get_running_loop()
    if r is None:  # кнопка-тема: route в пуле (внутри файл чтения индекса авторов)
        r = await loop.run_in_executor(
            None, lambda: route(text, allow_llm=rate_left(
                m.from_user.id if m.from_user else None)))
    if r["intent"] != "search":
        return
    if not r["query"].strip():
        # пустой после стоп-слов и фасета (эмодзи, «найди цитату») — это не
        # запрос: сразу уточнение с кнопками тем, а не 5 случайных карточек
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
        if prog:  # статус уже висит — превращаем его в ответ, а не мигаем
            try:
                await prog[0].edit_text(empty, reply_markup=clarify_kb())
                return
            except Exception:  # noqa: BLE001 — правка не прошла, шлём новым
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
    # route в пуле: длинный запрос запускает LLM-переформулировку (urllib,
    # до 30 с) — синхронный вызов замораживал бы весь бот. У исчерпавшего
    # дневной лимит переформулировку отключаем: сначала проверка, потом route
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
        # случайная ветка из непоказанных: колода из ~5 карточек, чтобы
        # «Ещё цитата» жила и после Sic (раньше колода была из одной)
        branch = await asyncio.get_running_loop().run_in_executor(
            None, random_branch, q.from_user.id)
        items = cards.load_norm()
        cs = [cards.card(p, items.get(str(p["id"]))) for p in branch]
        key = new_deck(cs, "")
        await send_card(q.message, bot, cs[0], key, 0,
                        similar=True, user_id=q.from_user.id)
        return await q.answer()
    if data.startswith("t:"):  # кнопка-тема из уточнения
        await q.answer()
        return await do_search(q.message, bot, data[2:])
    _, key, idx = (data.split(":") + ["", ""])[:3]
    deck = DECKS.get(key)
    if not deck:
        return await q.answer("Колода устарела, поищи заново", show_alert=True)
    cs = deck["cards"]
    # isdigit() пропускает юникодные «цифры» (²) — int() от них падает;
    # только ASCII-цифры
    i = int(idx) % len(cs) if idx.isascii() and idx.isdigit() and cs else 0
    if data[0] == "n":  # следующая цитата колоды
        i = (i + 1) % len(cs)
        await send_card(q.message, bot, cs[i], key, i, user_id=q.from_user.id)
        return await q.answer()
    if data[0] == "l":  # похожая цитата: ветка ближайших соседей без повторов
        if not rate_ok(q.from_user.id):
            return await q.answer(RATE_LIMIT_TEXT, show_alert=True)
        await q.answer("ищу похожую...")
        loop = asyncio.get_running_loop()
        typing = asyncio.create_task(_keep_typing(bot, q.message.chat.id))
        try:
            seen = store.shown_posts(q.from_user.id)
        except Exception:  # noqa: BLE001 — память не критична
            seen = set()
        try:
            branch = await loop.run_in_executor(
                None, similar_branch, cs[i]["text"], cs[i]["id"], seen)
        except Exception:  # noqa: BLE001
            log.exception("похожая цитата упала")
            branch = []
        finally:
            typing.cancel()
        if not branch:  # ближайшие исчерпаны — рандомная непоказанная ветка
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
    if data[0] == "f":  # читать полностью
        await q.message.answer(cards.card_text(cs[i], full=True))
        return await q.answer()
    if data[0] in "ap":  # справка: об авторе / о картине (reference.py)
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
    try:  # досшиваем векторы новых постов сразу; embedding-API иногда
        # роняет соединение на заливке больших батчей — поэтому несколько
        # попыток; после последней не роняем старт, первый поиск дошьёт сам
        # (индекс инкрементален)
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
    try:  # семантическая карта картин: дошиваем новые посты (без векторов
        # в рантайме — один embed_query на новый пост, потом только json).
        # Только карточные kind — reflection карточкой не становится,
        # шить его в карту бессмысленно
        norm = cards.load_norm()
        added = images.extend_painting_map(
            {p["id"]: p["text"] for p in load_posts()
             if norm.get(str(p["id"]), {}).get("kind", "quote") in PUBLIC_KINDS})
        if added:
            log.info("painting_map: дошито %s постов", added)
    except Exception:  # noqa: BLE001
        log.exception("досшивка painting_map не удалась")
    mig = store.migrate()  # SQLite — зеркало корпуса, синк на каждом старте
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
