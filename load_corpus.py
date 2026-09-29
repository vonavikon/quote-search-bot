"""Загрузчик корпуса из Telegram Desktop export (result.json).

Возвращает посты с непустым текстом. Один пост = один чанк (короткие
цитаты, дробить нечего).

Серверный режим: если файла экспорта нет (его на сервере не держим),
посты берутся из corpus_meta.json — мастер живёт в данных, экспорт
приходит только через /resync.
"""
import json
import os
import sys

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "result.json")

# папка данных: corpus.npz / corpus_meta.json / normalization.json.
# В docker монтируется volume и задаётся BOT_DATA=/data.
DATA_DIR = os.environ.get("BOT_DATA", os.path.dirname(os.path.abspath(__file__)))
META_PATH = os.path.join(DATA_DIR, "corpus_meta.json")

# манифест склеек едет с кодом (data/ в git не входит, а сплиту знать надо
# и серверу, и локальному экспорту)
SPLITS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json")

# тексты image-only постов, распознанные OCR (gpt-4.1-mini, разовый бэкфилл):
# в экспорте у них нет поля text, серверный корпус их вовсе не знает
OCR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr.json")


def load_ocr():
    """{post_id: {"date", "author", "text"}} распознанных image-only постов.

    Битый файл не роняет бот: деградация до корпуса без OCR-текстов.
    """
    if not os.path.exists(OCR_PATH):
        return {}
    try:
        raw = json.load(open(OCR_PATH, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — файл правится руками/скриптом
        print(f"!! ocr.json не читается ({e}); OCR-посты не добавляются",
              file=sys.stderr)
        return {}
    out = {}
    for k, v in raw.items():
        try:
            if str(int(k)) == k and (v.get("text") or "").strip():
                out[int(k)] = {"date": v.get("date") or "",
                               "author": v.get("author") or "",
                               "text": v["text"].strip()}
        except (AttributeError, TypeError, ValueError):
            print(f"!! ocr.json: запись {k!r} не по схеме, пропущена",
                  file=sys.stderr)
    return out


def load_splits():
    """{id поста: [блоки]} или {}. Ключи с "_" — служебные, не сплиты.

    Битый файл (ручная правка, обрыв) не роняет бот: деградация до
    неразрезанных склеек с записью в лог.
    """
    if not os.path.exists(SPLITS_PATH):
        return {}
    try:
        raw = json.load(open(SPLITS_PATH, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — манифест правится руками, крах недопустим
        print(f"!! splits.json не читается ({e}); склейки не режутся",
              file=sys.stderr)
        return {}
    out = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        try:
            ok = str(int(k)) == k and isinstance(v, list)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            print(f"!! splits.json: запись {k!r} не по схеме (int-ключ, list-блоки), "
                  f"пропущена", file=sys.stderr)
            continue
        out[k] = v
    return out


def split_child_ids():
    """Дочерние id всех склеек строками (формат ключей нормализации)."""
    return {f"{int(pid) * 100 + i}"
            for pid, blocks in load_splits().items()
            for i in range(1, len(blocks) + 1)}


def apply_splits(posts):
    """Посты со склейками, разобранными на дочерние (id*100 + №).

    Тексты детей живут в манифесте (курируемый контент): corpus_meta.json
    после index.save уже содержит детей, и повторное применение не только
    норма, а способ доставки правки — текст ребёнка обновляется из
    манифеста, инкрементальность index (post_id + текст) дошивает вектор.
    """
    splits = load_splits()
    if not splits:
        return posts
    child_text = {int(pid) * 100 + i: b["text"].strip()
                  for pid, blocks in splits.items()
                  for i, b in enumerate(blocks, 1)}
    out, seen = [], set()
    for p in posts:
        seen.add(p["id"])
        blocks = splits.get(str(p["id"]))
        if blocks:
            out += [{"id": p["id"] * 100 + i, "date": p["date"],
                     "author": p["author"], "text": b["text"].strip()}
                    for i, b in enumerate(blocks, 1)]
        elif p["id"] in child_text:
            out.append({**p, "text": child_text[p["id"]]})
        else:
            out.append(p)
    # предупреждение оставляем только для реальной опечатки куратора:
    # родителя нет И дети не появились
    missing = [k for k in splits
               if int(k) not in seen and int(k) * 100 + 1 not in seen]
    if missing:
        print(f"!! splits.json: посты {missing} не найдены в корпусе, "
              f"сплиты по ним не применены", file=sys.stderr)
    return out


def flatten(t):
    """text в export бывает строкой или списком сегментов {'text': ...}."""
    if t is None:
        return ""
    if isinstance(t, str):
        return t
    out = []
    for seg in t:
        out.append(seg.get("text", "") if isinstance(seg, dict) else str(seg))
    return "".join(out)


def _parse_export(data):
    ocr = load_ocr()
    posts = []
    for m in data.get("messages", []):
        if m.get("type") != "message":
            continue
        text = flatten(m.get("text")).strip()
        if not text:
            o = ocr.get(m.get("id"))  # image-only пост, распознанный OCR
            if o:
                text = o["text"]
        if not text:
            continue
        posts.append({
            "id": m.get("id"),
            "date": (m.get("date") or "")[:10],
            "author": m.get("forwarded_from") or "",
            "text": text,
        })
    return posts


def merge_ocr(posts):
    """Посты + image-only записи OCR, которых в корпусе нет.

    Серверный корпус (corpus_meta.json) image-only посты не содержит:
    после деплоя с ocr.json они добавляются здесь, инкрементальность
    index (post_id + текст) дошивает векторы на первом же старте.
    Текст существующего OCR-поста тоже берётся из ocr.json: правки OCR
    (мусор распознавания в тексте) доезжают до серверного корпуса без
    ручной правки corpus_meta.json — как тексты детей в splits.json.
    """
    ocr = load_ocr()
    if not ocr:
        return posts
    out, seen = [], set()
    for p in posts:
        o = ocr.get(p["id"])
        if o and o["text"] != p.get("text"):
            p = {**p, "text": o["text"]}
        out.append(p)
        seen.add(p["id"])
    for pid in sorted(ocr):
        if pid not in seen:
            out.append({"id": pid, **ocr[pid]})
    return out


def load_posts(path=None):
    if path is None:
        path = DEFAULT_PATH
        if not os.path.exists(path):
            with open(META_PATH, encoding="utf-8") as f:
                meta = json.load(f)
            posts = meta["posts"] if isinstance(meta, dict) else meta
            return merge_ocr(apply_splits(posts))
    with open(path, encoding="utf-8") as f:
        return apply_splits(_parse_export(json.load(f)))


if __name__ == "__main__":
    posts = load_posts()
    print(f"постов с текстом: {len(posts)}")
    for p in posts[:3]:
        head = f"[{p['id']}] {p['date']}"
        if p["author"]:
            head += f" · {p['author']}"
        print(f"\n{head}\n{p['text'][:200]}")
