"""Загрузка кураторского пула PD-art с Wikimedia Commons в paintings/.

Манифест paintings.json задаёт поисковую строку (search); скрипт ищет файл
через API Commons (namespace File, bitmap), берёт миниатюру 1280px и пишет
{id}.jpg. Повторный запуск докачивает только недостающее.

Запуск: python -X utf8 fetch_paintings.py [--force]
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "quote-search-bot/1.0 "
    "(PD-art pool fetcher; https://github.com/vonavikon/quote-search-bot)"}
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "paintings", "paintings.json")
WIDTH = 1280


def _get(url, retries=4):
    """GET с ретраями: Commons ловит серии запросов 429, ждём и повторяем."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _polite():
    time.sleep(1.5)


def find_file(query):
    """Поиск файла в namespace 6 (File:) по строке. -> заголовок File:...|None."""
    q = urllib.parse.quote(f"{query} filetype:bitmap")
    data = json.loads(_get(f"{API}?action=query&list=search&srnamespace=6"
                           f"&srlimit=5&srsearch={q}&format=json"))
    _polite()
    hits = data.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def thumb_url(file_title):
    """URL миниатюры WIDTHpx по заголовку файла. -> (url, width, height)|None."""
    t = urllib.parse.quote(file_title)
    data = json.loads(_get(f"{API}?action=query&titles={t}&prop=imageinfo"
                           f"&iiprop=url|size&iiurlwidth={WIDTH}&format=json"))
    _polite()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        ii = page.get("imageinfo")
        if ii:
            return ii[0].get("thumburl"), ii[0].get("thumbwidth"), ii[0].get("thumbheight")
    return None


def fetch(entry, force=False):
    out = os.path.join(HERE, "paintings", entry["file"])
    if os.path.exists(out) and os.path.getsize(out) > 30_000 and not force:
        return "ok (cached)"
    title = find_file(entry["search"])
    if not title:
        return "НЕ НАЙДЕНО поиском"
    info = thumb_url(title)
    if not info or not info[0]:
        return f"нет imageinfo: {title}"
    url, w, h = info
    blob = _get(url)
    _polite()
    if len(blob) < 30_000:
        return f"подозрительно мал ({len(blob)} B): {url}"
    with open(out, "wb") as f:
        f.write(blob)
    return f"ok {w}x{h}, {len(blob) // 1024} KB <- {title}"


if __name__ == "__main__":
    force = "--force" in sys.argv
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    bad = 0
    for e in manifest:
        try:
            r = fetch(e, force=force)
        except Exception as ex:  # noqa: BLE001
            r = f"ОШИБКА {ex.__class__.__name__}: {ex}"
        if not r.startswith("ok"):
            bad += 1
        print(f"{e['id']}: {r}")
    print(f"\nитог: {len(manifest) - bad}/{len(manifest)} ok, ошибок {bad}")
