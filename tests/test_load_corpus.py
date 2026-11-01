"""load_corpus: export parsing and the server fallback to meta."""
import json

import load_corpus


def test_flatten_variants():
    assert load_corpus.flatten(None) == ""
    assert load_corpus.flatten("просто текст") == "просто текст"
    assert load_corpus.flatten([{"text": "а"}, "б", {"text": "в"}]) == "абв"


def test_parse_export_filters_and_fields():
    data = {"messages": [
        {"id": 1, "type": "service", "text": "пином"},                      # a service message
        {"id": 2, "type": "message", "text": "  "},                          # empty text
        {"id": 3, "type": "message", "date": "2024-03-05T10:00:00",
         "text": "цитата", "forwarded_from": "Канал"},
        {"id": 4, "type": "message", "date": "2024-03-06",
         "text": [{"text": "сегмент"}]},
    ]}
    posts = load_corpus._parse_export(data)
    assert [p["id"] for p in posts] == [3, 4]
    p3 = posts[0]
    assert p3["author"] == "Канал" and p3["date"] == "2024-03-05" and p3["text"] == "цитата"
    assert posts[1]["text"] == "сегмент"


def test_load_posts_meta_fallback(tmp_path, monkeypatch):
    """No export (server) — posts come from corpus_meta.json."""
    meta = {"model": "m", "posts": [{"id": 1, "date": "2024-01-01",
                                     "author": "", "text": "текст"}]}
    meta_path = tmp_path / "corpus_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(load_corpus, "DEFAULT_PATH", str(tmp_path / "нет_экспорта.json"))
    monkeypatch.setattr(load_corpus, "META_PATH", str(meta_path))
    # the live ocr.json sits next to the module and would pull in 169 posts
    monkeypatch.setattr(load_corpus, "OCR_PATH", str(tmp_path / "нет_ocr.json"))
    assert load_corpus.load_posts() == meta["posts"]


def test_load_posts_meta_plain_list(tmp_path, monkeypatch):
    """The old meta format — a plain list of posts."""
    posts = [{"id": 9, "date": "2024-01-01", "author": "", "text": "x"}]
    meta_path = tmp_path / "corpus_meta.json"
    meta_path.write_text(json.dumps(posts, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(load_corpus, "DEFAULT_PATH", str(tmp_path / "нет.json"))
    monkeypatch.setattr(load_corpus, "META_PATH", str(meta_path))
    monkeypatch.setattr(load_corpus, "OCR_PATH", str(tmp_path / "нет_ocr.json"))
    assert load_corpus.load_posts() == posts


# ---------- OCR backfill of image-only posts ----------

def _write_ocr(tmp_data, manifest):
    with open(tmp_data / "ocr.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)


def test_parse_export_fills_ocr_text(tmp_data):
    """A post without text but with OCR recognition enters the corpus
    with it."""
    _write_ocr(tmp_data, {"7": {"date": "2024-03-07", "author": "",
                                "text": "цитата с картинки"}})
    data = {"messages": [
        {"id": 7, "type": "message", "date": "2024-03-07", "photo": "photos/x.jpg"},
        {"id": 8, "type": "message", "date": "2024-03-08"},  # no OCR and no text
    ]}
    posts = load_corpus._parse_export(data)
    assert [p["id"] for p in posts] == [7]
    assert posts[0]["text"] == "цитата с картинки"


def test_merge_ocr_appends_missing_to_meta(tmp_data):
    """The server corpus doesn't know image-only posts — merge adds
    them."""
    _write_ocr(tmp_data, {"7": {"date": "2024-03-07", "author": "",
                                "text": "цитата с картинки"}})
    meta = {"model": "m", "posts": [{"id": 1, "date": "2024-01-01",
                                     "author": "", "text": "текст"}]}
    (tmp_data / "corpus_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    out = load_corpus.load_posts()
    assert [p["id"] for p in out] == [1, 7]
    assert out[1]["text"] == "цитата с картинки" and out[1]["date"] == "2024-03-07"


def test_merge_ocr_no_duplicates(tmp_data):
    """The post is already in the corpus (say, after the first
    index.save) — no duplicate."""
    _write_ocr(tmp_data, {"7": {"date": "2024-03-07", "author": "",
                                "text": "цитата с картинки"}})
    posts = [{"id": 7, "date": "2024-03-07", "author": "", "text": "цитата с картинки"}]
    assert load_corpus.merge_ocr(posts) == posts


def test_merge_ocr_updates_stale_text(tmp_data):
    """An OCR fix (recognition garbage cleaned out of ocr.json) reaches
    the server corpus_meta.json with the old text — without hand-editing
    the meta."""
    _write_ocr(tmp_data, {"7": {"date": "2024-03-07", "author": "",
                                "text": "чистый текст"}})
    meta = {"model": "m", "posts": [
        {"id": 7, "date": "2024-03-07", "author": "",
         "text": "Канал\nyesterday at 23:05\n\nчистый текст"}]}
    (tmp_data / "corpus_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    out = load_corpus.load_posts()
    assert out[0]["text"] == "чистый текст"  # ocr.json — the source of an OCR post's text


def test_load_ocr_broken_file_degrades(tmp_data):
    (tmp_data / "ocr.json").write_text('{"7": ', encoding="utf-8")
    assert load_corpus.load_ocr() == {}  # broken JSON doesn't kill the corpus load


def test_load_ocr_empty_text_skipped(tmp_data):
    """A post with empty recognition (a garbage picture) doesn't make
    the corpus."""
    _write_ocr(tmp_data, {"7": {"date": "2024-03-07", "author": "", "text": "  "}})
    assert load_corpus.load_ocr() == {}
