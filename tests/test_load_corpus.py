"""load_corpus: разбор экспорта и серверный фолбэк на meta."""
import json

import load_corpus


def test_flatten_variants():
    assert load_corpus.flatten(None) == ""
    assert load_corpus.flatten("просто текст") == "просто текст"
    assert load_corpus.flatten([{"text": "а"}, "б", {"text": "в"}]) == "абв"


def test_parse_export_filters_and_fields():
    data = {"messages": [
        {"id": 1, "type": "service", "text": "пином"},                      # сервисное
        {"id": 2, "type": "message", "text": "  "},                          # пустой текст
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
    """Нет экспорта (сервер) — посты берутся из corpus_meta.json."""
    meta = {"model": "m", "posts": [{"id": 1, "date": "2024-01-01",
                                     "author": "", "text": "текст"}]}
    meta_path = tmp_path / "corpus_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(load_corpus, "DEFAULT_PATH", str(tmp_path / "нет_экспорта.json"))
    monkeypatch.setattr(load_corpus, "META_PATH", str(meta_path))
    # боевой ocr.json лежит рядом с модулем и втащил бы 169 постов
    monkeypatch.setattr(load_corpus, "OCR_PATH", str(tmp_path / "нет_ocr.json"))
    assert load_corpus.load_posts() == meta["posts"]


def test_load_posts_meta_plain_list(tmp_path, monkeypatch):
    """Старый формат meta — просто список постов."""
    posts = [{"id": 9, "date": "2024-01-01", "author": "", "text": "x"}]
    meta_path = tmp_path / "corpus_meta.json"
    meta_path.write_text(json.dumps(posts, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(load_corpus, "DEFAULT_PATH", str(tmp_path / "нет.json"))
    monkeypatch.setattr(load_corpus, "META_PATH", str(meta_path))
    monkeypatch.setattr(load_corpus, "OCR_PATH", str(tmp_path / "нет_ocr.json"))
    assert load_corpus.load_posts() == posts


# ---------- OCR-бэкфилл image-only постов ----------

def _write_ocr(tmp_data, manifest):
    with open(tmp_data / "ocr.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)


def test_parse_export_fills_ocr_text(tmp_data):
    """Пост без текста, но с распознанным OCR, входит в корпус с ним."""
    _write_ocr(tmp_data, {"7": {"date": "2024-03-07", "author": "",
                                "text": "цитата с картинки"}})
    data = {"messages": [
        {"id": 7, "type": "message", "date": "2024-03-07", "photo": "photos/x.jpg"},
        {"id": 8, "type": "message", "date": "2024-03-08"},  # без OCR и без текста
    ]}
    posts = load_corpus._parse_export(data)
    assert [p["id"] for p in posts] == [7]
    assert posts[0]["text"] == "цитата с картинки"


def test_merge_ocr_appends_missing_to_meta(tmp_data):
    """Серверный корпус image-only постов не знает — merge добавляет."""
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
    """Пост уже в корпусе (например, после первого index.save) — дубля нет."""
    _write_ocr(tmp_data, {"7": {"date": "2024-03-07", "author": "",
                                "text": "цитата с картинки"}})
    posts = [{"id": 7, "date": "2024-03-07", "author": "", "text": "цитата с картинки"}]
    assert load_corpus.merge_ocr(posts) == posts


def test_merge_ocr_updates_stale_text(tmp_data):
    """Правка OCR (мусор распознавания в ocr.json вычищен) доезжает до
    серверного corpus_meta.json со старым текстом — без ручной правки meta."""
    _write_ocr(tmp_data, {"7": {"date": "2024-03-07", "author": "",
                                "text": "чистый текст"}})
    meta = {"model": "m", "posts": [
        {"id": 7, "date": "2024-03-07", "author": "",
         "text": "Канал\nyesterday at 23:05\n\nчистый текст"}]}
    (tmp_data / "corpus_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    out = load_corpus.load_posts()
    assert out[0]["text"] == "чистый текст"  # ocr.json — источник текста OCR-поста


def test_load_ocr_broken_file_degrades(tmp_data):
    (tmp_data / "ocr.json").write_text('{"7": ', encoding="utf-8")
    assert load_corpus.load_ocr() == {}  # битый JSON не роняет загрузку корпуса


def test_load_ocr_empty_text_skipped(tmp_data):
    """Пост с пустым распознаванием (мусорная картинка) в корпус не идёт."""
    _write_ocr(tmp_data, {"7": {"date": "2024-03-07", "author": "", "text": "  "}})
    assert load_corpus.load_ocr() == {}
