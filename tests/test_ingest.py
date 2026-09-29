"""Живой ингест: идемпотентность, инкремент, порядок постов, resync."""
import json

import numpy as np

import index
import ingest
import load_corpus
import normalize


def _norm_run_stub(monkeypatch, calls):
    def run():
        calls.append(1)
        return {"model": "test", "items": {}}, {"n": 0}, 0.0
    monkeypatch.setattr(normalize, "run", run)


def _embed_stub(monkeypatch):
    def fake_embed_all(posts):
        return [[0.5, 0.5]] * len(posts)
    monkeypatch.setattr(index, "_embed_all", fake_embed_all)


def test_add_posts_idempotent(tiny_corpus, monkeypatch):
    calls = []
    _norm_run_stub(monkeypatch, calls)
    _embed_stub(monkeypatch)
    r = ingest.add_posts(tiny_corpus)
    assert r == {"added": 0, "normalized": 0, "skipped": []}
    assert not calls  # LLM-нормализация не зована
    assert np.load(index.CACHE)["vectors"].shape == (2, 2)


def test_add_posts_new_and_edited(tiny_corpus, monkeypatch):
    calls = []
    _norm_run_stub(monkeypatch, calls)
    _embed_stub(monkeypatch)
    edited = dict(tiny_corpus[0], text="обновлённый текст поста")
    new = {"id": 201, "date": "2024-02-01", "author": "", "text": "новый пост"}
    r = ingest.add_posts([edited, new])
    assert r["added"] == 2
    meta = json.loads(open(index.META, encoding="utf-8").read())
    # порядок старых сохранён, новый в конце, текст обновлён на месте
    assert [p["id"] for p in meta["posts"]] == [101, 102, 201]
    assert meta["posts"][0]["text"] == "обновлённый текст поста"
    assert np.load(index.CACHE)["vectors"].shape == (3, 2)
    assert calls  # нормализация добирает новые


def test_add_posts_skips_ocr_edits(tiny_corpus, monkeypatch):
    """Live-правка OCR-поста не инжестится: текст курируется в ocr.json,
    иначе merge_ocr маскировал бы правку старым распознанным текстом."""
    calls = []
    _norm_run_stub(monkeypatch, calls)
    _embed_stub(monkeypatch)
    with open(load_corpus.OCR_PATH, "w", encoding="utf-8") as f:
        json.dump({"102": {"date": "2024-01-02", "author": "",
                           "text": "распознанный текст"}}, f, ensure_ascii=False)
    edited = dict(tiny_corpus[1], text="live-правка поверх OCR")
    r = ingest.add_posts([edited])
    assert r == {"added": 0, "normalized": 0,
                 "skipped": [(102, "OCR-пост: текст курируется в ocr.json, "
                                   "live-правку перенеси туда")]}
    assert not calls  # перешива вектора и LLM не было


def test_resync_from_export(tmp_data, tiny_corpus, monkeypatch):
    calls = []
    _norm_run_stub(monkeypatch, calls)
    _embed_stub(monkeypatch)
    export = tmp_data / "result.json"
    export.write_text(json.dumps({"messages": [
        {"id": 101, "type": "message", "text": tiny_corpus[0]["text"]},  # без изменений
        {"id": 300, "type": "message", "text": "из экспорта"},
    ]}, ensure_ascii=False), encoding="utf-8")
    r = ingest.resync(str(export))
    assert r["added"] == 1
    meta = json.loads(open(index.META, encoding="utf-8").read())
    assert [p["id"] for p in meta["posts"]] == [101, 102, 300]


def test_stats(tmp_data, tiny_corpus, norm_cache):
    # у 102 reflection, у 101 author; 103 вне корпуса-фикстуры — не мешает
    s = ingest.stats()
    assert s["постов"] == 2 and s["векторов"] == 2
    assert s["размечено"] == 3 and s["автор подписан"] == 1
