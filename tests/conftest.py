"""Общие фикстуры: все пути данных уводятся в tmp, чтобы тесты не трогали
реальный корпус и не ходили в сеть."""
import json

import numpy as np
import pytest

import cards
import images
import index
import load_corpus
import normalize
import router


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    """Перенаправляет corpus/normalization/file_ids пути в tmp_path."""
    monkeypatch.setattr(index, "CACHE", tmp_path / "corpus.npz")
    monkeypatch.setattr(index, "META", tmp_path / "corpus_meta.json")
    monkeypatch.setattr(normalize, "CACHE", tmp_path / "normalization.json")
    monkeypatch.setattr(router, "NORM_CACHE", tmp_path / "normalization.json")
    monkeypatch.setattr(cards, "NORM_PATH", tmp_path / "normalization.json")
    monkeypatch.setattr(images, "FILE_IDS_PATH", tmp_path / "file_ids.json")
    # load_posts без экспорта читает META_PATH; экспорт указываем несуществующим
    monkeypatch.setattr(load_corpus, "META_PATH", str(tmp_path / "corpus_meta.json"))
    monkeypatch.setattr(load_corpus, "DEFAULT_PATH", str(tmp_path / "result.json"))
    # манифест склеек тоже в tmp: тесты пишут свой, боевой не мешает
    monkeypatch.setattr(load_corpus, "SPLITS_PATH", str(tmp_path / "splits.json"))
    monkeypatch.setattr(load_corpus, "OCR_PATH", str(tmp_path / "ocr.json"))
    # SQLite-база строит путь из DATA_DIR динамически — тоже в tmp
    monkeypatch.setattr(load_corpus, "DATA_DIR", str(tmp_path))
    images._manifest.cache_clear()
    images._cached_theme_index.cache_clear()
    return tmp_path


@pytest.fixture
def norm_cache(tmp_data):
    """Мини-кэш нормализации: Пастернак (quote, подписан), размышление, поэма."""
    items = {
        "101": {"kind": "quote", "author": "Борис Пастернак", "source": None,
                "author_evidence": "signed"},
        "102": {"kind": "reflection", "author": None, "source": None,
                "author_evidence": "none"},
        "103": {"kind": "poem", "author": None, "source": "Каменный гость",
                "author_evidence": "none"},
    }
    with open(normalize.CACHE, "w", encoding="utf-8") as f:
        json.dump({"model": "test", "items": items}, f, ensure_ascii=False)
    return items


@pytest.fixture
def tiny_corpus(tmp_data):
    """Индекс на 2 поста + мок эмбеддинга (детерминированные векторы, без API)."""
    posts = [
        {"id": 101, "date": "2024-01-01", "author": "",
         "text": "Любовь не стихия, но гаснет, как стихия."},
        {"id": 102, "date": "2024-01-02", "author": "",
         "text": "Одиночество: разговор с самим собой."},
    ]
    np.savez(index.CACHE, vectors=np.eye(2, dtype=np.float32))
    with open(index.META, "w", encoding="utf-8") as f:
        json.dump({"model": "test-model", "posts": posts}, f, ensure_ascii=False)
    return posts
