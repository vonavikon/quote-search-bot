"""Shared fixtures: all data paths are redirected into tmp so the tests
never touch the real corpus and never hit the network."""
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
    """Redirects the corpus/normalization/file_ids paths into tmp_path."""
    monkeypatch.setattr(index, "CACHE", tmp_path / "corpus.npz")
    monkeypatch.setattr(index, "META", tmp_path / "corpus_meta.json")
    monkeypatch.setattr(normalize, "CACHE", tmp_path / "normalization.json")
    monkeypatch.setattr(router, "NORM_CACHE", tmp_path / "normalization.json")
    monkeypatch.setattr(cards, "NORM_PATH", tmp_path / "normalization.json")
    monkeypatch.setattr(images, "FILE_IDS_PATH", tmp_path / "file_ids.json")
    # load_posts without an export reads META_PATH; point the export at a
    # nonexistent file
    monkeypatch.setattr(load_corpus, "META_PATH", str(tmp_path / "corpus_meta.json"))
    monkeypatch.setattr(load_corpus, "DEFAULT_PATH", str(tmp_path / "result.json"))
    # the glue manifest goes to tmp too: tests write their own, the live
    # one stays out of the way
    monkeypatch.setattr(load_corpus, "SPLITS_PATH", str(tmp_path / "splits.json"))
    monkeypatch.setattr(load_corpus, "OCR_PATH", str(tmp_path / "ocr.json"))
    # the SQLite base builds its path from DATA_DIR dynamically — tmp too
    monkeypatch.setattr(load_corpus, "DATA_DIR", str(tmp_path))
    images._manifest.cache_clear()
    images._cached_theme_index.cache_clear()
    return tmp_path


@pytest.fixture
def norm_cache(tmp_data):
    """A mini normalization cache: Pasternak (quote, signed), a
    reflection, a poem."""
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
    """A 2-post index + an embedding mock (deterministic vectors, no
    API)."""
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
