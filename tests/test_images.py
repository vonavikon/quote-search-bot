"""The painting pool: theme matching, the semantic map, the stable hash
fallback, the file_id cache."""
import json
import os

import images


POOL = [
    {"id": "a", "file": "a.jpg", "title": "Море", "author": "X", "year": 1900,
     "themes": ["море", "шторм"]},
    {"id": "b", "file": "b.jpg", "title": "Одиночество", "author": "Y", "year": 1901,
     "themes": ["одиночество", "туман"]},
    {"id": "c", "file": "c.jpg", "title": "Свеча", "author": "Z", "year": 1902,
     "themes": ["свеча", "ночь"]},
]


def _clear_caches():
    images._manifest.cache_clear()
    images._cached_theme_index.cache_clear()
    images._painting_map.cache_clear()
    images._by_id.cache_clear()


def _pool(tmp_path, monkeypatch):
    p = tmp_path / "paintings.json"
    p.write_text(json.dumps(POOL, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(images, "MANIFEST_PATH", str(p))
    # the repo map must not leak into tests with a tmp pool
    monkeypatch.setattr(images, "MAP_SEED_PATH", str(tmp_path / "map_seed.json"))
    monkeypatch.setattr(images, "MAP_PATH", str(tmp_path / "map.json"))
    monkeypatch.setattr(images, "VECS_PATH", str(tmp_path / "vecs.npz"))
    _clear_caches()
    return POOL


def _seed_map(tmp_path, mapping):
    with open(images.MAP_SEED_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
    images._painting_map.cache_clear()


def test_theme_candidates_exact_and_inflection(tmp_path, monkeypatch):
    _pool(tmp_path, monkeypatch)
    assert images.theme_candidates("одиночество")[0]["id"] == "b"
    assert images.theme_candidates("море")[0]["id"] == "a"
    # case and ending: «свечи» ~ «свеча», «одиночестве» ~ «одиночество»
    assert images.theme_candidates("зажги свечи")[0]["id"] == "c"
    assert images.theme_candidates("об одиночестве")[0]["id"] == "b"


def test_theme_candidates_no_hit(tmp_path, monkeypatch):
    _pool(tmp_path, monkeypatch)
    assert images.theme_candidates("про работу и труд") == []


def test_pick_theme_by_query(tmp_path, monkeypatch):
    _pool(tmp_path, monkeypatch)
    assert images.pick(1, "о море")["id"] == "a"


def test_semantic_map_between_query_and_hash(tmp_path, monkeypatch):
    """A post in the map — the painting from the map, before text and
    hash."""
    _pool(tmp_path, monkeypatch)
    _seed_map(tmp_path, {"7": ["b"]})
    assert images.pick(7)["id"] == "b"
    # the query theme beats the precomputed pick
    assert images.pick(7, "о море")["id"] == "a"
    # a painting taken in the deck is skipped, walk the top-K
    _seed_map(tmp_path, {"7": ["b", "a"]})
    assert images.pick(7, exclude={"b"})["id"] == "a"


def test_semantic_map_missing_post_falls_back(tmp_path, monkeypatch):
    """A post missing from the map — text and hash as before."""
    _pool(tmp_path, monkeypatch)
    _seed_map(tmp_path, {"7": ["b"]})
    assert images.pick(42, text="выходи в шторм добровольно")["id"] == "a"
    assert images.pick(42)["id"] == images.pick(42)["id"]


def test_semantic_map_unknown_painting_ignored(tmp_path, monkeypatch):
    """A painting id from the map that is not in the manifest — the entry
    is skipped."""
    _pool(tmp_path, monkeypatch)
    _seed_map(tmp_path, {"7": ["ghost", "c"]})
    assert images.pick(7)["id"] == "c"


def test_extend_map_noop_without_vecs(tmp_path, monkeypatch):
    """No description matrix — a silent 0, the posts stay on themes and
    the hash."""
    _pool(tmp_path, monkeypatch)
    assert images.extend_painting_map({9: "текст нового поста"}) == 0
    assert images.pick(9)["id"] in {p["id"] for p in POOL}


def test_pick_text_fallback_before_hash(tmp_path, monkeypatch):
    """The query matched no tags — the theme comes from the quote text
    itself; the hash stays the last resort."""
    _pool(tmp_path, monkeypatch)
    got = images.pick(42, "непонятная тема", text="выходи в шторм добровольно")
    assert got["id"] == "a"


def test_pick_stable_hash_fallback(tmp_path, monkeypatch):
    """With no query theme, the painting is stable for a post and always
    from the pool."""
    _pool(tmp_path, monkeypatch)
    ids = {p["id"] for p in POOL}
    assert images.pick(42)["id"] == images.pick(42)["id"]
    assert images.pick(42)["id"] in ids
    assert images.pick(42, "непонятная тема")["id"] in ids


def test_pick_exclude_no_collision(tmp_path, monkeypatch):
    """exclude (taken in the deck) is skipped — both in the theme path
    and the hash path."""
    _pool(tmp_path, monkeypatch)
    busy = images.pick(42, "о море")["id"]
    second = images.pick(43, "о море", exclude={busy})
    assert second["id"] != busy and second["id"] in {p["id"] for p in POOL}
    # no theme: the walk from the hash leaves the taken position
    first = images.pick(42)["id"]
    other = images.pick(42, exclude={first})
    assert other["id"] != first
    # the whole pool taken — the reserve: any painting, no KeyError
    assert images.pick(42, exclude={"a", "b", "c"})["id"] in {"a", "b", "c"}


def test_file_id_roundtrip(tmp_data):
    assert images.get_file_id("a") is None
    images.save_file_id("a", "AgAC-file-id")
    assert images.get_file_id("a") == "AgAC-file-id"
    images.save_file_id("a", "AgAC-file-id")  # writing the same id again — a no-op
    assert json.loads(open(images.FILE_IDS_PATH, encoding="utf-8").read()) == {"a": "AgAC-file-id"}


def test_real_manifest_valid():
    """The live manifest: unique ids, non-empty themes, the description
    vectors aligned with the manifest. jpgs are downloaded separately by
    fetch_paintings.py, so their completeness is checked only when the
    pool is already downloaded."""
    _clear_caches()  # a tmp pool may have been cached before us
    manifest = images._manifest()
    ids = {p["id"] for p in manifest}
    assert len(manifest) == len(ids) and len(manifest) >= 40
    if os.path.exists(images.VECS_PATH):
        import numpy as np
        assert np.load(images.VECS_PATH)["v"].shape[0] == len(manifest), \
            "painting_vecs.npz рассинхронизирован с манифестом"
    # the working map is built on the fly; if already built — no ghost paintings
    if os.path.exists(images.MAP_PATH):
        pmap = json.loads(open(images.MAP_PATH, encoding="utf-8").read())
        ghosts = {pid for top in pmap.values() for pid in top if pid not in ids}
        assert not ghosts, f"в карте нет таких картин: {sorted(ghosts)[:5]}"
    files = [p["file"] for p in manifest]
    present = [f for f in files if os.path.exists(os.path.join(images.POOL_DIR, f))]
    if len(present) < len(files):
        import pytest
        pytest.skip(f"пул докачан не целиком ({len(present)}/{len(files)}); "
                    "докачать: python -X utf8 fetch_paintings.py")
    # the pool is fully downloaded — there must be no missing files
    assert len(present) == len(files)
