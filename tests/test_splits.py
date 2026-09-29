"""Склейки: пост из нескольких цитат режется на дочерние посты (splits.json).
Дочерние id = id*100 + №, метаданные курируемые, LLM-нормализация их не трогает.
"""
import json

import load_corpus
import normalize
import store


def _write_splits(tmp_path, manifest):
    with open(tmp_path / "splits.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)


GLUED = {
    "5": [
        {"text": "Первая цитата.", "kind": "quote", "author": "Автор Первый"},
        {"text": "Вторая цитата.", "kind": "quote", "author": None,
         "source": "Книга"},
    ]
}


def test_apply_splits_children_and_bystander(tmp_data):
    _write_splits(tmp_data, GLUED)
    posts = [{"id": 5, "date": "2026-01-01", "author": "", "text": "склейка"},
             {"id": 6, "date": "2026-01-02", "author": "", "text": "мимо"}]
    out = load_corpus.apply_splits(posts)
    assert [p["id"] for p in out] == [501, 502, 6]
    assert out[0]["text"] == "Первая цитата."      # strip() отmanifesta
    assert out[0]["date"] == "2026-01-01"           # дата родителя наследуется
    assert out[2]["text"] == "мимо"                 # обычный пост не тронут


def test_apply_splits_no_manifest_passthrough(tmp_data):
    assert load_corpus.apply_splits([{"id": 1, "text": "x"}]) == [{"id": 1, "text": "x"}]


def test_load_norm_merges_split_meta(tmp_data):
    import cards
    _write_splits(tmp_data, GLUED)
    items = cards.load_norm()  # normalization.json нет вовсе
    assert items["501"]["author"] == "Автор Первый"
    assert items["501"]["kind"] == "quote"
    assert items["502"]["author"] is None
    assert items["502"]["source"] == "Книга"


def test_load_norm_manifest_overrides_cache(tmp_data):
    import cards
    _write_splits(tmp_data, GLUED)
    with open(cards.NORM_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": "t", "items": {
            "501": {"kind": "reflection", "author": None, "source": None,
                    "author_evidence": "none"}}}, f, ensure_ascii=False)
    assert cards.load_norm()["501"]["author"] == "Автор Первый"  # манифест важнее


def test_migrate_replaces_parent_with_children(tmp_data):
    _write_splits(tmp_data, GLUED)
    posts = [{"id": 5, "date": "2026-01-01", "author": "", "text": "склейка"}]
    with open(tmp_data / "corpus_meta.json", "w", encoding="utf-8") as f:
        json.dump({"model": "t", "posts": posts}, f, ensure_ascii=False)
    con = store.connect()
    con.execute("INSERT INTO posts(post_id, text) VALUES(999, 'старьё')")
    con.commit()
    r = store.migrate(con)
    assert r["migrated"] is False  # база была непуста (строка 999)
    ids = {row[0] for row in con.execute("SELECT post_id FROM posts")}
    assert ids == {501, 502}       # родитель 5 и мусор 999 исчезли
    kind, author = con.execute(
        "SELECT kind, author FROM posts WHERE post_id=501").fetchone()
    assert (kind, author) == ("quote", "Автор Первый")
    con.close()


def test_normalize_skips_split_children(tmp_data, monkeypatch):
    _write_splits(tmp_data, GLUED)
    posts = [{"id": 5, "date": "2026-01-01", "author": "", "text": "склейка"}]
    with open(tmp_data / "corpus_meta.json", "w", encoding="utf-8") as f:
        json.dump({"model": "t", "posts": posts}, f, ensure_ascii=False)
    monkeypatch.setattr(normalize, "run_call",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("LLM позван для ребёнка склейки")))
    normalize.run()  # не должен взорваться: дети курируемые, LLM не нужен


def test_split_child_ids(tmp_data):
    _write_splits(tmp_data, GLUED)
    assert load_corpus.split_child_ids() == {"501", "502"}


def test_load_splits_broken_file_degrades(tmp_data):
    (tmp_data / "splits.json").write_text('{"5": [}', encoding="utf-8")
    assert load_corpus.load_splits() == {}  # битый JSON не роняет загрузку корпуса


def test_load_splits_bad_schema_skips_entry(tmp_data):
    _write_splits(tmp_data, {"abc": [{"text": "x"}], "5": GLUED["5"]})
    splits = load_corpus.load_splits()
    assert list(splits) == ["5"]  # неint-ключ отсеян, валидная запись жива


def test_apply_splits_unknown_parent_warns_but_passes(tmp_data):
    _write_splits(tmp_data, {"999": [{"text": "нет такого"}]})
    posts = [{"id": 1, "text": "x"}]
    assert load_corpus.apply_splits(posts) == posts  # тишина по данным, не крах


def test_apply_splits_already_split_corpus_no_warning(tmp_data, capsys):
    # corpus_meta.json после index.save хранит детей вместо родителей:
    # старт сервера читает уже разобранный корпус — предупреждения быть не должно
    _write_splits(tmp_data, GLUED)
    posts = [{"id": 501, "date": "2026-01-01", "author": "", "text": "Первая цитата."},
             {"id": 502, "date": "2026-01-01", "author": "", "text": "Вторая цитата."}]
    assert load_corpus.apply_splits(posts) == posts
    assert "не найдены" not in capsys.readouterr().err


def test_apply_splits_updates_child_text(tmp_data):
    """Правка текста ребёнка в манифесте доезжает до уже разобранного
    корпуса: дети в corpus_meta.json не замораживают старую редакцию."""
    _write_splits(tmp_data, {"5": [
        {"text": "Новая редакция первой.", "kind": "quote", "author": None}]})
    posts = [{"id": 501, "date": "2026-01-01", "author": "", "text": "старая редакция"}]
    out = load_corpus.apply_splits(posts)
    assert out[0]["text"] == "Новая редакция первой."
    assert posts[0]["text"] == "старая редакция"  # вход не мутирован


def test_migrate_empty_corpus_no_sql_error(tmp_data):
    with open(tmp_data / "corpus_meta.json", "w", encoding="utf-8") as f:
        json.dump({"model": "t", "posts": []}, f, ensure_ascii=False)
    r = store.migrate()  # раньше: DELETE ... NOT IN () — синтаксическая ошибка
    assert r["posts"] == 0


def test_add_posts_skips_split_parent(tmp_data, monkeypatch):
    _write_splits(tmp_data, GLUED)
    cached = [{"id": 501, "date": "2026-01-01", "author": "", "text": "Первая цитата."},
              {"id": 502, "date": "2026-01-01", "author": "", "text": "Вторая цитата."}]
    import index
    import numpy as np
    np.savez(index.CACHE, vectors=np.zeros((2, 2)))
    with open(tmp_data / "corpus_meta.json", "w", encoding="utf-8") as f:
        json.dump({"model": "test", "posts": cached}, f, ensure_ascii=False)
    monkeypatch.setattr(index, "_embed_all",
                        lambda posts: (_ for _ in ()).throw(
                            AssertionError("embed позван для родителя склейки")))
    import ingest
    r = ingest.add_posts([{"id": 5, "date": "2026-01-01", "author": "",
                           "text": "правка склейки"}])
    assert r["added"] == 0
    assert r["skipped"] and r["skipped"][0][0] == 5
    meta = json.load(open(index.META, encoding="utf-8"))
    assert [p["id"] for p in meta["posts"]] == [501, 502]  # родитель не вернулся
