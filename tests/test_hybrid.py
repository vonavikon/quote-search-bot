"""Retrieval-слой: токенизация, BM25, RRF, правила подключения лексики."""
from hybrid import BM25, rrf_fusion, tokenize, wants_lexical


def test_tokenize():
    assert tokenize("Привет, мир-38!") == ["привет", "мир", "38"]


def test_bm25_ranks_matching_doc():
    bm = BM25([["море", "шторм"], ["любовь", "тишина"], ["море", "тишина"]])
    scores = bm.scores(["море"])
    assert scores[0] > 0 and scores[2] > 0 and scores[1] == 0
    assert sorted(range(3), key=lambda i: -scores[i])[0] in (0, 2)


def test_bm25_rare_term_beats_frequent():
    """Редкое слово весит больше частого (idf)."""
    docs = [["слово"]] * 5 + [["уникум", "слово"]]
    bm = BM25(docs)
    s = bm.scores(["слово", "уникум"])
    assert s[5] > s[0]


def test_rrf_fusion_prefers_double_hit():
    fused, score = rrf_fusion([3, 1, 2], [3, 2, 1])
    assert fused[0] == 3  # топ обоих списков
    assert score[3] > score[1]


def test_wants_lexical_weak_top1():
    ok, why = wants_lexical("мусорный запрос", BM25([["а"]]), 0.30)
    assert ok and "top-1" in why


def test_wants_lexical_rare_name():
    bm = BM25([["война", "и", "мир"], ["мир"]])  # «Война» df=1
    ok, why = wants_lexical("Война как она есть", bm, 0.60)
    assert ok and "df=1" in why


def test_wants_lexical_quotes():
    ok, why = wants_lexical("«жалость к себе»", BM25([["а"]]), 0.90)
    assert ok and "кавычк" in why


def test_wants_lexical_strong_dense_passes():
    bm = BM25([["мир"], ["мир"]])  # частое слово, df=2 <= 3 — но заглавных нет
    ok, _ = wants_lexical("про мир и покой", bm, 0.70)
    assert not ok
