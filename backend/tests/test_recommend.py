from backend import main


def test_recommend_returns_list(collections):
    emoji_coll, fb_coll = collections
    result = main.recommend(emoji_coll, fb_coll, "fröhlich", weight=0.7)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(hasattr(r, "emoji") for r in result)
    assert all(hasattr(r, "score") for r in result)
    assert all(hasattr(r, "description") for r in result)


def test_recommend_respects_n_results(collections):
    emoji_coll, fb_coll = collections
    result = main.recommend(emoji_coll, fb_coll, "fröhlich", weight=0.7, n_results=2)
    assert len(result) <= 2


def test_recommend_returns_list_entries(collections):
    emoji_coll, fb_coll = collections
    result = main.recommend(emoji_coll, fb_coll, "fröhlich", weight=0.7)
    for entry in result:
        assert isinstance(entry, main.ListEntry)


def test_recommend_without_feedback(collections, empty_feedback):
    emoji_coll, _ = collections
    result = main.recommend(emoji_coll, empty_feedback, "fröhlich", weight=0.7)
    assert len(result) > 0


def test_recommend_sorted_by_score_desc(collections):
    emoji_coll, fb_coll = collections
    result = main.recommend(emoji_coll, fb_coll, "fröhlich", weight=0.7)
    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=True)


def test_weight_zero(collections):
    emoji_coll, fb_coll = collections
    result = main.recommend(emoji_coll, fb_coll, "fröhlich", weight=0.0)
    assert len(result) > 0
    assert all(r.score >= 0 for r in result)


def test_negative_weight(collections):
    emoji_coll, fb_coll = collections
    result = main.recommend(emoji_coll, fb_coll, "fröhlich", weight=-1.0)
    assert len(result) > 0


def test_high_weight(collections):
    emoji_coll, fb_coll = collections
    result = main.recommend(emoji_coll, fb_coll, "fröhlich", weight=999.0)
    assert len(result) > 0
