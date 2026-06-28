from unittest.mock import patch
from fastapi.testclient import TestClient
from backend import main
from .conftest import make_collections


@patch("backend.main.init_db")
def test_generate_returns_200(mock_init_db):
    emoji_coll, fb_coll = make_collections()
    mock_init_db.return_value = (emoji_coll, fb_coll)
    with TestClient(main.app) as client:
        resp = client.post("/generate", data={"input": "feiern"})
    assert resp.status_code == 200


@patch("backend.main.init_db")
def test_generate_returns_list(mock_init_db):
    emoji_coll, fb_coll = make_collections()
    mock_init_db.return_value = (emoji_coll, fb_coll)
    with TestClient(main.app) as client:
        resp = client.post("/generate", data={"input": "feiern"})
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "emoji" in data[0]
    assert "description" in data[0]
    assert "score" in data[0]


@patch("backend.main.init_db")
def test_generate_without_llm(mock_init_db):
    emoji_coll, fb_coll = make_collections()
    mock_init_db.return_value = (emoji_coll, fb_coll)
    with TestClient(main.app) as client:
        resp = client.post("/generate", data={"input": "feiern", "use_llm": "false"})
    assert resp.status_code == 200


@patch("backend.main.init_db")
def test_feedback_stores_entry(mock_init_db):
    emoji_coll, fb_coll = make_collections()
    mock_init_db.return_value = (emoji_coll, fb_coll)
    with TestClient(main.app) as client:
        resp = client.post("/feedback", data={"emoji": "😂", "emoji_feedback": "super lustig"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert fb_coll.count() == 3  # 2 initial + 1 neu


@patch("backend.main.init_db")
def test_llm_status(mock_init_db):
    emoji_coll, fb_coll = make_collections()
    mock_init_db.return_value = (emoji_coll, fb_coll)
    with TestClient(main.app) as client:
        resp = client.get("/llm/status")
    assert resp.status_code == 200
    assert "available" in resp.json()
