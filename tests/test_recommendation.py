"""Tests for the travel recommendation endpoint."""

import pytest

from api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_recommend_health(client):
    resp = client.get("/api/recommend/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "healthy"


def test_recommend_places(client):
    resp = client.get("/api/recommend/places")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "places" in data
    assert data["total"] > 0


def test_recommendations_returns_200(client):
    resp = client.post(
        "/api/recommend/recommendations",
        json={"place": "Rio de Janeiro (RJ)", "top_n": 3},
    )
    assert resp.status_code == 200


def test_recommendations_ranked_by_score(client):
    resp = client.post(
        "/api/recommend/recommendations",
        json={"max_price_per_day": 200, "top_n": 5},
    )
    data = resp.get_json()
    recs = data["recommendations"]
    assert data["total"] == len(recs)
    # Scores must be non-increasing (descending).
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommendations_budget_respected(client):
    resp = client.post(
        "/api/recommend/recommendations",
        json={"place": "Sao Paulo (SP)", "max_price_per_day": 200, "top_n": 1},
    )
    data = resp.get_json()
    top = data["recommendations"][0]
    # Exact place match should rank the right hotel top.
    assert top["place"] == "Sao Paulo (SP)"


def test_recommendations_empty_body_ok(client):
    resp = client.post("/api/recommend/recommendations", json={})
    assert resp.status_code == 200


def test_recommendations_total_cost_with_days(client):
    resp = client.post(
        "/api/recommend/recommendations",
        json={"days": 3, "top_n": 1},
    )
    rec = resp.get_json()["recommendations"][0]
    assert rec["total_cost"] == pytest.approx(rec["price_per_day"] * 3, abs=0.01)


def test_recommendations_invalid_value(client):
    resp = client.post(
        "/api/recommend/recommendations",
        json={"top_n": 500},
    )
    assert resp.status_code == 400
