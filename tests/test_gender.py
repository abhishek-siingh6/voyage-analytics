"""Tests for the gender classification endpoint."""

import pytest

from api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _payload(name="Robert Braun", age=33, company="4You"):
    return {"user name": name, "age": age, "company": company}


def test_gender_health(client):
    resp = client.get("/api/gender/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "healthy"


def test_gender_predict_returns_200(client):
    resp = client.post("/api/gender/predict", json=_payload())
    assert resp.status_code == 200


def test_gender_predict_has_expected_fields(client):
    resp = client.post("/api/gender/predict", json=_payload())
    data = resp.get_json()
    assert set(["gender", "probability", "model_version"]).issubset(data.keys())
    assert data["gender"] in {"male", "female", "none"}
    assert 0.0 <= data["probability"] <= 1.0


def test_gender_predict_missing_field(client):
    resp = client.post("/api/gender/predict", json={"name": "Robert Braun"})
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "validation_failed"


def test_gender_predict_invalid_json(client):
    resp = client.post(
        "/api/gender/predict", data="not json", content_type="application/json"
    )
    assert resp.status_code == 400


def test_gender_predict_returns_json(client):
    resp = client.post("/api/gender/predict", json=_payload())
    assert resp.content_type == "application/json"
