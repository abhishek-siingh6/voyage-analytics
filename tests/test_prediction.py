"""Test prediction endpoint with XGBoost model."""

import pytest
import os
from api.app import app
from config.settings import settings

# The flight-price model artifact is supplied by the ML team and is deliberately
# git-ignored (large binary). Skip these tests when it is not present locally
# or in CI so a fresh checkout still runs the rest of the suite.
pytestmark = pytest.mark.skipif(
    not os.path.exists(settings.model_path),
    reason="Flight-price model artifact not present",
)


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predict_endpoint_returns_200(client):
    payload = {
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        "flightType": "firstClass",
        "agency": "FlyingDrops",
        "time": 1.76,
        "distance": 676.53,
        "flight_year": 2019,
        "flight_month": 9,
        "flight_day": 26,
        "flight_dayofweek": 3,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200


def test_predict_endpoint_returns_json(client):
    payload = {
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        "flightType": "firstClass",
        "agency": "FlyingDrops",
        "time": 1.76,
        "distance": 676.53,
        "flight_year": 2019,
        "flight_month": 9,
        "flight_day": 26,
        "flight_dayofweek": 3,
    }
    response = client.post("/api/predict", json=payload)
    assert response.content_type == "application/json"


def test_predict_endpoint_has_predicted_price(client):
    payload = {
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        "flightType": "firstClass",
        "agency": "FlyingDrops",
        "time": 1.76,
        "distance": 676.53,
        "flight_year": 2019,
        "flight_month": 9,
        "flight_day": 26,
        "flight_dayofweek": 3,
    }
    response = client.post("/api/predict", json=payload)
    data = response.get_json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], (int, float))


def test_predict_valid_price(client):
    payload = {
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        "flightType": "firstClass",
        "agency": "FlyingDrops",
        "time": 1.76,
        "distance": 676.53,
        "flight_year": 2019,
        "flight_month": 9,
        "flight_day": 26,
        "flight_dayofweek": 3,
    }
    response = client.post("/api/predict", json=payload)
    price = response.get_json()["predicted_price"]
    assert price > 0
    assert price < 100000


def test_predict_missing_field(client):
    payload = {
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        # missing flightType
        "agency": "FlyingDrops",
        "time": 1.76,
        "distance": 676.53,
        "flight_year": 2019,
        "flight_month": 9,
        "flight_day": 26,
        "flight_dayofweek": 3,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400


def test_predict_invalid_value(client):
    payload = {
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        "flightType": "invalidClass",
        "agency": "FlyingDrops",
        "time": 1.76,
        "distance": 676.53,
        "flight_year": 2019,
        "flight_month": 9,
        "flight_day": 26,
        "flight_dayofweek": 3,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400


def test_predict_invalid_json(client):
    response = client.post(
        "/api/predict",
        data="not valid json",
        content_type="application/json",
    )
    assert response.status_code == 400


def test_predict_empty_body(client):
    response = client.post(
        "/api/predict",
        data="{}",
        content_type="application/json",
    )
    assert response.status_code == 400