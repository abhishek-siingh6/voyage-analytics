"""Test health endpoint."""

import pytest
from api.app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test the /api/health endpoint returns healthy status."""
    response = client.get("/api/health")

    assert response.status_code == 200
    json_data = response.get_json()

    assert json_data["status"] == "healthy"


def test_health_check_json_response(client):
    """Test the health endpoint returns proper JSON."""
    response = client.get("/api/health")

    assert response.content_type == "application/json"


def test_health_check_200_status(client):
    """Test the health endpoint returns HTTP 200."""
    response = client.get("/api/health")

    assert response.status_code == 200
