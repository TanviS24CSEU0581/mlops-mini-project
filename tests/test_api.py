import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success():
    payload = {
        "age": 28.0,
        "screen_hours": 4.5,
        "apps_installed": 45,
        "plan_gb_limit": 2.5,
        "video_minutes": 120.0,
        "wifi_hours": 3.0,
        "phone_tier": "flagship",
        "circle": "Urban",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


def test_predict_rejects_excessive_apps():
    # Rule 1: apps_installed > 350 must return 422
    payload = {
        "age": 28.0,
        "screen_hours": 4.5,
        "apps_installed": 351,
        "plan_gb_limit": 2.5,
        "video_minutes": 120.0,
        "wifi_hours": 3.0,
        "phone_tier": "flagship",
        "circle": "Urban",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_plan_limit():
    # Rule 2: plan_gb_limit < 0 must return 422
    payload = {
        "age": 28.0,
        "screen_hours": 4.5,
        "apps_installed": 45,
        "plan_gb_limit": -1.0,
        "video_minutes": 120.0,
        "wifi_hours": 3.0,
        "phone_tier": "flagship",
        "circle": "Urban",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
