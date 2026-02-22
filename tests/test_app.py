"""
Unit tests for the VisionSense API.

VisionClassifier is mocked so no PyTorch model weights are required at test
time. Tests cover all routes, error handling, and core predict logic.
"""

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Mock the classifier before app.py is imported so the heavy PyTorch model
# is never loaded during tests.
# ---------------------------------------------------------------------------
MOCK_PREDICTIONS = [
    {"label": "cat", "confidence": 0.8800},
    {"label": "dog", "confidence": 0.0700},
    {"label": "bird", "confidence": 0.0250},
    {"label": "deer", "confidence": 0.0150},
    {"label": "horse", "confidence": 0.0100},
]

mock_classifier = MagicMock()
mock_classifier.predict.return_value = MOCK_PREDICTIONS
mock_classifier.device = "cpu"

with patch("src.classifier.VisionClassifier", return_value=mock_classifier):
    from app import app  # noqa: E402  (import after patch)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(width: int = 32, height: int = 32) -> bytes:
    """Return raw bytes of a valid small PNG image."""
    img = Image.new("RGB", (width, height), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200


def test_root_returns_welcome_message():
    response = client.get("/")
    body = response.json()
    assert "message" in body
    assert "VisionSense" in body["message"]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_expected_fields():
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert "model_loaded" in body
    assert "device" in body
    assert "message" in body


def test_health_model_loaded_is_true():
    body = client.get("/health").json()
    assert body["model_loaded"] == "True"


# ---------------------------------------------------------------------------
# GET /info
# ---------------------------------------------------------------------------

def test_info_returns_200():
    response = client.get("/info")
    assert response.status_code == 200


def test_info_returns_expected_fields():
    body = client.get("/info").json()
    assert "service" in body
    assert "version" in body
    assert "model" in body
    assert "framework" in body
    assert "description" in body


def test_info_service_name():
    body = client.get("/info").json()
    assert body["service"] == "VisionSense API"


def test_info_framework():
    body = client.get("/info").json()
    assert body["framework"] == "FastAPI"


# ---------------------------------------------------------------------------
# GET /dashboard
# ---------------------------------------------------------------------------

def test_dashboard_returns_200():
    response = client.get("/dashboard")
    assert response.status_code == 200


def test_dashboard_returns_html():
    response = client.get("/dashboard")
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type


# ---------------------------------------------------------------------------
# GET /logs
# ---------------------------------------------------------------------------

def test_logs_returns_200():
    response = client.get("/logs")
    assert response.status_code == 200


def test_logs_returns_list_when_no_log_file(tmp_path, monkeypatch):
    """When the log file does not exist the endpoint returns an empty list."""
    # Point LOG_FILE at a path that does not exist
    import app as app_module
    monkeypatch.setattr(app_module, "LOG_FILE", str(tmp_path / "nonexistent.log"))
    response = client.get("/logs")
    assert response.status_code == 200
    body = response.json()
    assert "logs" in body
    assert body["logs"] == []


def test_logs_limit_parameter():
    """The limit query parameter is accepted without error."""
    response = client.get("/logs?limit=5")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /predict — success path
# ---------------------------------------------------------------------------

def test_predict_with_valid_image_returns_200():
    png_bytes = _make_png_bytes()
    response = client.post(
        "/predict",
        files={"file": ("test.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert response.status_code == 200


def test_predict_success_flag_is_true():
    png_bytes = _make_png_bytes()
    body = client.post(
        "/predict",
        files={"file": ("test.png", io.BytesIO(png_bytes), "image/png")},
    ).json()
    assert body["success"] is True


def test_predict_returns_result_list():
    png_bytes = _make_png_bytes()
    body = client.post(
        "/predict",
        files={"file": ("test.png", io.BytesIO(png_bytes), "image/png")},
    ).json()
    assert "result" in body
    assert isinstance(body["result"], list)
    assert len(body["result"]) == 5


def test_predict_result_items_have_label_and_confidence():
    png_bytes = _make_png_bytes()
    body = client.post(
        "/predict",
        files={"file": ("test.png", io.BytesIO(png_bytes), "image/png")},
    ).json()
    for item in body["result"]:
        assert "label" in item
        assert "confidence" in item


def test_predict_calls_classifier():
    """Confirm the route delegates to the VisionClassifier.predict method."""
    mock_classifier.predict.reset_mock()
    png_bytes = _make_png_bytes()
    client.post(
        "/predict",
        files={"file": ("test.png", io.BytesIO(png_bytes), "image/png")},
    )
    mock_classifier.predict.assert_called_once()


# ---------------------------------------------------------------------------
# POST /predict — error path
# ---------------------------------------------------------------------------

def test_predict_with_corrupt_data_returns_500():
    """Sending non-image bytes should trigger the exception handler."""
    mock_classifier.predict.side_effect = Exception("cannot identify image file")
    response = client.post(
        "/predict",
        files={"file": ("bad.png", io.BytesIO(b"not-an-image"), "image/png")},
    )
    assert response.status_code == 500
    body = response.json()
    assert body["success"] is False
    assert "error" in body
    # Restore normal behaviour for subsequent tests
    mock_classifier.predict.side_effect = None
    mock_classifier.predict.return_value = MOCK_PREDICTIONS


def test_predict_error_body_contains_message():
    mock_classifier.predict.side_effect = ValueError("bad input")
    response = client.post(
        "/predict",
        files={"file": ("bad.png", io.BytesIO(b"garbage"), "image/png")},
    )
    body = response.json()
    assert body["success"] is False
    assert len(body["error"]) > 0
    mock_classifier.predict.side_effect = None
    mock_classifier.predict.return_value = MOCK_PREDICTIONS
