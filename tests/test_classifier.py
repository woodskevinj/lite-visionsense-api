"""
Unit tests for the VisionClassifier (onnxruntime-based).

onnxruntime.InferenceSession is mocked so no real ONNX model file is needed.
Tests cover: initialisation errors, preprocessing output shape/dtype/range,
and the predict() softmax + top-k logic.
"""

import json

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


SAMPLE_LABELS = ["airplane", "automobile", "bird", "cat", "deer"]

# Logits that make class index 3 ("cat") the clear winner
MOCK_LOGITS = np.array([[0.1, 0.2, 0.3, 5.0, 0.4]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_files(tmp_path):
    """Write minimal placeholder files so path-existence checks pass."""
    onnx_path = tmp_path / "resnet18.onnx"
    onnx_path.write_bytes(b"placeholder")

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(SAMPLE_LABELS))

    return str(onnx_path), str(labels_path)


@pytest.fixture
def mock_session():
    """InferenceSession mock with controllable logit output."""
    session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    session.get_inputs.return_value = [mock_input]
    session.run.return_value = [MOCK_LOGITS]
    return session


@pytest.fixture
def classifier(model_files, mock_session):
    from src.classifier import VisionClassifier
    onnx_path, labels_path = model_files
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        clf = VisionClassifier(model_path=onnx_path, labels_path=labels_path)
    return clf


@pytest.fixture
def sample_image(tmp_path):
    """Small RGB PNG on disk."""
    img = Image.new("RGB", (64, 64), color=(100, 149, 237))
    path = tmp_path / "test.png"
    img.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Initialisation — error handling
# ---------------------------------------------------------------------------

def test_raises_if_onnx_missing(tmp_path):
    from src.classifier import VisionClassifier
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(SAMPLE_LABELS))

    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        VisionClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            labels_path=str(labels_path),
        )


def test_raises_if_labels_missing(tmp_path):
    from src.classifier import VisionClassifier
    onnx_path = tmp_path / "resnet18.onnx"
    onnx_path.write_bytes(b"placeholder")

    with patch("onnxruntime.InferenceSession"):
        with pytest.raises(FileNotFoundError, match="Labels file not found"):
            VisionClassifier(
                model_path=str(onnx_path),
                labels_path=str(tmp_path / "nonexistent.json"),
            )


# ---------------------------------------------------------------------------
# Initialisation — attributes
# ---------------------------------------------------------------------------

def test_device_is_cpu(classifier):
    assert classifier.device == "cpu"


def test_labels_loaded_correctly(classifier):
    assert classifier.labels == SAMPLE_LABELS


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------

def test_preprocess_output_shape(classifier, sample_image):
    arr = classifier._preprocess(sample_image)
    assert arr.shape == (1, 3, 224, 224)


def test_preprocess_output_dtype(classifier, sample_image):
    arr = classifier._preprocess(sample_image)
    assert arr.dtype == np.float32


def test_preprocess_produces_negative_values(classifier, sample_image):
    """ImageNet normalisation shifts some channels below zero."""
    arr = classifier._preprocess(sample_image)
    assert arr.min() < 0


def test_preprocess_produces_positive_values(classifier, sample_image):
    arr = classifier._preprocess(sample_image)
    assert arr.max() > 0


# ---------------------------------------------------------------------------
# predict — return structure
# ---------------------------------------------------------------------------

def test_predict_returns_list(classifier, sample_image):
    assert isinstance(classifier.predict(sample_image), list)


def test_predict_default_top_k_length(classifier, sample_image):
    """Default top_k=5 with 5 labels → all labels returned."""
    assert len(classifier.predict(sample_image)) == 5


def test_predict_respects_top_k(classifier, sample_image):
    assert len(classifier.predict(sample_image, top_k=2)) == 2


def test_predict_items_have_label_and_confidence(classifier, sample_image):
    for item in classifier.predict(sample_image):
        assert "label" in item
        assert "confidence" in item


def test_predict_label_is_string(classifier, sample_image):
    for item in classifier.predict(sample_image):
        assert isinstance(item["label"], str)


def test_predict_confidence_is_float(classifier, sample_image):
    for item in classifier.predict(sample_image):
        assert isinstance(item["confidence"], float)


# ---------------------------------------------------------------------------
# predict — correctness of softmax + top-k logic
# ---------------------------------------------------------------------------

def test_predict_top_result_matches_highest_logit(classifier, sample_image):
    """Index 3 ('cat') has the highest logit in MOCK_LOGITS."""
    results = classifier.predict(sample_image)
    assert results[0]["label"] == "cat"


def test_predict_confidences_are_descending(classifier, sample_image):
    confs = [r["confidence"] for r in classifier.predict(sample_image)]
    assert confs == sorted(confs, reverse=True)


def test_predict_confidences_sum_to_one(classifier, sample_image):
    """Returning all labels → softmax probabilities sum to 1."""
    total = sum(r["confidence"] for r in classifier.predict(sample_image))
    assert abs(total - 1.0) < 0.01


def test_predict_confidence_values_are_between_0_and_1(classifier, sample_image):
    for item in classifier.predict(sample_image):
        assert 0.0 <= item["confidence"] <= 1.0


def test_predict_calls_onnx_session(classifier, sample_image, mock_session):
    classifier.predict(sample_image)
    mock_session.run.assert_called_once()


def test_predict_passes_correct_input_name(classifier, sample_image, mock_session):
    classifier.predict(sample_image)
    call_kwargs = mock_session.run.call_args[0]
    feed_dict = call_kwargs[1]
    assert "input" in feed_dict
