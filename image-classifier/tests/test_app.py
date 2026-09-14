"""Tests for image-classifier/serve/app.py.

Runs /predict against the real, committed ONNX model
(image-classifier/train/image_classifier.onnx) -- see conftest.py for how
that file (plus the normalization stats and class list) is substituted in
for app.py's hardcoded /opt path. No mocking of the model itself: it's a
small CIFAR-10-sized network and CPU inference on a 32x32 image is fast
enough to run directly in tests.
"""
import base64
import io

import app as app_module
import pytest
from PIL import Image


def _png_b64(color=(120, 40, 200), size=(32, 32)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture()
def client():
    return app_module.app.test_client()


@pytest.fixture()
def valid_image_b64():
    return _png_b64()


# --- /health -------------------------------------------------------------


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["classes"] == app_module.classes
    assert len(body["classes"]) > 0


# --- /predict: valid input ------------------------------------------------


def test_predict_valid_image_returns_ranked_scores(client, valid_image_b64):
    resp = client.post("/predict", json={"image": valid_image_b64})
    assert resp.status_code == 200
    body = resp.get_json()

    classes = app_module.classes
    assert body["label"] in classes
    assert len(body["scores"]) == len(classes)
    assert {s["label"] for s in body["scores"]} == set(classes)

    probs = [s["probability"] for s in body["scores"]]
    assert probs == sorted(probs, reverse=True)
    assert body["scores"][0]["label"] == body["label"]
    assert body["scores"][0]["probability"] == pytest.approx(body["confidence"])
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0, abs=1e-3)
    assert isinstance(body["elapsed_ms"], (int, float))


def test_predict_accepts_data_url_prefix(client, valid_image_b64):
    # app.py explicitly tolerates a "data:image/png;base64,..." prefix by
    # stripping everything up to (and including) the first comma.
    resp = client.post(
        "/predict", json={"image": f"data:image/png;base64,{valid_image_b64}"}
    )
    assert resp.status_code == 200
    assert resp.get_json()["label"] in app_module.classes


def test_predict_is_deterministic_for_same_image(client, valid_image_b64):
    r1 = client.post("/predict", json={"image": valid_image_b64}).get_json()
    r2 = client.post("/predict", json={"image": valid_image_b64}).get_json()
    assert r1["label"] == r2["label"]
    assert r1["confidence"] == r2["confidence"]
    assert r1["scores"] == r2["scores"]


# --- /predict: missing / malformed input ----------------------------------


def test_predict_missing_image_key_is_400(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "image is required"


def test_predict_empty_image_string_is_400(client):
    resp = client.post("/predict", json={"image": ""})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "image is required"


def test_predict_no_body_at_all_is_400(client):
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "image is required"


def test_predict_malformed_base64_is_400(client):
    resp = client.post("/predict", json={"image": "not-valid-base64!!"})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "image must be base64-encoded"


def test_predict_valid_base64_non_image_bytes_is_400(client):
    # Well-formed base64, but the decoded bytes aren't a real image -- PIL
    # should fail to open it and the handler should turn that into a 400,
    # not a 500.
    garbage = base64.b64encode(b"just some plain text, not image data at all").decode()
    resp = client.post("/predict", json={"image": garbage})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "could not decode image"


def test_predict_image_too_large_is_400(client):
    oversized = base64.b64encode(b"0" * (3 * 1024 * 1024 + 1)).decode()
    resp = client.post("/predict", json={"image": oversized})
    assert resp.status_code == 400
    assert "too large" in resp.get_json()["error"]


def test_predict_non_rgb_image_is_converted_and_succeeds(client):
    # Grayscale ("L" mode) input exercises the .convert("RGB") step in
    # preprocess() rather than only ever testing images already in RGB.
    img = Image.new("L", (32, 32), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    resp = client.post("/predict", json={"image": b64})
    assert resp.status_code == 200
    assert resp.get_json()["label"] in app_module.classes


def test_predict_non_square_image_is_resized_and_succeeds(client):
    # preprocess() resizes to a fixed 32x32 regardless of input dimensions.
    img = Image.new("RGB", (64, 16), color=(10, 200, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    resp = client.post("/predict", json={"image": b64})
    assert resp.status_code == 200
    assert resp.get_json()["label"] in app_module.classes
