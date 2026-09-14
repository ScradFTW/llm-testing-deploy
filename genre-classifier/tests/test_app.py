"""Tests for genre-classifier/serve/app.py.

Runs /predict against the real, committed scikit-learn pipeline
(genre-classifier/train/genre_pipeline.joblib) -- see conftest.py for how
that file is substituted in for app.py's hardcoded /opt path. No mocking:
the model is small and inference is fast enough to run directly.
"""
import app as app_module
import pytest


@pytest.fixture()
def client():
    # Deliberately not setting TESTING=True: that flag makes Flask propagate
    # unhandled exceptions to the test instead of returning them as the 500
    # response a real client would receive, which is what
    # test_predict_non_string_title_is_rejected_not_silently_accepted below
    # needs to observe.
    return app_module.app.test_client()


# --- /health -----------------------------------------------------------


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    # Labels come straight from the loaded pipeline's classes_.
    assert isinstance(body["labels"], list)
    assert len(body["labels"]) > 0


# --- /predict: valid input ----------------------------------------------


def test_predict_valid_title_returns_ranked_scores(client):
    resp = client.post("/predict", json={"title": "Bohemian Rhapsody"})
    assert resp.status_code == 200
    body = resp.get_json()

    assert body["title"] == "Bohemian Rhapsody"
    labels = app_module.labels
    assert body["genre"] in labels

    # scores covers every known label exactly once, sorted descending by
    # probability, and the top entry matches the reported genre/confidence.
    assert len(body["scores"]) == len(labels)
    assert {s["genre"] for s in body["scores"]} == set(labels)
    probs = [s["probability"] for s in body["scores"]]
    assert probs == sorted(probs, reverse=True)
    assert body["scores"][0]["genre"] == body["genre"]
    assert body["scores"][0]["probability"] == pytest.approx(body["confidence"])

    # Probabilities from predict_proba should be valid and sum to ~1.
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0, abs=1e-3)

    assert isinstance(body["elapsed_ms"], (int, float))


def test_predict_strips_surrounding_whitespace(client):
    resp = client.post("/predict", json={"title": "  Hey Jude  "})
    assert resp.status_code == 200
    assert resp.get_json()["title"] == "Hey Jude"


def test_predict_is_deterministic_for_same_title(client):
    r1 = client.post("/predict", json={"title": "Hey Jude"}).get_json()
    r2 = client.post("/predict", json={"title": "Hey Jude"}).get_json()
    assert r1["genre"] == r2["genre"]
    assert r1["confidence"] == r2["confidence"]
    assert r1["scores"] == r2["scores"]


# --- /predict: missing / malformed input --------------------------------


def test_predict_missing_title_key_is_400(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "title is required"


def test_predict_blank_title_is_400(client):
    resp = client.post("/predict", json={"title": "   "})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "title is required"


def test_predict_no_body_at_all_is_400(client):
    # request.get_json(silent=True) returns None for a body that isn't
    # valid JSON, and the handler treats that the same as an empty dict.
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "title is required"


def test_predict_title_too_long_is_400(client):
    resp = client.post("/predict", json={"title": "x" * 201})
    assert resp.status_code == 400
    assert "too long" in resp.get_json()["error"]


def test_predict_title_at_max_length_is_accepted(client):
    resp = client.post("/predict", json={"title": "x" * 200})
    assert resp.status_code == 200


def test_predict_non_string_title_is_rejected_not_silently_accepted(client):
    # `title` is truthy-checked before `.strip()`, so a non-string value
    # (e.g. a number) is not silently coerced into a valid prediction --
    # it fails loudly (500) rather than returning a bogus classification.
    resp = client.post("/predict", json={"title": 12345})
    assert resp.status_code == 500
