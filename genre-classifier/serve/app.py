import logging
import threading
import time
from collections import Counter, deque

import joblib
from flask import Flask, request, jsonify, Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("genre-classifier")

MODEL_PATH = "/opt/genre-classifier/genre_pipeline.joblib"
MAX_TITLE_LEN = 200

app = Flask(__name__)
pipeline = joblib.load(MODEL_PATH)
labels = list(pipeline.named_steps["clf"].classes_)
log.info("loaded model, labels=%s", labels)

# --- lightweight in-memory telemetry (same pattern as agent-orchestrator) ---
_lock = threading.Lock()
_requests_total = 0
_errors_total = 0
_genre_counts = Counter()
_recent_latencies_ms = deque(maxlen=200)


@app.get("/health")
def health():
    return jsonify(status="ok", labels=labels)


@app.get("/metrics")
def metrics_endpoint():
    with _lock:
        req_total, err_total = _requests_total, _errors_total
        lat = list(_recent_latencies_ms)
    lines = [
        f"genre_classifier_requests_total {req_total}",
        f"genre_classifier_errors_total {err_total}",
    ]
    if lat:
        lat_sorted = sorted(lat)
        p50 = lat_sorted[len(lat_sorted) // 2]
        p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1]
        lines.append(f"genre_classifier_latency_ms_p50 {p50:.2f}")
        lines.append(f"genre_classifier_latency_ms_p95 {p95:.2f}")
    return Response("\n".join(lines) + "\n", mimetype="text/plain")


@app.get("/stats")
def stats():
    with _lock:
        req_total, err_total = _requests_total, _errors_total
        lat = list(_recent_latencies_ms)
        genre_counts = dict(_genre_counts.most_common())
    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else None
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1] if lat_sorted else None
    return jsonify(
        requests_total=req_total,
        errors_total=err_total,
        latency_ms={"p50": p50, "p95": p95, "samples": len(lat)},
        genre_counts=genre_counts,
    )


@app.post("/predict")
def predict():
    started = time.monotonic()
    global _requests_total, _errors_total
    with _lock:
        _requests_total += 1

    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()

    if not title:
        with _lock:
            _errors_total += 1
        return jsonify(error="title is required"), 400
    if len(title) > MAX_TITLE_LEN:
        with _lock:
            _errors_total += 1
        return jsonify(error=f"title too long (max {MAX_TITLE_LEN} chars)"), 400

    probs = pipeline.predict_proba([title])[0]
    ranked = sorted(zip(labels, probs), key=lambda kv: kv[1], reverse=True)

    elapsed_ms = (time.monotonic() - started) * 1000
    with _lock:
        _recent_latencies_ms.append(elapsed_ms)
        _genre_counts[ranked[0][0]] += 1
    log.info("predict title=%r top=%s elapsed_ms=%.1f", title, ranked[0], elapsed_ms)

    return jsonify(
        title=title,
        genre=ranked[0][0],
        confidence=round(float(ranked[0][1]), 4),
        scores=[{"genre": g, "probability": round(float(p), 4)} for g, p in ranked],
        elapsed_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8083)
