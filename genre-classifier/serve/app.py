import logging
import time

import joblib
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("genre-classifier")

MODEL_PATH = "/opt/genre-classifier/genre_pipeline.joblib"
MAX_TITLE_LEN = 200

app = Flask(__name__)
pipeline = joblib.load(MODEL_PATH)
labels = list(pipeline.named_steps["clf"].classes_)
log.info("loaded model, labels=%s", labels)


@app.get("/health")
def health():
    return jsonify(status="ok", labels=labels)


@app.post("/predict")
def predict():
    started = time.monotonic()
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()

    if not title:
        return jsonify(error="title is required"), 400
    if len(title) > MAX_TITLE_LEN:
        return jsonify(error=f"title too long (max {MAX_TITLE_LEN} chars)"), 400

    probs = pipeline.predict_proba([title])[0]
    ranked = sorted(zip(labels, probs), key=lambda kv: kv[1], reverse=True)

    elapsed_ms = (time.monotonic() - started) * 1000
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
