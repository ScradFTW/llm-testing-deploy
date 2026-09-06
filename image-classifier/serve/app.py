import base64
import io
import logging
import threading
import time
from collections import Counter, deque

import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, Response
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("image-classifier")

MODEL_DIR = "/opt/image-classifier"
MAX_IMAGE_BYTES = 3 * 1024 * 1024  # 3MB

app = Flask(__name__)

session = ort.InferenceSession(f"{MODEL_DIR}/image_classifier.onnx", providers=["CPUExecutionProvider"])
mean = np.load(f"{MODEL_DIR}/normalize_mean.npy").reshape(1, 3, 1, 1)
std = np.load(f"{MODEL_DIR}/normalize_std.npy").reshape(1, 3, 1, 1)
classes = open(f"{MODEL_DIR}/classes.txt").read().splitlines()
log.info("loaded ONNX model, classes=%s", classes)

_lock = threading.Lock()
_requests_total = 0
_errors_total = 0
_class_counts = Counter()
_recent_latencies_ms = deque(maxlen=200)


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]  # HWC -> NCHW
    arr = (arr - mean) / std
    return arr.astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


@app.get("/health")
def health():
    return jsonify(status="ok", classes=classes)


@app.get("/metrics")
def metrics_endpoint():
    with _lock:
        req_total, err_total = _requests_total, _errors_total
        lat = list(_recent_latencies_ms)
    lines = [
        f"image_classifier_requests_total {req_total}",
        f"image_classifier_errors_total {err_total}",
    ]
    if lat:
        lat_sorted = sorted(lat)
        p50 = lat_sorted[len(lat_sorted) // 2]
        p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1]
        lines.append(f"image_classifier_latency_ms_p50 {p50:.2f}")
        lines.append(f"image_classifier_latency_ms_p95 {p95:.2f}")
    return Response("\n".join(lines) + "\n", mimetype="text/plain")


@app.get("/stats")
def stats():
    with _lock:
        req_total, err_total = _requests_total, _errors_total
        lat = list(_recent_latencies_ms)
        class_counts = dict(_class_counts.most_common())
    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else None
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1] if lat_sorted else None
    return jsonify(
        requests_total=req_total,
        errors_total=err_total,
        latency_ms={"p50": p50, "p95": p95, "samples": len(lat)},
        class_counts=class_counts,
    )


@app.post("/predict")
def predict():
    started = time.monotonic()
    global _requests_total, _errors_total
    with _lock:
        _requests_total += 1

    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image") or ""
    if "," in image_b64[:60]:  # tolerate a data: URL prefix
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception:
        with _lock:
            _errors_total += 1
        return jsonify(error="image must be base64-encoded"), 400

    if not image_bytes:
        with _lock:
            _errors_total += 1
        return jsonify(error="image is required"), 400
    if len(image_bytes) > MAX_IMAGE_BYTES:
        with _lock:
            _errors_total += 1
        return jsonify(error="image too large (max 3MB)"), 400

    try:
        x = preprocess(image_bytes)
    except Exception:
        with _lock:
            _errors_total += 1
        return jsonify(error="could not decode image"), 400

    logits = session.run(None, {"image": x})[0][0]
    probs = softmax(logits)
    ranked = sorted(zip(classes, probs.tolist()), key=lambda kv: kv[1], reverse=True)

    elapsed_ms = (time.monotonic() - started) * 1000
    with _lock:
        _recent_latencies_ms.append(elapsed_ms)
        _class_counts[ranked[0][0]] += 1
    log.info("predict top=%s elapsed_ms=%.1f", ranked[0], elapsed_ms)

    return jsonify(
        label=ranked[0][0],
        confidence=round(float(ranked[0][1]), 4),
        scores=[{"label": l, "probability": round(float(p), 4)} for l, p in ranked],
        elapsed_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8085)
