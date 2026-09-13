# app.py's MODEL_DIR is hardcoded to /opt/image-classifier — kept as-is
# (no code change) by reproducing that path inside the image instead.
FROM python:3.12-slim
WORKDIR /app

COPY image-classifier/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY image-classifier/serve/app.py .
COPY image-classifier/train/image_classifier.onnx /opt/image-classifier/image_classifier.onnx
COPY image-classifier/train/normalize_mean.npy /opt/image-classifier/normalize_mean.npy
COPY image-classifier/train/normalize_std.npy /opt/image-classifier/normalize_std.npy
COPY image-classifier/train/classes.txt /opt/image-classifier/classes.txt

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "waitress-serve --host=0.0.0.0 --port=${PORT} --threads=2 app:app"]
