# app.py's MODEL_PATH is hardcoded to /opt/genre-classifier/... — kept
# as-is (no code change) by reproducing that path inside the image instead.
FROM python:3.12-slim
WORKDIR /app

COPY genre-classifier/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY genre-classifier/serve/app.py .
COPY genre-classifier/train/genre_pipeline.joblib /opt/genre-classifier/genre_pipeline.joblib

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "waitress-serve --host=0.0.0.0 --port=${PORT} --threads=2 app:app"]
