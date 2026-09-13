FROM python:3.12-slim
WORKDIR /app

COPY agent-orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent-orchestrator/serve/app.py .

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "waitress-serve --host=0.0.0.0 --port=${PORT} --threads=4 app:app"]
