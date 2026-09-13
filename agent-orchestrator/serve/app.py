import json
import logging
import os
import re
import threading
import time
from collections import deque

import requests
from flask import Flask, request, jsonify, Response
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.id_token import fetch_id_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent-orchestrator")

# Both point at other Cloud Run services post-migration (see
# bradjobe-dev-infra/cloud_run.tf, which sets these env vars) instead of
# the old same-VPS 127.0.0.1 addresses.
LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8082/v1/chat/completions")
GENRE_URL = os.environ.get("GENRE_URL", "http://127.0.0.1:8083/predict")
MAX_MESSAGE_LEN = 300
MAX_TITLE_LEN = 120

_google_auth_request = GoogleAuthRequest()


def _genre_classifier_auth_header():
    """Cloud Run requires an identity token to call genre-classifier, whose
    ingress is locked to internal + load-balancer traffic (see
    bradjobe-dev-infra's cloud_run.tf INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER
    and the run.invoker grant scoped to this service's own account). Not
    needed — and fetch_id_token would fail — when running locally against
    127.0.0.1, so this degrades to no header outside Cloud Run.
    """
    if not GENRE_URL.startswith("https://"):
        return {}
    audience = GENRE_URL.split("/predict", 1)[0]
    try:
        token = fetch_id_token(_google_auth_request, audience)
        return {"Authorization": f"Bearer {token}"}
    except Exception:
        log.exception("failed to mint ID token for genre-classifier; calling unauthenticated")
        return {}

# The model is only shown one tool. It is NOT trusted to decide on its own
# whether the tool is relevant -- see GENRE_INTENT_RE below, which is the
# real gate. This prompt exists only to get the *shape* of a tool call out
# of a 0.5B model, which is already unreliable enough (see README).
TOOL_SYSTEM_PROMPT = (
    'You classify user messages into two types. TYPE A: the user is asking '
    'what genre/kind of music a specific song is. TYPE B: anything else. '
    'For TYPE A, respond with ONLY this JSON: {"tool": "get_song_genre", '
    '"title": "<song title>"}. For TYPE B, respond normally in plain text, '
    "never JSON.\n\n"
    "Examples:\n"
    "User: what genre is Hey Jude?\n"
    'Assistant: {"tool": "get_song_genre", "title": "Hey Jude"}\n'
    "User: hello, how are you?\n"
    "Assistant: I am doing well, thanks for asking! How can I help you?\n"
    "User: what is the capital of France?\n"
    "Assistant: The capital of France is Paris."
)

PLAIN_SYSTEM_PROMPT = (
    "You are a small, friendly demo assistant running locally on a "
    "single-core VPS. Keep replies short (a few sentences at most)."
)

# Deterministic guardrail: the model's own judgment about *when* to call the
# tool is not trusted (measured false-trigger rate without this gate was
# ~80%+ in testing -- it proposed the tool for "hello, how are you?" and
# "what is the capital of France?"). This regex is the actual gate; the
# model only supplies the shape of the call (the extracted title).
GENRE_INTENT_RE = re.compile(
    r"\b(genre|kind of music|type of (music|song)|classify|categori[sz]e)\b",
    re.IGNORECASE,
)

TOOL_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

app = Flask(__name__)

# --- lightweight in-memory telemetry (no external deps / DB needed) ---
_lock = threading.Lock()
metrics = {
    "requests_total": 0,
    "errors_total": 0,
    "tool_proposed_total": 0,
    "tool_blocked_total": 0,
    "tool_malformed_total": 0,
    "tool_executed_total": 0,
    "tool_exec_errors_total": 0,
}
recent_latencies_ms = deque(maxlen=200)
recent_events = deque(maxlen=50)  # small ring buffer for the dashboard/trace


def record_event(**fields):
    fields["ts"] = time.time()
    with _lock:
        recent_events.append(fields)


def bump(key, n=1):
    with _lock:
        metrics[key] += n


def call_llama(messages, max_tokens=80):
    resp = requests.post(
        LLAMA_URL,
        json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.2},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_genre_classifier(title):
    resp = requests.post(
        GENRE_URL,
        json={"title": title},
        headers=_genre_classifier_auth_header(),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def try_parse_tool_call(raw_text):
    """Returns a dict if `raw_text` looks like our tool-call JSON, else None."""
    match = TOOL_JSON_RE.search(raw_text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("tool") != "get_song_genre":
        return None
    title = obj.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    return {"title": title.strip()[:MAX_TITLE_LEN]}


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/metrics")
def metrics_endpoint():
    with _lock:
        m = dict(metrics)
        lat = list(recent_latencies_ms)
    lines = [f"agent_orchestrator_{k} {v}" for k, v in m.items()]
    if lat:
        lat_sorted = sorted(lat)
        p50 = lat_sorted[len(lat_sorted) // 2]
        p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1]
        lines.append(f"agent_orchestrator_latency_ms_p50 {p50:.1f}")
        lines.append(f"agent_orchestrator_latency_ms_p95 {p95:.1f}")
    return Response("\n".join(lines) + "\n", mimetype="text/plain")


@app.get("/stats")
def stats():
    with _lock:
        m = dict(metrics)
        lat = list(recent_latencies_ms)
        events = list(recent_events)[::-1]
    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else None
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1] if lat_sorted else None
    tool_attempts = m["tool_proposed_total"] + m["tool_malformed_total"]
    trigger_precision = None
    if tool_attempts > 0:
        trigger_precision = m["tool_executed_total"] / tool_attempts
    return jsonify(
        metrics=m,
        latency_ms={"p50": p50, "p95": p95, "samples": len(lat)},
        tool_trigger_precision=trigger_precision,
        recent_events=events,
    )


@app.post("/chat")
def chat():
    started = time.monotonic()
    bump("requests_total")

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify(error="message is required"), 400
    if len(user_message) > MAX_MESSAGE_LEN:
        return jsonify(error=f"message too long (max {MAX_MESSAGE_LEN} chars)"), 400

    trace = {"tool_proposed": False, "tool_executed": False, "blocked_reason": None}

    try:
        raw = call_llama(
            [
                {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )
        trace["raw_model_output"] = raw

        proposed = try_parse_tool_call(raw)

        if proposed is None and TOOL_JSON_RE.search(raw):
            # The model emitted *some* JSON-shaped output, but not our
            # expected schema -- e.g. it invented an unrequested tool
            # ("get_user_status") for a plain greeting. This is a real
            # failure mode observed while testing this 0.5B model, not a
            # hypothetical: never surface raw model JSON to the user for
            # it, re-ground it with a plain prompt instead.
            trace["blocked_reason"] = "guardrail: malformed/unrecognized tool call from model"
            bump("tool_malformed_total")
            record_event(type="tool_malformed", user_message=user_message, raw_model_output=raw)
            reply = call_llama(
                [
                    {"role": "system", "content": PLAIN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]
            )
        elif proposed is None:
            reply = raw
        else:
            trace["tool_proposed"] = True
            trace["proposed_title"] = proposed["title"]
            bump("tool_proposed_total")

            if not GENRE_INTENT_RE.search(user_message):
                # Guardrail: the model wants to call the tool, but the
                # user's actual message doesn't look like a genre question.
                # Don't trust the model's judgment -- fall back instead.
                trace["blocked_reason"] = "guardrail: no genre-intent keywords in user message"
                bump("tool_blocked_total")
                record_event(type="tool_blocked", user_message=user_message, proposed=proposed["title"])
                reply = call_llama(
                    [
                        {"role": "system", "content": PLAIN_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                try:
                    result = call_genre_classifier(proposed["title"])
                    trace["tool_executed"] = True
                    trace["tool_result"] = result
                    bump("tool_executed_total")
                    record_event(
                        type="tool_executed",
                        title=proposed["title"],
                        genre=result["genre"],
                        confidence=result["confidence"],
                    )
                    reply = (
                        f"I classified \"{result['title']}\" as **{result['genre']}** "
                        f"({result['confidence'] * 100:.0f}% confidence), using the "
                        f"genre-classifier model (not the language model itself)."
                    )
                except Exception:
                    trace["blocked_reason"] = "tool execution failed"
                    bump("tool_exec_errors_total")
                    log.exception("genre-classifier call failed")
                    reply = (
                        f"I tried to look up the genre for \"{proposed['title']}\" "
                        "but the classifier service didn't respond. Please try again."
                    )

        elapsed_ms = (time.monotonic() - started) * 1000
        recent_latencies_ms.append(elapsed_ms)
        trace["elapsed_ms"] = round(elapsed_ms, 1)
        log.info("chat message=%r trace=%s", user_message, {k: v for k, v in trace.items() if k != "raw_model_output"})
        return jsonify(reply=reply, trace=trace)

    except Exception:
        bump("errors_total")
        log.exception("chat request failed")
        return jsonify(error="internal error"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8084)
