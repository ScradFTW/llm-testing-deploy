"""Tests for agent-orchestrator/serve/app.py.

Covers:
  - try_parse_tool_call: JSON parsing/schema-validation of the model's
    proposed tool call (valid schema, wrong schema, not JSON at all).
  - GENRE_INTENT_RE: the deterministic guardrail that gates whether a
    proposed tool call is actually allowed to execute.
  - _genre_classifier_auth_header: degrades to no header against a plain
    HTTP GENRE_URL (local/dev), mints a bearer token via a mocked
    fetch_id_token against an https:// GENRE_URL, and degrades gracefully
    if minting fails.
  - /chat: the full request handler, with requests.post to LLAMA_URL /
    GENRE_URL mocked so no network call is ever made.
"""
from unittest import mock

import app as app_module
import pytest


@pytest.fixture()
def client():
    return app_module.app.test_client()


def _fake_response(json_body, raise_exc=None):
    resp = mock.Mock()
    resp.raise_for_status.side_effect = raise_exc
    resp.json.return_value = json_body
    return resp


def _llama_content(text):
    return {"choices": [{"message": {"content": text}}]}


# --- try_parse_tool_call: JSON parsing / schema validation ----------------


def test_try_parse_tool_call_valid_schema():
    raw = '{"tool": "get_song_genre", "title": "Hey Jude"}'
    assert app_module.try_parse_tool_call(raw) == {"title": "Hey Jude"}


def test_try_parse_tool_call_extracts_json_from_surrounding_text():
    raw = 'Sure thing! {"tool": "get_song_genre", "title": "Hey Jude"} Hope that helps.'
    assert app_module.try_parse_tool_call(raw) == {"title": "Hey Jude"}


def test_try_parse_tool_call_strips_and_truncates_title():
    raw = '{"tool": "get_song_genre", "title": "  Hey Jude  "}'
    assert app_module.try_parse_tool_call(raw) == {"title": "Hey Jude"}

    long_title = "x" * 200
    result = app_module.try_parse_tool_call(
        '{"tool": "get_song_genre", "title": "%s"}' % long_title
    )
    assert result["title"] == long_title[: app_module.MAX_TITLE_LEN]


def test_try_parse_tool_call_not_json_at_all():
    assert app_module.try_parse_tool_call("hello, how are you?") is None
    assert app_module.try_parse_tool_call("The capital of France is Paris.") is None


def test_try_parse_tool_call_wrong_tool_name():
    raw = '{"tool": "get_user_status", "title": "Hey Jude"}'
    assert app_module.try_parse_tool_call(raw) is None


def test_try_parse_tool_call_malformed_json():
    # Trailing comma makes this invalid JSON, not just an unexpected shape.
    raw = '{"tool": "get_song_genre", "title": "Hey Jude",}'
    assert app_module.try_parse_tool_call(raw) is None


def test_try_parse_tool_call_missing_or_blank_title():
    assert app_module.try_parse_tool_call('{"tool": "get_song_genre"}') is None
    assert app_module.try_parse_tool_call('{"tool": "get_song_genre", "title": ""}') is None
    assert app_module.try_parse_tool_call('{"tool": "get_song_genre", "title": "   "}') is None


def test_try_parse_tool_call_non_string_title():
    raw = '{"tool": "get_song_genre", "title": 123}'
    assert app_module.try_parse_tool_call(raw) is None


def test_try_parse_tool_call_json_but_not_an_object_shape():
    # {} matches TOOL_JSON_RE but isn't our dict schema once parsed further
    # up in the tool-name check.
    assert app_module.try_parse_tool_call('{"unexpected": true}') is None


# --- GENRE_INTENT_RE: the deterministic guardrail -------------------------


@pytest.mark.parametrize(
    "message",
    [
        "what genre is Hey Jude?",
        "what kind of music is this?",
        "what type of music is this song?",
        "what type of song is this?",
        "can you classify this track?",
        "please categorize this song",
        "please categorise this song",
        "GENRE please",  # case-insensitive
    ],
)
def test_genre_intent_regex_matches_genre_questions(message):
    assert app_module.GENRE_INTENT_RE.search(message)


@pytest.mark.parametrize(
    "message",
    [
        "hello, how are you?",
        "what is the capital of France?",
        "what is your favorite song?",  # no genre-ish keyword
    ],
)
def test_genre_intent_regex_does_not_match_unrelated_messages(message):
    assert app_module.GENRE_INTENT_RE.search(message) is None


# --- _genre_classifier_auth_header ----------------------------------------


def test_auth_header_empty_when_genre_url_not_https(monkeypatch):
    monkeypatch.setattr(app_module, "GENRE_URL", "http://127.0.0.1:8083/predict")
    fetch = mock.Mock()
    monkeypatch.setattr(app_module, "fetch_id_token", fetch)

    assert app_module._genre_classifier_auth_header() == {}
    fetch.assert_not_called()


def test_auth_header_bearer_token_when_genre_url_https(monkeypatch):
    monkeypatch.setattr(
        app_module, "GENRE_URL", "https://genre-classifier-abc-uc.a.run.app/predict"
    )
    fetch = mock.Mock(return_value="fake-id-token")
    monkeypatch.setattr(app_module, "fetch_id_token", fetch)

    header = app_module._genre_classifier_auth_header()

    assert header == {"Authorization": "Bearer fake-id-token"}
    # audience passed to fetch_id_token is the URL with /predict stripped.
    args, _ = fetch.call_args
    assert args[1] == "https://genre-classifier-abc-uc.a.run.app"


def test_auth_header_degrades_to_no_header_on_fetch_failure(monkeypatch):
    monkeypatch.setattr(
        app_module, "GENRE_URL", "https://genre-classifier-abc-uc.a.run.app/predict"
    )
    monkeypatch.setattr(
        app_module, "fetch_id_token", mock.Mock(side_effect=RuntimeError("no metadata server"))
    )

    assert app_module._genre_classifier_auth_header() == {}


# --- /chat: input validation -----------------------------------------------


def test_chat_missing_message_is_400(client):
    resp = client.post("/chat", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "message is required"


def test_chat_blank_message_is_400(client):
    resp = client.post("/chat", json={"message": "   "})
    assert resp.status_code == 400


def test_chat_message_too_long_is_400(client):
    resp = client.post("/chat", json={"message": "x" * 301})
    assert resp.status_code == 400
    assert "too long" in resp.get_json()["error"]


# --- /chat: full flow, outbound HTTP mocked -------------------------------


def test_chat_plain_message_passthrough_when_no_tool_proposed(client, monkeypatch):
    fake_post = mock.Mock(
        return_value=_fake_response(_llama_content("I am doing well, thanks!"))
    )
    monkeypatch.setattr(app_module.requests, "post", fake_post)

    resp = client.post("/chat", json={"message": "hello, how are you?"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["reply"] == "I am doing well, thanks!"
    assert body["trace"]["tool_proposed"] is False
    assert body["trace"]["tool_executed"] is False
    fake_post.assert_called_once()
    assert fake_post.call_args[0][0] == app_module.LLAMA_URL


def test_chat_tool_proposed_and_intent_matches_executes_tool(client, monkeypatch):
    llama_resp = _fake_response(
        _llama_content('{"tool": "get_song_genre", "title": "Hey Jude"}')
    )
    genre_resp = {"title": "Hey Jude", "genre": "Rock", "confidence": 0.8}

    def fake_post(url, json=None, headers=None, timeout=None):
        if url == app_module.LLAMA_URL:
            return llama_resp
        if url == app_module.GENRE_URL:
            assert json == {"title": "Hey Jude"}
            return _fake_response(genre_resp)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(app_module.requests, "post", fake_post)

    resp = client.post("/chat", json={"message": "what genre is Hey Jude?"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert "Rock" in body["reply"]
    assert "Hey Jude" in body["reply"]
    trace = body["trace"]
    assert trace["tool_proposed"] is True
    assert trace["tool_executed"] is True
    assert trace["blocked_reason"] is None
    assert trace["tool_result"] == genre_resp


def test_chat_tool_proposed_but_no_genre_intent_is_blocked(client, monkeypatch):
    # The model proposes the tool, but the user's own message has no
    # genre-ish keywords -- the deterministic guardrail should block it and
    # fall back to a plain-prompt reply, and the genre-classifier must
    # never be called.
    llama_responses = iter(
        [
            _llama_content('{"tool": "get_song_genre", "title": "Hey Jude"}'),
            _llama_content("Sure, here's a plain reply about Hey Jude."),
        ]
    )
    genre_post = mock.Mock()

    def fake_post(url, json=None, headers=None, timeout=None):
        if url == app_module.LLAMA_URL:
            return _fake_response(next(llama_responses))
        if url == app_module.GENRE_URL:
            genre_post(url, json)
            return _fake_response({})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(app_module.requests, "post", fake_post)

    resp = client.post("/chat", json={"message": "tell me about Hey Jude"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["reply"] == "Sure, here's a plain reply about Hey Jude."
    trace = body["trace"]
    assert trace["tool_proposed"] is True
    assert trace["tool_executed"] is False
    assert trace["blocked_reason"] == "guardrail: no genre-intent keywords in user message"
    genre_post.assert_not_called()


def test_chat_malformed_tool_json_is_reground_with_plain_prompt(client, monkeypatch):
    # The model emits JSON, but not our schema (wrong/unrequested tool) --
    # this must never be surfaced to the user, and must trigger a second,
    # plain-prompted call rather than executing anything.
    llama_responses = iter(
        [
            _llama_content('{"tool": "get_user_status", "user": "bob"}'),
            _llama_content("I'm just a demo assistant, how can I help?"),
        ]
    )
    genre_post = mock.Mock()

    def fake_post(url, json=None, headers=None, timeout=None):
        if url == app_module.LLAMA_URL:
            return _fake_response(next(llama_responses))
        if url == app_module.GENRE_URL:
            genre_post()
            return _fake_response({})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(app_module.requests, "post", fake_post)

    resp = client.post("/chat", json={"message": "how's it going?"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["reply"] == "I'm just a demo assistant, how can I help?"
    trace = body["trace"]
    assert trace["tool_proposed"] is False
    assert trace["blocked_reason"] == "guardrail: malformed/unrecognized tool call from model"
    genre_post.assert_not_called()


def test_chat_tool_execution_failure_is_handled_gracefully(client, monkeypatch):
    llama_resp = _fake_response(
        _llama_content('{"tool": "get_song_genre", "title": "Hey Jude"}')
    )

    def fake_post(url, json=None, headers=None, timeout=None):
        if url == app_module.LLAMA_URL:
            return llama_resp
        if url == app_module.GENRE_URL:
            raise app_module.requests.exceptions.ConnectionError("genre-classifier unreachable")
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(app_module.requests, "post", fake_post)

    resp = client.post("/chat", json={"message": "what genre is Hey Jude?"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert "didn't respond" in body["reply"]
    trace = body["trace"]
    assert trace["tool_proposed"] is True
    assert trace["tool_executed"] is False
    assert trace["blocked_reason"] == "tool execution failed"


def test_chat_llama_failure_returns_500_internal_error(client, monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        raise app_module.requests.exceptions.ConnectionError("llama-server unreachable")

    monkeypatch.setattr(app_module.requests, "post", fake_post)

    resp = client.post("/chat", json={"message": "hello there"})

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "internal error"
