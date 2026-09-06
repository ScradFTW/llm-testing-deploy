# agent-orchestrator — reasoning agent with one guarded tool

Live at: https://bradjobe.dev/agent-demo/

This is the piece that ties `llm-testing` and `genre-classifier` together,
and the one most directly aimed at "reasoning agent infrastructure": tool
execution, state management, and guardrails around an LLM that can't fully
be trusted to decide for itself when to act.

## The finding that shaped the design

The first version trusted the LLM's own judgment about when to call the
tool. Testing it immediately showed that was wrong:

| User message | Model's response |
|---|---|
| "what genre is Bohemian Rhapsody?" | `{"tool": "get_song_genre", "title": "Bohemian Rhapsody"}` (correct) |
| "what is the capital of France?" | `{"tool": "get_song_genre", "title": "Capital of France"}` |
| "hello, how are you?" | `{"tool": "get_user_status", "status": "good"}` (a tool that doesn't exist) |
| "tell me a joke" | `{"tool": "get_joke", "title": "Why did the tomato turn red?"}` (also invented) |

A 0.5B model isn't reliable enough to gate its own tool use, even with
few-shot examples in the prompt. So it doesn't: `serve/app.py` treats the
model's proposed tool call as a *suggestion*, not a decision.

## Request flow

1. Send the user's message to the LLM with a prompt describing one tool.
2. Parse the response. Three outcomes:
   - **Not JSON at all** → it's a normal answer, return it directly.
   - **JSON, but not our schema** (`tool != "get_song_genre"`, or missing
     fields) → the model hallucinated some other tool call. Never surface
     that raw JSON to the user — re-ask the model with a plain prompt
     instead. (`tool_malformed`)
   - **Valid schema** → check the *user's original message* against a
     keyword regex (`genre`, `kind of music`, `classify`, ...). This regex,
     not the model, is the actual gate.
     - Keywords absent → the model wanted to call the tool, but the
       message doesn't look like a genre question. Block it, re-ask with a
       plain prompt. (`tool_blocked`)
     - Keywords present → call `genre-classifier` over loopback, then ask
       the LLM once more to phrase a final answer using the result.
       (`tool_executed`)
3. Every outcome is counted and the last 50 are kept in memory for
   `/stats` — see `../status-dashboard/`.

## Why this is the interesting part of the project

Anyone can wire an LLM to a function and call it "agentic" when the demo
inputs are cherry-picked. The useful engineering here is: measuring the
failure rate honestly (tool-trigger precision is often under 60% in casual
use — visible live on `/status`), and designing the system so a highly
unreliable component (a 0.5B model's judgment) can't cause a bad outcome
even when it's wrong most of the time.

## Serving

Same pattern as the other two services: Flask + waitress,
`systemd/agent-orchestrator.service` (`Restart=always`, `MemoryMax`,
sandboxed, loopback-only on `127.0.0.1:8084`), reverse-proxied and rate
limited by nginx, behind the same Basic Auth.
