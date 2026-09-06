# status-dashboard — live telemetry

Live at: https://bradjobe.dev/status/

Not a new backend service — a static page (`frontend/status/`) that polls
each existing service's own `/stats` JSON endpoint every 5 seconds and
renders it: request counts, p50/p95 latency, the genre-classifier's live
prediction distribution, the agent-orchestrator's tool-trigger precision,
and a running event log of every agent decision (executed / blocked /
malformed).

## Why in-process counters instead of Prometheus/Grafana

At three low-traffic services on a single 2GB box, running a full metrics
stack would cost more memory than the services being monitored. Both
`genre-classifier` and `agent-orchestrator` keep a small in-memory counter
set (`threading.Lock` + a bounded `deque` for latency samples and recent
events) and expose it two ways:

- `/stats` — JSON, consumed by this dashboard.
- `/metrics` — Prometheus text exposition format, so the same data could be
  scraped by a real Prometheus instance without any code changes if this
  ever needed to graduate off a single box.

nginx proxies `/status/api/agent-stats` → `agent-orchestrator:8084/stats`
and `/status/api/genre-stats` → `genre-classifier:8083/stats`.
`llama-server` has no HTTP stats endpoint of its own; its per-request
timing (prompt/eval tokens-per-second) goes to the systemd journal instead
(`journalctl -u llama-server`) and is represented on the dashboard
indirectly, through the agent-orchestrator calls that depend on it.

## What it's for

Every number on this page is real, not a mockup — it reflects whatever
traffic (yours or a demo run) actually hit these services since they last
restarted. The tool-trigger-precision figure in particular is the
project's one online evaluation metric: how often the LLM's own tool
decision agreed with the deterministic guardrail in `agent-orchestrator`.
