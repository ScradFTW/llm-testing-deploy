# status-dashboard — live telemetry

Live at: https://bradjobe.dev/status/

Not a new backend service — a static page (`frontend/status/`) that polls
each existing service's own `/stats` JSON endpoint every 5 seconds and
renders it: request counts, p50/p95 latency, the genre-classifier's live
prediction distribution, the agent-orchestrator's tool-trigger precision,
and a running event log of every agent decision (executed / blocked /
malformed).

## Why in-process counters instead of Prometheus/Grafana

For three low-traffic Cloud Run services, standing up a full metrics stack
(Cloud Monitoring aside) would be more operational surface than the thing
being monitored. Both `genre-classifier` and `agent-orchestrator` keep a
small in-memory counter set (`threading.Lock` + a bounded `deque` for
latency samples and recent events) and expose it two ways:

- `/stats` — JSON, consumed by this dashboard.
- `/metrics` — Prometheus text exposition format, so the same data could be
  scraped by a real Prometheus instance without any code changes, if this
  ever grew past a couple of Cloud Run services worth graphing that way.

The load balancer's URL map rewrites `/status/api/agent-stats` →
`agent-orchestrator`'s `/stats` and `/status/api/genre-stats` →
`genre-classifier`'s `/stats` (see `bradjobe-dev-infra`'s `lb.tf` route
rules — each is its own Cloud Run backend service now, not a port on a
shared box). `llama-server` has no HTTP stats endpoint of its own; its
per-request timing (prompt/eval tokens-per-second) goes to its GKE pod
logs instead and is represented on the dashboard indirectly, through the
agent-orchestrator calls that depend on it.

## What it's for

Every number on this page is real, not a mockup — it reflects whatever
traffic (yours or a demo run) actually hit these services since they last
restarted. The tool-trigger-precision figure in particular is the
project's one online evaluation metric: how often the LLM's own tool
decision agreed with the deterministic guardrail in `agent-orchestrator`.
