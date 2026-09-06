# llm-testing — self-hosted LLM demo on bradjobe.dev

Live at: https://bradjobe.dev/llm-testing/

A small, self-hosted chat demo running entirely on the same 1 vCPU / 2GB
Linode VPS that serves bradjobe.dev — built as a production-infra portfolio
piece (model serving, reliability, observability, guardrails), not just a
chat widget wired to a hosted API.

## Architecture

```
Browser
  │  HTTPS (bradjobe.dev's existing cert)
  ▼
nginx (existing bradjobe.dev server block)
  ├─ /llm-testing/          → static files, served from /var/www/llm-testing/
  │                            (deliberately OUTSIDE /var/www/html — see below)
  └─ /llm-testing/api/      → reverse proxy, rate-limited, streaming (SSE)
                                  │
                                  ▼
                        llama-server (127.0.0.1:8082 only)
                        systemd-managed, memory-capped, sandboxed
                        Qwen2.5-0.5B-Instruct, Q4_K_M GGUF (~470MB)
```

This VPS also hosts an unrelated project (`ccaas`, in `/opt/ccaas`) on the
same box. Two collisions came up while integrating this demo, both now
resolved:
- `ccaas-backend` and `llama-server` both defaulted to port 8080 — moved
  `llama-server` to **8082** (ccaas is on 8081).
- A portfolio redeploy did a full sync/replace of `/var/www/html`, which
  silently deleted the `llm-testing` static files. Fixed by serving them
  from `/var/www/llm-testing/` (via nginx `alias`) instead of nesting them
  under the portfolio's own webroot — so no future portfolio deploy can
  touch them.

## Access control

Both `/llm-testing/` and `/llm-testing/api/` are gated with nginx HTTP Basic
Auth (`auth_basic` + `/etc/nginx/.htpasswd-llm-testing`, bcrypt-hashed via
`htpasswd -B`). The htpasswd file is server-only and intentionally not in
this repo. Credentials were shared with Brad directly, not committed
anywhere.

## Why these choices

- **Model**: Qwen2.5-0.5B-Instruct, Q4_K_M quantization. Small enough to run
  comfortably on one shared CPU core (~15-25 tok/s) while leaving headroom
  for nginx and the rest of the box.
- **Serving**: [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
  `llama-server`, which exposes an OpenAI-compatible `/v1/chat/completions`
  endpoint. Bound to `127.0.0.1` only — never reachable directly from the
  internet, only through the nginx proxy.
- **Reliability**: a systemd unit (`systemd/llama-server.service`) with
  `Restart=always`, a hard `MemoryMax` cap, and sandboxing
  (`ProtectSystem=strict`, `NoNewPrivileges`, etc.) so a runaway model
  process can't take down the other sites on this box.
- **Guardrails / abuse protection**: a small 2048-token context window
  bounds worst-case compute per request; nginx `limit_req` rate-limits by
  IP (`nginx/llm-testing.conf`); request bodies are capped at 8KB; a fixed
  system prompt scopes the assistant's behavior.
- **Observability**: a dedicated JSON nginx log format
  (`/var/log/nginx/llm-testing.log`) captures latency/status/bytes per
  request; llama-server's own per-request timing stats (prompt/eval tok/s)
  flow to the systemd journal (`journalctl -u llama-server`).
- **Streaming**: SSE passthrough (`proxy_buffering off`) so the frontend
  renders tokens incrementally.

## Layout

- `nginx/llm-testing.conf` — http-level `limit_req_zone` + JSON `log_format`
  (deployed to `/etc/nginx/conf.d/llm-testing.conf`)
- `nginx/bradjobe.dev-site.conf` — full site config as deployed, for
  reference/diffing (deployed to `/etc/nginx/sites-enabled/default`); the
  `location /llm-testing/api/` block is the addition relevant to this
  project
- `systemd/llama-server.service` — model server unit
  (`/etc/systemd/system/llama-server.service`)
- `frontend/llm-testing/` — static chat UI
  (`/var/www/llm-testing/` — intentionally not under `/var/www/html/`)

## Server setup (for reference — done manually, not scripted/CI'd)

1. Downloaded a prebuilt CPU release of llama.cpp
   (`llama-b<build>-bin-ubuntu-x64.tar.gz`) to `/opt/llm/llama.cpp/`
   (`apt install libgomp1` is the only runtime dependency needed).
2. Downloaded `qwen2.5-0.5b-instruct-q4_k_m.gguf` from the
   Qwen2.5-0.5B-Instruct-GGUF Hugging Face repo to `/opt/llm/models/`.
3. Created a dedicated unprivileged `llm` system user; `/opt/llm` owned by it.
4. Installed the systemd unit, `enable --now`.
5. Added the nginx config, `nginx -t`, `systemctl reload nginx`.
6. Deployed the static frontend files.

## Known limitations / what's explicitly out of scope here

This is intentionally scoped as a solid single-box demo, not the
full production stack:
- No CI/CD — deploys are manual (see steps above).
- No separate app-layer guardrail/moderation service — guardrails are
  nginx- and context-window-based only.
- No metrics dashboard — telemetry is raw JSON logs + journald, not
  scraped/visualized (e.g. Prometheus/Grafana).

## What would change at "millions of users" scale

Worth noting explicitly since this demo runs on a $5/mo VPS: at real
production scale (the kind this project is meant to speak to in an
interview), the design would change substantially:
- **Serving**: vLLM or TensorRT-LLM on GPU fleets instead of CPU llama.cpp;
  continuous batching, paged KV-cache, speculative decoding.
- **Scaling**: horizontal autoscaling behind a load balancer, model
  routing/fallback across model sizes or providers, canary deploys.
- **Reliability**: multi-region failover, circuit breakers, graceful
  degradation (e.g. fall back to a smaller/faster model under load).
- **Evaluation**: offline regression suites + online quality/latency
  monitoring, hallucination/policy-compliance scoring, automated canary
  evals gating rollout.
- **Observability**: structured tracing across the agent/tool-call graph,
  Prometheus/Grafana or equivalent, SLO-based alerting.
- **Security**: proper secrets management, IAM-scoped service accounts,
  input/output moderation as a real service, audit logging.
- **IaC/CI-CD**: Terraform-managed infra, GitHub Actions (or similar)
  deploying through dev → staging → prod with automated gates.
