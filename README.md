# ML infra demos on bradjobe.dev

Three self-hosted pieces, each its own Cloud Run service on GCP, built as
production-infra portfolio pieces rather than API-key-and-a-widget demos.
**Start at [bradjobe.dev/ai](https://bradjobe.dev/ai/)** — it's the
interviewer-facing overview that ties everything together (including the
two pieces that now live in other repos — see below) with an explicit map
to the job this was built for.

- **[`/genre-classifier`](genre-classifier/README.md)** — training a text
  model from scratch: data cleaning, evaluating a neural net against a
  classical baseline, and shipping the one that actually won.
- **[`/image-classifier`](image-classifier/README.md)** — training a
  computer-vision model from scratch: a CNN trained on GPU, exported to
  ONNX, and served on CPU only — the non-text half of the project.
- **[`/agent-demo`](agent-orchestrator/README.md)** — a reasoning agent that
  ties the LLM and the text classifier together: the LLM can call the
  classifier as a tool, gated by a deterministic guardrail because the
  LLM's own judgment about *when* to call it turned out not to be
  trustworthy (measured, not assumed).
- **`/status`** (`status-dashboard/`) — live telemetry polled straight from
  each service's own in-process metrics.

Each is its own Cloud Run service (own container, own least-privilege IAM
service account), provisioned by Terraform and deployed by Cloud Build on
every push to `main` — see the `bradjobe-dev-infra` repo. Service-to-service
calls (agent-orchestrator → genre-classifier) are authenticated with a
Google-minted ID token rather than relying on network-path trust.
Rate limiting is Cloud Armor at the load balancer, not nginx `limit_req`.

**`/llm-testing`** — the LLM chat demo — used to live in this repo
(`llama-server` on the same VPS, behind nginx + Basic Auth). It has since
moved to its own repo, **`qwen-llm-gke`**, and now runs on a small GKE
cluster (4-node CPU pool, GPU node pool provisioned and pending quota
approval) behind its own subdomain (`llm.bradjobe.dev`) and Cloud Armor
policy — no more Basic Auth, no more sharing a box with `ccaas`. See that
repo for the current architecture, and `git log` here for the original
single-VPS version if the history's useful.

## What changed moving off the single VPS

This project originally ran as four services sharing one $5/mo 1-vCPU/2GB
Linode VPS behind a single nginx vhost — systemd units, loopback-only
ports, one shared filesystem. Migrating to GCP meant trading that for:
- **Compute**: Cloud Run per service instead of systemd units on one box —
  each service scales, restarts, and fails independently now.
- **Networking**: a Global External HTTPS Load Balancer with Cloud Armor
  replaces nginx's reverse proxy + `limit_req` zones (same rate-limit
  thresholds, different mechanism); every Cloud Run service is
  ingress-locked to load-balancer-only traffic, no direct `*.run.app` URL.
- **Identity**: one IAM service account per service, with explicit
  `run.invoker` grants for service-to-service calls, instead of trusting
  whatever else happened to be running on 127.0.0.1.
- **IaC/CI-CD**: Terraform-managed infra (applied only from Cloud Build,
  never a laptop) and a Cloud Build trigger per service on push to `main`,
  instead of the manual server-setup steps this repo used to document here.

## What would change at "millions of users" scale

Even on GCP, this is still a handful of low-traffic Cloud Run services, not
a production-scale deployment. At real scale the design would change
further:
- **Serving**: vLLM or TensorRT-LLM on GPU fleets instead of CPU llama.cpp;
  continuous batching, paged KV-cache, speculative decoding.
- **Scaling**: multi-region deployment, model routing/fallback across model
  sizes or providers, canary deploys.
- **Reliability**: multi-region failover, circuit breakers, graceful
  degradation (e.g. fall back to a smaller/faster model under load).
- **Evaluation**: offline regression suites + online quality/latency
  monitoring, hallucination/policy-compliance scoring, automated canary
  evals gating rollout.
- **Observability**: structured tracing across the agent/tool-call graph,
  Prometheus/Grafana or equivalent, SLO-based alerting.
- **Security**: a secrets manager rotation policy, finer-grained IAM,
  input/output moderation as a real service, audit logging.
