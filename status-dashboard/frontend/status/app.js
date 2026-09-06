(() => {
  "use strict";

  const serviceGrid = document.getElementById("serviceGrid");
  const precisionTile = document.getElementById("precisionTile");
  const genreBars = document.getElementById("genreBars");
  const eventsTable = document.getElementById("eventsTable");
  const lastUpdated = document.getElementById("lastUpdated");

  function statusDot(level) {
    return `<span class="status-dot ${level}"></span>`;
  }

  function tile(name, value, sub, level) {
    return `
      <div class="tile">
        <div class="tile-name">${level ? statusDot(level) : ""}${name}</div>
        <div class="tile-value">${value}</div>
        <div class="tile-sub">${sub || "&nbsp;"}</div>
      </div>`;
  }

  function fmtMs(v) {
    if (v === null || v === undefined) return "—";
    return `${v.toFixed(0)}ms`;
  }

  async function fetchJson(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) return { ok: false, status: res.status };
      return { ok: true, data: await res.json() };
    } catch (e) {
      return { ok: false, error: String(e) };
    }
  }

  function renderEvents(events) {
    if (!events || events.length === 0) {
      eventsTable.innerHTML = '<p class="bars-empty">No agent requests yet — try /agent-demo.</p>';
      return;
    }
    const rows = events
      .slice(0, 15)
      .map((e) => {
        const time = new Date(e.ts * 1000).toLocaleTimeString();
        const tag = `<span class="evt-tag ${e.type}">${e.type.replace("tool_", "")}</span>`;
        let detail = "";
        if (e.type === "tool_executed") {
          detail = `"${e.title}" → ${e.genre} (${(e.confidence * 100).toFixed(0)}%)`;
        } else if (e.type === "tool_blocked") {
          detail = `"${e.user_message}" (model proposed: "${e.proposed}")`;
        } else if (e.type === "tool_malformed") {
          detail = `"${e.user_message}" (raw: ${JSON.stringify(e.raw_model_output).slice(0, 60)})`;
        }
        return `<tr><td>${time}</td><td>${tag}</td><td>${detail}</td></tr>`;
      })
      .join("");
    eventsTable.innerHTML = `
      <table class="events">
        <thead><tr><th>time</th><th>outcome</th><th>detail</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  function renderGenreBars(counts) {
    const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
    if (entries.length === 0) {
      genreBars.innerHTML = '<p class="bars-empty">No predictions yet — try /genre-classifier.</p>';
      return;
    }
    const max = entries[0][1];
    genreBars.innerHTML = entries
      .map(
        ([genre, count]) => `
        <div class="bar-row">
          <div class="bar-label">${genre}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(4, (count / max) * 100)}%"></div></div>
          <div class="bar-count">${count}</div>
        </div>`
      )
      .join("");
  }

  function renderImageBars(counts) {
    const el = document.getElementById("imageBars");
    if (!el) return;
    const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
    if (entries.length === 0) {
      el.innerHTML = '<p class="bars-empty">No predictions yet — try /image-classifier.</p>';
      return;
    }
    const max = entries[0][1];
    el.innerHTML = entries
      .map(
        ([label, count]) => `
        <div class="bar-row">
          <div class="bar-label">${label}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(4, (count / max) * 100)}%"></div></div>
          <div class="bar-count">${count}</div>
        </div>`
      )
      .join("");
  }

  async function refresh() {
    const [agentRes, genreRes, imageRes] = await Promise.all([
      fetchJson("/status/api/agent-stats"),
      fetchJson("/status/api/genre-stats"),
      fetchJson("/status/api/image-stats"),
    ]);

    const tiles = [];

    if (agentRes.ok) {
      const a = agentRes.data;
      tiles.push(tile("agent-orchestrator", a.metrics.requests_total, `p50 ${fmtMs(a.latency_ms.p50)} · p95 ${fmtMs(a.latency_ms.p95)}`, "good"));
      const precisionPct = a.tool_trigger_precision === null ? "—" : `${(a.tool_trigger_precision * 100).toFixed(0)}%`;
      precisionTile.innerHTML = `${precisionPct} <span style="font-size:0.9rem;color:var(--muted);font-weight:400">
        (${a.metrics.tool_executed_total} executed / ${a.metrics.tool_proposed_total + a.metrics.tool_malformed_total} proposed —
        ${a.metrics.tool_blocked_total} off-topic-blocked, ${a.metrics.tool_malformed_total} malformed)</span>`;
      renderEvents(a.recent_events);
    } else {
      tiles.push(tile("agent-orchestrator", "down", agentRes.status ? `HTTP ${agentRes.status}` : "unreachable", "critical"));
    }

    if (genreRes.ok) {
      const g = genreRes.data;
      tiles.push(tile("genre-classifier", g.requests_total, `p50 ${fmtMs(g.latency_ms.p50)} · p95 ${fmtMs(g.latency_ms.p95)}`, "good"));
      renderGenreBars(g.genre_counts);
    } else {
      tiles.push(tile("genre-classifier", "down", genreRes.status ? `HTTP ${genreRes.status}` : "unreachable", "critical"));
    }

    if (imageRes.ok) {
      const im = imageRes.data;
      tiles.push(tile("image-classifier", im.requests_total, `p50 ${fmtMs(im.latency_ms.p50)} · p95 ${fmtMs(im.latency_ms.p95)}`, "good"));
      renderImageBars(im.class_counts);
    } else {
      tiles.push(tile("image-classifier", "down", imageRes.status ? `HTTP ${imageRes.status}` : "unreachable", "critical"));
    }

    // llama-server has no HTTP /stats of its own (kept minimal on purpose —
    // its per-request timings go to the systemd journal instead), so this
    // tile reflects it indirectly via the orchestrator, which calls it on
    // every request.
    tiles.push(tile("llama-server", agentRes.ok ? "reachable" : "unknown", "via agent-orchestrator calls; see journalctl -u llama-server for token/sec", agentRes.ok ? "good" : "warning"));

    serviceGrid.innerHTML = tiles.join("");
    lastUpdated.textContent = `updated ${new Date().toLocaleTimeString()} — refreshes every 5s`;
  }

  refresh();
  setInterval(refresh, 5000);
})();
