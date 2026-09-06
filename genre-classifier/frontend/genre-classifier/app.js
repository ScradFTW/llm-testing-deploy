(() => {
  "use strict";

  const API_URL = "/genre-classifier/api/predict";

  const composerEl = document.getElementById("composer");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const statusEl = document.getElementById("statusline");
  const resultEl = document.getElementById("result");
  const topGenreEl = document.getElementById("topGenre");
  const topConfEl = document.getElementById("topConf");
  const barsEl = document.getElementById("bars");
  const detailsToggle = document.getElementById("detailsToggle");
  const details = document.getElementById("details");

  detailsToggle.addEventListener("click", () => {
    const expanded = detailsToggle.getAttribute("aria-expanded") === "true";
    detailsToggle.setAttribute("aria-expanded", String(!expanded));
    details.hidden = expanded;
  });

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function setBusy(v) {
    sendBtn.disabled = v;
    inputEl.disabled = v;
  }

  function renderResult(data) {
    resultEl.hidden = false;
    topGenreEl.textContent = data.genre;
    topConfEl.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

    barsEl.innerHTML = "";
    for (const { genre, probability } of data.scores.slice(0, 6)) {
      const row = document.createElement("div");
      row.className = "bar-row";

      const label = document.createElement("div");
      label.className = "bar-label";
      label.textContent = genre;

      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = `${Math.max(2, probability * 100)}%`;
      track.appendChild(fill);

      const pct = document.createElement("div");
      pct.className = "bar-pct";
      pct.textContent = `${(probability * 100).toFixed(1)}%`;

      row.append(label, track, pct);
      barsEl.appendChild(row);
    }
  }

  function renderError(message) {
    resultEl.hidden = false;
    barsEl.innerHTML = "";
    topGenreEl.textContent = "";
    topConfEl.textContent = "";
    const err = document.createElement("div");
    err.className = "error-msg";
    err.textContent = message;
    barsEl.appendChild(err);
  }

  composerEl.addEventListener("submit", async (e) => {
    e.preventDefault();
    const title = inputEl.value.trim();
    if (!title) return;

    setBusy(true);
    setStatus("classifying…");

    const startedAt = performance.now();
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });

      if (res.status === 429) {
        renderError("Rate limited — this demo allows a few requests per minute per visitor. Try again shortly.");
        setStatus("rate limited");
        return;
      }

      const data = await res.json();
      if (!res.ok) {
        renderError(data.error || `Backend returned ${res.status}`);
        setStatus("error");
        return;
      }

      renderResult(data);
      const elapsedS = (performance.now() - startedAt) / 1000;
      setStatus(`${elapsedS.toFixed(2)}s round trip · ${data.elapsed_ms}ms model inference`);
    } catch (err) {
      renderError("Something went wrong reaching the classifier. Please try again.");
      setStatus("error");
    } finally {
      setBusy(false);
      inputEl.focus();
    }
  });
})();
