(() => {
  "use strict";

  const API_URL = "/image-classifier/api/predict";
  const MAX_BYTES = 3 * 1024 * 1024;

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const preview = document.getElementById("preview");
  const dzContent = document.querySelector(".dz-content");
  const statusEl = document.getElementById("statusline");
  const resultEl = document.getElementById("result");
  const topLabelEl = document.getElementById("topLabel");
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

  function renderResult(data) {
    resultEl.hidden = false;
    topLabelEl.textContent = data.label;
    topConfEl.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;
    barsEl.innerHTML = "";
    for (const { label, probability } of data.scores.slice(0, 6)) {
      const row = document.createElement("div");
      row.className = "bar-row";
      const lbl = document.createElement("div");
      lbl.className = "bar-label";
      lbl.textContent = label;
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = `${Math.max(2, probability * 100)}%`;
      track.appendChild(fill);
      const pct = document.createElement("div");
      pct.className = "bar-pct";
      pct.textContent = `${(probability * 100).toFixed(1)}%`;
      row.append(lbl, track, pct);
      barsEl.appendChild(row);
    }
  }

  function renderError(message) {
    resultEl.hidden = false;
    barsEl.innerHTML = "";
    topLabelEl.textContent = "";
    topConfEl.textContent = "";
    const err = document.createElement("div");
    err.className = "error-msg";
    err.textContent = message;
    barsEl.appendChild(err);
  }

  async function classifyFile(file) {
    if (!file.type.startsWith("image/")) {
      renderError("Please choose an image file.");
      return;
    }
    if (file.size > MAX_BYTES) {
      renderError("Image too large (max 3MB).");
      return;
    }

    const dataUrl = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.readAsDataURL(file);
    });

    preview.src = dataUrl;
    preview.hidden = false;
    dzContent.hidden = true;

    setStatus("classifying…");
    const startedAt = performance.now();

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
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
    }
  }

  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });
  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) classifyFile(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) classifyFile(file);
  });
})();
