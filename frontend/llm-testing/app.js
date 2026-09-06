(() => {
  "use strict";

  const API_URL = "/llm-testing/api/chat/completions";
  const SYSTEM_PROMPT =
    "You are a small, friendly demo assistant running locally on a single-core VPS. " +
    "Keep replies short (a few sentences at most) since you are generation-constrained on CPU.";
  const MAX_HISTORY_TURNS = 6; // keep the KV cache / context small on a 2048-token window

  const messagesEl = document.getElementById("messages");
  const composerEl = document.getElementById("composer");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const statusEl = document.getElementById("statusline");
  const detailsToggle = document.getElementById("detailsToggle");
  const details = document.getElementById("details");

  detailsToggle.addEventListener("click", () => {
    const expanded = detailsToggle.getAttribute("aria-expanded") === "true";
    detailsToggle.setAttribute("aria-expanded", String(!expanded));
    details.hidden = expanded;
  });

  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  });

  /** @type {{role: "user"|"assistant", content: string}[]} */
  let history = [];
  let busy = false;

  function addBubble(role, text, opts = {}) {
    const wrap = document.createElement("div");
    wrap.className = `msg msg-${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble" + (opts.error ? " error" : "") + (opts.pending ? " pending" : "");
    bubble.textContent = text;
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function setBusy(v) {
    busy = v;
    sendBtn.disabled = v;
    inputEl.disabled = v;
  }

  async function sendMessage(userText) {
    history.push({ role: "user", content: userText });
    history = history.slice(-MAX_HISTORY_TURNS * 2);

    const assistantBubble = addBubble("assistant", "", { pending: true });
    setBusy(true);
    setStatus("thinking…");

    const startedAt = performance.now();
    let chunkCount = 0;
    let fullText = "";

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [{ role: "system", content: SYSTEM_PROMPT }, ...history],
          max_tokens: 200,
          temperature: 0.7,
          stream: true,
        }),
      });

      if (res.status === 429) {
        assistantBubble.classList.remove("pending");
        assistantBubble.classList.add("error");
        assistantBubble.textContent =
          "Rate limited — this demo allows a few messages per minute per visitor. Try again shortly.";
        history.pop();
        setStatus("rate limited");
        return;
      }

      if (!res.ok || !res.body) {
        throw new Error(`Backend returned ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const lines = buf.split("\n");
        buf = lines.pop(); // keep incomplete line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === "[DONE]") continue;
          try {
            const json = JSON.parse(payload);
            const delta = json.choices?.[0]?.delta?.content;
            if (delta) {
              fullText += delta;
              chunkCount += 1;
              assistantBubble.textContent = fullText;
              messagesEl.scrollTop = messagesEl.scrollHeight;
            }
          } catch {
            // ignore partial/non-JSON keep-alive lines
          }
        }
      }

      assistantBubble.classList.remove("pending");
      if (!fullText) {
        assistantBubble.textContent = "(empty response — try rephrasing)";
      } else {
        history.push({ role: "assistant", content: fullText });
      }

      const elapsedS = (performance.now() - startedAt) / 1000;
      const approxTokPerSec = chunkCount > 0 ? (chunkCount / elapsedS).toFixed(1) : "0";
      setStatus(`~${approxTokPerSec} tok/s · ${elapsedS.toFixed(1)}s · single CPU core`);
    } catch (err) {
      assistantBubble.classList.remove("pending");
      assistantBubble.classList.add("error");
      assistantBubble.textContent = "Something went wrong reaching the model server. Please try again.";
      history.pop();
      setStatus("error");
    } finally {
      setBusy(false);
      inputEl.focus();
    }
  }

  composerEl.addEventListener("submit", (e) => {
    e.preventDefault();
    if (busy) return;
    const text = inputEl.value.trim();
    if (!text) return;
    addBubble("user", text);
    inputEl.value = "";
    inputEl.style.height = "auto";
    sendMessage(text);
  });

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composerEl.requestSubmit();
    }
  });
})();
