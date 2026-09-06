(() => {
  "use strict";

  const API_URL = "/agent-demo/api/chat";

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
    return { wrap, bubble };
  }

  function traceTag(trace) {
    if (trace.tool_executed) return { cls: "executed", label: "🔧 tool executed" };
    if (trace.blocked_reason && trace.blocked_reason.startsWith("guardrail: malformed"))
      return { cls: "malformed", label: "⚠ malformed tool call blocked" };
    if (trace.blocked_reason && trace.blocked_reason.startsWith("guardrail:"))
      return { cls: "blocked", label: "🛡 tool call blocked (off-topic)" };
    if (trace.blocked_reason === "tool execution failed")
      return { cls: "malformed", label: "⚠ tool execution failed" };
    return { cls: "none", label: "no tool involved" };
  }

  function addTrace(bubbleWrap, trace) {
    const tag = traceTag(trace);
    const el = document.createElement("div");
    el.className = "trace";
    const span = document.createElement("span");
    span.className = `tag ${tag.cls}`;
    span.textContent = tag.label;
    el.appendChild(span);
    el.appendChild(document.createTextNode(`${trace.elapsed_ms}ms`));
    bubbleWrap.bubble.parentElement.appendChild(el);
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
    const { wrap, bubble } = addBubble("assistant", "…", { pending: true });
    setBusy(true);
    setStatus("thinking…");

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText }),
      });

      if (res.status === 429) {
        bubble.classList.remove("pending");
        bubble.classList.add("error");
        bubble.textContent = "Rate limited — try again in a moment.";
        setStatus("rate limited");
        return;
      }

      const data = await res.json();
      bubble.classList.remove("pending");

      if (!res.ok) {
        bubble.classList.add("error");
        bubble.textContent = data.error || `Backend returned ${res.status}`;
        setStatus("error");
        return;
      }

      bubble.textContent = data.reply;
      addTrace({ wrap, bubble }, data.trace);
      setStatus(`${data.trace.elapsed_ms}ms total`);
    } catch (err) {
      bubble.classList.remove("pending");
      bubble.classList.add("error");
      bubble.textContent = "Something went wrong reaching the agent. Please try again.";
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
