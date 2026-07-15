// Fake News Detective — popup controller.
// Flow: inject content.js into the active tab -> ask it to extract the
// article -> POST to the local analysis API -> render the response.
// "Highlight patterns on page" replays the returned spans through content.js.

const API_BASE = "http://127.0.0.1:8000";

const analyzeBtn = document.getElementById("analyzeBtn");
const highlightBtn = document.getElementById("highlightBtn");
const clearBtn = document.getElementById("clearBtn");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const errorBanner = document.getElementById("errorBanner");
const stubBanner = document.getElementById("stubBanner");

const scoreValue = document.getElementById("scoreValue");
const meterFill = document.getElementById("meterFill");
const statusDot = document.getElementById("statusDot");
const statusWord = document.getElementById("statusWord");
const labelValue = document.getElementById("labelValue");
const confidenceValue = document.getElementById("confidenceValue");
const signalsList = document.getElementById("signalsList");

let lastSpans = [];

function humanize(name) {
  return name
    .split("_")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

function scoreBand(score) {
  if (score < 25) return { word: "Low", color: "var(--fnd-good)" };
  if (score < 50) return { word: "Moderate", color: "var(--fnd-warning)" };
  if (score < 75) return { word: "Elevated", color: "var(--fnd-serious)" };
  return { word: "High", color: "var(--fnd-critical)" };
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

function setBusy(busy) {
  analyzeBtn.disabled = busy;
  loading.hidden = !busy;
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id) throw new Error("No active tab.");
  if (!/^https?:/.test(tab.url || "")) {
    throw new Error("Open a news article in this tab first.");
  }
  return tab;
}

async function ensureContentScript(tabId) {
  await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
}

function sendMessage(tabId, message) {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      resolve(response);
    });
  });
}

function renderResults(data) {
  results.hidden = false;
  stubBanner.hidden = !data.model_is_stub;

  const score = Math.max(0, Math.min(100, Math.round(data.manipulation_score)));
  const band = scoreBand(score);
  scoreValue.textContent = String(score);
  meterFill.style.width = `${score}%`;
  meterFill.style.background = band.color;
  statusDot.style.background = band.color;
  statusWord.textContent = band.word;

  labelValue.textContent = data.label === "likely_fake" ? "Likely fake (style)" : "Likely real (style)";
  confidenceValue.textContent = data.confidence.toFixed(2);

  signalsList.innerHTML = "";
  for (const sig of data.signals || []) {
    const li = document.createElement("li");
    li.className = "fnd-signal";
    li.innerHTML = `
      <span class="fnd-signal-icon" aria-hidden="true"></span>
      <span class="fnd-signal-body">
        <span class="fnd-signal-name">
          <span>${humanize(sig.name)}</span>
          <span class="fnd-signal-value">${sig.value}</span>
        </span>
        <span class="fnd-signal-explanation">${sig.explanation}</span>
      </span>
    `;
    signalsList.appendChild(li);
  }

  lastSpans = data.highlighted_spans || [];
  highlightBtn.hidden = lastSpans.length === 0;
  highlightBtn.textContent = "Highlight patterns on page";
  clearBtn.hidden = true;
}

async function handleAnalyze() {
  clearError();
  results.hidden = true;
  setBusy(true);
  try {
    const tab = await getActiveTab();
    await ensureContentScript(tab.id);
    const article = await sendMessage(tab.id, { type: "FND_EXTRACT" });
    if (!article || !article.text || article.text.trim().split(/\s+/).length < 20) {
      throw new Error("Couldn't find enough article text on this page (need ~20+ words).");
    }

    const resp = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: article.text, title: article.title || "" }),
    });

    if (!resp.ok) {
      const detail = await resp.json().catch(() => null);
      throw new Error((detail && detail.detail) || `Analysis failed (${resp.status}).`);
    }

    const data = await resp.json();
    renderResults(data);
  } catch (err) {
    const msg = /Failed to fetch/i.test(err.message)
      ? "Can't reach the local analysis API at localhost:8000 — is the backend running?"
      : err.message;
    showError(msg);
  } finally {
    setBusy(false);
  }
}

async function handleHighlight() {
  try {
    const tab = await getActiveTab();
    const res = await sendMessage(tab.id, { type: "FND_HIGHLIGHT", spans: lastSpans });
    highlightBtn.hidden = true;
    clearBtn.hidden = false;
    clearBtn.textContent = `Clear highlights (${res?.count ?? 0})`;
  } catch (err) {
    showError(err.message);
  }
}

async function handleClear() {
  try {
    const tab = await getActiveTab();
    await sendMessage(tab.id, { type: "FND_CLEAR" });
    clearBtn.hidden = true;
    highlightBtn.hidden = false;
  } catch (err) {
    showError(err.message);
  }
}

analyzeBtn.addEventListener("click", handleAnalyze);
highlightBtn.addEventListener("click", handleHighlight);
clearBtn.addEventListener("click", handleClear);
