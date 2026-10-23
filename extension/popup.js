// Fake News Detective — popup controller.
// Flow: inject content.js into the active tab -> ask it to extract the
// article -> POST to the local analysis API -> render the response.
// "Highlight patterns on page" replays the returned spans through content.js.

const API_BASE = "http://127.0.0.1:8000";

const $ = (id) => document.getElementById(id);

const analyzeBtn = $("analyzeBtn");
const analyzeLabel = $("analyzeLabel");
const highlightBtn = $("highlightBtn");
const clearBtn = $("clearBtn");
const loading = $("loading");
const results = $("results");
const empty = $("empty");
const errorBanner = $("errorBanner");
const errorText = $("errorText");
const stubBanner = $("stubBanner");
const apiPill = $("apiPill");
const apiPillText = $("apiPillText");

const scoreValue = $("scoreValue");
const gauge = $("gauge");
const gaugeFill = $("gaugeFill");
const bandChip = $("bandChip");
const statusWord = $("statusWord");
const bandNote = $("bandNote");
const labelChip = $("labelChip");
const confFill = $("confFill");
const confidenceValue = $("confidenceValue");
const signalsList = $("signalsList");
const flagCount = $("flagCount");
const toggleClean = $("toggleClean");

// Circumference of the gauge ring (r = 52 in the SVG viewBox).
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52;

let lastSpans = [];
let cleanShown = false;
let cleanCount = 0;
let scoreAnimation = null;

function humanize(name) {
  return name
    .split("_")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

// `color` paints the ring and the dot; `ink` paints text, which needs a
// contrast-safe variant of the same palette entry on light backgrounds.
const BANDS = [
  { max: 25, word: "Low", color: "var(--fnd-good)", ink: "var(--fnd-good-ink)", note: "Reads like conventional, plainly-written reporting." },
  { max: 50, word: "Moderate", color: "var(--fnd-warning)", ink: "var(--fnd-warning-ink)", note: "Some persuasive styling — worth a second look." },
  { max: 75, word: "Elevated", color: "var(--fnd-serious)", ink: "var(--fnd-serious-ink)", note: "Several manipulation patterns stacked together." },
  { max: 101, word: "High", color: "var(--fnd-critical)", ink: "var(--fnd-critical-ink)", note: "Heavily styled for emotion rather than information." },
];

function scoreBand(score) {
  return BANDS.find((b) => score < b.max);
}

// The backend reports every signal as a bare float, but the units differ:
// some are 0-1 ratios, some are counts, readability is a Flesch score.
// Format each so the chip reads as a quantity instead of "0.047".
const PERCENT_SIGNALS = new Set([
  "caps_ratio",
  "allcaps_words",
  "quote_ratio",
  "subjectivity",
]);
const COUNT_SIGNALS = new Set(["clickbait_terms", "sourcing_markers", "readability"]);

function formatValue(name, value) {
  if (PERCENT_SIGNALS.has(name)) return `${Math.round(value * 100)}%`;
  if (COUNT_SIGNALS.has(name)) return String(Math.round(value));
  // Densities and the variance coefficient: keep 2dp so a round 1.0 still
  // reads as a rate rather than a count.
  return value.toFixed(2);
}

function showError(message) {
  errorText.textContent = message;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.hidden = true;
  errorText.textContent = "";
}

function setBusy(busy) {
  analyzeBtn.disabled = busy;
  loading.hidden = !busy;
  if (busy) empty.hidden = true;
  analyzeLabel.textContent = busy ? "Analyzing…" : "Analyze this page";
}

function setApiState(state, text) {
  apiPill.dataset.state = state;
  apiPillText.textContent = text;
}

async function pingBackend() {
  try {
    const resp = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!resp.ok) throw new Error("bad status");
    const health = await resp.json();
    setApiState("online", health.model_is_stub ? "stub model" : "connected");
    stubBanner.hidden = !health.model_is_stub;
  } catch {
    setApiState("offline", "offline");
  }
}

/** Count the score up so the number and the ring animate together. */
function animateScore(target) {
  cancelAnimationFrame(scoreAnimation);
  const start = performance.now();
  const duration = 900;
  const step = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    scoreValue.textContent = String(Math.round(target * eased));
    if (t < 1) scoreAnimation = requestAnimationFrame(step);
  };
  scoreAnimation = requestAnimationFrame(step);
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

function buildSignal(sig, index) {
  // `triggered`/`weight` are optional so an older backend still renders.
  const flagged = sig.triggered === true;
  const li = document.createElement("li");
  li.className = "fnd-signal";
  li.dataset.flagged = String(flagged);
  li.dataset.clean = String(!flagged);
  li.style.setProperty("--fnd-sig-color", flagged ? "var(--fnd-serious)" : "var(--fnd-good)");
  li.style.setProperty("--fnd-sig-ink", flagged ? "var(--fnd-serious-ink)" : "var(--fnd-good-ink)");
  li.style.animationDelay = `${Math.min(index, 8) * 35}ms`;

  const icon = document.createElement("span");
  icon.className = "fnd-signal-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = flagged ? "!" : "✓";

  const body = document.createElement("span");
  body.className = "fnd-signal-body";

  const nameRow = document.createElement("span");
  nameRow.className = "fnd-signal-name";
  const nameText = document.createElement("span");
  nameText.textContent = humanize(sig.name);
  const valueChip = document.createElement("span");
  valueChip.className = "fnd-signal-value";
  valueChip.textContent = formatValue(sig.name, sig.value);
  nameRow.append(nameText, valueChip);

  const explanation = document.createElement("span");
  explanation.className = "fnd-signal-explanation";
  explanation.textContent = sig.explanation;

  body.append(nameRow, explanation);

  // Only flagged signals actually contribute to the score, so the weight bar
  // would be misleading on a clean one.
  if (flagged && typeof sig.weight === "number") {
    const bar = document.createElement("span");
    bar.className = "fnd-signal-weight";
    bar.title = `Adds up to ${Math.round(sig.weight * 100)}% of the style score`;
    const fill = document.createElement("i");
    // Heaviest signal is 0.20, so scale against that for a readable bar.
    requestAnimationFrame(() => {
      fill.style.width = `${Math.min(100, (sig.weight / 0.2) * 100)}%`;
    });
    bar.appendChild(fill);
    body.appendChild(bar);
  }

  li.append(icon, body);
  return li;
}

function applyCleanVisibility() {
  for (const li of signalsList.querySelectorAll('[data-clean="true"]')) {
    li.hidden = !cleanShown;
  }
  toggleClean.textContent = cleanShown
    ? "Hide clean signals"
    : `Show ${cleanCount} clean signal${cleanCount === 1 ? "" : "s"}`;
}

function renderSignals(signals) {
  // Flagged first: those are the ones that explain the score.
  const ordered = [...signals].sort(
    (a, b) => Number(b.triggered === true) - Number(a.triggered === true)
  );
  const flagged = ordered.filter((s) => s.triggered === true).length;
  cleanCount = ordered.length - flagged;

  signalsList.replaceChildren(...ordered.map(buildSignal));
  flagCount.textContent = `${flagged} of ${ordered.length} flagged`;

  // With no `triggered` field nothing is "clean" — show everything, no toggle.
  toggleClean.hidden = cleanCount === 0 || flagged === 0;
  cleanShown = toggleClean.hidden;
  applyCleanVisibility();
}

function renderResults(data) {
  empty.hidden = true;
  results.hidden = false;
  stubBanner.hidden = !data.model_is_stub;

  const score = Math.max(0, Math.min(100, Math.round(data.manipulation_score)));
  const band = scoreBand(score);
  for (const el of [gauge, bandChip]) {
    el.style.setProperty("--fnd-band-color", band.color);
    el.style.setProperty("--fnd-band-ink", band.ink);
  }
  statusWord.textContent = band.word;
  bandNote.textContent = band.note;
  animateScore(score);
  // Reset first so re-analyzing replays the sweep instead of jumping.
  gaugeFill.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
  requestAnimationFrame(() => {
    gaugeFill.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE * (1 - score / 100));
  });

  const isFake = data.label === "likely_fake";
  labelChip.textContent = isFake ? "Likely fake (style)" : "Likely real (style)";
  labelChip.dataset.tone = isFake ? "fake" : "real";
  confidenceValue.textContent = data.confidence.toFixed(2);
  requestAnimationFrame(() => {
    confFill.style.width = `${Math.round(data.confidence * 100)}%`;
  });

  renderSignals(data.signals || []);

  lastSpans = data.highlighted_spans || [];
  highlightBtn.hidden = lastSpans.length === 0;
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
    setApiState("online", data.model_is_stub ? "stub model" : "connected");
    renderResults(data);
  } catch (err) {
    const offline = /Failed to fetch/i.test(err.message);
    if (offline) setApiState("offline", "offline");
    showError(
      offline
        ? "Can't reach the local analysis API at localhost:8000 — is the backend running?"
        : err.message
    );
    if (results.hidden) empty.hidden = false;
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
toggleClean.addEventListener("click", () => {
  cleanShown = !cleanShown;
  applyCleanVisibility();
  // Revealing five rows re-flows the list; keep the button under the cursor.
  toggleClean.scrollIntoView({ block: "nearest" });
});

pingBackend();
