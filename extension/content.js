// Fake News Detective — content script.
// Injected on demand by popup.js (chrome.scripting.executeScript). Handles:
//   FND_EXTRACT   -> pull article title + body text off the page
//   FND_HIGHLIGHT -> wrap the analysis API's highlighted_spans in the page
//   FND_CLEAR     -> remove any highlights this script added
//
// Guarded so re-injecting on an already-instrumented page (popup reopened)
// doesn't register the listener twice.
if (!window.__fndContentLoaded) {
  window.__fndContentLoaded = true;

  const ARTICLE_SELECTORS = [
    "article",
    "[role='article']",
    "main article",
    "#content article",
    ".article-body",
    ".article-content",
    ".story-body",
    "main",
  ];

  // Comment widgets (Disqus, Viafoura, OpenWeb/Spot.IM, generic "comments"
  // sections) commonly wrap each comment in a semantic <article>, so a naive
  // "first <article> on the page" pick can land on a 2-sentence comment
  // instead of the real story. Anything nested under one of these is
  // disqualified as a candidate.
  const NON_ARTICLE_RE = /comment|disqus|viafoura|openweb|spot-im|discussion/i;

  function isInNonArticleContainer(el) {
    let node = el;
    while (node && node !== document.body) {
      if (NON_ARTICLE_RE.test(node.id || "") || NON_ARTICLE_RE.test(node.className || "")) {
        return true;
      }
      node = node.parentElement;
    }
    return false;
  }

  function wordCount(el) {
    return ((el.innerText || "").trim().match(/\S+/g) || []).length;
  }

  function findArticleRoot() {
    // Collect every candidate across all selectors and pick the one with the
    // most text — rather than the first match — so a small comment <article>
    // never beats the real (much longer) story body.
    let best = null;
    let bestWords = 40; // minimum worth considering
    for (const sel of ARTICLE_SELECTORS) {
      for (const el of document.querySelectorAll(sel)) {
        if (isInNonArticleContainer(el)) continue;
        const words = wordCount(el);
        if (words > bestWords) {
          best = el;
          bestWords = words;
        }
      }
    }
    if (best) return best;

    // Fallback: the <body>'s largest cluster of <p> text.
    const paragraphs = Array.from(document.querySelectorAll("p")).filter(
      (p) => !isInNonArticleContainer(p)
    );
    if (paragraphs.length) {
      // Group paragraphs by nearest common ancestor 2 levels up, pick the
      // group with the most combined text.
      const groups = new Map();
      for (const p of paragraphs) {
        const key = p.parentElement || document.body;
        const text = p.innerText || "";
        groups.set(key, (groups.get(key) || "") + "\n" + text);
      }
      let bestText = null;
      let bestLen = 0;
      for (const [, text] of groups) {
        if (text.length > bestLen) {
          bestLen = text.length;
          bestText = text;
        }
      }
      if (bestText) return bestText;
    }
    return document.body;
  }

  function extractArticle() {
    const root = findArticleRoot();
    const text = typeof root === "string" ? root : root.innerText || "";
    const titleEl = document.querySelector("h1");
    const title = (titleEl && titleEl.innerText.trim()) || document.title || "";
    return { title, text: text.trim() };
  }

  // ---- Highlighting -------------------------------------------------

  const HL_CLASS = "fnd-highlight";
  const STYLE_ID = "fnd-highlight-style";

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      mark.${HL_CLASS} {
        color: inherit;
        border-radius: 2px;
        padding: 0 1px;
        box-decoration-break: clone;
        -webkit-box-decoration-break: clone;
      }
      mark.${HL_CLASS}[data-fnd-dir="fake"] {
        background: color-mix(in srgb, #f0eb20 calc(var(--fnd-w, 0.6) * 100%), transparent);
        border-bottom: 2px solid #b8b300;
      }
      mark.${HL_CLASS}[data-fnd-dir="real"] {
        background: color-mix(in srgb, #7c7be8 calc(var(--fnd-w, 0.6) * 100%), transparent);
        border-bottom: 2px solid #3c3bbb;
      }
    `;
    document.head.appendChild(style);
  }

  function escapeRegExp(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function textNodesUnder(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent) return NodeFilter.FILTER_REJECT;
        const tag = parent.tagName;
        if (tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT" || tag === "TEXTAREA") {
          return NodeFilter.FILTER_REJECT;
        }
        if (parent.closest(`.${HL_CLASS}`)) return NodeFilter.FILTER_REJECT;
        if (!node.data || !node.data.trim()) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    let n;
    while ((n = walker.nextNode())) nodes.push(n);
    return nodes;
  }

  function wrapRange(range, direction, weight) {
    const mark = document.createElement("mark");
    mark.className = HL_CLASS;
    mark.dataset.fndDir = direction;
    mark.style.setProperty("--fnd-w", String(Math.min(Math.max(weight, 0.15), 1)));
    const frag = range.extractContents();
    mark.appendChild(frag);
    range.insertNode(mark);
  }

  function highlightSpans(spans) {
    ensureStyle();
    const nodes = textNodesUnder(document.body);
    if (!nodes.length) return 0;

    let fullText = "";
    const offsets = []; // offsets[i] = starting index of nodes[i] in fullText
    for (const node of nodes) {
      offsets.push(fullText.length);
      fullText += node.data;
    }

    // Sort spans strongest-first so the most important ones win when two
    // spans would otherwise overlap; then re-sort matches by position.
    const ordered = [...spans].sort((a, b) => (b.weight || 0) - (a.weight || 0));

    const matches = [];
    const claimed = []; // [start, end) ranges already used
    for (const span of ordered) {
      const words = span.text.trim().split(/\s+/).map(escapeRegExp);
      if (!words.length) continue;
      const pattern = words.join("\\s+");
      let re;
      try {
        re = new RegExp(pattern, "i");
      } catch {
        continue;
      }
      const m = re.exec(fullText);
      if (!m) continue;
      const start = m.index;
      const end = start + m[0].length;
      const overlaps = claimed.some(([s, e]) => start < e && end > s);
      if (overlaps) continue;
      claimed.push([start, end]);
      matches.push({ start, end, direction: span.direction, weight: span.weight ?? 0.6 });
    }

    matches.sort((a, b) => a.start - b.start);

    let count = 0;
    // Process in reverse document order so earlier DOM mutations don't
    // invalidate offsets/node references for matches still pending.
    for (let i = matches.length - 1; i >= 0; i--) {
      const { start, end, direction, weight } = matches[i];

      let startNode = null, startOffset = 0;
      let endNode = null, endOffset = 0;
      for (let j = 0; j < nodes.length; j++) {
        const nodeStart = offsets[j];
        const nodeEnd = nodeStart + nodes[j].data.length;
        if (startNode === null && start >= nodeStart && start < nodeEnd) {
          startNode = nodes[j];
          startOffset = start - nodeStart;
        }
        if (end > nodeStart && end <= nodeEnd) {
          endNode = nodes[j];
          endOffset = end - nodeStart;
          if (startNode !== null) break;
        }
      }
      if (!startNode || !endNode) continue;

      try {
        const range = document.createRange();
        range.setStart(startNode, startOffset);
        range.setEnd(endNode, endOffset);
        wrapRange(range, direction, weight);
        count++;
      } catch (e) {
        // Range spanned a boundary that couldn't be wrapped cleanly; skip it.
      }
    }
    return count;
  }

  function clearHighlights() {
    const marks = document.querySelectorAll(`mark.${HL_CLASS}`);
    for (const mark of marks) {
      const parent = mark.parentNode;
      if (!parent) continue;
      while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
      parent.removeChild(mark);
      parent.normalize();
    }
    return marks.length;
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || !msg.type) return;
    if (msg.type === "FND_EXTRACT") {
      sendResponse(extractArticle());
    } else if (msg.type === "FND_HIGHLIGHT") {
      sendResponse({ count: highlightSpans(msg.spans || []) });
    } else if (msg.type === "FND_CLEAR") {
      sendResponse({ count: clearHighlights() });
    }
    return true;
  });
}
