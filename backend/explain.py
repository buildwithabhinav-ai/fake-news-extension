"""
Token/sentence attribution for the "highlight on page" feature.

explain(text) returns a list of spans:
    { "text": "...", "weight": 0.0-1.0, "direction": "fake" | "real" }

`weight` is relative importance (used for highlight opacity in the
extension); `direction` says which class the span pushed toward.

--------------------------------------------------------------------------
PLACEHOLDER — wire in SHAP or LIME here.

For a scikit-learn TF-IDF + LogisticRegression pipeline the cheapest real
implementation is not even SHAP: LogisticRegression coefficients * TF-IDF
values give exact per-token contributions. Sketch:

    import numpy as np
    from model import _model, _vectorizer          # after load_model()

    def explain(text):
        X = _vectorizer.transform([text])
        contrib = X.multiply(_model.coef_[0]).toarray()[0]  # per-feature
        vocab = _vectorizer.get_feature_names_out()
        top = np.argsort(np.abs(contrib))[::-1][:20]
        ...map tokens back to sentences containing them...

For SHAP proper:

    import shap
    explainer = shap.LinearExplainer(_model, background_data)
    shap_values = explainer.shap_values(_vectorizer.transform([text]))

For the DistilBERT variant, use shap.Explainer with the HF pipeline, or
transformers-interpret / captum for token-level attributions.
--------------------------------------------------------------------------

Until then, this module ships a transparent heuristic: sentences containing
clickbait terms or shouting are marked "fake"-leaning, sentences with
sourcing language are marked "real"-leaning. It exists purely so the
highlight UI is testable end-to-end.
"""

import re

from signals import CLICKBAIT_TERMS, SOURCING_PATTERNS

# Split on sentence-ish boundaries; good enough for highlighting.
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


def explain(text: str) -> list[dict]:
    spans: list[dict] = []

    for match in _SENTENCE_RE.finditer(text):
        sentence = match.group().strip()
        if len(sentence) < 20:  # skip fragments
            continue
        lower = sentence.lower()

        clickbait_hits = sum(t in lower for t in CLICKBAIT_TERMS)
        letters = [c for c in sentence if c.isalpha()]
        shouty = letters and sum(c.isupper() for c in letters) / len(letters) > 0.3
        exclaimy = sentence.count("!") >= 2
        sourced = any(re.search(p, lower) for p in SOURCING_PATTERNS)

        if clickbait_hits or shouty or exclaimy:
            weight = min(0.4 + 0.2 * clickbait_hits + 0.2 * shouty + 0.2 * exclaimy, 1.0)
            spans.append({"text": sentence, "weight": round(weight, 2), "direction": "fake"})
        elif sourced:
            spans.append({"text": sentence, "weight": 0.5, "direction": "real"})

    # Cap the payload — the extension only needs the strongest spans.
    spans.sort(key=lambda s: s["weight"], reverse=True)
    return spans[:15]
