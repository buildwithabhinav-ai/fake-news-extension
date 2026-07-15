"""
Model loading + prediction.

The classifier is deliberately treated as ONE weak signal among several, not
the verdict. Out-of-distribution testing (see README "Known Limitations")
showed the TF-IDF models lean on dataset artifacts — wire datelines, the
disjoint `subject` field — so its probability is blended with stylometric
signals rather than reported as "truth".

To plug in your real artifacts, drop them next to this file (or set the
env vars) and restart the server:

    MODEL_PATH       default: backend/artifacts/model.pkl
    VECTORIZER_PATH  default: backend/artifacts/vectorizer.pkl

If either file is missing, a stub model is used so the whole pipeline
(extension included) works end-to-end before you've exported anything.
"""

import os
import re
from pathlib import Path

import joblib

from textclean import clean_text

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ARTIFACT_DIR / "model.pkl"))
VECTORIZER_PATH = Path(os.environ.get("VECTORIZER_PATH", ARTIFACT_DIR / "vectorizer.pkl"))

_model = None
_vectorizer = None
_using_stub = True


def _preprocess(text: str) -> str:
    """
    Mirror training preprocessing exactly. train.py fits the vectorizer on
    textclean.clean_text() output (artifact boilerplate stripped); the
    vectorizer itself handles lowercasing, tokenization, and stopwords. Both
    sides import the same clean_text, so train/serve can't drift — skipping
    or diverging here is a common, easy-to-miss bug when wiring a trained
    model into a live API.
    """
    return clean_text(text)


def load_model() -> None:
    """Called once at server startup. Falls back to stub if artifacts absent."""
    global _model, _vectorizer, _using_stub
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _using_stub = False
        print(f"[model] loaded {MODEL_PATH.name} + {VECTORIZER_PATH.name}")
    else:
        print(
            f"[model] artifacts not found at {MODEL_PATH} / {VECTORIZER_PATH} "
            "— using STUB model. Export your trained model with "
            "joblib.dump() to those paths to use the real one."
        )


def using_stub() -> bool:
    return _using_stub


def predict_proba_fake(text: str) -> float:
    """
    Return P(fake) in [0, 1].

    Real path: vectorize -> predict_proba, taking the column that corresponds
    to the "fake" class. Adjust FAKE_CLASS below if your label encoding
    differs (e.g. you trained with 1 = real).
    """
    if not _using_stub:
        X = _vectorizer.transform([_preprocess(text)])
        proba = _model.predict_proba(X)[0]
        # --- ADJUST TO YOUR LABEL ENCODING -------------------------------
        # classes_ tells you the column order, e.g. array([0, 1]) or
        # array(['FAKE', 'REAL']). Pick the index of your fake label:
        # backend/train.py labels Fake.csv as 0 and True.csv as 1, so
        # 0 = fake for the exported model.pkl.
        FAKE_CLASS = 0  # <-- change me if needed
        # ------------------------------------------------------------------
        try:
            idx = list(_model.classes_).index(FAKE_CLASS)
        except ValueError:
            idx = 0
        return float(proba[idx])

    return _stub_predict(text)


def _stub_predict(text: str) -> float:
    """
    Crude heuristic stand-in so the API returns plausible values before the
    real artifacts are wired up. NOT a classifier — delete-safe.
    """
    lower = text.lower()
    score = 0.5
    if "!" in text:
        score += min(text.count("!") * 0.03, 0.2)
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.12:
        score += 0.15
    if re.search(r"\((?:reuters|ap|afp)\)", lower):
        score -= 0.3
    if "according to" in lower or "officials said" in lower:
        score -= 0.1
    return max(0.02, min(score, 0.98))
