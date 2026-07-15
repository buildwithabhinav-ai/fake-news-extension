"""
Fake News Detective — analysis API.

Run from the backend/ directory:

    uvicorn main:app --reload --port 8000

One endpoint: POST /analyze  { "text": "...", "title": "..." }

Design note on framing: the response never claims to measure truth. `label`
and `confidence` come from a classifier known (from our own OOD testing) to
key on stylistic/dataset artifacts, so `manipulation_score` blends that
probability with independent, human-verifiable stylometric signals, and the
UI copy presents everything as *style* analysis. Keep it that way.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import model
from explain import explain
from signals import compute_signals, style_score

app = FastAPI(title="Fake News Detective API", version="1.0.0")

# CORS: the popup fetches from a chrome-extension:// origin, which varies by
# install, so allow the scheme by regex. localhost is handy for curl/tests.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"chrome-extension://.*",
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Article body text")
    title: str = Field(default="", description="Article title/headline")


class SignalOut(BaseModel):
    name: str
    value: float
    explanation: str


class SpanOut(BaseModel):
    text: str
    weight: float
    direction: str  # "fake" | "real"


class AnalyzeResponse(BaseModel):
    label: str  # "likely_real" | "likely_fake"
    confidence: float
    manipulation_score: int  # 0-100
    signals: list[SignalOut]
    highlighted_spans: list[SpanOut]
    model_is_stub: bool  # so the UI can warn while artifacts aren't wired up
    disclaimer: str


DISCLAIMER = (
    "This tool flags stylistic manipulation patterns, not factual accuracy. "
    "A low score does not mean an article is true, and a high score does not "
    "mean it is false."
)


@app.on_event("startup")
def _startup() -> None:
    model.load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_is_stub": model.using_stub()}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    text = req.text.strip()
    if len(text.split()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Text too short to analyze meaningfully (need ~20+ words).",
        )

    # 1) Classifier probability — treated as one signal, not the verdict.
    p_fake = model.predict_proba_fake(f"{req.title} {text}".strip())

    # 2) Independent stylometric signals.
    sigs = compute_signals(text, req.title)

    # 3) Blend: half classifier, half interpretable style score. The blend
    #    keeps a bad classifier from dominating and makes the number legible:
    #    a user can see most of their score itemized in the signals list.
    score_style = style_score(sigs)
    manipulation = round(100 * (0.5 * p_fake + 0.5 * score_style))

    # Label/confidence reflect only the classifier, since that's what they
    # are: its opinion. Confidence = distance from the 0.5 coin-flip line.
    label = "likely_fake" if p_fake >= 0.5 else "likely_real"
    confidence = round(abs(p_fake - 0.5) * 2, 3)

    return AnalyzeResponse(
        label=label,
        confidence=confidence,
        manipulation_score=manipulation,
        signals=[SignalOut(**s.to_api()) for s in sigs],
        highlighted_spans=[SpanOut(**s) for s in explain(text)],
        model_is_stub=model.using_stub(),
        disclaimer=DISCLAIMER,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
