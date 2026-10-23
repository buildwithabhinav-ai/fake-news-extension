"""
Stylometric signal extraction.

Each signal is an interpretable, hand-computable feature of the *style* of a
text — deliberately independent of the ML model. The whole point of surfacing
these is honesty: the classifier's probability is opaque and known to lean on
dataset artifacts, while these signals are things a reader can verify with
their own eyes ("yes, this headline really is 40% capital letters").

Every signal returns:
    name        machine-readable id
    value       the raw measured value (float)
    triggered   whether it crossed the "worth mentioning" threshold
    explanation plain-English sentence for the popup UI
    weight      0.0-1.0 contribution to the manipulation score when triggered
"""


import re
import statistics
from dataclasses import dataclass, asdict

import textstat
from textblob import TextBlob

# Words/phrases common in sensationalist or engagement-bait writing.
# Extend this freely — it's a signal, not a verdict, so false positives are
# cheap as long as the UI explains *why* something was flagged.
CLICKBAIT_TERMS = [
    "shocking", "you won't believe", "you wont believe", "destroyed",
    "slams", "bombshell", "explosive", "unbelievable", "mind-blowing",
    "what happens next", "the truth about", "they don't want you to know",
    "wake up", "sheeple", "mainstream media won't", "banned", "exposed",
    "miracle", "secret", "outrageous", "insane", "epic", "furious",
    "breaking", "urgent", "share before it's deleted", "go viral",
]

# Markers of conventional news sourcing: wire services, datelines,
# attribution verbs. Their *presence* is weak evidence of standard editorial
# practice; their absence in a long "news" piece is a mild warning sign.
SOURCING_PATTERNS = [
    r"\(reuters\)", r"\(ap\)", r"\(afp\)", r"\bassociated press\b",
    r"\baccording to\b", r"\bofficials said\b", r"\bspokesperson\b",
    r"\bin a statement\b", r"\btold reporters\b", r"\bpress release\b",
    r"\bcited\b", r"\bsources? (?:said|told|confirmed)\b",
]

# A wire dateline like "WASHINGTON (Reuters) -" at the start of the body.
DATELINE_RE = re.compile(r"^[A-Z][A-Z\s,./]{2,30}\((?:Reuters|AP|AFP)\)", re.MULTILINE)


@dataclass
class Signal:
    name: str
    value: float
    triggered: bool
    explanation: str
    weight: float  # contribution to manipulation score when triggered

    def to_api(self) -> dict:
        """Shape returned to the extension.

        `triggered` and `weight` are exposed so the popup can colour a signal
        by whether it actually fired and show how much of the manipulation
        score it is responsible for — the score stays fully itemized in the UI.
        """
        d = asdict(self)
        d["value"] = round(d["value"], 3)
        return d


def _caps_ratio(text: str) -> Signal:
    letters = [c for c in text if c.isalpha()]
    ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
    triggered = ratio > 0.12  # normal prose sits around 0.02-0.06
    return Signal(
        name="caps_ratio",
        value=ratio,
        triggered=triggered,
        explanation=(
            f"{ratio:.0%} of letters are UPPERCASE. Heavy capitalization is a "
            "common attention-grabbing device in manipulative writing."
            if triggered else
            f"Capitalization looks normal ({ratio:.0%} uppercase)."
        ),
        weight=0.10,
    )


def _allcaps_words(text: str) -> Signal:
    # Whole SHOUTED words (4+ letters, so acronyms like FBI/GOP don't count).
    words = re.findall(r"[A-Za-z]{4,}", text)
    shouted = [w for w in words if w.isupper()]
    ratio = len(shouted) / max(len(words), 1)
    triggered = ratio > 0.03  # straight news is essentially 0
    return Signal(
        name="allcaps_words",
        value=ratio,
        triggered=triggered,
        explanation=(
            f"{ratio:.1%} of words are fully SHOUTED (e.g. "
            f"{', '.join(shouted[:3])}). Wire copy virtually never "
            "capitalizes whole words for emphasis."
            if triggered else
            "No unusual use of ALL-CAPS words."
        ),
        weight=0.05,
    )


def _exclamation_density(text: str) -> Signal:
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    exclaims = text.count("!")
    density = exclaims / sentences
    triggered = density > 0.10  # more than ~1 exclamation per 10 sentences
    return Signal(
        name="exclamation_density",
        value=density,
        triggered=triggered,
        explanation=(
            f"{exclaims} exclamation mark(s) across ~{sentences} sentences. "
            "Wire-style news reporting almost never uses exclamation marks."
            if triggered else
            "Exclamation mark usage is within the range typical of straight news."
        ),
        weight=0.15,
    )


def _question_density(text: str) -> Signal:
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    questions = text.count("?")
    density = questions / sentences
    triggered = density > 0.15  # rhetorical-question barrage
    return Signal(
        name="question_density",
        value=density,
        triggered=triggered,
        explanation=(
            f"{questions} question mark(s) across ~{sentences} sentences. "
            "Chains of rhetorical questions (“Coincidence? You decide!”) are "
            "a persuasion device, not a reporting style."
            if triggered else
            "Question mark usage looks typical of straight news."
        ),
        weight=0.10,
    )


def _clickbait_terms(text: str) -> Signal:
    lower = text.lower()
    hits = [t for t in CLICKBAIT_TERMS if t in lower]
    triggered = len(hits) >= 2  # one hit alone is too noisy
    shown = ", ".join(f"“{h}”" for h in hits[:5])
    return Signal(
        name="clickbait_terms",
        value=float(len(hits)),
        triggered=triggered,
        explanation=(
            f"Contains sensationalist phrasing: {shown}. Emotional-trigger "
            "vocabulary is designed to provoke sharing rather than inform."
            if triggered else
            "Little or no sensationalist vocabulary detected."
        ),
        weight=0.20,
    )


def _readability(text: str) -> Signal:
    # Flesch Reading Ease: higher = easier. Wire copy usually lands 40-60.
    # Extremely easy text (>80) in a "news" register can indicate copy tuned
    # for virality; we flag it gently and at low weight because readable
    # writing is obviously not evidence of deception on its own.
    score = float(textstat.flesch_reading_ease(text))
    triggered = score > 80
    return Signal(
        name="readability",
        value=score,
        triggered=triggered,
        explanation=(
            f"Flesch reading ease is {score:.0f} (very simple language). "
            "Not suspicious by itself, but viral misinformation is often "
            "written far below the register of standard news copy."
            if triggered else
            f"Reading level (Flesch {score:.0f}) is typical of published news."
        ),
        weight=0.05,
    )


def _sentence_length_variance(text: str) -> Signal:
    lengths = [len(s.split()) for s in re.split(r"[.!?]+", text) if s.split()]
    if len(lengths) < 5:
        cv = 0.0  # too short to judge
    else:
        cv = statistics.pstdev(lengths) / max(statistics.mean(lengths), 1)
    triggered = cv > 0.9
    return Signal(
        name="sentence_length_variance",
        value=cv,
        triggered=triggered,
        explanation=(
            "Sentence lengths swing wildly (very short punchy fragments mixed "
            "with long rants) — a rhythm typical of persuasion writing, not "
            "edited news copy."
            if triggered else
            "Sentence rhythm is consistent with edited prose."
        ),
        weight=0.05,
    )


def _subjectivity(text: str) -> Signal:
    # TextBlob's pattern-based subjectivity: 0 = objective, 1 = opinion.
    # Straight news usually lands 0.3-0.45. Capped input keeps latency low.
    score = float(TextBlob(text[:10000]).sentiment.subjectivity)
    # Wire copy with vivid-but-factual language can reach ~0.6, so the
    # trigger sits above that; sustained opinion writing lands 0.65+.
    triggered = score > 0.65
    return Signal(
        name="subjectivity",
        value=score,
        triggered=triggered,
        explanation=(
            f"Language is heavily opinionated (subjectivity {score:.2f} on a "
            "0-1 scale). News reporting separates fact from commentary; this "
            "text doesn't."
            if triggered else
            f"Subjectivity ({score:.2f}) is within the normal range for "
            "reporting."
        ),
        weight=0.10,
    )


def _quote_ratio(text: str) -> Signal:
    # Fraction of characters inside quotation marks. Real reporting quotes
    # its sources; long "news" with zero quoted material is a warning sign.
    quoted = sum(len(m) for m in re.findall(r"[\"“]([^\"”]{2,300})[\"”]", text))
    ratio = quoted / max(len(text), 1)
    long_enough = len(text.split()) > 150
    triggered = long_enough and ratio < 0.005
    return Signal(
        name="quote_ratio",
        value=ratio,
        triggered=triggered,
        explanation=(
            "Almost no directly quoted material for an article of this "
            "length. Legitimate reporting quotes the people it covers."
            if triggered else
            f"{ratio:.0%} of the text is quoted material — consistent with "
            "sourced reporting."
        ),
        weight=0.10,
    )


def _sourcing(text: str) -> Signal:
    lower = text.lower()
    hits = sum(bool(re.search(p, lower)) for p in SOURCING_PATTERNS)
    has_dateline = bool(DATELINE_RE.search(text))
    value = float(hits + (1 if has_dateline else 0))
    # Only meaningful for article-length text; a tweet lacking "officials said"
    # tells us nothing.
    long_enough = len(text.split()) > 150
    triggered = long_enough and value == 0
    return Signal(
        name="sourcing_markers",
        value=value,
        triggered=triggered,
        explanation=(
            "No attribution language found (“according to…”, "
            "“officials said…”, wire datelines). Legitimate "
            "reporting nearly always names its sources."
            if triggered else
            f"Found {int(value)} sourcing/attribution marker(s) — consistent "
            "with conventional editorial practice."
        ),
        weight=0.10,
    )


def compute_signals(text: str, title: str = "") -> list[Signal]:
    """
    Run every stylometric detector over title + body.

    Weights are normalized: the ten signals' weights sum to exactly 1.0, so
    the style score is a true 0-1 scale and no single detector can dominate
    the manipulation score (worst offender caps at 0.20).
    """
    combined = f"{title}\n{text}".strip()
    return [
        _caps_ratio(combined),          # 0.10
        _allcaps_words(combined),       # 0.05
        _exclamation_density(combined), # 0.15
        _question_density(combined),    # 0.10
        _clickbait_terms(combined),     # 0.20
        _readability(text or combined), # 0.05
        _sentence_length_variance(text or combined),  # 0.05
        _subjectivity(text or combined),              # 0.10
        _quote_ratio(text or combined),               # 0.10
        _sourcing(text or combined),                  # 0.10
    ]


def style_score(signals: list[Signal]) -> float:
    """
    Collapse triggered signals into a 0.0-1.0 'stylistic manipulation' score.
    Simple weighted sum — kept dumb on purpose so it stays explainable:
    the score IS the sum of the reasons shown in the UI, nothing hidden.
    """
    return min(sum(s.weight for s in signals if s.triggered), 1.0)
