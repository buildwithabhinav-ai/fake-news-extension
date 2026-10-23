"""
Shared text cleaning for training AND serving.

The ISOT dataset (Fake.csv / True.csv) leaks its own construction into the
vocabulary: 99.2% of real articles carry a "(Reuters)" wire dateline, while
fake articles carry tweet embeds ("pic.twitter.com/…"), image credits
("Featured image via Getty Images"), and outlet self-references (21st Century
Wire, Breitbart). A classifier trained on raw text learns "has a dateline"
instead of "reads like manipulation". Everything here removes those artifacts
BEFORE vectorization so the model is forced onto style and content words.

This module is imported by both train.py (fit time) and model.py (inference
time) — keeping one implementation guarantees the two can never drift.
"""

import re

# --- Structural boilerplate stripped by regex ------------------------------

# Wire dateline at the start of a body: "WASHINGTON (Reuters) - ..." or
# "LONDON/PARIS (Reuters) -". Also catches AP/AFP variants.
_DATELINE_RE = re.compile(
    r"^[A-Z][A-Z\s,./'&-]{1,45}\((?:Reuters|Reuters Breakingviews|AP|AFP)\)\s*[-–—]*\s*",
    re.MULTILINE,
)
# Any remaining inline agency attribution: "(Reuters)", "(AP)", "(AFP)".
_AGENCY_RE = re.compile(r"\((?:Reuters|AP|AFP)\)", re.IGNORECASE)
# URLs, shorteners, and twitter picture links.
_URL_RE = re.compile(r"(?:https?://|www\.|bit\.ly/|pic\.twitter\.com/)\S+", re.IGNORECASE)
# @handles from embedded tweets.
_HANDLE_RE = re.compile(r"@\w+")
# Image/photo credit lines common in the fake corpus:
# "Featured image via Getty Images", "Photo by ...", "Image source: ...".
_IMAGE_CREDIT_RE = re.compile(
    r"(?:featured\s+)?(?:image|photo|picture|screenshot|screen\s*(?:shot|grab|capture))s?\s*"
    r"(?:via|by|from|credit|source|courtesy)[:\s][^.\n]*",
    re.IGNORECASE,
)
# "READ MORE:", "WATCH:", "VIDEO:" section markers.
_SECTION_RE = re.compile(r"\b(?:READ MORE|READ|WATCH|VIDEO|UPDATE)\s*:", re.IGNORECASE)
# Byline at the start of a line: "By Steve Holland" (only when line-initial,
# so ordinary prose like "by the committee" is untouched).
_BYLINE_RE = re.compile(r"^\s*By\s+[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){0,3}\s*$",
                        re.MULTILINE)
# Wire-copy timestamps: "8:30 p.m. EDT", "1100 GMT".
_TIMESTAMP_RE = re.compile(
    r"\b\d{1,2}[:.]?\d{0,2}\s*(?:a\.?m\.?|p\.?m\.?)?\s*(?:EDT|EST|GMT|ET|PT)\b",
    re.IGNORECASE,
)
# Aggregator/scraper page furniture (30k corpus): "18 minutes ago", "Read
# full article Read in NewsBreak", "3 min read", share-button "Copied".
_AGGREGATOR_RE = re.compile(
    r"\b\d+\s+(?:seconds?|minutes?|hours?|days?)\s+ago\b"
    r"|read\s+(?:full\s+article|in\s+newsbreak)\b"
    r"|\b\d+\s*min\s+read\b",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Strip dataset boilerplate; leaves style/punctuation intact."""
    text = _DATELINE_RE.sub("", text)
    text = _AGENCY_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _HANDLE_RE.sub(" ", text)
    text = _IMAGE_CREDIT_RE.sub(" ", text)
    text = _SECTION_RE.sub(" ", text)
    text = _BYLINE_RE.sub(" ", text)
    text = _TIMESTAMP_RE.sub(" ", text)
    text = _AGGREGATOR_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# --- Artifact stopwords (fed to the TF-IDF vectorizer) ----------------------
# Curated from the Step-1 audit: every entry was in (or adjacent to) the top-50
# weighted features of the leaky baseline and encodes WHICH corpus an article
# came from, not HOW it is written. Genuine style words the audit surfaced
# ("even", "really", "actually", "breaking", "know") are deliberately NOT here.

ARTIFACT_STOPWORDS = sorted({
    # wire agencies & outlets
    "reuters", "ap", "afp", "breitbart", "fox", "cnn", "nbc", "cbs", "abc",
    "msnbc", "bbc", "infowars", "wnd", "wire", "21wire", "21st", "daily",
    "century", "mail", "factbox", "breakingviews",
    # tweet/image-embed boilerplate
    "via", "video", "videos", "image", "images", "featured", "getty", "pic",
    "pics", "twitter", "tweet", "tweets", "com", "www", "http", "https",
    "watch", "read", "screengrab", "screenshot", "photo", "photos", "file",
    "caption", "ly", "flickr", "youtube", "instagram", "facebook",
    # dateline cities / countries over-represented in one corpus
    "washington", "london", "moscow", "york", "beijing", "geneva", "brussels",
    "paris", "berlin", "tokyo", "havana", "madrid", "rome", "istanbul",
    "ankara", "cairo", "jerusalem", "baghdad", "kabul", "islamabad", "riyadh",
    "britain", "china", "myanmar", "us", "capital",
    # politicians / people (topic bias, not fakeness)
    "trump", "donald", "hillary", "clinton", "obama", "barack", "michelle",
    "pence", "putin", "comey", "flynn", "mueller", "macron", "merkel",
    "sanders", "cruz", "pelosi", "mcconnell", "ryan", "bannon", "sessions",
    "kushner", "ivanka", "melania",
    # honorifics & political shorthand
    "mr", "mrs", "ms", "rep", "sen", "gop",
    # days, months, timezones (publication metadata bleed)
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "edt", "est", "gmt",
    # corporate boilerplate ("Thomson Reuters Inc")
    "inc", "corp", "thomson",
    # 30k-corpus source identity: aggregator/outlet names & page furniture
    # (top-weighted in the raw audit; they encode the scrape source, not style)
    "newsbreak", "news18", "hindustantimes", "huffpost", "huffington",
    "yahoo", "patch", "newsroom", "advertisement", "copied", "amp",
    "coronavirusfacts", "factcheck",
})