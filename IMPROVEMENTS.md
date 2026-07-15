# Model Improvements — Artifact-Free Retraining

**Date:** 2026-07-11 · **Scope:** `backend/` (classifier + stylometric signals). API schema unchanged; the Chrome extension needs no changes.

## The problem

The TF-IDF + Logistic Regression classifier scored ~98.6% accuracy, but a
vocabulary audit showed it was largely reading **which corpus an article came
from**, not **how it was written**. The ISOT dataset (`dataset/Fake.csv` /
`dataset/True.csv`) leaks its own construction:

- `(Reuters)` wire datelines appear in **99.2%** of real articles and 0.04% of
  fake ones. The single token `reuters`, used alone as a presence rule,
  classifies **99.4%** of the corpus.
- Fake articles carry tweet embeds (`pic.twitter.com/…`, `@handles`), image
  credits (`Featured image via Getty Images`), and outlet self-references
  (`21st Century Wire`, Breitbart).
- Weekday/month names and `EDT`/`EST` timestamps (Reuters copy is dated),
  dateline cities (`WASHINGTON`, `LONDON`, `MOSCOW`), and politician names
  (`hillary`, `obama`) rank among the strongest features — topic bias, not
  manipulation.

A model leaning on these collapses on any article that isn't wire copy or a
2016-era political blog post.

## What changed

1. **`backend/textclean.py` (new)** — shared by training *and* serving so they
   can't drift. Regex-strips datelines, `(Reuters)`/`(AP)`/`(AFP)`
   attributions, URLs, twitter handles, image-credit lines, `READ MORE:`
   markers, bylines, and wire timestamps. Also defines ~140 **artifact
   stopwords** (agencies, outlets, dateline cities, politician names,
   honorifics, day/month names, timezone codes) curated from the audit and fed
   to the vectorizer. Genuine style words the audit surfaced (`even`,
   `really`, `actually`, `breaking`, `know`) are deliberately kept.
2. **`backend/train.py` (new)** — reproducible pipeline: dedupe (5,793 exact
   duplicates and 630 near-empty bodies removed → 38,591 articles), audit
   table (feature | weight | class | keep/remove + reason), 5-fold stratified
   CV, cross-domain test, single-token leakage probe, artifact export with
   `.bak` backups. Cleaned vectorizer uses alphabetic-only tokens (drops
   years/dates), `sublinear_tf=True`, `min_df=5` — chosen by a cross-domain
   probe.
3. **`backend/model.py`** — inference preprocessing now calls
   `textclean.clean_text()` (matching training) instead of the old
   NLTK tokenize-and-rejoin step. Label encoding unchanged (0 = fake).
4. **`backend/signals.py`** — five new stylometric signals: ALL-CAPS word
   ratio, question-mark density, sentence-length variance, subjectivity
   (TextBlob), and quote-to-text ratio. All ten signal weights now sum to
   exactly **1.0** (largest single weight 0.20), so no detector can dominate
   the 0–100 manipulation score.
5. **`backend/requirements.txt`** — added `textblob`, `pandas`, `nltk`.

## Before / after (same deduped data, same LR classifier)

### Stratified 5-fold cross-validation (fake = positive class)

| Metric | Original | Cleaned |
|---|---|---|
| Accuracy | 0.9855 | 0.9736 |
| Precision (fake) | 0.9898 | 0.9784 |
| Recall (fake) | 0.9779 | 0.9626 |
| F1 (fake) | 0.9838 | 0.9704 |
| Confusion matrix [fake, real] | [[17012, 384], [176, 21019]] | [[16745, 651], [369, 20826]] |

The ~1.2-point drop is the point: that margin was leaked dataset identity,
not detection skill.

### Single-token leakage probe (flag if any token alone > 90%)

| Text | Strongest single-token rule | Verdict |
|---|---|---|
| Raw | `reuters` → **99.4%** | **LEAK** |
| Cleaned | `you` → 73.2% | ok — a genuine style signal (direct address) |

### Cross-domain test (proxy for train-on-dataset1 / test-on-dataset2)

There is no second corpus in the repo, so domains were built from the
disjoint `subject` fields: **A** = US politics (True `politicsNews` + Fake
`politics`/`Government News`/`left-news`), **B** = world/general (True
`worldnews` + Fake `News`/`US_News`/`Middle-east`).

| Direction | Original | Cleaned |
|---|---|---|
| A → B accuracy | 0.9396 | 0.9165 |
| B → A accuracy | 0.6789 | 0.6575 |

Caveat: the original's edge here is itself an artifact — `(Reuters)` exists
in *both* domains' real articles, so an in-dataset topic split can't punish
that leak; only the token probe exposes it. Both models degrade B→A (real
US-politics articles get misread as fake), which is honest evidence that
**any** bag-of-words model on this dataset overfits topic. This is exactly
why `/analyze` blends the classifier 50/50 with the stylometric signals.

### What the model relies on now

- **Fake:** `even, breaking, america, like, fact, entire, actually,
  reportedly, really, know` — informal register, emotive emphasis, hedged
  sourcing.
- **Real:** `said, republican, minister, spokesman, reporters, told,
  statement, comment, agency, percent` — attribution verbs and institutional
  sourcing language.

## Reproduce

```bash
cd backend
pip install -r requirements.txt
python3 train.py        # audit + validation + exports artifacts/
uvicorn main:app --reload --port 8000
```

Full numbers: `backend/artifacts/metrics.json`. Previous artifacts kept as
`model.pkl.bak` / `vectorizer.pkl.bak`.

---

# Retrain on the 30k/30k multi-source corpus

**Date:** 2026-07-12 · **Scope:** `backend/train.py` (`--dataset` flag),
`backend/artifacts/`. Same cleaned pipeline as above; no API or extension
changes. **The live `model.pkl`/`vectorizer.pkl` are now trained on this
corpus**; the ISOT versions are preserved as `model_isot.pkl`,
`vectorizer_isot.pkl`, `metrics_isot.json`.

## The corpus

`dataset/fake_30k.csv` + `dataset/real_30k.csv` — 30,000 rows each, drawn
from multiple sources rather than ISOT's two:

| Class | Sources |
|---|---|
| Fake | webhose_lowtrust_v2 (15,055) · daslab_fake_fullbody (11,329) · constraint_covid19 (3,616) |
| Real | rcv1_reuters_fulltext (9,479) · huffpost_news_category (6,993) · constraint_covid19 (5,596) · reuters_headline_archive (3,630) · webz_whitelisted_outlets (2,195) · bbc_news_2004_05 (2,107) |

After dropping <20-word bodies and exact duplicates: **52,722 articles**
(27,868 fake / 24,854 real). The multi-source mix matters: no single outlet's
boilerplate can define a class the way `(Reuters)` did in ISOT.

## Results (same cleaned pipeline: `clean_text` + artifact stopwords + alphabetic-only TF-IDF)

### Stratified 5-fold cross-validation (fake = positive class)

| Metric | Original (raw text) | Cleaned |
|---|---|---|
| Accuracy | 0.9304 | 0.8959 |
| Precision (fake) | 0.9345 | 0.8986 |
| Recall (fake) | 0.9338 | 0.9053 |
| F1 (fake) | 0.9341 | 0.9019 |

Lower than ISOT's 0.9736 — expected and healthier: six real sources and
three fake ones are harder to separate than two corpora with disjoint
boilerplate.

### Single-token leakage probe (flag if any token alone > 90%)

| Text | Strongest single-token rule | Verdict |
|---|---|---|
| Raw | `by` → 68.3% | ok |
| Cleaned | `by` → 68.2% | ok |

No ISOT-style leak exists in this corpus even before cleaning; the strongest
"rule" is a common function word. Aggregator furniture that did surface in
the audit (`newsbreak`, `news18`, `huffpost`, `advertisement`, …) was added
to the artifact stopword list (see the "aggregator/outlet furniture (30k)"
group in `train.py` / `textclean.py`).

### Cross-domain test (A = constraint_covid19 slice, B = everything else)

constraint_covid19 is the only source present in *both* classes, so it forms
a true benchmark slice: train on covid articles → test on everything else,
and vice versa.

| Direction | Original | Cleaned |
|---|---|---|
| A → B accuracy | 0.5076 | 0.5233 |
| B → A accuracy | 0.6276 | 0.5450 |

Both models land near chance across this topic shift (covid short-form
social-media claims vs long-form general news — a domain gap, not just a
topic gap). Same conclusion as ISOT: bag-of-words style classifiers do not
transfer across domains, which is why `/analyze` blends the classifier 50/50
with the interpretable stylometric signals and the UI frames everything as
*style* analysis.

## Verification (2026-07-12)

- Serving path (`model.py` → `clean_text` → vectorizer → LR) smoke-tested:
  clickbait-style text → P(fake) 0.849; wire-style text → P(fake) 0.022.
- 200+200 sampled corpus rows through the serving path: 93.0% / 92.5%,
  consistent with CV — no train/serve preprocessing drift.
- Live API: `/health` reports `model_is_stub: false`; `/analyze` returns
  `likely_fake` (score 71) for a clickbait sample and `likely_real` (score 1)
  for a wire-copy sample, with all 10 signals populated.

## Reproduce / switch datasets

```bash
cd backend
python3 train.py --dataset 30k    # retrain on the 30k corpus (current live model)
python3 train.py                  # retrain on ISOT instead
```

Full numbers: `backend/artifacts/metrics.json` (30k) and
`backend/artifacts/metrics_isot.json` (ISOT).
