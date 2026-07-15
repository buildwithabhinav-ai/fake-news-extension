"""
Train + validate the Fake News Detective classifier.

Run from the repo root or backend/:

    python3 backend/train.py                  # ISOT: full audit + eval + export
    python3 backend/train.py --dataset 30k    # fake_30k/real_30k corpus

Produces:
    backend/artifacts/model.pkl / vectorizer.pkl   (cleaned model, serving)
    backend/artifacts/metrics.json                 (before/after numbers)

Pipeline compared side by side:
  ORIGINAL  raw text -> TF-IDF(nltk stopwords) -> LogisticRegression
  CLEANED   textclean.clean_text -> TF-IDF(nltk + artifact stopwords,
            alphabetic-only tokens) -> LogisticRegression

Labels match model.py's FAKE_CLASS: 0 = fake, 1 = real.
"""

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))
from textclean import ARTIFACT_STOPWORDS, clean_text  # noqa: E402

import nltk  # noqa: E402

try:
    from nltk.corpus import stopwords
    NLTK_STOP = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    NLTK_STOP = stopwords.words("english")

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
SEED = 42

# Reasons shown in the Step-1 audit table for flagged features.
ARTIFACT_REASONS = {
    "agency/outlet name": {"reuters", "ap", "afp", "breitbart", "fox", "wire",
                           "21wire", "21st", "daily", "press"},
    "tweet/image boilerplate": {"via", "video", "image", "images", "featured",
                                "getty", "pic", "com", "https", "http", "www",
                                "twitter", "read", "watch", "ly", "photo"},
    "dateline city / country topic bias": {"washington", "london", "moscow",
                                           "york", "britain", "china",
                                           "myanmar", "us", "capital"},
    "politician name (topic bias)": {"trump", "donald", "hillary", "clinton",
                                     "obama", "barack", "comey", "flynn",
                                     "mueller", "putin"},
    "honorific/political shorthand": {"mr", "mrs", "rep", "sen", "gop"},
    "date/time metadata": {"monday", "tuesday", "wednesday", "thursday",
                           "friday", "saturday", "sunday", "january",
                           "february", "march", "april", "june", "july",
                           "august", "september", "october", "november",
                           "december", "nov", "dec", "oct", "sept", "edt",
                           "est", "gmt"},
    "corporate boilerplate": {"inc", "corp", "thomson"},
    "aggregator/outlet furniture (30k)": {"newsbreak", "news18",
                                          "hindustantimes", "huffpost",
                                          "huffington", "yahoo", "patch",
                                          "newsroom", "advertisement",
                                          "copied", "amp", "coronavirusfacts",
                                          "factcheck"},
}


def flag(feature: str) -> str:
    for reason, words in ARTIFACT_REASONS.items():
        if feature in words:
            return f"REMOVE — {reason}"
    return "keep — plausible style/content signal"


def load_data(dataset: str = "isot") -> pd.DataFrame:
    if dataset == "isot":
        fake = pd.read_csv(ROOT / "dataset/Fake.csv")
        true = pd.read_csv(ROOT / "dataset/True.csv")
    else:  # 30k: multi-source corpus with a shared covid benchmark slice
        fake = pd.read_csv(ROOT / "dataset/fake_30k.csv")
        true = pd.read_csv(ROOT / "dataset/real_30k.csv")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    df["full"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    n0 = len(df)
    df = df[df["full"].str.split().str.len() >= 20].drop_duplicates(subset="full")
    df = df.reset_index(drop=True)
    # domain column drives the cross-domain eval:
    #   isot -> US-politics subjects vs world/general subjects
    #   30k  -> constraint_covid19 source (in both classes) vs everything else
    if dataset == "isot":
        a_subjects = {"politicsNews", "politics", "Government News", "left-news"}
        df["domain"] = np.where(df.subject.isin(a_subjects), "A", "B")
        dom_names = {"A": "us-politics", "B": "world/general"}
    else:
        df["domain"] = np.where(df.source_dataset == "constraint_covid19", "A", "B")
        dom_names = {"A": "covid19", "B": "non-covid sources"}
    df.attrs["dom_names"] = dom_names
    print(f"[data] {n0} rows -> {len(df)} after dropping <20-word bodies and "
          f"exact duplicates ({dict(df.label.value_counts())}; 0=fake, 1=real)")
    return df


def make_original():
    return (TfidfVectorizer(lowercase=True, stop_words=NLTK_STOP,
                            max_features=50000),
            LogisticRegression(max_iter=1000, C=1.0, random_state=SEED))


def make_cleaned():
    # sublinear_tf + min_df=5 chosen by a cross-domain probe: they gave the
    # best topic-shift generalization (B->A) without inflating the model.
    return (TfidfVectorizer(lowercase=True,
                            stop_words=sorted(set(NLTK_STOP) | set(ARTIFACT_STOPWORDS)),
                            token_pattern=r"(?u)\b[a-z]{2,}\b",  # letters only: drops years/dates
                            max_features=50000, sublinear_tf=True, min_df=5),
            LogisticRegression(max_iter=1000, C=1.0, random_state=SEED))


def audit(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("STEP 1 — VOCABULARY AUDIT OF THE ORIGINAL (UNCLEANED) MODEL")
    print("=" * 78)
    vec, clf = make_original()
    X = vec.fit_transform(df["full"])
    clf.fit(X, df["label"])
    feats = vec.get_feature_names_out()
    coef = clf.coef_[0]  # >0 pushes toward real (1), <0 toward fake (0)
    order = np.argsort(coef)

    header = f"{'feature':<18} {'weight':>8}  {'class':<5}  verdict"
    for cls, idxs in (("fake", order[:50]), ("real", order[::-1][:50])):
        print(f"\n--- top 50 features for the {cls.upper()} class ---")
        print(header)
        print("-" * len(header) * 2)
        for i in idxs:
            print(f"{feats[i]:<18} {coef[i]:>+8.3f}  {cls:<5}  {flag(feats[i])}")


def evaluate_cv(name, texts, y):
    """Stratified 5-fold CV; returns metrics dict for the FAKE class (label 0)."""
    vec, clf = make_original() if name == "original" else make_cleaned()
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(vec, clf)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(pipe, texts, y, cv=skf, n_jobs=-1)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, pos_label=0,
                                                  average="binary")
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision_fake": round(p, 4),
        "recall_fake": round(r, 4),
        "f1_fake": round(f1, 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),  # rows: true fake, true real
    }


def evaluate_cross_domain(df):
    """
    Simulate 'train on dataset1, test on dataset2' with the domain split set
    up in load_data (topic split for ISOT, covid-vs-rest source split for
    the 30k corpus). A model that memorized artifacts collapses here; one
    that learned style should hold up.
    """
    dom_names = df.attrs.get("dom_names", {"A": "A", "B": "B"})
    dom_a = df[df.domain == "A"]
    dom_b = df[df.domain == "B"]
    print(f"[cross-domain] A({dom_names['A']})={len(dom_a)} "
          f"{dict(dom_a.label.value_counts())} | "
          f"B({dom_names['B']})={len(dom_b)} {dict(dom_b.label.value_counts())}")

    results = {}
    for name in ("original", "cleaned"):
        col = "full" if name == "original" else "clean"
        for tr, te, tag in ((dom_a, dom_b, "A->B"), (dom_b, dom_a, "B->A")):
            vec, clf = make_original() if name == "original" else make_cleaned()
            clf.fit(vec.fit_transform(tr[col]), tr["label"])
            pred = clf.predict(vec.transform(te[col]))
            p, r, f1, _ = precision_recall_fscore_support(
                te["label"], pred, pos_label=0, average="binary")
            results[f"{name} {tag}"] = {
                "accuracy": round(accuracy_score(te["label"], pred), 4),
                "precision_fake": round(p, 4),
                "recall_fake": round(r, 4),
                "f1_fake": round(f1, 4),
            }
    return results


def leakage_probe(df, top_k=200):
    """
    STEP 4 leakage check: can ANY single token, used alone as a presence/
    absence rule, hit >90% accuracy? Tested on both raw and cleaned text.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    findings = {}
    for col in ("full", "clean"):
        cv = CountVectorizer(lowercase=True, binary=True, max_features=20000,
                             token_pattern=r"(?u)\b[a-z]{2,}\b")
        X = cv.fit_transform(df[col])
        y = df["label"].to_numpy()
        feats = cv.get_feature_names_out()
        # accuracy of rule "token present => real" (and its negation) per token
        n = len(y)
        pos = np.asarray(X[y == 1].sum(axis=0)).ravel()   # real docs containing token
        neg = np.asarray(X[y == 0].sum(axis=0)).ravel()   # fake docs containing token
        n_real, n_fake = (y == 1).sum(), (y == 0).sum()
        acc_if_real = (pos + (n_fake - neg)) / n          # present=>real
        acc_if_fake = (neg + (n_real - pos)) / n          # present=>fake
        best = np.maximum(acc_if_real, acc_if_fake)
        order = np.argsort(best)[::-1][:10]
        findings[col] = [(feats[i], round(float(best[i]), 4)) for i in order]
    return findings


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("isot", "30k"), default="isot")
    args = ap.parse_args()
    df = load_data(args.dataset)
    audit(df)

    print("\n" + "=" * 78)
    print("STEP 2 — cleaning text (datelines, agencies, URLs, handles, bylines)")
    print("=" * 78)
    df["clean"] = df["full"].map(clean_text)
    print(f"[clean] avg chars {df['full'].str.len().mean():.0f} -> "
          f"{df['clean'].str.len().mean():.0f}; "
          f"{len(ARTIFACT_STOPWORDS)} artifact stopwords added to vectorizer")

    print("\n" + "=" * 78)
    print("STEP 4 — validation")
    print("=" * 78)
    metrics = {"cv_5fold": {}, "cross_domain": {}, "leakage_probe": {}}

    for name in ("original", "cleaned"):
        col = "full" if name == "original" else "clean"
        m = evaluate_cv(name, df[col], df["label"])
        metrics["cv_5fold"][name] = m
        print(f"[5-fold CV] {name:<9} acc={m['accuracy']} "
              f"P(fake)={m['precision_fake']} R(fake)={m['recall_fake']} "
              f"F1(fake)={m['f1_fake']} cm={m['confusion_matrix']}")

    metrics["cross_domain"] = evaluate_cross_domain(df)
    for k, v in metrics["cross_domain"].items():
        print(f"[cross-domain] {k:<15} acc={v['accuracy']} "
              f"P(fake)={v['precision_fake']} R(fake)={v['recall_fake']} "
              f"F1(fake)={v['f1_fake']}")

    metrics["leakage_probe"] = leakage_probe(df)
    for col, toks in metrics["leakage_probe"].items():
        label = "RAW text" if col == "full" else "CLEANED text"
        worst = toks[0]
        print(f"[leakage] {label}: strongest single-token rule = "
              f"'{worst[0]}' at {worst[1]:.1%} accuracy "
              f"{'*** LEAK (>90%) ***' if worst[1] > 0.9 else '(ok, <90%)'}")
        print(f"          top tokens: {toks}")

    # ---- STEP 5: export the cleaned model for the API --------------------
    print("\n" + "=" * 78)
    print("STEP 5 — exporting cleaned model to backend/artifacts/")
    print("=" * 78)
    for f in ("model.pkl", "vectorizer.pkl"):
        src = ARTIFACT_DIR / f
        if src.exists():
            shutil.copy(src, ARTIFACT_DIR / f"{f}.bak")
    vec, clf = make_cleaned()
    X = vec.fit_transform(df["clean"])
    clf.fit(X, df["label"])
    joblib.dump(clf, ARTIFACT_DIR / "model.pkl")
    joblib.dump(vec, ARTIFACT_DIR / "vectorizer.pkl")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[export] wrote model.pkl, vectorizer.pkl (+ .bak backups), "
          f"metrics.json to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
