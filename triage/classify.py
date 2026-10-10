"""Failure classifier: predict BGL's alert category (or BENIGN) for each failure line.

Ground truth is BGL's real per-line alert label, so accuracy numbers are honest.
Training rows are de-duplicated (masked text, label) pairs, each weighted equally: frequency
weighting let a few huge classes (e.g. KERNDTLB, 150k lines) swamp rare ones. The weighting and
C were picked on a Jun-Jul train / Aug validation split, never on the test period.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

from . import config


def features(df: pd.DataFrame) -> pd.Series:
    return "comp_" + df["component"] + " lvl_" + df["level"] + " " + df["masked"]


class FailureClassifier:
    def __init__(self, state_dir: Path):
        self.path = state_dir / "classifier.joblib"
        self.model = joblib.load(self.path) if self.path.exists() else None

    def fit(self, df: pd.DataFrame) -> "FailureClassifier":
        agg = df.assign(text=features(df))[["text", "label"]].drop_duplicates()
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"[^\s:;,()\[\]=]+", sublinear_tf=True),
            LogisticRegression(max_iter=3000, C=10.0),
        )
        self.model.fit(agg["text"], agg["label"])
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (labels, confidence) per row. Prediction runs once per distinct text."""
        text = features(df)
        uniq = text.unique()
        proba = self.model.predict_proba(uniq)
        classes = self.model.classes_
        lab = dict(zip(uniq, classes[proba.argmax(1)]))
        conf = dict(zip(uniq, proba.max(1)))
        return text.map(lab).to_numpy(), text.map(conf).to_numpy()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.path)


def lookup_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Majority training label per Drain template; unseen templates default to BENIGN."""
    majority = train.groupby("template_id")["label"].agg(lambda s: s.value_counts().idxmax())
    return test["template_id"].map(majority).fillna(config.BENIGN_LABEL).to_numpy()


def score(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true))
    alert_true = y_true != config.BENIGN_LABEL
    alert_pred = y_pred != config.BENIGN_LABEL
    p, r, f, _ = precision_recall_fscore_support(alert_true, alert_pred, average="binary", zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "alert_detection": {"precision": float(p), "recall": float(r), "f1": float(f)},
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }
