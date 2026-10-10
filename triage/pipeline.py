"""Orchestration: bootstrap (train + evaluate) and per-run triage of new logs."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_score, normalized_mutual_info_score

from . import buckets, config, report
from .classify import FailureClassifier, lookup_baseline, score
from .embed import Embedder
from .ingest import read_failures_file
from .parsing import TemplateParser, mask
from .signatures import SignatureStore
from .summarize import summarize

BUCKET_ORDER = {b: i for i, b in enumerate(["new_bug", "known_bug", "flaky", "environment", "noise"])}


class Triage:
    def __init__(self, state_dir: Path = config.STATE_DIR):
        self.state_dir = state_dir
        self.parser = TemplateParser(state_dir / "drain.pkl")
        self.embedder = Embedder(state_dir)
        self.sigs = SignatureStore(state_dir)
        self.clf = FailureClassifier(state_dir)

    def save(self) -> None:
        self.parser.save()
        self.sigs.save()
        self.clf.save()

    # ---- shared steps ------------------------------------------------------
    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["masked"] = df["message"].map(mask)
        df["template_id"] = self.parser.parse(df["message"])
        return df

    def attach_signatures(self, df: pd.DataFrame) -> set[str]:
        """Map every row to a signature, creating signatures for unseen templates."""
        templates = self.parser.templates()
        unseen = sorted(set(df["template_id"].unique()) - set(self.sigs.template_to_sig))
        created: set[str] = set()
        if unseen:
            vecs = self.embedder.encode([templates[t] for t in unseen])
            created = set(self.sigs.assign_new(unseen, vecs))
        df["signature"] = df["template_id"].map(self.sigs.template_to_sig)
        return created

    # ---- bootstrap ---------------------------------------------------------
    def bootstrap(self, df: pd.DataFrame, train_end: str) -> dict:
        t0 = time.time()
        train = self.parse(df[df["date"] <= train_end])
        templates = self.parser.templates()
        tids = sorted(train["template_id"].unique())
        vecs = self.embedder.encode([templates[t] for t in tids], fit=True)
        self.sigs.bootstrap(tids, vecs)
        train["signature"] = train["template_id"].map(self.sigs.template_to_sig)
        self.sigs.record(train)
        self.clf.fit(train)
        self.save()
        print(f"trained on {len(train):,} failure lines in {time.time() - t0:.0f}s")

        # Evaluate on the later period with a fresh copy of the saved state, learning
        # templates/signatures online exactly as daily runs would.
        evaluator = Triage(self.state_dir)
        test = evaluator.parse(df[df["date"] > train_end])
        created = evaluator.attach_signatures(test)
        pred, _ = evaluator.clf.predict(test)
        base = lookup_baseline(train, test)
        unseen = ~test["template_id"].isin(set(train["template_id"]))
        novel = ~test["label"].isin(set(train["label"]))  # alert classes never seen in training
        in_new = test["signature"].isin(created)
        daily = test.groupby("date").agg(lines=("label", "size"), signatures=("signature", "nunique"))

        metrics = {
            "dataset": "Loghub BGL",
            "train_period": [df["date"].min(), train_end],
            "test_period": [test["date"].min(), test["date"].max()],
            "failure_lines": {"train": len(train), "test": len(test)},
            "templates": int(len(self.parser.templates())),
            "signatures": {"after_bootstrap": len(self.sigs.sigs), "after_test": len(evaluator.sigs.sigs)},
            "classifier_test": score(test["label"], pred),
            "baseline_template_lookup_test": score(test["label"], base),
            "unseen_template_lines_test": {
                "n": int(unseen.sum()),
                "classifier": score(test["label"][unseen], pred[unseen]) if unseen.any() else None,
                "baseline": score(test["label"][unseen], base[unseen]) if unseen.any() else None,
            },
            "novel_class_detection_test": {
                "novel_class_lines": int(novel.sum()),
                "novel_classes": sorted(set(test["label"][novel])),
                "recall_flagged_as_new_signature": float(in_new[novel].mean()) if novel.any() else None,
                "lines_in_new_signatures": int(in_new.sum()),
                "precision_new_signature_is_novel_class": float(novel[in_new].mean()) if in_new.any() else None,
            },
            "signature_quality": {
                "train_homogeneity": float(homogeneity_score(train["label"], train["signature"])),
                "test_homogeneity": float(homogeneity_score(test["label"], test["signature"])),
                "train_nmi": float(normalized_mutual_info_score(train["label"], train["signature"])),
                "line_weighted_purity_test": purity(test),
            },
            "daily_compression_test": {
                "days": int(len(daily)),
                "median_failure_lines_per_day": float(daily["lines"].median()),
                "median_signatures_per_day": float(daily["signatures"].median()),
                "total_failure_lines": int(daily["lines"].sum()),
                "total_signature_reviews": int(daily["signatures"].sum()),
            },
            "config": {"embed_backend": config.EMBED_BACKEND, "cluster_distance": config.CLUSTER_DISTANCE},
        }
        return metrics

    # ---- daily run ---------------------------------------------------------
    def run(self, df: pd.DataFrame, run_day: str | None = None, reports_dir: Path = config.REPORTS_DIR,
            email: bool = False) -> Path:
        t0 = time.time()
        run_day = run_day or df["date"].max()
        df = self.parse(df)
        created = self.attach_signatures(df)
        df["pred"], df["conf"] = self.clf.predict(df)
        templates = self.parser.templates()

        items = []
        for sid, g in df.groupby("signature"):
            sig = self.sigs.sigs[sid]
            category = Counter(g["pred"]).most_common(1)[0][0]
            alerts = g["pred"] != config.BENIGN_LABEL
            alert_cat = Counter(g["pred"][alerts]).most_common(1)[0][0] if alerts.any() else category
            component = Counter(g["component"]).most_common(1)[0][0]
            tids = Counter(g["template_id"]).most_common(5)
            items.append({
                "id": sid,
                "count": int(len(g)),
                "nodes": int(g["node"].nunique()),
                "component": component,
                "level": Counter(g["level"]).most_common(1)[0][0],
                "category": alert_cat,
                "alert_share": float(alerts.mean()),
                "confidence": float(g["conf"].mean()),
                "templates": [templates[t] for t, _ in tids],
                "samples": g["message"].drop_duplicates().head(3).tolist(),
                "first_seen": sig["first_seen"] or run_day,
                "bucket": buckets.bucket({**sig, "component": component}, run_day, sid in created,
                                         alert_cat, float(alerts.mean())),
            })
        self.sigs.record(df)
        for it in items:
            it["history"] = buckets.history_window(self.sigs.sigs[it["id"]], run_day)
        items.sort(key=lambda it: (BUCKET_ORDER[it["bucket"]], -it["count"]))
        backend = summarize(items, self.sigs.sigs)

        label_check = None
        if (df["label"] != config.BENIGN_LABEL).any():  # replayed BGL data carries real labels
            label_check = score(df["label"], df["pred"])
            label_check.pop("per_class")
        run = {
            "date": run_day,
            "failure_lines": int(len(df)),
            "signatures": len(items),
            "new_signatures": len(created),
            "bucket_counts": dict(Counter(it["bucket"] for it in items)),
            "summary_backend": backend,
            "label_check": label_check,
            "seconds": round(time.time() - t0, 1),
        }
        self.save()
        path = report.write(run, items, reports_dir)
        if email:
            report.send_email(path, run)
        return path


def purity(df: pd.DataFrame) -> float:
    """Share of lines whose label equals their signature's majority label."""
    majority = df.groupby("signature")["label"].agg(lambda s: s.value_counts().iloc[0]).sum()
    return float(majority / len(df))


def load_failures(path: Path = config.RAW_LOG) -> pd.DataFrame:
    cache = config.DATA_DIR / "bgl_failures.parquet"
    if path == config.RAW_LOG and cache.exists():
        return pd.read_parquet(cache)
    df = read_failures_file(path)
    if path == config.RAW_LOG:
        try:
            df.to_parquet(cache)
        except ImportError:
            pass
    return df


def write_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
