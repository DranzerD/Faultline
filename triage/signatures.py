"""Group log templates into recurring failure signatures and track their history.

A signature is a cluster of semantically similar Drain templates. Bootstrap clusters
all known templates with agglomerative clustering on embedding cosine distance; later
runs attach each unseen template to the nearest signature centroid, or open a new
signature when nothing is close enough (a candidate new bug).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from . import config


def cluster_vectors(vecs: np.ndarray, distance: float = config.CLUSTER_DISTANCE) -> np.ndarray:
    if len(vecs) < 2:
        return np.zeros(len(vecs), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=distance
    )
    return model.fit_predict(vecs)


class SignatureStore:
    def __init__(self, state_dir: Path):
        self.json_path = state_dir / "signatures.json"
        self.vec_path = state_dir / "template_vectors.npz"
        self.sigs: dict[str, dict] = {}
        self.template_to_sig: dict[int, str] = {}
        self.next_id = 1
        self.vectors: dict[int, np.ndarray] = {}
        if self.json_path.exists():
            data = json.loads(self.json_path.read_text())
            self.sigs = data["signatures"]
            self.template_to_sig = {int(k): v for k, v in data["template_to_sig"].items()}
            self.next_id = data["next_id"]
            npz = np.load(self.vec_path)
            self.vectors = dict(zip(npz["ids"].tolist(), npz["vecs"]))

    # ---- persistence -------------------------------------------------------
    def save(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self.next_id,
            "template_to_sig": {str(k): v for k, v in self.template_to_sig.items()},
            "signatures": self.sigs,
        }
        self.json_path.write_text(json.dumps(data, indent=1))
        ids = np.array(list(self.vectors.keys()), dtype=np.int64)
        vecs = np.stack(list(self.vectors.values())) if self.vectors else np.zeros((0, 1))
        np.savez_compressed(self.vec_path, ids=ids, vecs=vecs)

    # ---- building ----------------------------------------------------------
    def _new_sig(self, templates: list[int]) -> str:
        sid = f"SIG-{self.next_id:04d}"
        self.next_id += 1
        self.sigs[sid] = {"id": sid, "templates": templates, "first_seen": None, "last_seen": None,
                          "daily_counts": {}, "component": None, "level": None, "summary": None}
        for t in templates:
            self.template_to_sig[t] = sid
        return sid

    def bootstrap(self, template_ids: list[int], vecs: np.ndarray, distance: float = config.CLUSTER_DISTANCE) -> None:
        labels = cluster_vectors(vecs, distance)
        groups: dict[int, list[int]] = {}
        for tid, lab in zip(template_ids, labels):
            groups.setdefault(int(lab), []).append(int(tid))
        for tids in sorted(groups.values(), key=lambda g: min(g)):
            self._new_sig(tids)
        self.vectors.update({int(t): v for t, v in zip(template_ids, vecs)})

    def _centroids(self) -> tuple[list[str], np.ndarray]:
        sids, cents = [], []
        for sid, sig in self.sigs.items():
            vs = [self.vectors[t] for t in sig["templates"] if t in self.vectors]
            if vs:
                c = np.mean(vs, axis=0)
                sids.append(sid)
                cents.append(c / (np.linalg.norm(c) or 1.0))
        return sids, np.array(cents)

    def assign_new(self, template_ids: list[int], vecs: np.ndarray) -> list[str]:
        """Attach unseen templates to the closest signature or open new ones. Returns new signature ids."""
        created: list[str] = []
        sids, cents = self._centroids()
        for tid, v in zip(template_ids, vecs):
            tid = int(tid)
            self.vectors[tid] = v
            best, sim = None, -1.0
            if len(cents):
                sims = cents @ v
                i = int(np.argmax(sims))
                best, sim = sids[i], float(sims[i])
            if best is not None and sim >= config.ASSIGN_SIMILARITY:
                self.sigs[best]["templates"].append(tid)
                self.template_to_sig[tid] = best
            else:
                sid = self._new_sig([tid])
                created.append(sid)
                sids.append(sid)
                cents = np.vstack([cents, v[None, :]]) if len(cents) else v[None, :]
        return created

    # ---- history -----------------------------------------------------------
    def record(self, df: pd.DataFrame) -> None:
        """Add per-day occurrence counts for events in df (needs columns date, signature, component, level)."""
        for (sid, day), n in df.groupby(["signature", "date"]).size().items():
            sig = self.sigs[sid]
            sig["daily_counts"][day] = sig["daily_counts"].get(day, 0) + int(n)
            sig["first_seen"] = min(filter(None, [sig["first_seen"], day]))
            sig["last_seen"] = max(filter(None, [sig["last_seen"], day]))
        for sid, g in df.groupby("signature"):
            self.sigs[sid]["component"] = Counter(g["component"]).most_common(1)[0][0]
            self.sigs[sid]["level"] = Counter(g["level"]).most_common(1)[0][0]
