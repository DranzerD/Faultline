"""Embed log templates as L2-normalised vectors.

Default backend is a small local sentence-transformers model (free, ~90 MB, CPU-friendly).
The TF-IDF backend needs no download and is useful for quick CI runs.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np

from . import config


def readable(template: str) -> str:
    """Strip Drain/mask placeholders so the embedding focuses on the words."""
    text = re.sub(r"<[A-Z*]+>|<\*>", " ", template)
    return re.sub(r"\s+", " ", text).strip() or template


class Embedder:
    def __init__(self, state_dir: Path, backend: str = config.EMBED_BACKEND):
        self.backend = backend
        self.path = state_dir / "tfidf.joblib"
        self._model = None

    def _load(self, fit_texts: list[str] | None = None):
        if self._model is not None:
            return self._model
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        elif self.path.exists():
            self._model = joblib.load(self.path)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            if not fit_texts:
                raise RuntimeError("TF-IDF embedder has no saved state; run bootstrap first")
            self._model = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(fit_texts)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model, self.path)
        return self._model

    def encode(self, templates: list[str], fit: bool = False) -> np.ndarray:
        texts = [readable(t) for t in templates]
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        model = self._load(texts if fit else None)
        if self.backend == "sentence-transformers":
            vecs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        else:
            vecs = model.transform(texts).toarray()
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1, norms)
        return np.asarray(vecs, dtype=np.float32)
