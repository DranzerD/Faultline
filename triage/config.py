"""Central configuration: paths, thresholds and domain mappings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("TRIAGE_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
RAW_LOG = DATA_DIR / "raw" / "BGL.log"
STATE_DIR = Path(os.environ.get("TRIAGE_STATE_DIR", ROOT / "state"))
REPORTS_DIR = Path(os.environ.get("TRIAGE_REPORTS_DIR", ROOT / "reports"))

BGL_URL = "https://zenodo.org/records/8196385/files/BGL.zip?download=1"

# Severity levels treated as "failures" worth triaging. In BGL every real alert
# is FATAL or FAILURE, but ~640k FATAL/ERROR/SEVERE lines are labelled non-alert,
# so the classifier has to separate real alert categories from benign noise.
FAILURE_LEVELS = {"FATAL", "FAILURE", "SEVERE", "ERROR"}
BENIGN_LABEL = "BENIGN"

# Embedding backend: "sentence-transformers" (local, free) or "tfidf" (no download).
EMBED_BACKEND = os.environ.get("TRIAGE_EMBED_BACKEND", "sentence-transformers")
EMBED_MODEL = os.environ.get("TRIAGE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Templates whose embeddings are closer than this (cosine distance) form one signature.
CLUSTER_DISTANCE = float(os.environ.get("TRIAGE_CLUSTER_DISTANCE", "0.25"))
# A new template joins an existing signature if cosine similarity >= this.
ASSIGN_SIMILARITY = 1.0 - CLUSTER_DISTANCE

# LLM summaries via a local Ollama server (free). Falls back to rule-based text.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
LLM_ENABLED = os.environ.get("TRIAGE_LLM", "auto")  # auto | off
MAX_SUMMARIES = int(os.environ.get("TRIAGE_MAX_SUMMARIES", "15"))

# Alert categories (BGL label prefixes/names) that indicate infrastructure or
# environment problems rather than software bugs.
ENV_CATEGORY_PREFIXES = ("LINK", "MON", "MAS")
ENV_CATEGORIES = {"KERNPOW", "KERNNOETH", "KERNSOCK", "KERNMNTF", "KERNMNT", "APPUNAV", "KERNCON"}
ENV_COMPONENTS = {"HARDWARE", "LINKCARD", "MONITOR", "DISCOVERY", "CMCS", "BGLMASTER", "SERV_NET"}

# Flaky = a known signature that keeps coming back intermittently.
FLAKY_MIN_ACTIVE_DAYS = 7      # history span (first..last seen) must cover at least this
FLAKY_MIN_OCCURRENCES = 3      # seen on at least this many distinct days
FLAKY_MAX_PRESENCE = 0.5       # ...but on at most this fraction of days in its span
