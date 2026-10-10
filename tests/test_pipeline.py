"""Fast tests on synthetic BGL-format lines (TF-IDF embeddings, LLM off, no dataset needed)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from triage import buckets, config
from triage.ingest import parse_line, read_failures
from triage.parsing import mask


def bgl(label: str, day: str, node: str, component: str, level: str, msg: str, sec: int = 0) -> str:
    epoch = 1125532800 + sec
    d = day.replace("-", ".")
    return f"{label} {epoch} {d} {node} {day}-00.00.{sec:02d}.000000 {node} RAS {component} {level} {msg}"


TRAIN = (
    [bgl("KERNDTLB", "2005-09-01", f"R0{i % 3}-M0-N0-C:J12-U11", "KERNEL", "FATAL", "data TLB error interrupt", i) for i in range(30)]
    + [bgl("-", "2005-09-01", "R01-M1-N2", "KERNEL", "FATAL", f"instruction address: 0x{i:08x}", i) for i in range(30)]
    + [bgl("APPREAD", "2005-09-02", "R02-M1-N3", "APP", "FATAL",
           f"ciod: failed to read message prefix on control stream (CioStream socket to 172.16.96.{i}:3385{i % 9}", i) for i in range(30)]
    + [bgl("-", "2005-09-02", "R02-M1-N3", "APP", "FATAL",
           f"ciod: Error loading /home/user{i}/job.rts: invalid or missing program image, No such file or directory", i) for i in range(30)]
    + [bgl("-", "2005-09-02", "R03-M0-N1", "KERNEL", "INFO", "generating core.123", 1)]
)
NEW_DAY = (
    [bgl("KERNDTLB", "2005-09-03", "R00-M0-N0-C:J12-U11", "KERNEL", "FATAL", "data TLB error interrupt", i) for i in range(5)]
    + [bgl("KERNPAN", "2005-09-03", "R05-M0-N0", "KERNEL", "FATAL",
           "kernel panic: quantum flux capacitor overheated in rack fan tray", i) for i in range(4)]
)


def test_mask_replaces_variable_fields():
    m = mask("ciod: failed on socket to 172.16.96.116:33851 at 0x00004ed8 loading /p/gb1/x.rts node R02-M1-N0-C:J12-U11 count 42")
    assert m == "ciod: failed on socket to <IP> at <HEX> loading <PATH> node <LOC> count <NUM>"


def test_parse_line_keeps_failures_only():
    assert parse_line(TRAIN[0])[:2] == ("KERNDTLB", 1125532800)
    assert parse_line(TRAIN[30])[0] == config.BENIGN_LABEL
    assert parse_line(TRAIN[-1]) is None  # INFO is not a failure level
    assert parse_line("garbage") is None


def test_bucket_rules():
    sig = {"component": "KERNEL", "daily_counts": {"2005-09-01": 3, "2005-09-05": 1, "2005-09-12": 2}}
    assert buckets.bucket(sig, "2005-09-20", True, "KERNDTLB", 1.0) == "new_bug"
    assert buckets.bucket(sig, "2005-09-20", False, "BENIGN", 0.0) == "noise"
    assert buckets.bucket(sig, "2005-09-20", False, "LINKIAP", 1.0) == "environment"
    assert buckets.bucket(sig, "2005-09-20", False, "KERNDTLB", 1.0) == "flaky"  # 3 days seen over a 12-day span
    steady = {"component": "KERNEL", "daily_counts": {f"2005-09-{d:02d}": 1 for d in range(1, 13)}}
    assert buckets.bucket(steady, "2005-09-20", False, "KERNDTLB", 1.0) == "known_bug"
    assert buckets.history_window(sig, "2005-09-12", days=3) == [0, 0, 2]


@pytest.fixture
def triage_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBED_BACKEND", "tfidf")
    monkeypatch.setattr(config, "LLM_ENABLED", "off")
    from triage import embed
    monkeypatch.setattr(embed.Embedder.__init__, "__defaults__", ("tfidf",))
    return tmp_path


def test_end_to_end_bootstrap_and_run(triage_env):
    from triage.pipeline import Triage

    state, reports = triage_env / "state", triage_env / "reports"
    train = read_failures(TRAIN)
    metrics = Triage(state).bootstrap(pd.concat([train, read_failures(NEW_DAY[:5])]), "2005-09-02")
    assert metrics["failure_lines"] == {"train": 120, "test": 5}
    assert metrics["classifier_test"]["accuracy"] == 1.0

    path = Triage(state).run(read_failures(NEW_DAY), reports_dir=reports)
    assert path.exists() and (reports / "index.html").exists()
    data = json.loads((reports / "2005-09-03.json").read_text())
    by_bucket = {s["bucket"] for s in data["signatures"]}
    assert "new_bug" in by_bucket  # the never-seen kernel panic message
    assert data["run"]["failure_lines"] == 9
    assert all(s["summary_source"] == "rules" for s in data["signatures"])
    assert "<script" not in path.read_text(encoding="utf-8")
