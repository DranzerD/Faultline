"""Read BGL-format logs into a DataFrame of failure events.

BGL line layout (space separated):
    label epoch date node datetime node2 type component level message...
`label` is "-" for non-alert lines, otherwise an alert category such as KERNDTLB.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from . import config

COLUMNS = ["label", "epoch", "date", "node", "component", "level", "message"]


def download_bgl(dest: Path = config.RAW_LOG) -> Path:
    """Fetch the Loghub BGL dataset (57 MB zip, ~743 MB unzipped) from Zenodo."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(config.BGL_URL, timeout=600)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extract("BGL.log", dest.parent)
    return dest


def parse_line(line: str) -> tuple | None:
    parts = line.rstrip("\n").split(" ", 9)
    if len(parts) < 10:
        return None
    label, epoch, date, node, _dt, _node2, _typ, component, level, message = parts
    if level not in config.FAILURE_LEVELS:
        return None
    try:
        epoch_i = int(epoch)
    except ValueError:
        return None
    label = config.BENIGN_LABEL if label == "-" else label
    return (label, epoch_i, date.replace(".", "-"), node, component, level, message.strip())


def read_failures(lines: Iterable[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Keep failure-level lines, optionally restricted to dates in [start, end] (YYYY-MM-DD)."""
    rows = []
    for line in lines:
        rec = parse_line(line)
        if rec is None:
            continue
        if start and rec[2] < start:
            continue
        if end and rec[2] > end:
            continue
        rows.append(rec)
    return pd.DataFrame(rows, columns=COLUMNS)


def read_failures_file(path: Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return read_failures(fh, start, end)


def write_day(date: str, out: Path, raw: Path = config.RAW_LOG) -> int:
    """Copy all raw lines for one day (YYYY-MM-DD) into `out`, to simulate a new CI log drop."""
    token = " " + date.replace("-", ".") + " "
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(raw, encoding="utf-8", errors="replace") as src, open(out, "w", encoding="utf-8") as dst:
        for line in src:
            if token in line[:40]:
                dst.write(line)
                n += 1
    return n
