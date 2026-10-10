"""Turn raw failure messages into log templates with Drain (via drain3).

Messages are masked with regexes first (IPs, node locations, paths, hex, numbers)
and de-duplicated, so Drain only sees each distinct masked string once. That keeps
parsing ~1M lines fast while producing the same template ids per line.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

MASKS = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\bR\d{2}-M\d(?:-[A-Z0-9]+(?::J\d{2}-U\d{2})?)*"), "<LOC>"),
    (re.compile(r"(?<![\w<])/[\w./+\-]+"), "<PATH>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<HEX>"),
    (re.compile(r"\.{3,}"), " "),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "<NUM>"),
    (re.compile(r"\s+"), " "),
]


def mask(message: str) -> str:
    for pattern, repl in MASKS:
        message = pattern.sub(repl, message)
    return message.strip()


class TemplateParser:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        cfg = TemplateMinerConfig()
        cfg.drain_sim_th = 0.5
        cfg.drain_depth = 4
        cfg.drain_extra_delimiters = ["=", ","]
        cfg.parametrize_numeric_tokens = True
        cfg.masking_instructions = []  # masking is done by `mask` above
        self.miner = TemplateMiner(config=cfg)
        if state_path.exists():
            with open(state_path, "rb") as fh:
                self.miner.drain = pickle.load(fh)

    @property
    def max_id(self) -> int:
        return max(self.miner.drain.id_to_cluster.keys(), default=0)

    def parse(self, messages: pd.Series) -> np.ndarray:
        """Return a Drain template id for every message (learning new templates online)."""
        masked = messages.map(mask)
        ids = {m: self.miner.add_log_message(m)["cluster_id"] for m in masked.unique()}
        return masked.map(ids).to_numpy()

    def templates(self) -> dict[int, str]:
        return {c.cluster_id: c.get_template() for c in self.miner.drain.clusters}

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "wb") as fh:
            pickle.dump(self.miner.drain, fh)
