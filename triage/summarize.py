"""Root-cause summaries per signature.

Uses a local Ollama model when one is reachable (free, nothing leaves the machine),
otherwise falls back to a rule-based summary so the pipeline always completes.
Summaries are cached on the signature and only regenerated when its templates change.
"""
from __future__ import annotations

import hashlib
import re

import requests

from . import config

HINTS = [
    (r"parity|ecc|ddr|correctable|cache.*error|memory", "memory or cache hardware fault on the node"),
    (r"tlb|storage interrupt|machine check|program interrupt|alignment", "kernel-level exception, usually a faulty node or a bad application memory access"),
    (r"ciod|socket|stream|connection|link|torus|network|ethernet", "I/O or interconnect communication failure between compute and I/O nodes"),
    (r"loading|program image|no such file|exec format", "job launch failure: missing or invalid executable path"),
    (r"mount|nfs|lustre|filesystem", "filesystem mount or availability problem"),
    (r"power|voltage|temperature|fan|clock", "power, cooling or clock hardware alarm"),
    (r"assert|terminated|panic|microloader", "software assertion or kernel termination; inspect the job and firmware level"),
    (r"node card|assembly|service|idoproxy|mmcs", "control-system / service-node issue rather than the user job"),
]


def _key(templates: list[str]) -> str:
    return hashlib.sha1("\n".join(sorted(templates)).encode()).hexdigest()[:12]


def rule_summary(item: dict) -> str:
    text = " ".join(item["templates"]).lower()
    cause = next((h for p, h in HINTS if re.search(p, text)), "unrecognised failure pattern; needs manual review")
    return (
        f"Likely cause: {cause}. Seen {item['count']:,} times today on {item['nodes']} node(s) "
        f"(component {item['component']}, level {item['level']}); classifier says {item['category']}."
    )


def _ollama_available() -> bool:
    """True when the Ollama server is up and can actually load the configured model."""
    if config.LLM_ENABLED == "off":
        return False
    try:
        resp = requests.post(f"{config.OLLAMA_URL}/api/show", json={"model": config.OLLAMA_MODEL}, timeout=5)
    except requests.RequestException:
        print(f"LLM summaries off: no Ollama server at {config.OLLAMA_URL}; using rule-based summaries")
        return False
    if resp.status_code != 200:
        print(f"LLM summaries off: Ollama cannot load {config.OLLAMA_MODEL} ({resp.text.strip()[:160]}); "
              f"try `ollama pull {config.OLLAMA_MODEL}`. Using rule-based summaries")
        return False
    return True


def llm_summary(item: dict) -> str:
    prompt = (
        "You are an SRE triaging failures from a BlueGene/L supercomputer test/production log.\n"
        "Write a root-cause summary in at most 3 short sentences: the likely cause, what is affected, "
        "and one concrete next debugging step. No preamble, no markdown.\n\n"
        f"Signature: {item['id']}  bucket: {item['bucket']}  predicted category: {item['category']}\n"
        f"Component: {item['component']}  level: {item['level']}\n"
        f"Occurrences today: {item['count']}  distinct nodes: {item['nodes']}  first seen: {item['first_seen']}\n"
        "Templates:\n" + "\n".join(f"- {t}" for t in item["templates"][:5]) + "\n"
        "Sample lines:\n" + "\n".join(f"- {s}" for s in item["samples"][:3])
    )
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/generate",
        json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.2, "num_predict": 160}},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def summarize(items: list[dict], sigs: dict[str, dict]) -> str:
    """Fill item['summary'] / item['summary_source'] in place. Returns the backend used."""
    use_llm = _ollama_available()
    backend = f"ollama:{config.OLLAMA_MODEL}" if use_llm else "rules"
    for i, item in enumerate(items):
        sig = sigs[item["id"]]
        key = _key(item["templates"])
        cached = sig.get("summary")
        if cached and cached["key"] == key and (cached["source"] != "rules" or not use_llm):
            item["summary"], item["summary_source"] = cached["text"], cached["source"]
            continue
        text, source = rule_summary(item), "rules"
        if use_llm and i < config.MAX_SUMMARIES:
            try:
                text, source = llm_summary(item), backend
            except (requests.RequestException, KeyError, ValueError) as exc:
                print(f"LLM summary failed for {item['id']}: {exc}")
        item["summary"], item["summary_source"] = text, source
        if source != "rules":
            sig["summary"] = {"key": key, "text": text, "source": source}
    return backend
