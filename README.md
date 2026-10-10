# Faultline

[![ci](https://github.com/DranzerD/Faultline/actions/workflows/ci.yml/badge.svg)](https://github.com/DranzerD/Faultline/actions/workflows/ci.yml)
[![triage](https://github.com/DranzerD/Faultline/actions/workflows/triage.yml/badge.svg)](https://github.com/DranzerD/Faultline/actions/workflows/triage.yml)

**[Live reports](https://dranzerd.github.io/Faultline/)** · [Sample report (2005-09-03)](https://dranzerd.github.io/Faultline/2005-09-03.html)

**LLM-powered failure triage.** Raw logs are parsed into templates (Drain), embedded and
clustered into recurring **failure signatures**, classified, bucketed as
**new bug / known bug / flaky / environment / benign noise**, summarised by a local LLM, and
published as a daily HTML report by a GitHub Actions pipeline whenever new logs arrive.

Everything is free: the dataset is public, embeddings use a local `all-MiniLM-L6-v2`, and
summaries come from a local [Ollama](https://ollama.com) model. If no LLM is available, the
pipeline falls back to rule-based summaries.

```
raw logs ─► failure lines ─► Drain templates ─► embeddings ─► signatures ─┬─► triage bucket ─► LLM summary ─► HTML / email / Pages
 (BGL)      (FATAL/ERROR/…)   (masked, deduped)  (MiniLM)    (agglomerative)└─► classifier (alert category or BENIGN)
```

## Results (measured, `metrics/metrics.json`)

Dataset: [Loghub](https://github.com/logpai/loghub) **BGL** (BlueGene/L, 4.75M lines, real per-line alert labels).
Failure lines are those with level FATAL, FAILURE, SEVERE or ERROR: **987,940 lines**, of which 348k are real
alerts in 41 categories and the rest are benign noise at failure severity.

**Time-based split:** train on 2005-06-03 → 2005-08-31 (691,497 lines), test on 2005-09-01 → 2006-01-04
(296,443 lines). Model settings were picked on a Jun–Jul / Aug validation split inside the training
period, so the test period was never used for tuning.

| Metric (test period) | Value |
|---|---|
| Classifier accuracy (41 alert classes + BENIGN) | **96.9%** (baseline template lookup: 87.2%) |
| Macro-F1 | **0.63** (baseline: 0.56) |
| Alert detection precision / recall | **1.00 / 0.92** |
| Accuracy on lines whose template never appeared in training (47k lines) | **82.4%** (baseline: 21.3%) |
| Lines from alert classes absent from training that are flagged as a *new signature* | **69%** of 9,061 (precision 58%) |
| Signature purity (line-weighted majority-label share) | **98.3%** (homogeneity 0.93) |
| Templates → signatures | 191 Drain templates → 130 signatures (169 after the test period) |
| Daily review load (median over 120 test days) | **1,202 failure lines → 28 signatures** |

Known limitation: the 0-F1 classes (KERNREC, APPTO, APPOUT, KERNSERV) have **zero** training examples.
The supervised classifier can't label them, and the new-signature detector is what surfaces them.

The triage buckets come from transparent rules (`triage/buckets.py`) over signature history. BGL has no
ground truth for "flaky" or "known bug", so the buckets are **not** part of the accuracy figures above.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
python -m triage download                           # 57 MB zip from Zenodo -> data/raw/BGL.log
python -m triage bootstrap                          # parse, cluster, train, evaluate -> metrics/metrics.json
python -m triage replay --start 2005-09-01 --days 7 # simulate a week of daily runs -> reports/
python -m triage run logs/samples/bgl-2005-09-09.log
python -m pytest -q
```

LLM summaries: install Ollama, then run `ollama pull gemma3:4b`, or set `OLLAMA_MODEL` to any model you
have pulled. Set `TRIAGE_LLM=off` to force rule-based summaries. Summaries are cached per signature and
only regenerated when the signature's templates change.

## CI/CD

* **`triage.yml`**: runs on push to `logs/incoming/**`, daily at 06:00 UTC, or manually. It triages every
  new `*.log`, moves it to `logs/processed/`, commits the updated `state/` and `reports/`, uploads the
  reports as an artifact, and deploys them to GitHub Pages.
  * Optional: repo variable `USE_OLLAMA=true` installs Ollama in the runner for LLM summaries
    (`OLLAMA_MODEL`, default `qwen2.5:1.5b`).
  * Optional email: secrets `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `REPORT_TO`.
* **`ci.yml`**: runs the tests on pushes and PRs.
* **`bootstrap.yml`**: manual. Downloads BGL, retrains from scratch, and commits the new state and metrics.

To try it, copy a sample drop into `logs/incoming/` and push:
`cp logs/samples/bgl-2005-09-09.log logs/incoming/`.
For Pages, set the repo's Settings → Pages → Source to **GitHub Actions**.

## Layout

| Path | What |
|---|---|
| `triage/ingest.py` | BGL parsing, failure filtering, dataset download |
| `triage/parsing.py` | Regex masking + Drain3 template mining (dedup before mining keeps ~1M lines fast) |
| `triage/embed.py` | Local sentence-transformers embeddings (TF-IDF fallback) |
| `triage/signatures.py` | Agglomerative clustering into signatures, online assignment, daily history |
| `triage/classify.py` | TF-IDF + logistic regression classifier, baseline, metrics |
| `triage/buckets.py` | new / known / flaky / environment / noise rules |
| `triage/summarize.py` | Ollama summaries with rule-based fallback and caching |
| `triage/report.py` | HTML report, index page, SMTP email |
| `state/` | Persisted Drain tree, signatures + history, classifier (≈2 MB, committed so CI can resume) |
