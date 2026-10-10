"""Render the per-run HTML report, keep a JSON copy and an index page, optionally email it."""
from __future__ import annotations

import json
import os
import smtplib
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from html import escape
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

LABELS = {"new_bug": "New bug", "known_bug": "Known bug", "flaky": "Flaky",
          "environment": "Environment", "noise": "Benign noise"}

_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                   autoescape=select_autoescape(["html", "j2"]))


def write(run: dict, items: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    end = date.fromisoformat(run["date"])
    days = [(end - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]
    html = _env.get_template("report.html.j2").render(
        run=run, items=items, labels=LABELS, days=days,
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    path = out_dir / f"{run['date']}.html"
    path.write_text(html, encoding="utf-8")
    (out_dir / f"{run['date']}.json").write_text(json.dumps({"run": run, "signatures": items}, indent=1))
    write_index(out_dir)
    return path


def write_index(out_dir: Path) -> None:
    runs = []
    for p in sorted(out_dir.glob("*.json"), reverse=True):
        try:
            runs.append(json.loads(p.read_text())["run"])
        except (KeyError, ValueError):
            continue
    rows = "\n".join(
        f'<tr><td><a href="{escape(r["date"])}.html">{escape(r["date"])}</a></td>'
        f'<td>{r["failure_lines"]:,}</td><td>{r["signatures"]}</td><td>{r["new_signatures"]}</td></tr>'
        for r in runs
    )
    (out_dir / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Faultline Reports</title>
<style>:root{{color-scheme:light dark;--s:#fcfcfb;--t:#0b0b0b;--b:#e4e3df}}
@media (prefers-color-scheme:dark){{:root{{--s:#1a1a19;--t:#fff;--b:#383835}}}}
body{{margin:0;background:var(--s);color:var(--t);font:14px/1.5 system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:24px 16px}}table{{width:100%;border-collapse:collapse}}
td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--b);font-variant-numeric:tabular-nums}}</style>
</head><body><main><h1>Faultline reports</h1><table><thead><tr><th>Date</th><th>Failure lines</th>
<th>Signatures</th><th>New</th></tr></thead><tbody>{rows}</tbody></table></main></body></html>""", encoding="utf-8")


def send_email(report_path: Path, run: dict) -> None:
    """Email the report using SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / REPORT_TO env vars."""
    host, to = os.environ.get("SMTP_HOST"), os.environ.get("REPORT_TO")
    if not host or not to:
        print("email skipped: SMTP_HOST and REPORT_TO not set")
        return
    msg = EmailMessage()
    msg["Subject"] = (f"[Faultline] {run['date']}: {run['new_signatures']} new, "
                      f"{run['signatures']} signatures, {run['failure_lines']:,} failures")
    msg["From"] = os.environ.get("SMTP_USER", "triage@localhost")
    msg["To"] = to
    msg.set_content("This report is HTML; open it in an HTML-capable mail client.")
    msg.add_alternative(report_path.read_text(encoding="utf-8"), subtype="html")
    with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587"))) as smtp:
        smtp.starttls()
        if os.environ.get("SMTP_USER"):
            smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)
