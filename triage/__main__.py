"""CLI: python -m triage {download,bootstrap,run,replay,extract-day}."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="faultline", description="Faultline: LLM-assisted log failure triage")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="fetch the Loghub BGL dataset")

    b = sub.add_parser("bootstrap", help="parse, cluster, train on early data and evaluate on later data")
    b.add_argument("--train-end", default="2005-08-31", help="last day (YYYY-MM-DD) of the training period")
    b.add_argument("--metrics", type=Path, default=config.ROOT / "metrics" / "metrics.json")

    r = sub.add_parser("run", help="triage one BGL-format log file and write a report")
    r.add_argument("logfile", type=Path)
    r.add_argument("--date", help="report date (defaults to the latest date in the file)")
    r.add_argument("--email", action="store_true", help="email the report (SMTP_* env vars)")

    rp = sub.add_parser("replay", help="run day by day over raw BGL dates, simulating daily CI drops")
    rp.add_argument("--start", required=True)
    rp.add_argument("--days", type=int, default=7)

    pi = sub.add_parser("process-incoming", help="triage every *.log in a folder, then move it to processed/")
    pi.add_argument("--incoming", type=Path, default=config.ROOT / "logs" / "incoming")
    pi.add_argument("--processed", type=Path, default=config.ROOT / "logs" / "processed")
    pi.add_argument("--email", action="store_true")

    e = sub.add_parser("extract-day", help="write one day of raw BGL to a file (a fake 'new logs' drop)")
    e.add_argument("date")
    e.add_argument("out", type=Path)

    args = ap.parse_args(argv)

    if args.cmd == "download":
        from .ingest import download_bgl
        print(download_bgl())
    elif args.cmd == "bootstrap":
        from .pipeline import Triage, load_failures, write_metrics
        df = load_failures()
        metrics = Triage().bootstrap(df, args.train_end)
        write_metrics(metrics, args.metrics)
        print_metrics(metrics)
    elif args.cmd == "run":
        from .ingest import read_failures_file
        from .pipeline import Triage
        df = read_failures_file(args.logfile)
        if df.empty:
            print("no failure lines in", args.logfile)
            return
        print(Triage().run(df, args.date, email=args.email))
    elif args.cmd == "replay":
        from datetime import date, timedelta
        from .pipeline import Triage, load_failures
        df = load_failures()
        start = date.fromisoformat(args.start)
        for i in range(args.days):
            day = (start + timedelta(days=i)).isoformat()
            part = df[df["date"] == day]
            if part.empty:
                print(day, "no failures")
                continue
            print(day, Triage().run(part, day))
    elif args.cmd == "process-incoming":
        from .ingest import read_failures_file
        from .pipeline import Triage
        files = sorted(args.incoming.glob("*.log"))
        if not files:
            print("nothing to process in", args.incoming)
        args.processed.mkdir(parents=True, exist_ok=True)
        for f in files:
            df = read_failures_file(f)
            print(f.name, Triage().run(df, email=args.email) if not df.empty else "no failure lines")
            f.replace(args.processed / f.name)
    elif args.cmd == "extract-day":
        from .ingest import write_day
        print(write_day(args.date, args.out), "lines ->", args.out)


def print_metrics(m: dict) -> None:
    c, bl = m["classifier_test"], m["baseline_template_lookup_test"]
    u = m["unseen_template_lines_test"]
    d = m["daily_compression_test"]
    print(f"\nTrain {m['train_period']}  test {m['test_period']}  "
          f"lines {m['failure_lines']['train']:,}/{m['failure_lines']['test']:,}")
    print(f"Templates {m['templates']}  signatures {m['signatures']}")
    print(f"Classifier  acc {c['accuracy']:.4f}  macro-F1 {c['macro_f1']:.3f}  alert P/R "
          f"{c['alert_detection']['precision']:.3f}/{c['alert_detection']['recall']:.3f}")
    print(f"Baseline    acc {bl['accuracy']:.4f}  macro-F1 {bl['macro_f1']:.3f}")
    if u["n"]:
        print(f"Unseen-template lines {u['n']:,}: classifier acc {u['classifier']['accuracy']:.3f} "
              f"vs baseline {u['baseline']['accuracy']:.3f}")
    n = m["novel_class_detection_test"]
    print(f"Novel alert classes ({n['novel_class_lines']:,} lines): {n['recall_flagged_as_new_signature']:.3f} "
          f"land in new signatures; precision {n['precision_new_signature_is_novel_class']:.3f}")
    print(f"Signature quality {m['signature_quality']}")
    print(f"Daily: median {d['median_failure_lines_per_day']:,.0f} lines -> "
          f"{d['median_signatures_per_day']:.0f} signatures over {d['days']} days")


if __name__ == "__main__":
    main()
