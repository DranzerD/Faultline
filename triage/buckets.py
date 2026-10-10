"""Assign each signature in a run to a triage bucket: new_bug, environment, flaky, known_bug.

These are transparent rules over signature history and the classifier's prediction.
BGL has no ground truth for these buckets, so they are not part of the accuracy metrics.
"""
from __future__ import annotations

from datetime import date, timedelta

from . import config

BUCKETS = ["new_bug", "environment", "flaky", "known_bug", "noise"]


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days + 1


def presence(sig: dict, before: str) -> tuple[int, int]:
    """(days seen, span in days) for history strictly before `before`."""
    days = sorted(d for d in sig["daily_counts"] if d < before)
    if not days:
        return 0, 0
    return len(days), _days(days[0], days[-1])


def is_env(category: str, component: str | None) -> bool:
    return (
        category in config.ENV_CATEGORIES
        or category.startswith(config.ENV_CATEGORY_PREFIXES)
        or (component or "") in config.ENV_COMPONENTS
    )


def bucket(sig: dict, run_day: str, is_new: bool, category: str, alert_share: float) -> str:
    if is_new:
        return "new_bug"
    if alert_share < 0.5:
        return "noise"
    if is_env(category, sig.get("component")):
        return "environment"
    seen, span = presence(sig, run_day)
    if (
        span >= config.FLAKY_MIN_ACTIVE_DAYS
        and seen >= config.FLAKY_MIN_OCCURRENCES
        and seen / span <= config.FLAKY_MAX_PRESENCE
    ):
        return "flaky"
    return "known_bug"


def history_window(sig: dict, end: str, days: int = 14) -> list[int]:
    d_end = date.fromisoformat(end)
    return [sig["daily_counts"].get((d_end - timedelta(days=i)).isoformat(), 0) for i in range(days - 1, -1, -1)]
