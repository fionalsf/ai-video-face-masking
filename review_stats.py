"""Review statistics: build and persist review_report.json + review_statistics.json."""

from __future__ import annotations

import json
import os
import time
from statistics import mean

from event_quality import (
    QUALITY_HIGH,
    QUALITY_LEVELS,
    QUALITY_LOW,
    QUALITY_MEDIUM,
    load_event_quality,
    quality_by_event_id,
)

STATUS_ACCEPTED = "accepted"
STATUS_ACCEPTED_FIRST_HALF = "accepted_first_half"
STATUS_ACCEPTED_SECOND_HALF = "accepted_second_half"
STATUS_REJECTED = "rejected"
STATUS_SKIPPED = "skipped"
ACCEPT_STATUSES = {STATUS_ACCEPTED, STATUS_ACCEPTED_FIRST_HALF, STATUS_ACCEPTED_SECOND_HALF}

REPORT_NAME = "review_report.json"
STATISTICS_NAME = "review_statistics.json"


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 1) if total > 0 else 0.0


def _avg(items: list[dict], key: str) -> float | None:
    vals = [e[key] for e in items if e.get(key) is not None]
    return round(float(mean(vals)), 4) if vals else None


def build_quality_gate_stats(
    events: list[dict],
    decisions: dict[str, str],
    quality_map: dict[str, dict],
) -> dict:
    counts = {QUALITY_HIGH: 0, QUALITY_MEDIUM: 0, QUALITY_LOW: 0}
    by_quality = {
        q: {"accepted": 0, "rejected": 0} for q in QUALITY_LEVELS
    }
    for ev in events:
        eid = ev["event_id"]
        q = quality_map.get(eid, {}).get("quality")
        if q in counts:
            counts[q] += 1
        decision = decisions.get(eid)
        if q in by_quality:
            if decision in ACCEPT_STATUSES:
                by_quality[q]["accepted"] += 1
            elif decision == STATUS_REJECTED:
                by_quality[q]["rejected"] += 1
    return {
        "counts": counts,
        "by_quality": by_quality,
    }


def build_review_report(
    video: str,
    events: list[dict],
    decisions: dict[str, str],
    *,
    quality_map: dict[str, dict] | None = None,
) -> dict:
    total = len(events)
    accepted_n = sum(1 for e in events if decisions.get(e["event_id"]) in ACCEPT_STATUSES)
    rejected_n = sum(1 for e in events if decisions.get(e["event_id"]) == STATUS_REJECTED)
    skipped_n = sum(1 for e in events if decisions.get(e["event_id"]) == STATUS_SKIPPED)
    remaining = total - accepted_n - rejected_n

    accepted = [e for e in events if decisions.get(e["event_id"]) in ACCEPT_STATUSES]
    rejected = [e for e in events if decisions.get(e["event_id"]) == STATUS_REJECTED]

    report = {
        "video": os.path.basename(video) if video else "",
        "video_path": os.path.abspath(video) if video else "",
        "total_events": total,
        "accepted": accepted_n,
        "rejected": rejected_n,
        "skipped": skipped_n,
        "remaining": remaining,
        "accept_rate": _pct(accepted_n, total),
        "reject_rate": _pct(rejected_n, total),
        "skip_rate": _pct(skipped_n, total),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "review_finished_at": (
            time.strftime("%Y-%m-%d %H:%M:%S") if remaining == 0 and total > 0 else None
        ),
        "analysis": {
            "accepted_avg_peak_confidence": _avg(accepted, "peak_confidence"),
            "rejected_avg_peak_confidence": _avg(rejected, "peak_confidence"),
            "accepted_avg_duration_sec": _avg(accepted, "duration_sec"),
            "rejected_avg_duration_sec": _avg(rejected, "duration_sec"),
            "accepted_avg_frame_count": _avg(accepted, "frame_count"),
            "rejected_avg_frame_count": _avg(rejected, "frame_count"),
        },
    }
    if quality_map is not None:
        report["quality_gate"] = build_quality_gate_stats(events, decisions, quality_map)
    return report


def save_review_report(output_dir: str, report: dict) -> str:
    path = os.path.join(output_dir, REPORT_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def save_review_statistics(output_dir: str, report: dict) -> str:
    path = os.path.join(output_dir, STATISTICS_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def load_review_report(output_dir: str) -> dict | None:
    path = os.path.join(output_dir, REPORT_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def update_review_report(output_dir: str, video: str, events: list[dict], decisions: dict[str, str]) -> dict:
    quality_map = quality_by_event_id(load_event_quality(output_dir))
    report = build_review_report(video, events, decisions, quality_map=quality_map)
    save_review_report(output_dir, report)
    save_review_statistics(output_dir, report)
    return report
