"""Bootstrap silver ground truth from existing events JSON (for pipeline testing only)."""

from __future__ import annotations

import argparse
import json
import os
import sys

from benchmark.ground_truth import GT_SCHEMA_VERSION, save_json


def events_to_gt(events_doc: dict, *, video: str, source: str) -> dict:
    events = events_doc.get("events") or events_doc
    if isinstance(events, dict):
        events = []
    gt_events = []
    for i, ev in enumerate(events, start=1):
        gt_events.append({
            "gt_event_id": str(ev.get("gt_event_id") or ev.get("event_id") or ev.get("behavior_event_id") or f"gt_{i:04d}"),
            "start_time": float(ev["start_time"]),
            "end_time": float(ev["end_time"]),
            "should_mask": ev.get("should_mask", True),
            "label": ev.get("label", "face_presence"),
        })
    duration = float(events_doc.get("duration_sec") or 0)
    if duration <= 0 and gt_events:
        duration = max(float(e["end_time"]) for e in gt_events)
    return {
        "schema_version": GT_SCHEMA_VERSION,
        "video": video,
        "duration_sec": duration,
        "provenance": {
            "type": "silver_bootstrap",
            "source_file": os.path.abspath(source),
            "warning": "Not human verified — for benchmark plumbing only",
        },
        "events": gt_events,
        "clips": [],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Bootstrap silver GT from events JSON")
    p.add_argument("--events", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.events, encoding="utf-8") as f:
        doc = json.load(f)
    gt = events_to_gt(doc, video=args.video, source=args.events)
    save_json(args.out, gt)
    print(f"[gt] {os.path.abspath(args.out)} events={len(gt['events'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
