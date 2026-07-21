#!/usr/bin/env python3
"""Export a review pack from an existing pipeline output directory."""

from __future__ import annotations

import argparse
import json
import os
import sys

from event_builder import FaceEvent, TIER_REVIEW
from export_events import export_review_pack, write_confirmed_events_template


def parse_args():
    p = argparse.ArgumentParser(description="Export review thumbnails from face_events.json")
    p.add_argument("--output-dir", required=True, help="Pipeline output directory containing face_events.json")
    return p.parse_args()


def face_event_from_dict(data: dict) -> FaceEvent:
    return FaceEvent(
        event_id=str(data["event_id"]),
        track_id=int(data.get("track_id", -1)),
        tier=str(data.get("tier", TIER_REVIEW)),
        start_time=float(data.get("start_time", 0.0)),
        end_time=float(data.get("end_time", 0.0)),
        start_frame=int(data.get("start_frame", 0)),
        end_frame=int(data.get("end_frame", 0)),
        avg_confidence=float(data.get("avg_confidence", 0.0)),
        peak_confidence=float(data.get("peak_confidence", 0.0)),
        detection_count=int(data.get("detection_count", len(data.get("trajectory") or []))),
        trajectory=list(data.get("trajectory") or []),
        rule_hints=list(data.get("rule_hints") or []),
    )


def main() -> int:
    out_dir = os.path.abspath(args.output_dir)
    face_events_path = os.path.join(out_dir, "face_events.json")
    if not os.path.isfile(face_events_path):
        print(f"[error] face_events.json not found: {face_events_path}", file=sys.stderr)
        return 1

    with open(face_events_path, encoding="utf-8-sig") as f:
        data = json.load(f)

    video_path = data["video"]
    review_dir = os.path.join(out_dir, "review")
    review_events = [
        face_event_from_dict(ev)
        for ev in data.get("events", [])
        if ev.get("tier") == TIER_REVIEW
    ]

    pending_path = export_review_pack(video_path, review_dir, review_events)
    write_confirmed_events_template(review_dir, video_path)
    print(f"[Review] {len(review_events)} events pending -> {pending_path}")
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main())
