#!/usr/bin/env python3
"""Build v2 mask-timeline review pack from an existing pipeline output."""

from __future__ import annotations

import argparse
import json
import os
import sys

from mask_timeline import (
    build_mask_timeline,
    export_mask_review_pack,
    save_mask_timeline,
    write_mask_confirmed_template,
)
from video_meta import get_video_meta


def parse_args():
    p = argparse.ArgumentParser(description="Export v2 mask proposal review pack from face_events.json")
    p.add_argument("--output-dir", required=True, help="Pipeline output directory containing face_events.json")
    p.add_argument("--expand", type=float, default=None, help="Preview mosaic expansion ratio")
    return p.parse_args()


def load_runtime(out_dir: str) -> dict:
    path = os.path.join(out_dir, "review_report.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8-sig") as f:
        report = json.load(f)
    return report.get("runtime") or {}


def main() -> int:
    args = parse_args()
    out_dir = os.path.abspath(args.output_dir)
    face_events_path = os.path.join(out_dir, "face_events.json")
    if not os.path.isfile(face_events_path):
        print(f"[error] face_events.json not found: {face_events_path}", file=sys.stderr)
        return 1

    with open(face_events_path, encoding="utf-8-sig") as f:
        face_data = json.load(f)

    video_path = face_data["video"]
    meta = get_video_meta(video_path)
    runtime = load_runtime(out_dir)
    timeline = build_mask_timeline(video_path, face_data.get("events") or [], meta, runtime)
    timeline_path = save_mask_timeline(out_dir, timeline)

    review_dir = os.path.join(out_dir, "review")
    pending_path = export_mask_review_pack(
        timeline,
        review_dir,
        expand=float(args.expand if args.expand is not None else runtime.get("expand") or 0.20),
    )
    with open(pending_path, encoding="utf-8-sig") as f:
        pending = json.load(f)
    decisions_path = write_mask_confirmed_template(review_dir, pending.get("events") or [])

    review_pending = len(pending.get("events") or [])
    print(f"[Timeline] {timeline['proposal_count']} proposals -> {timeline_path}")
    print(f"[Review] {review_pending} mask proposals pending -> {pending_path}")
    print(f"[Review] decisions -> {decisions_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
