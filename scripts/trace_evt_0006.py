#!/usr/bin/env python3
"""Trace Event Builder merge logic for evt_0006."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_builder import (
    HIGH_CONF_RECOVERY,
    _build_chunks_legacy,
    _build_chunks_merged,
    bbox_iou,
    build_events,
    should_merge_detections,
)
from video_meta import get_video_meta, sec_to_timecode

OD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
    "detection",
    "DJI_20260511100755_0008_D",
)
GAP = 1.0


def main() -> int:
    with open(os.path.join(OD, "tracked_detections.json"), encoding="utf-8") as f:
        tracked = json.load(f)
    with open(os.path.join(OD, "detection_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    fps = float(summary["fps"])
    meta = get_video_meta(summary["video"])
    events = build_events(
        tracked,
        gap_sec=GAP,
        frame_h=meta["height"],
        frame_w=meta["width"],
        fps=fps,
        total_frames=meta["frames"],
        detect_interval=summary.get("detect_interval"),
    )
    ev6 = next(e for e in events if e.event_id == "evt_0006")
    track21 = sorted([d for d in tracked if d["track_id"] == 21], key=lambda x: x["frame"])

    print("=== 1. evt_0006 three detections ===")
    for i, p in enumerate(ev6.trajectory, start=1):
        print(
            f"  det{i}: frame={p['frame']}  timestamp={p['t']:.3f}s "
            f"({sec_to_timecode(p['t'])})  track_id={ev6.track_id}  conf={p['conf']:.4f}"
        )

    print(f"\n=== track 21: {len(track21)} total detections on track ===")
    near = [d for d in track21 if 900 <= d["frame"] <= 1650]
    print(f"  in evt window (frame 900-1650): {len(near)} detections")

    print("\n=== legacy split (gap >= 1.0s => new chunk) ===")
    for i, c in enumerate(_build_chunks_legacy(track21, GAP), start=1):
        fs = [d["frame"] for d in c]
        print(f"  chunk{i}: {len(c)} dets, frames {min(fs)}-{max(fs)}")

    print("\n=== merged split (should_merge_detections) ===")
    for i, c in enumerate(_build_chunks_merged(track21, GAP), start=1):
        fs = [d["frame"] for d in c]
        print(f"  chunk{i}: {len(c)} dets, frames {min(fs)}-{max(fs)}")

    print("\n=== 3. step-by-step should_merge_detections ===")
    dets = ev6.trajectory
    for i in range(1, len(dets)):
        prev, curr = dets[i - 1], dets[i]
        g = curr["t"] - prev["t"]
        iou = bbox_iou(prev["bbox"], curr["bbox"])
        merge = should_merge_detections(prev, curr, GAP)
        print(f"\n  det{i} -> det{i + 1}:")
        print(f"    gap = {g:.3f}s")
        print(f"    prev_conf={prev['conf']:.4f}, curr_conf={curr['conf']:.4f}, iou={iou:.3f}")
        if g < GAP:
            print(f"    rule A: gap < {GAP}s => merge=True")
        elif g <= 2 * GAP:
            print(f"    rule B: gap <= {2 * GAP}s => merge=True (same-track dropout)")
        else:
            both = prev["conf"] > HIGH_CONF_RECOVERY and curr["conf"] > HIGH_CONF_RECOVERY
            print(f"    rule C: gap > {2 * GAP}s => merge only if both conf > {HIGH_CONF_RECOVERY}")
            print(f"    both high conf? {both} => merge={merge}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
