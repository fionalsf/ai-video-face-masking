"""Backward-compatible wrapper; core logic lives in identity_stitching + identity_behavior_builder."""

from __future__ import annotations

from typing import Any

from identity_behavior_builder import build_identity_behavior_events
from identity_stitching import (
    DEFAULT_TEMPORAL_TAU_SEC,
    TARGET_EVENT_RANGE,
    run_identity_stitching,
)

TRACK_GRAPH_NAME = "track_graph.json"
LOWER_BOUND_ESTIMATE_NAME = "behavior_event_lower_bound_estimate.json"
DEFAULT_BEHAVIOR_GAP_SEC = 8.0
DEFAULT_APPEARANCE_MIN = 0.72


def run_track_stitching_analysis(
    output_dir: str,
    *,
    video: str | None = None,
    max_gap_sec: float = DEFAULT_TEMPORAL_TAU_SEC,
    appearance_min: float = DEFAULT_APPEARANCE_MIN,
    behavior_gap_sec: float = DEFAULT_BEHAVIOR_GAP_SEC,
) -> dict[str, Any]:
    import json
    import os
    import time

    output_dir = os.path.abspath(output_dir)
    tracked_path = os.path.join(output_dir, "tracked_detections.json")
    with open(tracked_path, encoding="utf-8") as f:
        tracked = json.load(f)

    stitch = run_identity_stitching(
        tracked,
        output_dir=output_dir,
        video=video,
        temporal_tau=max_gap_sec,
        appearance_min=appearance_min,
    )
    behavior = build_identity_behavior_events(
        tracked,
        stitch["clusters"],
        output_dir=output_dir,
        video=video,
        behavior_gap_sec=behavior_gap_sec,
    )

    target_lo, target_hi = TARGET_EVENT_RANGE
    count = behavior["behavior_event_count"]
    estimate_path = os.path.join(output_dir, LOWER_BOUND_ESTIMATE_NAME)
    with open(estimate_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Use identity_clusters.json + behavior_events.json (production pipeline)",
            "stitched_behavior_event_count": count,
            "identity_cluster_count": stitch["identity_cluster_count"],
            "target_event_range": list(TARGET_EVENT_RANGE),
            "target_achievable": target_lo <= count <= target_hi,
        }, f, ensure_ascii=False, indent=2)

    return {
        "track_graph_path": stitch.get("track_graph_path"),
        "estimate_path": estimate_path,
        "identity_clusters_path": stitch.get("identity_clusters_path"),
        "behavior_events_path": behavior.get("behavior_events_path"),
        "track_count": stitch["track_count"],
        "identity_cluster_count": stitch["identity_cluster_count"],
        "stitched_behavior_event_count": count,
        "track_level_behavior_event_count": count,
        "target_achievable": target_lo <= count <= target_hi,
        "linked_edge_count": stitch["linked_edge_count"],
    }


def main() -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Legacy wrapper for identity stitching pipeline")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--video", default=None)
    p.add_argument("--stitch-gap", type=float, default=DEFAULT_TEMPORAL_TAU_SEC)
    p.add_argument("--appearance-min", type=float, default=DEFAULT_APPEARANCE_MIN)
    p.add_argument("--behavior-gap", type=float, default=DEFAULT_BEHAVIOR_GAP_SEC)
    args = p.parse_args()

    stats = run_track_stitching_analysis(
        args.output_dir,
        video=args.video,
        max_gap_sec=args.stitch_gap,
        appearance_min=args.appearance_min,
        behavior_gap_sec=args.behavior_gap,
    )
    print(f"[clusters] {stats.get('identity_clusters_path')}")
    print(f"[behavior] {stats.get('behavior_events_path')}")
    print(
        f"  tracks={stats['track_count']} -> identities={stats['identity_cluster_count']}"
        f" | behavior={stats['stitched_behavior_event_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
