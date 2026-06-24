"""Run frozen stitching variant + behavior event builder."""

from __future__ import annotations

import json
import os
from typing import Any

from identity_behavior_builder import build_identity_behavior_events
from video_meta import get_video_meta

from benchmark.pipelines.stitching_v1_frozen import run_stitching_v1_frozen
from benchmark.pipelines.stitching_v2_frozen import run_stitching_v2_frozen


def load_tracked(output_dir: str) -> list[dict]:
    path = os.path.join(os.path.abspath(output_dir), "tracked_detections.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_variant_pipeline(
    variant: str,
    *,
    output_dir: str,
    video: str,
    write_artifacts: bool = False,
) -> dict[str, Any]:
    tracked = load_tracked(output_dir)
    meta = get_video_meta(video)
    fps = float(meta["fps"])

    if variant == "v1_greedy":
        stitch = run_stitching_v1_frozen(tracked, video=video)
        variant_out = os.path.join(output_dir, "benchmark", "v1_greedy") if write_artifacts else None
    elif variant == "v2_graph":
        variant_out = os.path.join(output_dir, "benchmark", "v2_graph") if write_artifacts else None
        stitch = run_stitching_v2_frozen(tracked, video=video, output_dir=variant_out)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    behavior = build_identity_behavior_events(
        tracked,
        stitch["clusters"],
        output_dir=variant_out,
        video=video,
        fps=fps,
        frame_h=meta["height"],
        frame_w=meta["width"],
        total_frames=meta["frames"],
    )

    doc = {
        "variant": variant,
        "stitching_layer": stitch.get("layer"),
        "identity_cluster_count": len(stitch["clusters"]),
        "linked_edge_count": stitch.get("linked_edge_count") or len(stitch.get("assigned_edges") or []),
        "behavior_event_count": behavior["behavior_event_count"],
        "events": behavior["behavior_events"],
        "stitch_stats": {
            "appearance_method": stitch.get("appearance_method"),
            "parameters_frozen": True,
        },
    }
    if write_artifacts and variant_out:
        os.makedirs(variant_out, exist_ok=True)
        with open(os.path.join(variant_out, "behavior_events.json"), "w", encoding="utf-8") as f:
            json.dump({
                "variant": variant,
                "video": video,
                "events": behavior["behavior_events"],
            }, f, ensure_ascii=False, indent=2)
    return doc
