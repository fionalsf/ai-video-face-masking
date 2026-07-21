#!/usr/bin/env python3
"""Compare review-generation variants against a baseline output."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def group_boxes(rows: list[dict[str, Any]]) -> dict[int, list[list[float]]]:
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    for row in rows:
        box = row.get("bbox")
        if box and len(box) == 4:
            grouped[int(row["frame"])].append([float(x) for x in box])
    return grouped


def match_box_maps(
    baseline: dict[int, list[list[float]]],
    candidate: dict[int, list[list[float]]],
    frames: list[int],
    threshold: float,
) -> dict[str, float | int]:
    baseline_count = candidate_count = matches = 0
    ious: list[float] = []
    for frame in frames:
        base_boxes = baseline.get(frame, [])
        cand_boxes = candidate.get(frame, [])
        baseline_count += len(base_boxes)
        candidate_count += len(cand_boxes)
        available = set(range(len(cand_boxes)))
        for base_box in base_boxes:
            ranked = sorted(
                ((iou(base_box, cand_boxes[j]), j) for j in available),
                reverse=True,
            )
            if ranked and ranked[0][0] >= threshold:
                score, index = ranked[0]
                available.remove(index)
                matches += 1
                ious.append(score)
    return {
        "baseline_boxes": baseline_count,
        "candidate_boxes": candidate_count,
        "matched_boxes": matches,
        "baseline_recall": matches / baseline_count if baseline_count else 1.0,
        "candidate_match_rate": matches / candidate_count if candidate_count else 1.0,
        "mean_matched_iou": sum(ious) / len(ious) if ious else 0.0,
    }


def summarize(output_dir: str) -> dict[str, Any]:
    report = load_json(os.path.join(output_dir, "review_report.json"))
    timeline = load_json(os.path.join(output_dir, "mask_timeline.json"))
    behavior = load_json(os.path.join(output_dir, "behavior_events.json"))
    detections = load_json(os.path.join(output_dir, "tracked_detections.json"))
    return {
        "report": report,
        "timeline": timeline,
        "behavior": behavior,
        "detections": detections,
        "detection_boxes": group_boxes(detections),
        "mask_boxes": group_boxes(timeline.get("entries") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    baseline = summarize(os.path.abspath(args.baseline))
    total_frames = int(baseline["timeline"].get("total_frames") or 0)
    rows = []
    base_time = float(baseline["report"]["stage_times"]["total_pipeline_sec"])
    for candidate_path in args.candidate:
        candidate = summarize(os.path.abspath(candidate_path))
        runtime = candidate["report"]["runtime"]
        interval = int(runtime["interval"])
        base_interval = int(baseline["report"]["runtime"]["interval"])
        common_step = math.lcm(base_interval, interval)
        common_frames = list(range(0, total_frames, common_step))
        all_frames = list(range(total_frames))
        cand_time = float(candidate["report"]["stage_times"]["total_pipeline_sec"])
        rows.append({
            "candidate": os.path.abspath(candidate_path),
            "interval": interval,
            "imgsz": int(runtime["imgsz"]),
            "total_sec": cand_time,
            "speedup": base_time / cand_time,
            "time_reduction": 1.0 - cand_time / base_time,
            "tracked_detections": len(candidate["detections"]),
            "behavior_events": int(candidate["behavior"]["behavior_event_count"]),
            "timeline_proposals": int(candidate["timeline"]["proposal_count"]),
            "timeline_entries": int(candidate["timeline"]["entry_count"]),
            "common_frame_detection_match": match_box_maps(
                baseline["detection_boxes"], candidate["detection_boxes"], common_frames, args.iou
            ),
            "final_mask_match": match_box_maps(
                baseline["mask_boxes"], candidate["mask_boxes"], all_frames, args.iou
            ),
        })

    payload = {
        "baseline": {
            "path": os.path.abspath(args.baseline),
            "total_sec": base_time,
            "interval": int(baseline["report"]["runtime"]["interval"]),
            "imgsz": int(baseline["report"]["runtime"]["imgsz"]),
            "tracked_detections": len(baseline["detections"]),
            "behavior_events": int(baseline["behavior"]["behavior_event_count"]),
            "timeline_proposals": int(baseline["timeline"]["proposal_count"]),
            "timeline_entries": int(baseline["timeline"]["entry_count"]),
        },
        "iou_threshold": args.iou,
        "candidates": rows,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
