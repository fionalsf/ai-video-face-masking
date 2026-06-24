#!/usr/bin/env python3
"""DEPRECATED — multi-scheme comparison only. Production uses Scheme C + event_merge.py."""

raise SystemExit(
    "compare_event_merge.py is deprecated. Production pipeline: Scheme C builder + event_merge.py"
)

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from statistics import mean
from typing import Any

from event_builder import (
    DEFAULT_HARD_SPLIT_GAP_SEC,
    FaceEvent,
    _build_chunks_legacy,
    _make_event,
    build_events,
    resolve_temporal_padding,
    should_merge_detections,
)
from event_quality import QUALITY_LOW, score_events

SCHEME_A = "A_conf_merge"
SCHEME_B = "B_time_only"
SCHEME_C = "C_time_hard_split"
REVIEW_SEC_PER_EVENT = 30.0


def build_events_time_only(
    detections: list[dict],
    gap_sec: float = 1.0,
    *,
    fps: float = 30.0,
    total_frames: int | None = None,
    detect_interval: int | None = None,
    frame_h: int = 1080,
    frame_w: int = 1920,
) -> list[FaceEvent]:
    """Scheme B: split when gap >= gap_sec; no confidence-based merge or singleton absorb."""
    from rules import suggest_rule_hints

    pre_pad, post_pad = resolve_temporal_padding(
        fps, detect_interval=detect_interval
    )
    by_track: dict[int, list[dict]] = {}
    for d in detections:
        by_track.setdefault(d["track_id"], []).append(d)
    for tid in by_track:
        by_track[tid].sort(key=lambda x: x["frame"])

    events: list[FaceEvent] = []
    evt_num = 0
    for track_id in sorted(by_track):
        chunks = _build_chunks_legacy(by_track[track_id], gap_sec)
        for chunk in chunks:
            evt_num += 1
            events.append(
                _make_event(
                    evt_num,
                    track_id,
                    chunk,
                    frame_h,
                    frame_w,
                    suggest_rule_hints,
                    fps=fps,
                    total_frames=total_frames,
                    pre_padding_sec=pre_pad,
                    post_padding_sec=post_pad,
                )
            )
    return events


def summarize_scheme(events: list[FaceEvent], *, fps: float) -> dict[str, Any]:
    quality_doc = score_events(events, fps=fps)
    durations = [float(e.end_time) - float(e.start_time) for e in events]
    single_frame = sum(1 for e in events if e.detection_count == 1)
    low_count = quality_doc["summary"]["low"]
    total = len(events)
    return {
        "event_total": total,
        "avg_duration_sec": round(float(mean(durations)), 3) if durations else 0.0,
        "single_frame_event_count": single_frame,
        "low_quality_event_count": low_count,
        "medium_quality_event_count": quality_doc["summary"]["medium"],
        "high_quality_event_count": quality_doc["summary"]["high"],
        "review_time_min_estimate": round(total * REVIEW_SEC_PER_EVENT / 60.0, 1),
        "review_time_sec_estimate": round(total * REVIEW_SEC_PER_EVENT, 0),
        "render_false_mask_count_manual": None,
        "render_false_mask_notes": "需人工播放 Render 对比后填写（overlay 或 preview 均可）",
    }


def load_context(output_dir: str) -> tuple[list[dict], dict]:
    tracked_path = os.path.join(output_dir, "tracked_detections.json")
    summary_path = os.path.join(output_dir, "detection_summary.json")
    if not os.path.isfile(tracked_path):
        raise FileNotFoundError(f"Missing {tracked_path}")
    with open(tracked_path, encoding="utf-8") as f:
        tracked = json.load(f)
    summary: dict = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    return tracked, summary


def build_comparison_report(
    output_dir: str,
    *,
    event_gap: float = 1.0,
) -> dict[str, Any]:
    from video_meta import get_video_meta

    output_dir = os.path.abspath(output_dir)
    tracked, summary = load_context(output_dir)
    fps = float(summary.get("fps") or 30.0)
    detect_interval = summary.get("detect_interval")
    video = summary.get("video", "")
    frame_h, frame_w, total_frames = 1080, 1920, None
    if video and os.path.isfile(video):
        meta = get_video_meta(video)
        frame_h, frame_w = meta["height"], meta["width"]
        total_frames = meta["frames"]

    common = dict(
        fps=fps,
        total_frames=total_frames,
        detect_interval=int(detect_interval) if detect_interval else None,
        frame_h=frame_h,
        frame_w=frame_w,
    )

    events_b = build_events_time_only(tracked, gap_sec=event_gap, **common)
    events_c = build_events(tracked, gap_sec=event_gap, **common)

    stats_b = summarize_scheme(events_b, fps=fps)
    stats_c = summarize_scheme(events_c, fps=fps)

    def delta_c_b(key: str) -> Any:
        vb, vc = stats_b[key], stats_c[key]
        if isinstance(vb, (int, float)) and isinstance(vc, (int, float)):
            return round(vc - vb, 3) if isinstance(vb, float) else vc - vb
        return None

    track21_c = [
        e for e in events_c if e.track_id == 21 and any(
            p["frame"] in (955, 1425, 1585) for p in e.trajectory
        )
    ]
    track21_b = [
        e for e in events_b if e.track_id == 21 and any(
            p["frame"] in (955, 1425, 1585) for p in e.trajectory
        )
    ]

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": output_dir,
        "video": os.path.basename(video) if video else "",
        "event_gap_sec": event_gap,
        "hard_split_gap_sec": DEFAULT_HARD_SPLIT_GAP_SEC,
        "production_scheme": SCHEME_C,
        "unchanged_modules": [
            "Detection",
            "Tracking",
            "Quality",
            "Review",
            "Timeline",
            "Render",
        ],
        "variable": "Event Builder merge strategy (Scheme C is production)",
        "schemes": {
            SCHEME_A: {
                "label": "历史：gap>2s 且双端 conf>0.75 可继续合并（已废弃）",
                "implementation": "见上一轮 comparison_report.json",
            },
            SCHEME_B: {
                "label": "实验：纯时间 gap>=1s 切分",
                "implementation": "_build_chunks_legacy() only",
            },
            SCHEME_C: {
                "label": "生产：时间连续性，gap>3s 硬切，禁止 conf 合并",
                "implementation": "build_events() + should_merge_detections(time-only)",
            },
        },
        "review_time_formula": f"{REVIEW_SEC_PER_EVENT}s per event (estimate)",
        SCHEME_B: stats_b,
        SCHEME_C: stats_c,
        "delta_C_minus_B": {
            "event_total": delta_c_b("event_total"),
            "avg_duration_sec": delta_c_b("avg_duration_sec"),
            "single_frame_event_count": delta_c_b("single_frame_event_count"),
            "low_quality_event_count": delta_c_b("low_quality_event_count"),
            "review_time_min_estimate": delta_c_b("review_time_min_estimate"),
        },
        "case_study_track21_evt_0006": {
            "scheme_B_time_only_1s": {
                "event_count_on_track": len(track21_b),
                "auto_split": len(track21_b) > 1,
                "events": [
                    {
                        "event_id": e.event_id,
                        "detection_count": e.detection_count,
                        "duration_sec": round(e.end_time - e.start_time, 3),
                        "frames": [p["frame"] for p in e.trajectory],
                    }
                    for e in track21_b
                ],
            },
            "scheme_C_production": {
                "event_count_on_track": len(track21_c),
                "auto_split": len(track21_c) > 1,
                "events": [
                    {
                        "event_id": e.event_id,
                        "detection_count": e.detection_count,
                        "duration_sec": round(e.end_time - e.start_time, 3),
                        "frames": [p["frame"] for p in e.trajectory],
                        "gaps_to_next_sec": None,
                    }
                    for e in track21_c
                ],
            },
        },
        "manual_observation_checklist": {
            "render_false_mask_count": "播放 Render 后统计误打码帧/段数，填入各 scheme 的 render_false_mask_count_manual",
            "suggested_clips": ["evt_0006 区间 31-54s overlay 预览"],
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Compare Event Builder merge schemes A vs B")
    p.add_argument(
        "--output-dir",
        default="output/detection/DJI_20260511100755_0008_D",
        help="Detection output dir with tracked_detections.json",
    )
    p.add_argument("--event-gap", type=float, default=1.0)
    p.add_argument(
        "--report",
        default=None,
        help="Output path (default: <output-dir>/comparison_report.json)",
    )
    args = p.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    report_path = args.report or os.path.join(out_dir, "comparison_report.json")

    report = build_comparison_report(out_dir, event_gap=args.event_gap)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    b, c = report[SCHEME_B], report[SCHEME_C]
    cs = report["case_study_track21_evt_0006"]["scheme_C_production"]
    print(f"[done] {report_path}")
    print(f"  C event_total={c['event_total']}  B={b['event_total']}")
    print(f"  track21 Scheme C: {cs['event_count_on_track']} events auto_split={cs['auto_split']}")
    for e in cs["events"]:
        print(f"    {e['event_id']} frames={e['frames']} duration={e['duration_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
