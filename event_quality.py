"""Event quality scoring for timeline interpolation reliability analysis."""

from __future__ import annotations

import json
import os
import time
from typing import Any

QUALITY_HIGH = "HIGH"
QUALITY_MEDIUM = "MEDIUM"
QUALITY_LOW = "LOW"
QUALITY_LEVELS = (QUALITY_HIGH, QUALITY_MEDIUM, QUALITY_LOW)

LOW_QUALITY_WARNING = "检测点过少，跨度过长，插值可靠性低。"

EVENT_QUALITY_NAME = "event_quality.json"


def _detection_gaps_sec(trajectory: list[dict]) -> tuple[float, float]:
    if len(trajectory) < 2:
        return 0.0, 0.0
    times = sorted(float(p["t"]) for p in trajectory)
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    return max(gaps), sum(gaps) / len(gaps)


def _longest_interpolation_gap_sec(
    trajectory: list[dict],
    start_time: float,
    end_time: float,
) -> float:
    """Longest continuous interval within event bounds with no detection."""
    duration = float(end_time) - float(start_time)
    if duration <= 0:
        return 0.0
    if not trajectory:
        return round(duration, 3)
    times = sorted(float(p["t"]) for p in trajectory)
    gaps = [max(0.0, times[0] - start_time)]
    gaps.extend(times[i + 1] - times[i] for i in range(len(times) - 1))
    gaps.append(max(0.0, end_time - times[-1]))
    return round(max(gaps), 3)


def build_low_quality_reasons(
    *,
    detection_count: int,
    duration_sec: float,
    max_detection_gap_sec: float,
    interpolation_ratio: float,
) -> list[str]:
    reasons: list[str] = []
    sparse = (detection_count <= 2 and duration_sec >= 3.0) or (
        detection_count <= 3 and duration_sec >= 8.0
    )
    if sparse:
        reasons.append(f"检测点过少（{detection_count}）")
    if detection_count <= 3 and duration_sec >= 8.0:
        reasons.append(f"持续时间过长（{duration_sec:.1f}s）")
    if max_detection_gap_sec >= 10.0:
        reasons.append(f"最大检测间隔（{max_detection_gap_sec:.2f}s）")
    if interpolation_ratio >= 0.90:
        reasons.append(f"插值比例（{interpolation_ratio:.2%}）")
    return reasons


def classify_event_quality(
    *,
    detection_count: int,
    duration_sec: float,
    max_detection_gap_sec: float,
    interpolation_ratio: float,
) -> str:
    if detection_count <= 2 and duration_sec >= 3.0:
        return QUALITY_LOW
    if detection_count <= 3 and duration_sec >= 8.0:
        return QUALITY_LOW
    if max_detection_gap_sec >= 10.0:
        return QUALITY_LOW
    if interpolation_ratio >= 0.90:
        return QUALITY_LOW

    if detection_count <= 4 and duration_sec >= 5.0:
        return QUALITY_MEDIUM
    if max_detection_gap_sec >= 5.0:
        return QUALITY_MEDIUM
    if interpolation_ratio >= 0.70:
        return QUALITY_MEDIUM

    return QUALITY_HIGH


def score_face_event(ev, *, fps: float = 30.0) -> dict[str, Any]:
    """Score a FaceEvent (or compatible object with trajectory + padded bounds)."""
    trajectory = ev.trajectory
    detection_count = len(trajectory)
    duration_sec = round(float(ev.end_time) - float(ev.start_time), 3)
    coverage_frames = int(ev.end_frame) - int(ev.start_frame) + 1

    if coverage_frames > 0:
        interpolation_ratio = round(
            max(0.0, (coverage_frames - detection_count) / coverage_frames),
            4,
        )
    else:
        interpolation_ratio = 0.0

    detection_density = (
        round(detection_count / duration_sec, 4) if duration_sec > 0 else 0.0
    )

    if detection_count >= 2:
        max_gap, avg_gap = _detection_gaps_sec(trajectory)
        max_gap = round(max_gap, 3)
        avg_gap = round(avg_gap, 3)
    else:
        max_gap = round(duration_sec, 3) if detection_count <= 1 else 0.0
        avg_gap = max_gap

    longest_interp_gap = _longest_interpolation_gap_sec(
        trajectory, float(ev.start_time), float(ev.end_time)
    )

    quality = classify_event_quality(
        detection_count=detection_count,
        duration_sec=duration_sec,
        max_detection_gap_sec=max_gap,
        interpolation_ratio=interpolation_ratio,
    )

    risk_reasons = (
        build_low_quality_reasons(
            detection_count=detection_count,
            duration_sec=duration_sec,
            max_detection_gap_sec=max_gap,
            interpolation_ratio=interpolation_ratio,
        )
        if quality == QUALITY_LOW
        else []
    )

    return {
        "event_id": ev.event_id,
        "track_id": ev.track_id,
        "quality": quality,
        "suggest_reject": quality == QUALITY_LOW,
        "warning": LOW_QUALITY_WARNING if quality == QUALITY_LOW else None,
        "risk_reasons": risk_reasons,
        "detection_count": detection_count,
        "duration_sec": duration_sec,
        "detection_density": detection_density,
        "max_detection_gap_sec": max_gap,
        "avg_detection_gap_sec": avg_gap,
        "longest_interpolation_gap_sec": longest_interp_gap,
        "interpolation_ratio": interpolation_ratio,
        "coverage_frames": coverage_frames,
        "peak_confidence": getattr(ev, "peak_confidence", None),
    }


def score_event_dict(ev: dict, *, fps: float = 30.0) -> dict[str, Any]:
    """Score a final/merged event dict (production merge layer output)."""

    class _Ev:
        pass

    obj = _Ev()
    obj.event_id = ev.get("event_id", "")
    obj.track_id = ev.get("track_id", 0)
    obj.start_time = ev.get("start_time", 0.0)
    obj.end_time = ev.get("end_time", 0.0)
    obj.start_frame = ev.get("start_frame", 0)
    obj.end_frame = ev.get("end_frame", 0)
    obj.trajectory = ev.get("trajectory") or []
    obj.peak_confidence = ev.get("peak_confidence")
    return score_face_event(obj, fps=fps)


def score_events(events: list, *, fps: float = 30.0) -> dict[str, Any]:
    rows = [score_face_event(ev, fps=fps) for ev in events]
    by_quality = {q: 0 for q in QUALITY_LEVELS}
    for row in rows:
        by_quality[row["quality"]] = by_quality.get(row["quality"], 0) + 1

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fps": fps,
        "event_count": len(rows),
        "summary": {
            "high": by_quality[QUALITY_HIGH],
            "medium": by_quality[QUALITY_MEDIUM],
            "low": by_quality[QUALITY_LOW],
        },
        "events": rows,
    }


def save_event_quality(output_dir: str, quality_doc: dict) -> str:
    path = os.path.join(os.path.abspath(output_dir), EVENT_QUALITY_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(quality_doc, f, ensure_ascii=False, indent=2)
    return path


def load_event_quality(output_dir: str) -> dict[str, Any] | None:
    path = os.path.join(os.path.abspath(output_dir), EVENT_QUALITY_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def quality_by_event_id(quality_doc: dict | None) -> dict[str, dict]:
    if not quality_doc:
        return {}
    return {row["event_id"]: row for row in quality_doc.get("events", [])}


def main() -> int:
    import argparse
    import sys

    from event_builder import build_events

    p = argparse.ArgumentParser(description="Generate event_quality.json from tracked detections")
    p.add_argument("--output-dir", required=True, help="Detection output directory")
    p.add_argument("--event-gap", type=float, default=1.0)
    args = p.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    tracked_path = os.path.join(out_dir, "tracked_detections.json")
    summary_path = os.path.join(out_dir, "detection_summary.json")
    if not os.path.isfile(tracked_path):
        print(f"[error] Missing {tracked_path}", file=sys.stderr)
        return 1

    with open(tracked_path, encoding="utf-8") as f:
        tracked = json.load(f)
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    fps = float(summary.get("fps") or 30.0)
    total_frames = int(summary.get("total_frames") or 0)
    detect_interval = summary.get("detect_interval")
    from video_meta import get_video_meta

    frame_h, frame_w = 1080, 1920
    video = summary.get("video")
    if video and os.path.isfile(video):
        meta = get_video_meta(video)
        frame_h, frame_w = meta["height"], meta["width"]
        if not total_frames:
            total_frames = meta["frames"]

    events = build_events(
        tracked,
        gap_sec=args.event_gap,
        frame_h=frame_h,
        frame_w=frame_w,
        fps=fps,
        total_frames=total_frames or None,
        detect_interval=int(detect_interval) if detect_interval else None,
    )
    doc = score_events(events, fps=fps)
    path = save_event_quality(out_dir, doc)
    s = doc["summary"]
    print(f"[done] {path}")
    print(f"  HIGH={s['high']} MEDIUM={s['medium']} LOW={s['low']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
