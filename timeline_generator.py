"""Generate timeline.json from accepted review decisions + event trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from event_builder import build_events, resolve_temporal_padding
from gap_analysis import DEFAULT_HARD_SPLIT_GAP_SEC
from review_stats import STATUS_ACCEPTED, load_review_report

TIMELINE_NAME = "timeline.json"
TIMELINE_DEBUG_NAME = "timeline_debug.json"

DEFAULT_NEAR_DETECTION_PAD_SEC = 0.4
DEFAULT_MAX_BRIDGE_GAP_SEC = DEFAULT_HARD_SPLIT_GAP_SEC


def _group_detections_by_gap(trajectory: list[dict], max_gap_frames: int) -> list[list[dict]]:
    """Cluster consecutive detections; break when frame gap exceeds bridge threshold."""
    if not trajectory:
        return []
    ordered = sorted(trajectory, key=lambda p: int(p["frame"]))
    groups: list[list[dict]] = [[ordered[0]]]
    for pt in ordered[1:]:
        if int(pt["frame"]) - int(groups[-1][-1]["frame"]) <= max_gap_frames:
            groups[-1].append(pt)
        else:
            groups.append([pt])
    return groups


def _fill_group_frames(
    group: list[dict],
    *,
    event_id: str,
    track_id: int,
    fps: float,
    pad_frames: int,
    clamp_start: int | None,
    clamp_end: int | None,
    by_frame: dict[int, dict],
) -> tuple[int, int]:
    """Fill timeline entries for one detection cluster; return (start_frame, end_frame)."""

    def append_entry(frame: int, bbox: list[float], conf: float | None) -> None:
        entry = {
            "frame": frame,
            "timestamp": round(frame / fps if fps > 0 else 0.0, 3),
            "track_id": track_id,
            "event_id": event_id,
            "bbox": [round(v, 1) for v in bbox],
            "confidence": round(conf, 4) if conf is not None else None,
        }
        by_frame[frame] = entry

    first_frame = int(group[0]["frame"])
    last_frame = int(group[-1]["frame"])
    seg_start = first_frame - pad_frames
    seg_end = last_frame + pad_frames
    if clamp_start is not None:
        seg_start = max(seg_start, clamp_start)
    if clamp_end is not None:
        seg_end = min(seg_end, clamp_end)
    if seg_start > seg_end:
        return first_frame, last_frame

    for pt in group:
        append_entry(int(pt["frame"]), list(pt["bbox"]), pt.get("conf"))

    for i in range(len(group) - 1):
        f0 = int(group[i]["frame"])
        f1 = int(group[i + 1]["frame"])
        b0 = np.asarray(group[i]["bbox"], dtype=np.float64)
        b1 = np.asarray(group[i + 1]["bbox"], dtype=np.float64)
        c0 = float(group[i].get("conf", 0))
        c1 = float(group[i + 1].get("conf", 0))
        gap = f1 - f0
        if gap <= 1:
            continue
        for f in range(f0 + 1, f1):
            t = (f - f0) / gap
            append_entry(f, (b0 + (b1 - b0) * t).tolist(), c0 + (c1 - c0) * t)

    first_bbox = list(group[0]["bbox"])
    last_bbox = list(group[-1]["bbox"])
    first_conf = group[0].get("conf")
    last_conf = group[-1].get("conf")
    for f in range(seg_start, first_frame):
        append_entry(f, first_bbox, first_conf)
    for f in range(last_frame + 1, seg_end + 1):
        append_entry(f, last_bbox, last_conf)

    return seg_start, seg_end


def interpolate_event_trajectory(
    trajectory: list[dict],
    event_id: str,
    track_id: int,
    fps: float,
    *,
    cover_start_frame: int | None = None,
    cover_end_frame: int | None = None,
    near_detection_pad_sec: float = DEFAULT_NEAR_DETECTION_PAD_SEC,
    max_bridge_gap_sec: float = DEFAULT_MAX_BRIDGE_GAP_SEC,
) -> list[dict]:
    """Segmented interpolation: short pad near detections, break on long no-detection gaps."""
    if not trajectory:
        return []

    pad_frames = max(1, int(round(near_detection_pad_sec * fps)))
    max_gap_frames = max(1, int(round(max_bridge_gap_sec * fps)))
    by_frame: dict[int, dict] = {}

    groups = _group_detections_by_gap(trajectory, max_gap_frames)
    for group in groups:
        _fill_group_frames(
            group,
            event_id=event_id,
            track_id=track_id,
            fps=fps,
            pad_frames=pad_frames,
            clamp_start=cover_start_frame,
            clamp_end=cover_end_frame,
            by_frame=by_frame,
        )

    entries = list(by_frame.values())
    entries.sort(key=lambda x: (x["frame"], x["event_id"]))
    return entries


def segment_count_for_trajectory(
    trajectory: list[dict],
    fps: float,
    *,
    max_bridge_gap_sec: float = DEFAULT_MAX_BRIDGE_GAP_SEC,
) -> int:
    if not trajectory:
        return 0
    max_gap_frames = max(1, int(round(max_bridge_gap_sec * fps)))
    return len(_group_detections_by_gap(trajectory, max_gap_frames))


def load_decisions(output_dir: str) -> dict[str, str]:
    path = os.path.join(output_dir, "confirmed_events.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and data and not isinstance(next(iter(data.values())), dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def load_tracked(output_dir: str) -> list[dict]:
    path = os.path.join(output_dir, "tracked_detections.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing tracked_detections.json in {output_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_video_meta(output_dir: str) -> tuple[str, float, int, int, int]:
    from video_meta import get_video_meta

    video = ""
    fps = 30.0
    for name in ("detection_summary.json", "event_preview.json"):
        p = os.path.join(output_dir, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            video = data.get("video") or video
            fps = float(data.get("fps") or fps)
    if not video or not os.path.isfile(video):
        raise FileNotFoundError(f"Video not found for timeline generation: {video}")
    meta = get_video_meta(video)
    return video, float(meta["fps"]), meta["width"], meta["height"], meta["frames"]


def load_detect_interval(output_dir: str) -> int | None:
    for name in ("detection_summary.json", "event_preview.json"):
        p = os.path.join(output_dir, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            interval = data.get("detect_interval")
            if interval is not None:
                return int(interval)
    return None


def _event_timeline_debug_row(
    ev,
    expanded: list[dict],
    fps: float,
    *,
    segment_count: int = 1,
) -> dict:
    """Per-event debug: event bounds vs actual segmented timeline span."""
    det_start = float(ev.trajectory[0]["t"]) if ev.trajectory else float(ev.start_time)
    det_end = float(ev.trajectory[-1]["t"]) if ev.trajectory else float(ev.end_time)
    pre_padding_sec = round(max(0.0, det_start - ev.start_time), 3)
    post_padding_sec = round(max(0.0, ev.end_time - det_end), 3)
    event_span_frames = int(ev.end_frame) - int(ev.start_frame) + 1

    if not expanded:
        return {
            "event_id": ev.event_id,
            "event_start_time": ev.start_time,
            "event_end_time": ev.end_time,
            "timeline_start_time": None,
            "timeline_end_time": None,
            "pre_padding_sec": pre_padding_sec,
            "post_padding_sec": post_padding_sec,
            "segment_count": segment_count,
            "event_span_frames": event_span_frames,
            "timeline_frame_count": 0,
            "uncovered_event_frames": event_span_frames,
            "extra_before_frames": 0,
            "extra_after_frames": 0,
        }

    frames = [int(e["frame"]) for e in expanded]
    tl_start_frame = min(frames)
    tl_end_frame = max(frames)
    tl_start_time = round(tl_start_frame / fps if fps > 0 else 0.0, 3)
    tl_end_time = round(tl_end_frame / fps if fps > 0 else 0.0, 3)

    extra_before = sum(1 for f in frames if f < ev.start_frame)
    extra_after = sum(1 for f in frames if f > ev.end_frame)
    covered_in_event = len({f for f in frames if ev.start_frame <= f <= ev.end_frame})

    return {
        "event_id": ev.event_id,
        "event_start_time": ev.start_time,
        "event_end_time": ev.end_time,
        "timeline_start_time": tl_start_time,
        "timeline_end_time": tl_end_time,
        "pre_padding_sec": pre_padding_sec,
        "post_padding_sec": post_padding_sec,
        "segment_count": segment_count,
        "event_span_frames": event_span_frames,
        "timeline_frame_count": len(expanded),
        "uncovered_event_frames": max(0, event_span_frames - covered_in_event),
        "extra_before_frames": extra_before,
        "extra_after_frames": extra_after,
    }


def _build_timeline_debug_doc(
    timeline: dict,
    debug_events: list[dict],
) -> dict:
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video": timeline.get("video"),
        "fps": timeline.get("fps"),
        "accepted_event_count": len(debug_events),
        "temporal_padding": timeline.get("temporal_padding"),
        "segmented_interpolation": timeline.get("segmented_interpolation"),
        "events": debug_events,
        "summary": {
            "events_with_extra_before_frames": sum(
                1 for e in debug_events if e["extra_before_frames"] > 0
            ),
            "events_with_extra_after_frames": sum(
                1 for e in debug_events if e["extra_after_frames"] > 0
            ),
            "total_extra_before_frames": sum(e["extra_before_frames"] for e in debug_events),
            "total_extra_after_frames": sum(e["extra_after_frames"] for e in debug_events),
            "events_with_multiple_segments": sum(
                1 for e in debug_events if e.get("segment_count", 1) > 1
            ),
            "total_uncovered_event_frames": sum(
                e.get("uncovered_event_frames", 0) for e in debug_events
            ),
        },
    }


def build_timeline(
    output_dir: str,
    event_gap: float = 1.0,
    *,
    near_detection_pad_sec: float = DEFAULT_NEAR_DETECTION_PAD_SEC,
    max_bridge_gap_sec: float = DEFAULT_MAX_BRIDGE_GAP_SEC,
) -> dict:
    output_dir = os.path.abspath(output_dir)
    video, fps, w, h, total_frames = load_video_meta(output_dir)
    tracked = load_tracked(output_dir)
    decisions = load_decisions(output_dir)
    detect_interval = load_detect_interval(output_dir)
    pre_pad, post_pad = resolve_temporal_padding(fps, detect_interval=detect_interval)

    events = build_events(
        tracked,
        gap_sec=event_gap,
        frame_h=h,
        frame_w=w,
        fps=fps,
        total_frames=total_frames,
        detect_interval=detect_interval,
    )
    event_by_id = {e.event_id: e for e in events}

    accepted_ids = [eid for eid, st in decisions.items() if st == STATUS_ACCEPTED]
    all_entries: list[dict] = []
    accepted_events_meta: list[dict] = []
    debug_events: list[dict] = []

    for eid in sorted(accepted_ids):
        ev = event_by_id.get(eid)
        if ev is None:
            continue
        traj = [
            {"frame": p["frame"], "bbox": p["bbox"], "conf": p["conf"], "t": p["t"]}
            for p in ev.trajectory
        ]
        expanded = interpolate_event_trajectory(
            traj,
            eid,
            ev.track_id,
            fps,
            cover_start_frame=ev.start_frame,
            cover_end_frame=ev.end_frame,
            near_detection_pad_sec=near_detection_pad_sec,
            max_bridge_gap_sec=max_bridge_gap_sec,
        )
        segments = segment_count_for_trajectory(
            traj, fps, max_bridge_gap_sec=max_bridge_gap_sec
        )
        all_entries.extend(expanded)
        debug_events.append(
            _event_timeline_debug_row(ev, expanded, fps, segment_count=segments)
        )
        accepted_events_meta.append({
            "event_id": eid,
            "track_id": ev.track_id,
            "start_time": ev.start_time,
            "end_time": ev.end_time,
            "start_frame": ev.start_frame,
            "end_frame": ev.end_frame,
            "frame_count": ev.detection_count,
            "timeline_frames": len(expanded),
            "segment_count": segments,
            "peak_confidence": ev.peak_confidence,
        })

    all_entries.sort(key=lambda x: (x["frame"], x["event_id"]))

    timeline = {
        "video": os.path.abspath(video),
        "fps": fps,
        "total_frames": total_frames,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "accepted_event_count": len(accepted_events_meta),
        "timeline_entry_count": len(all_entries),
        "detect_interval": detect_interval,
        "temporal_padding": {
            "pre_padding_sec": pre_pad,
            "post_padding_sec": post_pad,
            "padding_frames": detect_interval * 2 if detect_interval else None,
        },
        "segmented_interpolation": {
            "near_detection_pad_sec": near_detection_pad_sec,
            "max_bridge_gap_sec": max_bridge_gap_sec,
            "description": (
                "Draw bbox only near detections; break when no-detection gap "
                "exceeds max_bridge_gap_sec."
            ),
        },
        "accepted_events": accepted_events_meta,
        "entries": all_entries,
    }
    timeline["_debug"] = _build_timeline_debug_doc(timeline, debug_events)
    return timeline


def save_timeline(output_dir: str, timeline: dict) -> str:
    path = os.path.join(output_dir, TIMELINE_NAME)
    payload = {k: v for k, v in timeline.items() if k != "_debug"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def save_timeline_debug(output_dir: str, debug: dict) -> str:
    path = os.path.join(output_dir, TIMELINE_DEBUG_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
    return path


def generate_timeline(
    output_dir: str,
    event_gap: float = 1.0,
    *,
    near_detection_pad_sec: float = DEFAULT_NEAR_DETECTION_PAD_SEC,
    max_bridge_gap_sec: float = DEFAULT_MAX_BRIDGE_GAP_SEC,
) -> str:
    timeline = build_timeline(
        output_dir,
        event_gap=event_gap,
        near_detection_pad_sec=near_detection_pad_sec,
        max_bridge_gap_sec=max_bridge_gap_sec,
    )
    debug = timeline.pop("_debug")
    path = save_timeline(output_dir, timeline)
    save_timeline_debug(output_dir, debug)
    return path


def parse_args():
    p = argparse.ArgumentParser(description="Generate timeline.json from accepted events")
    p.add_argument("--output-dir", required=True, help="Detection/review output directory")
    p.add_argument("--event-gap", type=float, default=1.0)
    p.add_argument(
        "--near-detection-pad-sec",
        type=float,
        default=DEFAULT_NEAR_DETECTION_PAD_SEC,
        help="Extend bbox this many seconds before/after each detection cluster",
    )
    p.add_argument(
        "--max-bridge-gap-sec",
        type=float,
        default=DEFAULT_MAX_BRIDGE_GAP_SEC,
        help="Max no-detection gap (seconds) to linearly interpolate across",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"[error] Not a directory: {output_dir}", file=sys.stderr)
        return 1
    if not load_decisions(output_dir):
        print("[error] confirmed_events.json missing or empty", file=sys.stderr)
        return 1

    path = generate_timeline(
        output_dir,
        event_gap=args.event_gap,
        near_detection_pad_sec=args.near_detection_pad_sec,
        max_bridge_gap_sec=args.max_bridge_gap_sec,
    )
    with open(path, encoding="utf-8") as f:
        tl = json.load(f)
    print(f"[done] timeline.json -> {path}")
    print(f"  accepted events: {tl['accepted_event_count']}")
    print(f"  timeline entries: {tl['timeline_entry_count']}")
    debug_path = os.path.join(output_dir, TIMELINE_DEBUG_NAME)
    if os.path.isfile(debug_path):
        with open(debug_path, encoding="utf-8") as f:
            dbg = json.load(f)
        s = dbg.get("summary", {})
        print(f"  timeline_debug.json -> {debug_path}")
        print(
            f"  extra frames: before={s.get('total_extra_before_frames', 0)} "
            f"after={s.get('total_extra_after_frames', 0)}"
        )
    report = load_review_report(output_dir)
    if report:
        print(f"  review accept rate: {report.get('accept_rate')}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
