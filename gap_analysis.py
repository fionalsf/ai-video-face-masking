"""Gap semantic layer: presence / absence segments before Event Builder."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_HARD_SPLIT_GAP_SEC = 3.0
ABSENCE_SEGMENTS_NAME = "absence_segments.json"
EVENT_SEGMENT_MAP_NAME = "event_segment_map.json"


def group_by_track(detections: list[dict]) -> dict[int, list[dict]]:
    by_track: dict[int, list[dict]] = {}
    for d in detections:
        by_track.setdefault(int(d["track_id"]), []).append(d)
    for tid in by_track:
        by_track[tid].sort(key=lambda x: int(x["frame"]))
    return by_track


def split_presence_segments(
    seq: list[dict],
    hard_split_gap_sec: float = DEFAULT_HARD_SPLIT_GAP_SEC,
) -> list[list[dict]]:
    """Split track detections into presence segments (consecutive gaps <= hard_split)."""
    if not seq:
        return []
    ordered = sorted(seq, key=lambda x: int(x["frame"]))
    segments: list[list[dict]] = [[ordered[0]]]
    for det in ordered[1:]:
        gap = float(det["t"]) - float(segments[-1][-1]["t"])
        if gap > hard_split_gap_sec:
            segments.append([det])
        else:
            segments[-1].append(det)
    return segments


def presence_segment_id(track_id: int, segment_index: int) -> str:
    return f"pres_{track_id:04d}_{segment_index:02d}"


def absence_segment_id(track_id: int, segment_index: int) -> str:
    return f"abs_{track_id:04d}_{segment_index:02d}"


def compute_absence_segments(
    track_id: int,
    seq: list[dict],
    hard_split_gap_sec: float,
) -> list[dict[str, Any]]:
    """Absence segments: gaps between detections where gap > hard_split_gap_sec."""
    if len(seq) < 2:
        return []
    ordered = sorted(seq, key=lambda x: int(x["frame"]))
    absences: list[dict[str, Any]] = []
    for i in range(len(ordered) - 1):
        prev, curr = ordered[i], ordered[i + 1]
        gap = float(curr["t"]) - float(prev["t"])
        if gap <= hard_split_gap_sec:
            continue
        idx = len(absences)
        absences.append({
            "absence_id": absence_segment_id(track_id, idx),
            "track_id": track_id,
            "segment_index": idx,
            "gap_sec": round(gap, 3),
            "duration_sec": round(gap, 3),
            "start_time": round(float(prev["t"]), 3),
            "end_time": round(float(curr["t"]), 3),
            "start_frame": int(prev["frame"]),
            "end_frame": int(curr["frame"]),
            "after_detection_frame": int(prev["frame"]),
            "before_detection_frame": int(curr["frame"]),
            "after_detection_time": round(float(prev["t"]), 3),
            "before_detection_time": round(float(curr["t"]), 3),
        })
    return absences


def analyze_track(
    track_id: int,
    seq: list[dict],
    *,
    hard_split_gap_sec: float = DEFAULT_HARD_SPLIT_GAP_SEC,
) -> dict[str, Any]:
    presence_groups = split_presence_segments(seq, hard_split_gap_sec)
    presence_meta: list[dict[str, Any]] = []
    for idx, group in enumerate(presence_groups):
        presence_meta.append({
            "presence_segment_id": presence_segment_id(track_id, idx),
            "track_id": track_id,
            "segment_index": idx,
            "detection_count": len(group),
            "start_time": round(float(group[0]["t"]), 3),
            "end_time": round(float(group[-1]["t"]), 3),
            "start_frame": int(group[0]["frame"]),
            "end_frame": int(group[-1]["frame"]),
        })
    absences = compute_absence_segments(track_id, seq, hard_split_gap_sec)
    return {
        "track_id": track_id,
        "detection_count": len(seq),
        "presence_segment_count": len(presence_groups),
        "absence_segment_count": len(absences),
        "presence_segments": presence_meta,
        "absences": absences,
        "presence_groups": presence_groups,
    }


@dataclass
class GapAnalysisResult:
    hard_split_gap_sec: float
    fps: float
    track_count: int
    presence_segment_total: int
    absence_segment_total: int
    tracks: list[dict[str, Any]] = field(default_factory=list)
    absences: list[dict[str, Any]] = field(default_factory=list)
    presence_by_track: dict[int, list[list[dict]]] = field(default_factory=dict)

    def to_absence_doc(self) -> dict[str, Any]:
        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hard_split_gap_sec": self.hard_split_gap_sec,
            "fps": self.fps,
            "track_count": self.track_count,
            "absence_segment_count": self.absence_segment_total,
            "presence_segment_count": self.presence_segment_total,
            "tracks": [
                {
                    "track_id": t["track_id"],
                    "detection_count": t["detection_count"],
                    "presence_segment_count": t["presence_segment_count"],
                    "absence_segment_count": t["absence_segment_count"],
                    "presence_segments": t["presence_segments"],
                }
                for t in self.tracks
            ],
            "absences": self.absences,
        }


def analyze_gaps(
    detections: list[dict],
    *,
    hard_split_gap_sec: float = DEFAULT_HARD_SPLIT_GAP_SEC,
    fps: float = 30.0,
) -> GapAnalysisResult:
    by_track = group_by_track(detections)
    tracks: list[dict[str, Any]] = []
    all_absences: list[dict[str, Any]] = []
    presence_by_track: dict[int, list[list[dict]]] = {}
    presence_total = 0

    for track_id in sorted(by_track):
        row = analyze_track(track_id, by_track[track_id], hard_split_gap_sec=hard_split_gap_sec)
        tracks.append({k: v for k, v in row.items() if k != "presence_groups"})
        all_absences.extend(row["absences"])
        presence_by_track[track_id] = row["presence_groups"]
        presence_total += row["presence_segment_count"]

    return GapAnalysisResult(
        hard_split_gap_sec=hard_split_gap_sec,
        fps=fps,
        track_count=len(tracks),
        presence_segment_total=presence_total,
        absence_segment_total=len(all_absences),
        tracks=tracks,
        absences=all_absences,
        presence_by_track=presence_by_track,
    )


def save_absence_segments(output_dir: str, gap: GapAnalysisResult) -> str:
    path = os.path.join(os.path.abspath(output_dir), ABSENCE_SEGMENTS_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gap.to_absence_doc(), f, ensure_ascii=False, indent=2)
    return path


def build_event_segment_map_rows(
    event_bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event_count": len(event_bindings),
        "events": event_bindings,
    }


def save_event_segment_map(output_dir: str, rows: list[dict[str, Any]]) -> str:
    path = os.path.join(os.path.abspath(output_dir), EVENT_SEGMENT_MAP_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_event_segment_map_rows(rows), f, ensure_ascii=False, indent=2)
    return path


def save_gap_analysis_debug(
    output_dir: str,
    gap: GapAnalysisResult,
    event_bindings: list[dict[str, Any]],
) -> tuple[str, str]:
    absence_path = save_absence_segments(output_dir, gap)
    map_path = save_event_segment_map(output_dir, event_bindings)
    return absence_path, map_path


def main() -> int:
    import argparse
    import sys

    from event_builder import build_events
    from video_meta import get_video_meta

    p = argparse.ArgumentParser(description="Gap analysis layer + debug JSON export")
    p.add_argument("--output-dir", required=True)
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
    summary: dict = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    fps = float(summary.get("fps") or 30.0)
    video = summary.get("video")
    frame_h, frame_w, total_frames = 1080, 1920, None
    if video and os.path.isfile(video):
        meta = get_video_meta(video)
        frame_h, frame_w = meta["height"], meta["width"]
        total_frames = meta["frames"]

    events = build_events(
        tracked,
        gap_sec=args.event_gap,
        frame_h=frame_h,
        frame_w=frame_w,
        fps=fps,
        total_frames=total_frames,
        detect_interval=int(summary.get("detect_interval") or 0) or None,
        output_dir=out_dir,
    )
    print(f"[done] events={len(events)}")
    print(f"  {os.path.join(out_dir, ABSENCE_SEGMENTS_NAME)}")
    print(f"  {os.path.join(out_dir, EVENT_SEGMENT_MAP_NAME)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
