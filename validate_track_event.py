#!/usr/bin/env python3
"""Phase 2 validation: ByteTrack + Event build from Phase 1 detections.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from statistics import mean

import cv2

from event_builder import build_events
from identity_behavior_builder import build_identity_behavior_events, behavior_event_to_face_event
from identity_stitching import run_identity_stitching
from event_merge import save_segmentation_events
from event_quality import save_event_quality, score_event_dict, score_events
from export import draw_detections
from tracker import cpu_byte_track
from video_meta import get_video_meta, sec_to_timecode


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 — Tracking + Event validation")
    p.add_argument(
        "--detection-dir",
        help="Phase 1 output dir containing detections.json",
    )
    p.add_argument("--detections", help="Path to detections.json")
    p.add_argument("--video", help="Source video (default: from detection_summary.json)")
    p.add_argument("--event-gap", type=float, default=1.0, help="Event split gap (seconds)")
    p.add_argument("--output-dir", help="Output dir (default: same as detection-dir)")
    return p.parse_args()


def load_phase1(detection_dir: str | None, detections_path: str | None) -> tuple[str, list[dict], dict]:
    if detection_dir:
        detection_dir = os.path.abspath(detection_dir)
        detections_path = os.path.join(detection_dir, "detections.json")
        summary_path = os.path.join(detection_dir, "detection_summary.json")
    else:
        detections_path = os.path.abspath(detections_path)
        detection_dir = os.path.dirname(detections_path)
        summary_path = os.path.join(detection_dir, "detection_summary.json")

    if not os.path.isfile(detections_path):
        raise FileNotFoundError(f"detections.json not found: {detections_path}")

    with open(detections_path, encoding="utf-8") as f:
        records = json.load(f)

    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    return detection_dir, records, summary


def records_to_sparse(records: list[dict]) -> dict[int, list[tuple[list[float], float]]]:
    sparse: dict[int, list] = {}
    for rec in records:
        dets = rec.get("detections") or []
        if not dets:
            continue
        sparse[int(rec["frame"])] = [
            (d["bbox"], float(d["confidence"])) for d in dets
        ]
    return sparse


def per_frame_to_detections(
    per_frame: dict[int, list[tuple[int, list[float], float]]],
    fps: float,
) -> list[dict]:
    out = []
    for frame_idx in sorted(per_frame):
        t_sec = round(frame_idx / fps, 3) if fps > 0 else 0.0
        for track_id, bbox, conf in per_frame[frame_idx]:
            out.append({
                "frame": frame_idx,
                "t": t_sec,
                "track_id": track_id,
                "bbox": [round(v, 1) for v in bbox],
                "conf": round(conf, 4),
            })
    return out


def build_track_stats(tracked: list[dict]) -> dict:
    by_track: dict[int, list[dict]] = {}
    for d in tracked:
        by_track.setdefault(d["track_id"], []).append(d)

    tracks = []
    for track_id in sorted(by_track):
        seq = sorted(by_track[track_id], key=lambda x: x["frame"])
        duration = round(seq[-1]["t"] - seq[0]["t"], 3)
        tracks.append({
            "track_id": track_id,
            "start_time": seq[0]["t"],
            "end_time": seq[-1]["t"],
            "start_timecode": sec_to_timecode(seq[0]["t"]),
            "end_timecode": sec_to_timecode(seq[-1]["t"]),
            "duration_sec": duration,
            "detection_count": len(seq),
            "start_frame": seq[0]["frame"],
            "end_frame": seq[-1]["frame"],
        })

    durations = [t["duration_sec"] for t in tracks]
    det_counts = [t["detection_count"] for t in tracks]
    longest = max(tracks, key=lambda x: x["duration_sec"]) if tracks else None
    shortest = min(tracks, key=lambda x: x["duration_sec"]) if tracks else None

    return {
        "track_total": len(tracks),
        "track_avg_duration_sec": round(mean(durations), 3) if durations else 0.0,
        "track_avg_detection_count": round(mean(det_counts), 2) if det_counts else 0.0,
        "track_longest": longest,
        "track_shortest": shortest,
        "tracks": tracks,
    }


def _ev_get(e, key: str):
    return e[key] if isinstance(e, dict) else getattr(e, key)


def build_event_stats(events) -> dict:
    items = []
    for e in events:
        duration = round(_ev_get(e, "end_time") - _ev_get(e, "start_time"), 3)
        items.append({
            "event_id": _ev_get(e, "event_id"),
            "identity_id": _ev_get(e, "identity_id") if isinstance(e, dict) and "identity_id" in e else None,
            "track_id": _ev_get(e, "primary_track_id") if isinstance(e, dict) and "primary_track_id" in e else _ev_get(e, "track_id"),
            "start_time": _ev_get(e, "start_time"),
            "end_time": _ev_get(e, "end_time"),
            "start_timecode": sec_to_timecode(_ev_get(e, "start_time")),
            "end_timecode": sec_to_timecode(_ev_get(e, "end_time")),
            "duration_sec": duration,
            "frame_count": _ev_get(e, "detection_count"),
            "peak_confidence": _ev_get(e, "peak_confidence"),
            "avg_confidence": _ev_get(e, "avg_confidence"),
        })

    durations = [x["duration_sec"] for x in items]
    frame_counts = [x["frame_count"] for x in items]
    longest = max(items, key=lambda x: x["duration_sec"]) if items else None
    shortest = min(items, key=lambda x: x["duration_sec"]) if items else None

    return {
        "event_total": len(items),
        "event_avg_duration_sec": round(mean(durations), 3) if durations else 0.0,
        "event_avg_frame_count": round(mean(frame_counts), 2) if frame_counts else 0.0,
        "event_longest": longest,
        "event_shortest": shortest,
        "events": items,
    }


def middle_trajectory_point(trajectory: list[dict]) -> dict:
    return trajectory[len(trajectory) // 2]


def export_event_previews(
    events,
    video_path: str,
    preview_dir: str,
) -> list[dict]:
    os.makedirs(preview_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    previews = []
    for e in events:
        trajectory = _ev_get(e, "trajectory")
        event_id = _ev_get(e, "event_id")
        mid = middle_trajectory_point(trajectory)
        frame_idx = int(mid["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
            ok, frame = cap.read()
        if not ok:
            continue

        det = {
            "bbox": mid["bbox"],
            "confidence": mid["conf"],
        }
        annotated = draw_detections(frame, [det])
        img_name = f"{event_id}.jpg"
        img_path = os.path.join(preview_dir, img_name)
        cv2.imwrite(img_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        previews.append({
            "event_id": event_id,
            "identity_id": _ev_get(e, "identity_id") if isinstance(e, dict) else None,
            "track_id": _ev_get(e, "primary_track_id") if isinstance(e, dict) and "primary_track_id" in e else _ev_get(e, "track_id"),
            "start_time": _ev_get(e, "start_time"),
            "end_time": _ev_get(e, "end_time"),
            "start_timecode": sec_to_timecode(_ev_get(e, "start_time")),
            "end_timecode": sec_to_timecode(_ev_get(e, "end_time")),
            "duration_sec": round(_ev_get(e, "end_time") - _ev_get(e, "start_time"), 3),
            "peak_confidence": _ev_get(e, "peak_confidence"),
            "avg_confidence": _ev_get(e, "avg_confidence"),
            "frame_count": _ev_get(e, "detection_count"),
            "preview_frame": frame_idx,
            "preview_timecode": sec_to_timecode(mid["t"]),
            "preview_image": os.path.join("event_previews", img_name).replace("\\", "/"),
        })

    cap.release()
    return previews


def run_validation(args) -> int:
    if not args.detection_dir and not args.detections:
        print("[error] --detection-dir or --detections required", file=sys.stderr)
        return 1

    detection_dir, records, summary = load_phase1(args.detection_dir, args.detections)
    out_dir = os.path.abspath(args.output_dir or detection_dir)
    os.makedirs(out_dir, exist_ok=True)

    video_path = args.video or summary.get("video")
    if not video_path or not os.path.isfile(video_path):
        print(f"[error] Video not found: {video_path}", file=sys.stderr)
        return 1

    meta = get_video_meta(video_path)
    fps = float(summary.get("fps") or meta["fps"])

    raw_detections = sum(len(r.get("detections") or []) for r in records)
    print(f"[info] loaded {len(records)} sampled frames, {raw_detections} raw detections")
    print(f"[info] video: {video_path}")

    sparse = records_to_sparse(records)
    print(f"[1/3] ByteTrack on {len(sparse)} frames...")
    per_frame = cpu_byte_track(sparse)
    tracked = per_frame_to_detections(per_frame, fps)
    print(f"      tracked boxes: {len(tracked)}")

    print(f"[2/3] Event build — Scheme C segmentation (gap={args.event_gap}s)...")
    builder_stats: dict = {}
    segmentation_events = build_events(
        tracked,
        gap_sec=args.event_gap,
        frame_h=meta["height"],
        frame_w=meta["width"],
        merge_stats=builder_stats,
        fps=fps,
        total_frames=meta["frames"],
        detect_interval=int(summary.get("detect_interval") or 0) or None,
        output_dir=out_dir,
    )
    print(f"      segmentation events: {len(segmentation_events)}")
    if builder_stats:
        print(
            f"      builder: {builder_stats.get('original_event_count')} ->"
            f" {builder_stats.get('merged_event_count')}"
        )

    seg_path = save_segmentation_events(
        segmentation_events,
        out_dir,
        video=os.path.abspath(video_path),
        fps=fps,
    )
    print(f"      saved: {os.path.basename(seg_path)}")

    print("[2c/3] Identity stitching (pipeline core)...")
    stitch_stats = run_identity_stitching(
        tracked,
        output_dir=out_dir,
        video=os.path.abspath(video_path),
    )
    print(
        f"      stitch: {stitch_stats['track_count']} tracks ->"
        f" {stitch_stats['identity_cluster_count']} identities"
        f" | appearance={stitch_stats['appearance_method']}"
    )

    print("[2d/3] Identity behavior event builder (production)...")
    behavior_stats = build_identity_behavior_events(
        tracked,
        stitch_stats["clusters"],
        output_dir=out_dir,
        video=os.path.abspath(video_path),
        fps=fps,
        frame_h=meta["height"],
        frame_w=meta["width"],
        total_frames=meta["frames"],
        detect_interval=int(summary.get("detect_interval") or 0) or None,
    )
    production_events = behavior_stats["behavior_events"]
    print(
        f"      behavior: {behavior_stats['behavior_event_count']} events"
        f" | target_30_80={behavior_stats['target_in_range']}"
    )

    merge_stats = {"deprecated": "production uses behavior_events.json from identity pipeline"}

    track_summary = build_track_stats(tracked)
    event_summary = build_event_stats(production_events)
    quality_doc = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fps": fps,
        "pipeline": "identity_stitching + identity_behavior_builder",
        "event_count": len(production_events),
        "summary": {},
        "events": [ev["event_quality_score"] for ev in production_events if ev.get("event_quality_score")],
    }
    if production_events:
        scored = score_events(
            [behavior_event_to_face_event(ev) for ev in production_events],
            fps=fps,
        )
        quality_doc["summary"] = scored.get("summary", {})
        quality_doc["events"] = scored.get("events", [])

    compression = {
        "raw_detections": raw_detections,
        "tracked_detections": len(tracked),
        "track_total": track_summary["track_total"],
        "event_total": event_summary["event_total"],
        "detection_to_event_ratio": round(raw_detections / max(1, event_summary["event_total"]), 2),
        "avg_detections_per_event": round(raw_detections / max(1, event_summary["event_total"]), 2),
    }

    print("[3/3] Export event previews...")
    preview_dir = os.path.join(out_dir, "event_previews")
    os.makedirs(preview_dir, exist_ok=True)
    for old in os.listdir(preview_dir):
        if old.lower().endswith(".jpg"):
            try:
                os.remove(os.path.join(preview_dir, old))
            except OSError:
                pass
    event_previews = export_event_previews(production_events, video_path, preview_dir)

    with open(os.path.join(out_dir, "track_summary.json"), "w", encoding="utf-8") as f:
        json.dump(track_summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "event_summary.json"), "w", encoding="utf-8") as f:
        json.dump({**event_summary, "compression": compression, "builder_stats": builder_stats, "merge_stats": merge_stats}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "event_merge_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"builder": builder_stats, "merge_layer": merge_stats}, f, ensure_ascii=False, indent=2)
    quality_path = save_event_quality(out_dir, quality_doc)
    with open(os.path.join(out_dir, "event_preview.json"), "w", encoding="utf-8") as f:
        json.dump({
            "video": os.path.abspath(video_path),
            "event_gap_sec": args.event_gap,
            "compression": compression,
            "events": event_previews,
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "tracked_detections.json"), "w", encoding="utf-8") as f:
        json.dump(tracked, f, ensure_ascii=False, indent=2)

    print()
    print("[done] Phase 2 tracking + event validation")
    print(f"  output: {out_dir}")
    print(f"  tracks: {track_summary['track_total']}  |  avg duration: {track_summary['track_avg_duration_sec']}s")
    print(f"  events: {event_summary['event_total']}  |  avg duration: {event_summary['event_avg_duration_sec']}s")
    qs = quality_doc["summary"]
    print(f"  quality: HIGH={qs['high']} MEDIUM={qs['medium']} LOW={qs['low']} -> {quality_path}")
    print(f"  compression: {raw_detections} detections -> {event_summary['event_total']} events "
          f"({compression['detection_to_event_ratio']}:1)")
    return 0


def main():
    sys.exit(run_validation(parse_args()))


if __name__ == "__main__":
    main()
