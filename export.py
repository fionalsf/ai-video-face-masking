"""Phase 1 detection export: detections.json, review_images/, detection_summary.json."""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np


def frame_timestamp(frame_idx: int, fps: float) -> float:
    return round(frame_idx / fps if fps > 0 else 0.0, 3)


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        conf = det["confidence"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return vis


def confidence_histogram(all_confs: list[float], bin_width: float = 0.05) -> dict[str, int]:
    if not all_confs:
        return {}
    lo = max(0.0, min(all_confs) // bin_width * bin_width)
    lo = round(lo, 2)
    if lo > 0.35:
        lo = 0.35
    hi = 1.0
    edges = []
    cur = lo
    while cur < hi:
        edges.append(round(cur, 2))
        cur = round(cur + bin_width, 2)
    edges.append(hi)

    hist = {f"{edges[i]:.2f}-{edges[i + 1]:.2f}": 0 for i in range(len(edges) - 1)}
    for c in all_confs:
        placed = False
        for i in range(len(edges) - 1):
            a, b = edges[i], edges[i + 1]
            if a <= c < b or (b == hi and a <= c <= b):
                hist[f"{a:.2f}-{b:.2f}"] += 1
                placed = True
                break
        if not placed:
            last_key = f"{edges[-2]:.2f}-{edges[-1]:.2f}"
            hist[last_key] += 1
    return hist


def review_image_name(frame_idx: int, det_idx: int, confidence: float, multi: bool) -> str:
    conf_tag = f"{confidence:.3f}"
    if multi:
        return f"frame_{frame_idx:06d}_d{det_idx:02d}_conf_{conf_tag}.jpg"
    return f"frame_{frame_idx:06d}_conf_{conf_tag}.jpg"


def export_review_images(
    review_dir: str,
    frame_idx: int,
    frame_bgr: np.ndarray,
    detections: list[dict],
) -> None:
    if not detections:
        return
    os.makedirs(review_dir, exist_ok=True)
    multi = len(detections) > 1
    for det_idx, det in enumerate(detections):
        name = review_image_name(frame_idx, det_idx, det["confidence"], multi)
        annotated = draw_detections(frame_bgr, [det])
        cv2.imwrite(
            os.path.join(review_dir, name),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )


def build_detection_summary(
    video_path: str,
    fps: float,
    total_frames: int,
    sampled_frames: int,
    all_confs: list[float],
    interval: int,
    conf_threshold: float,
    imgsz: int,
) -> dict:
    total_detections = len(all_confs)
    duration_sec = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
    duration_min = max(duration_sec / 60.0, 1e-9)
    return {
        "video": os.path.abspath(video_path),
        "fps": round(fps, 4),
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "total_detections": total_detections,
        "avg_confidence": round(float(np.mean(all_confs)), 4) if all_confs else 0.0,
        "confidence_histogram": confidence_histogram(all_confs),
        "detections_per_minute": round(total_detections / duration_min, 3),
        "detect_interval": interval,
        "conf_threshold": conf_threshold,
        "imgsz": imgsz,
        "duration_sec": round(duration_sec, 3),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_phase1_outputs(
    out_dir: str,
    frame_records: list[dict],
    summary: dict,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "detections.json"), "w", encoding="utf-8") as f:
        json.dump(frame_records, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "detection_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
