"""Recommend pipeline parameters from video stats and Phase-1 detection summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from video_meta import get_video_meta

DEFAULT_CONF = 0.35
DEFAULT_EVENT_GAP = 1.0
DEFAULT_MOSAIC_LEVEL = "extreme"
MOSAIC_LEVELS = ("low", "medium", "high", "extreme")


@dataclass
class VideoStats:
    fps: float
    total_frames: int
    duration_sec: float
    sampled_frames: int = 0
    width: int | None = None
    height: int | None = None
    detect_interval: int = 5

    @classmethod
    def from_summary(cls, summary: dict) -> VideoStats:
        fps = float(summary.get("fps") or 30.0)
        total_frames = int(summary.get("total_frames") or 0)
        duration = float(summary.get("duration_sec") or 0.0)
        if duration <= 0 and total_frames > 0 and fps > 0:
            duration = total_frames / fps
        return cls(
            fps=fps,
            total_frames=total_frames,
            duration_sec=round(duration, 3),
            sampled_frames=int(summary.get("sampled_frames") or 0),
            detect_interval=int(summary.get("detect_interval") or 5),
        )

    @classmethod
    def from_video(cls, video_path: str, *, detect_interval: int = 5) -> VideoStats:
        meta = get_video_meta(video_path)
        fps = float(meta["fps"])
        frames = int(meta["frames"])
        return cls(
            fps=fps,
            total_frames=frames,
            duration_sec=round(frames / fps if fps > 0 else 0.0, 3),
            sampled_frames=max(1, frames // max(1, detect_interval)),
            width=int(meta["width"]),
            height=int(meta["height"]),
            detect_interval=detect_interval,
        )


@dataclass
class DetectionStats:
    total_detections: int
    avg_confidence: float
    detections_per_minute: float
    confidence_histogram: dict[str, int] = field(default_factory=dict)
    conf_threshold: float = DEFAULT_CONF

    @classmethod
    def from_summary(cls, summary: dict) -> DetectionStats:
        return cls(
            total_detections=int(summary.get("total_detections") or 0),
            avg_confidence=float(summary.get("avg_confidence") or 0.0),
            detections_per_minute=float(summary.get("detections_per_minute") or 0.0),
            confidence_histogram=dict(summary.get("confidence_histogram") or {}),
            conf_threshold=float(summary.get("conf_threshold") or DEFAULT_CONF),
        )


def _bin_midpoint(label: str) -> float:
    lo, hi = label.split("-", 1)
    return (float(lo) + float(hi)) / 2.0


def analyze_histogram(histogram: dict[str, int]) -> dict[str, float]:
    total = sum(histogram.values())
    if total <= 0:
        return {
            "total": 0,
            "low_ratio": 0.0,
            "borderline_ratio": 0.0,
            "high_ratio": 0.0,
            "very_low_ratio": 0.0,
        }

    low = borderline = high = very_low = 0
    for label, count in histogram.items():
        mid = _bin_midpoint(label)
        if mid < 0.45:
            very_low += count
        if mid < 0.50:
            low += count
        elif mid < 0.75:
            borderline += count
        else:
            high += count

    return {
        "total": total,
        "low_ratio": round(low / total, 4),
        "borderline_ratio": round(borderline / total, 4),
        "high_ratio": round(high / total, 4),
        "very_low_ratio": round(very_low / total, 4),
    }


def detection_density_per_sample(
    total_detections: int,
    sampled_frames: int,
) -> float:
    if sampled_frames <= 0:
        return 0.0
    return round(total_detections / sampled_frames, 4)


def recommend_conf(
    avg_confidence: float,
    hist_stats: dict[str, float],
    current_conf: float = DEFAULT_CONF,
) -> tuple[float, str]:
    conf = DEFAULT_CONF
    reasons: list[str] = []

    low_ratio = hist_stats.get("low_ratio", 0.0)
    very_low_ratio = hist_stats.get("very_low_ratio", 0.0)
    high_ratio = hist_stats.get("high_ratio", 0.0)

    if very_low_ratio >= 0.25:
        conf += 0.10
        reasons.append(f"very_low_conf_ratio={very_low_ratio:.2f}")
    elif low_ratio >= 0.40:
        conf += 0.08
        reasons.append(f"low_conf_ratio={low_ratio:.2f}")
    elif low_ratio >= 0.28:
        conf += 0.05
        reasons.append(f"elevated_low_conf={low_ratio:.2f}")

    if avg_confidence < 0.50:
        conf = max(conf, 0.45)
        reasons.append(f"avg_confidence={avg_confidence:.3f}")
    elif avg_confidence < 0.58:
        conf = max(conf, 0.40)
        reasons.append(f"avg_confidence={avg_confidence:.3f}")

    if high_ratio >= 0.45 and low_ratio < 0.20 and avg_confidence >= 0.65:
        conf = min(conf, DEFAULT_CONF)
        reasons.append("strong_high_conf_distribution")

    conf = round(min(0.55, max(0.30, conf)) * 20) / 20
    if not reasons:
        reasons.append("default_balanced_recall")
    return conf, "; ".join(reasons)


def recommend_event_gap(
    detections_per_minute: float,
    duration_sec: float,
    detect_interval: int,
    density_per_sample: float,
) -> tuple[float, str]:
    if detections_per_minute >= 120:
        gap = 0.8
        reason = "high_detection_density"
    elif detections_per_minute >= 70:
        gap = 1.0
        reason = "moderate_high_density"
    elif detections_per_minute >= 35:
        gap = 1.2
        reason = "moderate_density"
    else:
        gap = 1.5
        reason = "sparse_faces"

    if duration_sec >= 600 and detections_per_minute < 45:
        gap = max(gap, 1.5)
        reason = "long_video_sparse_faces"

    if detect_interval >= 8:
        gap = min(2.0, gap + 0.2)
        reason += "; wide_detect_stride"

    if density_per_sample >= 0.35:
        gap = max(0.7, gap - 0.2)
        reason += "; dense_sampled_frames"

    gap = round(min(2.0, max(0.5, gap)), 2)
    return gap, reason


def recommend_padding(
    fps: float,
    detect_interval: int,
    detections_per_minute: float,
) -> tuple[dict[str, Any], str]:
    interval = max(1, detect_interval)
    padding_frames = interval * 2
    pre_sec = padding_frames / fps if fps > 0 else 0.25
    post_sec = padding_frames / fps if fps > 0 else 0.40

    pre_sec = max(0.20, min(0.30, pre_sec))
    post_sec = max(0.30, min(0.50, post_sec))

    if detections_per_minute < 50:
        post_sec = min(0.50, round(post_sec * 1.2, 3))
        reason = "sparse_faces_extra_post_padding"
    elif detections_per_minute >= 100:
        post_sec = max(post_sec, 0.33)
        reason = "high_density_standard_padding"
    else:
        reason = "auto_from_detect_interval"

    if interval >= 5 and fps > 0:
        post_sec = max(post_sec, round((padding_frames + 2) / fps, 3))
        post_sec = min(0.50, post_sec)

    pre_frames = max(1, int(round(pre_sec * fps))) if fps > 0 else padding_frames
    post_frames = max(1, int(round(post_sec * fps))) if fps > 0 else padding_frames

    return {
        "pre_padding_sec": round(pre_sec, 3),
        "post_padding_sec": round(post_sec, 3),
        "pre_padding_frames": pre_frames,
        "post_padding_frames": post_frames,
        "padding_frames_base": padding_frames,
    }, reason


def recommend_mosaic_level(
    avg_confidence: float,
    hist_stats: dict[str, float],
    detections_per_minute: float,
    duration_sec: float,
) -> tuple[str, str]:
    low_ratio = hist_stats.get("low_ratio", 0.0)

    if detections_per_minute >= 90 or duration_sec >= 300:
        return "extreme", "privacy_first_factory_default"

    if avg_confidence < 0.60 or low_ratio >= 0.30:
        return "extreme", "uncertain_or_borderline_detections"

    if avg_confidence >= 0.68 and hist_stats.get("high_ratio", 0.0) >= 0.35:
        return "high", "stable_high_confidence_faces"

    return DEFAULT_MOSAIC_LEVEL, "default_privacy_pipeline"


def recommend_parameters(
    video: VideoStats,
    detection: DetectionStats,
) -> dict[str, Any]:
    hist_stats = analyze_histogram(detection.confidence_histogram)
    density_per_sample = detection_density_per_sample(
        detection.total_detections,
        video.sampled_frames,
    )

    conf, conf_reason = recommend_conf(
        detection.avg_confidence,
        hist_stats,
        detection.conf_threshold,
    )
    gap, gap_reason = recommend_event_gap(
        detection.detections_per_minute,
        video.duration_sec,
        video.detect_interval,
        density_per_sample,
    )
    padding, pad_reason = recommend_padding(
        video.fps,
        video.detect_interval,
        detection.detections_per_minute,
    )
    mosaic, mosaic_reason = recommend_mosaic_level(
        detection.avg_confidence,
        hist_stats,
        detection.detections_per_minute,
        video.duration_sec,
    )

    if mosaic not in MOSAIC_LEVELS:
        mosaic = DEFAULT_MOSAIC_LEVEL

    return {
        "recommended_conf": conf,
        "recommended_event_gap": gap,
        "recommended_padding": padding,
        "recommended_mosaic_level": mosaic,
        "inputs": {
            "video": {
                "fps": video.fps,
                "total_frames": video.total_frames,
                "duration_sec": video.duration_sec,
                "sampled_frames": video.sampled_frames,
                "detect_interval": video.detect_interval,
                "width": video.width,
                "height": video.height,
            },
            "detection": {
                "total_detections": detection.total_detections,
                "avg_confidence": detection.avg_confidence,
                "detections_per_minute": detection.detections_per_minute,
                "detections_per_sampled_frame": density_per_sample,
                "confidence_histogram": detection.confidence_histogram,
                "histogram_analysis": hist_stats,
            },
        },
        "rationale": {
            "recommended_conf": conf_reason,
            "recommended_event_gap": gap_reason,
            "recommended_padding": pad_reason,
            "recommended_mosaic_level": mosaic_reason,
        },
    }


def recommend_from_detection_summary(summary: dict) -> dict[str, Any]:
    video = VideoStats.from_summary(summary)
    detection = DetectionStats.from_summary(summary)
    return recommend_parameters(video, detection)


def load_detection_summary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recommend_from_detection_dir(detection_dir: str) -> dict[str, Any]:
    detection_dir = os.path.abspath(detection_dir)
    summary_path = os.path.join(detection_dir, "detection_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing detection_summary.json in {detection_dir}")
    summary = load_detection_summary(summary_path)
    out = recommend_from_detection_summary(summary)
    out["detection_dir"] = detection_dir
    out["video_path"] = summary.get("video")
    return out


def save_recommendations(detection_dir: str, recommendations: dict) -> str:
    path = os.path.join(os.path.abspath(detection_dir), "tuning_recommendations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recommend conf / event_gap / padding / mosaic_level from detection stats",
    )
    p.add_argument(
        "--detection-dir",
        help="Phase-1 output dir containing detection_summary.json",
    )
    p.add_argument(
        "--summary",
        help="Path to detection_summary.json (alternative to --detection-dir)",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Write tuning_recommendations.json into detection dir",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.detection_dir and not args.summary:
        print("[error] Provide --detection-dir or --summary", file=sys.stderr)
        return 1

    try:
        if args.detection_dir:
            rec = recommend_from_detection_dir(args.detection_dir)
            detection_dir = os.path.abspath(args.detection_dir)
        else:
            summary = load_detection_summary(os.path.abspath(args.summary))
            rec = recommend_from_detection_summary(summary)
            detection_dir = os.path.dirname(os.path.abspath(args.summary))
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(json.dumps({
        "recommended_conf": rec["recommended_conf"],
        "recommended_event_gap": rec["recommended_event_gap"],
        "recommended_padding": rec["recommended_padding"],
        "recommended_mosaic_level": rec["recommended_mosaic_level"],
    }, ensure_ascii=False, indent=2))

    if args.write:
        path = save_recommendations(detection_dir, rec)
        print(f"[done] {path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
