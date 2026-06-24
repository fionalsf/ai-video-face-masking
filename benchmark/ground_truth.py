"""Ground truth + evaluation clip schema for benchmark framework."""

from __future__ import annotations

import json
import os
from typing import Any

GT_SCHEMA_VERSION = "benchmark_gt_v1"
CLIPS_SCHEMA_VERSION = "benchmark_clips_v1"


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, doc: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def validate_ground_truth(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not doc.get("video"):
        errors.append("missing video path")
    events = doc.get("events") or []
    for i, ev in enumerate(events):
        if "start_time" not in ev or "end_time" not in ev:
            errors.append(f"events[{i}] missing start_time/end_time")
        if ev.get("should_mask") is False:
            continue
        if float(ev.get("end_time", 0)) <= float(ev.get("start_time", 0)):
            errors.append(f"events[{i}] invalid interval")
    clips = doc.get("clips") or []
    for i, c in enumerate(clips):
        if "clip_id" not in c:
            errors.append(f"clips[{i}] missing clip_id")
        if float(c.get("end_time", 0)) <= float(c.get("start_time", 0)):
            errors.append(f"clips[{i}] invalid interval")
    if not events and not clips:
        if not doc.get("duration_sec") and not doc.get("clips_ref"):
            errors.append("ground truth must include events, clips[], duration_sec, or clips_ref")
    return errors


def load_ground_truth(path: str) -> dict[str, Any]:
    doc = load_json(path)
    errs = validate_ground_truth(doc)
    if errs:
        raise ValueError(f"Invalid ground truth {path}: " + "; ".join(errs))
    return doc


def load_clips(path: str) -> list[dict[str, Any]]:
    doc = load_json(path)
    clips = doc.get("clips") or []
    if not clips:
        raise ValueError(f"No clips in {path}")
    return clips


def clips_from_ground_truth(gt: dict[str, Any]) -> list[dict[str, Any]]:
    if gt.get("clips"):
        return gt["clips"]
    duration = float(gt.get("duration_sec") or 0)
    if duration <= 0 and gt.get("events"):
        duration = max(float(e["end_time"]) for e in gt["events"])
    if duration <= 0:
        raise ValueError("Cannot infer clips: provide clips[] or duration_sec in ground truth")
    return generate_uniform_clips(duration, target_count=30)


def generate_uniform_clips(
    duration_sec: float,
    *,
    target_count: int = 30,
    min_clip_sec: float = 10.0,
    max_clip_sec: float = 45.0,
) -> list[dict[str, Any]]:
    target_count = max(20, min(50, target_count))
    clip_len = duration_sec / target_count
    clip_len = max(min_clip_sec, min(max_clip_sec, clip_len))
    clips: list[dict[str, Any]] = []
    t = 0.0
    idx = 1
    while t < duration_sec - 0.01:
        end = min(duration_sec, t + clip_len)
        clips.append({
            "clip_id": f"clip_{idx:03d}",
            "start_time": round(t, 3),
            "end_time": round(end, 3),
            "duration_sec": round(end - t, 3),
            "label": f"segment_{idx}",
        })
        t = end
        idx += 1
    return clips


def gt_events_for_clip(gt: dict[str, Any], clip: dict[str, Any]) -> list[dict]:
    cs = float(clip["start_time"])
    ce = float(clip["end_time"])
    out: list[dict] = []
    for ev in gt.get("events") or []:
        if float(ev["end_time"]) <= cs or float(ev["start_time"]) >= ce:
            continue
        out.append(ev)
    return out
