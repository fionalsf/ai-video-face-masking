"""Promote likely low-confidence faces into the review workflow."""

from __future__ import annotations

from typing import Any

import cv2

from event_builder import FaceEvent, TIER_LOW_CONF, TIER_REVIEW


PROMOTION_HINT = "low_conf_review_candidate"


def _event_duration(ev: FaceEvent) -> float:
    return max(0.0, float(ev.end_time) - float(ev.start_time))


def _midpoint(ev: FaceEvent) -> dict[str, Any] | None:
    if not ev.trajectory:
        return None
    mid_t = (float(ev.start_time) + float(ev.end_time)) * 0.5
    return min(ev.trajectory, key=lambda p: abs(float(p.get("t", 0.0)) - mid_t))


def _skin_ratio(crop) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mask_a = (y > 45) & (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 140)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_b = (v > 45) & (s > 18) & ((h <= 25) | (h >= 165))
    return float((mask_a | mask_b).mean())


def is_promotable_low_conf(
    ev: FaceEvent,
    cap,
    meta: dict,
    runtime: dict | None = None,
) -> bool:
    runtime = runtime or {}
    hints = set(ev.rule_hints or [])
    if ev.tier != TIER_LOW_CONF or "edge_clip" not in hints:
        return False
    if float(ev.peak_confidence or 0.0) < float(runtime.get("low_conf_standalone_min_peak") or 0.45):
        return False
    if _event_duration(ev) > float(runtime.get("low_conf_standalone_max_duration") or 2.5):
        return False

    point = _midpoint(ev)
    if point is None:
        return False
    bbox = point.get("bbox")
    if not bbox or len(bbox) != 4:
        return False

    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    if width <= 0 or height <= 0:
        return False

    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    box_w, box_h = max(0, x2 - x1), max(0, y2 - y1)
    if box_w <= 0 or box_h <= 0:
        return False

    area_ratio = (box_w * box_h) / max(1, width * height)
    aspect = box_w / max(1, box_h)
    max_area = float(runtime.get("low_conf_standalone_max_area") or 0.075)
    if not (0.0015 <= area_ratio <= max_area):
        return False
    if not (0.45 <= aspect <= 2.7):
        return False
    if y2 >= height * 0.96 and y1 >= height * 0.72:
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(point.get("frame", 0)))
    ok, frame = cap.read()
    if not ok:
        return False
    crop = frame[y1:y2, x1:x2]
    min_skin = float(runtime.get("low_conf_standalone_min_skin") or 0.16)
    return _skin_ratio(crop) >= min_skin


def promote_low_conf_review_candidates(
    video_path: str,
    events: list[FaceEvent],
    meta: dict,
    runtime: dict | None = None,
) -> int:
    if runtime is not None and not bool(runtime.get("low_conf_standalone", True)):
        return 0

    cap = cv2.VideoCapture(video_path)
    promoted = 0
    try:
        for ev in events:
            if not is_promotable_low_conf(ev, cap, meta, runtime):
                continue
            ev.tier = TIER_REVIEW
            if PROMOTION_HINT not in ev.rule_hints:
                ev.rule_hints.append(PROMOTION_HINT)
            promoted += 1
    finally:
        cap.release()
    return promoted
