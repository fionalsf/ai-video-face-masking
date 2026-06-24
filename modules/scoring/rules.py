"""Rule hints for scoring downgrade."""

from __future__ import annotations

from core.event import Event, TrackPoint


def box_cy_ratio(bbox: list[float], frame_h: int) -> float:
    return ((bbox[1] + bbox[3]) / 2.0) / max(1.0, frame_h)


def box_touches_edge(bbox: list[float], frame_w: int, frame_h: int, margin: float = 5.0) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= margin or y1 <= margin or x2 >= frame_w - margin or y2 >= frame_h - margin


def suggest_rule_hints(event: Event, frame_h: int, frame_w: int, peak_conf: float) -> list[str]:
    if not event.trajectory:
        return []
    peak_pt = max(event.trajectory, key=lambda p: p.conf)
    bbox = peak_pt.bbox
    hints: list[str] = []
    if box_cy_ratio(bbox, frame_h) > 0.60:
        hints.append("workzone_low")
    if box_touches_edge(bbox, frame_w, frame_h) and peak_conf < 0.85:
        hints.append("edge_clip")
    w = max(1.0, bbox[2] - bbox[0])
    h = max(1.0, bbox[3] - bbox[1])
    ar = w / h
    if ar < 0.45 or ar > 2.0:
        hints.append("aspect_ratio")
    return hints
