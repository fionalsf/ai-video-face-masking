"""Review-tier candidates for aggressive edge/partial-face fallback."""

from __future__ import annotations

from statistics import mean

import cv2

from event_builder import FaceEvent, TIER_REVIEW
from render import _edge_partial_face_boxes


def _bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlaps_existing(frame_idx: int, bbox: list[float], existing_by_frame: dict[int, list[list[float]]]) -> bool:
    for existing in existing_by_frame.get(frame_idx, []):
        if _bbox_iou(bbox, existing) >= 0.15:
            return True
    return False


def _merge_candidates(candidates: list[dict], *, max_gap_frames: int) -> list[list[dict]]:
    if not candidates:
        return []
    chunks: list[list[dict]] = [[candidates[0]]]
    for cand in candidates[1:]:
        prev = chunks[-1][-1]
        if cand["frame"] - prev["frame"] <= max_gap_frames:
            chunks[-1].append(cand)
        else:
            chunks.append([cand])
    return chunks


def build_edge_review_candidates(
    video_path: str,
    meta: dict,
    existing_events: list[FaceEvent],
    *,
    stride: int = 5,
    min_hits: int = 1,
    max_gap_frames: int | None = None,
) -> list[FaceEvent]:
    """Detect risky partial edge-face regions and return Review-tier events.

    These candidates are intentionally not auto-confirmed: they are meant to
    catch close-up partial faces while remaining rejectable in the review UI.
    """
    fps = float(meta.get("fps") or 30.0)
    total_frames = int(meta.get("frames") or 0)
    stride = max(1, int(stride))
    max_gap_frames = max_gap_frames or max(stride * 6, int(round(fps)))

    existing_by_frame: dict[int, list[list[float]]] = {}
    for ev in existing_events:
        for pt in ev.trajectory:
            existing_by_frame.setdefault(int(pt["frame"]), []).append(list(pt["bbox"]))

    cap = cv2.VideoCapture(video_path)
    candidates: list[dict] = []
    frame_idx = 0
    try:
        while total_frames <= 0 or frame_idx < total_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % stride == 0:
                boxes = _edge_partial_face_boxes(frame)
                for box in boxes:
                    if _overlaps_existing(frame_idx, box, existing_by_frame):
                        continue
                    candidates.append({
                        "frame": frame_idx,
                        "t": round(frame_idx / fps, 3) if fps > 0 else 0.0,
                        "bbox": [round(float(v), 1) for v in box],
                        "conf": 0.5,
                    })
            frame_idx += 1
    finally:
        cap.release()

    events: list[FaceEvent] = []
    event_num = 0
    for chunk in _merge_candidates(candidates, max_gap_frames=max_gap_frames):
        if len(chunk) < min_hits:
            continue
        event_num += 1
        confs = [float(c["conf"]) for c in chunk]
        start_frame = int(chunk[0]["frame"])
        end_frame = int(chunk[-1]["frame"])
        events.append(FaceEvent(
            event_id=f"bevt_edge_{event_num:04d}",
            track_id=-1000 - event_num,
            tier=TIER_REVIEW,
            start_time=round(start_frame / fps, 3) if fps > 0 else 0.0,
            end_time=round(end_frame / fps, 3) if fps > 0 else 0.0,
            start_frame=start_frame,
            end_frame=end_frame,
            avg_confidence=round(mean(confs), 4),
            peak_confidence=round(max(confs), 4),
            detection_count=len(chunk),
            trajectory=chunk,
            rule_hints=["edge_partial_face_candidate", "review_required"],
        ))
    return events
