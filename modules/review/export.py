"""Review pack export: 3 keyframes + pending_events.json."""

from __future__ import annotations

import json
import os
import time

import cv2

from core.event import Event, Tier
from utils.draw import draw_detections


def _pick_keyframe_indices(n: int) -> tuple[int, int, int]:
    if n == 1:
        return 0, 0, 0
    if n == 2:
        return 0, 1, 1
    mid = n // 2
    return 0, mid, n - 1


def _read_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def export_review_pack(video_path: str, review_dir: str, review_events: list[Event]) -> str:
    previews_dir = os.path.join(review_dir, "previews")
    os.makedirs(previews_dir, exist_ok=True)
    pending = []
    for ev in review_events:
        d = ev.to_dict()
        n = len(ev.trajectory)
        i0, im, i1 = _pick_keyframe_indices(n)
        preview_paths = {}
        for label, idx in [("start", i0), ("mid", im), ("end", i1)]:
            pt = ev.trajectory[idx]
            frame = _read_frame(video_path, pt.frame)
            if frame is None:
                continue
            annotated = draw_detections(frame, [{"bbox": pt.bbox, "confidence": pt.conf}])
            fname = f"{ev.event_id}_{label}.jpg"
            fpath = os.path.join(previews_dir, fname)
            cv2.imwrite(fpath, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            preview_paths[label] = os.path.join("previews", fname).replace("\\", "/")
        d["previews"] = preview_paths
        d["review_status"] = "pending"
        pending.append(d)
    payload = {
        "video": os.path.abspath(video_path),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": len(pending),
        "events": pending,
    }
    path = os.path.join(review_dir, "pending_events.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def write_confirmed_events_template(review_dir: str, video_path: str) -> str:
    path = os.path.join(review_dir, "confirmed_events.json")
    if os.path.isfile(path):
        return path
    payload = {
        "video": os.path.abspath(video_path),
        "review_dir": os.path.abspath(review_dir),
        "started_at": None,
        "updated_at": None,
        "total_review_events": 0,
        "decided_count": 0,
        "summary": {"confirmed_face": 0, "rejected_fp": 0, "skipped": 0},
        "events": [],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def filter_review_events(events: list[Event]) -> list[Event]:
    return [e for e in events if e.tier == Tier.REVIEW.value]
