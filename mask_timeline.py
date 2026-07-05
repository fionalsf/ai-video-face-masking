"""Frozen mask-timeline proposal layer for v2 review/render flow.

This module turns face events into short, reviewable mask proposals.  The
proposal timeline is the contract between candidate generation, human review,
and final rendering: confirm/render should not recompute tracking, optical flow,
or bbox refinement after a reviewer has made a decision.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any

import cv2
import numpy as np

MASK_TIMELINE_NAME = "mask_timeline.json"
MASK_TIMELINE_DEBUG_NAME = "mask_timeline_debug.json"
SUPPRESSED_REVIEW_NAME = "suppressed_events.json"

STATUS_ACCEPTED = "accepted"
STATUS_ACCEPTED_FIRST_HALF = "accepted_first_half"
STATUS_ACCEPTED_SECOND_HALF = "accepted_second_half"
STATUS_REJECTED = "rejected"
STATUS_SKIPPED = "skipped"
ACCEPT_STATUSES = {STATUS_ACCEPTED, STATUS_ACCEPTED_FIRST_HALF, STATUS_ACCEPTED_SECOND_HALF}

MAIN_REVIEW_EDGE_MIN_DURATION = 2.0
MAIN_REVIEW_LOW_CONF_MIN_PEAK = 0.70
MAIN_REVIEW_LOW_CONF_EDGE_MIN_PEAK = 0.60


def _clip_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1:
        x2 = min(float(width), x1 + 2.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 2.0)
    return [x1, y1, x2, y2]


def _edge_aware_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = _clip_bbox(bbox, width, height)
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    edge = max(4.0, min(width, height) * 0.01)
    if x1 <= edge:
        x2 += bw * 0.25
    if x2 >= width - edge:
        x1 -= bw * 0.25
    if y1 <= edge:
        y2 += bh * 0.45
    if y2 >= height - edge:
        y1 -= bh * 0.30
    return _clip_bbox([x1, y1, x2, y2], width, height)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_distance_ratio(a: list[float], b: list[float]) -> float:
    acx, acy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
    bcx, bcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    scale = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)
    return (((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / scale


def _area_ratio(a: list[float], b: list[float]) -> float:
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    ba = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return max(aa, ba) / min(aa, ba)


def parse_review_decisions(path_or_doc: str | dict) -> dict[str, str]:
    if isinstance(path_or_doc, str):
        if not os.path.isfile(path_or_doc):
            return {}
        with open(path_or_doc, encoding="utf-8-sig") as f:
            doc = json.load(f)
    else:
        doc = path_or_doc

    if not isinstance(doc, dict):
        return {}
    events = doc.get("events")
    if isinstance(events, list):
        out: dict[str, str] = {}
        for row in events:
            eid = row.get("event_id")
            status = row.get("status") or row.get("review_status")
            if not eid or not status:
                continue
            if status == "confirmed_face":
                status = STATUS_ACCEPTED
            elif status == "rejected_fp":
                status = STATUS_REJECTED
            out[str(eid)] = str(status)
        return out

    meta_keys = {
        "events", "summary", "video", "review_dir", "started_at", "updated_at",
        "total_review_events", "decided_count",
    }
    return {
        str(k): str(v)
        for k, v in doc.items()
        if str(k) not in meta_keys and isinstance(v, str)
    }


def _split_trajectory(
    trajectory: list[dict],
    *,
    max_gap_frames: int,
    min_iou: float,
    max_center_jump: float,
    max_area_ratio: float,
) -> list[list[dict]]:
    if not trajectory:
        return []
    ordered = sorted(trajectory, key=lambda p: int(p["frame"]))
    groups: list[list[dict]] = [[ordered[0]]]
    for pt in ordered[1:]:
        prev = groups[-1][-1]
        gap = int(pt["frame"]) - int(prev["frame"])
        track_changed = (
            pt.get("track_id") is not None
            and prev.get("track_id") is not None
            and int(pt["track_id"]) != int(prev["track_id"])
        )
        b0 = [float(v) for v in prev["bbox"]]
        b1 = [float(v) for v in pt["bbox"]]
        discontinuous = (
            gap > max_gap_frames
            or track_changed
            or (_bbox_iou(b0, b1) < min_iou and _center_distance_ratio(b0, b1) > max_center_jump)
            or _area_ratio(b0, b1) > max_area_ratio
        )
        if discontinuous:
            groups.append([pt])
        else:
            groups[-1].append(pt)
    return groups


def _append_entry(
    entries: dict[int, list[dict]],
    *,
    frame: int,
    fps: float,
    proposal_id: str,
    source_event_id: str,
    source_tier: str,
    track_id: int | None,
    bbox: list[float],
    confidence: float | None,
) -> None:
    entries[int(frame)].append({
        "frame": int(frame),
        "timestamp": round(int(frame) / fps if fps > 0 else 0.0, 3),
        "proposal_id": proposal_id,
        "event_id": proposal_id,
        "source_event_id": source_event_id,
        "source_tier": source_tier,
        "track_id": track_id,
        "bbox": [round(float(v), 1) for v in bbox],
        "confidence": round(float(confidence), 4) if confidence is not None else None,
    })


def _interpolate_segment(
    segment: list[dict],
    *,
    proposal_id: str,
    source_event_id: str,
    source_tier: str,
    fps: float,
    total_frames: int,
    width: int,
    height: int,
    extend_frames: int,
    max_interpolate_gap: int,
    edge_aware: bool,
) -> tuple[list[dict], dict]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    normalized = []
    for pt in segment:
        bbox = [float(v) for v in pt["bbox"]]
        bbox = _edge_aware_bbox(bbox, width, height) if edge_aware else _clip_bbox(bbox, width, height)
        normalized.append({**pt, "bbox": bbox})

    for pt in normalized:
        _append_entry(
            by_frame,
            frame=int(pt["frame"]),
            fps=fps,
            proposal_id=proposal_id,
            source_event_id=source_event_id,
            source_tier=source_tier,
            track_id=pt.get("track_id"),
            bbox=pt["bbox"],
            confidence=pt.get("conf"),
        )

    for i in range(len(normalized) - 1):
        p0, p1 = normalized[i], normalized[i + 1]
        f0, f1 = int(p0["frame"]), int(p1["frame"])
        gap = f1 - f0
        if gap <= 1 or gap > max_interpolate_gap:
            continue
        b0 = np.asarray(p0["bbox"], dtype=np.float64)
        b1 = np.asarray(p1["bbox"], dtype=np.float64)
        c0 = float(p0.get("conf") or 0.0)
        c1 = float(p1.get("conf") or c0)
        for frame in range(f0 + 1, f1):
            t = (frame - f0) / gap
            _append_entry(
                by_frame,
                frame=frame,
                fps=fps,
                proposal_id=proposal_id,
                source_event_id=source_event_id,
                source_tier=source_tier,
                track_id=p0.get("track_id"),
                bbox=(b0 + (b1 - b0) * t).tolist(),
                confidence=c0 + (c1 - c0) * t,
            )

    first = normalized[0]
    last = normalized[-1]
    first_frame = int(first["frame"])
    last_frame = int(last["frame"])
    start = max(0, first_frame - extend_frames)
    end = min(total_frames - 1, last_frame + extend_frames) if total_frames > 0 else last_frame + extend_frames
    for frame in range(start, first_frame):
        _append_entry(
            by_frame,
            frame=frame,
            fps=fps,
            proposal_id=proposal_id,
            source_event_id=source_event_id,
            source_tier=source_tier,
            track_id=first.get("track_id"),
            bbox=first["bbox"],
            confidence=first.get("conf"),
        )
    for frame in range(last_frame + 1, end + 1):
        _append_entry(
            by_frame,
            frame=frame,
            fps=fps,
            proposal_id=proposal_id,
            source_event_id=source_event_id,
            source_tier=source_tier,
            track_id=last.get("track_id"),
            bbox=last["bbox"],
            confidence=last.get("conf"),
        )

    entries = [row for frame in sorted(by_frame) for row in by_frame[frame]]
    debug = {
        "detection_count": len(segment),
        "timeline_frame_count": len({int(e["frame"]) for e in entries}),
        "start_frame": min(int(e["frame"]) for e in entries) if entries else first_frame,
        "end_frame": max(int(e["frame"]) for e in entries) if entries else last_frame,
    }
    return entries, debug


def _requires_explicit_accept(ev: dict, meta: dict) -> bool:
    hints = set(ev.get("rule_hints") or [])
    if hints & {"cross_presence_span", "low_conf_review_candidate"}:
        return True
    traj = ev.get("trajectory") or []
    if any(float(p.get("conf", 1.0)) < 0.55 for p in traj):
        return True
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    for p in traj:
        bbox = p.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if width > 0 and height > 0:
            area = (bw * bh) / max(1, width * height)
            edge = x1 <= 2 or y1 <= 12 or x2 >= width - 2 or y2 >= height - 2
            if edge and area > 0.03:
                return True
    return False


def _event_duration(ev: dict) -> float:
    return max(0.0, float(ev.get("end_time", 0.0)) - float(ev.get("start_time", 0.0)))


def _include_low_conf_for_review(ev: dict, runtime: dict) -> bool:
    if not bool(runtime.get("timeline_low_conf_review", True)):
        return False
    peak = float(ev.get("peak_confidence") or 0.0)
    duration = _event_duration(ev)
    hints = set(ev.get("rule_hints") or [])
    min_peak = float(runtime.get("low_conf_bridge_min_peak") or 0.35)
    max_duration = float(runtime.get("low_conf_bridge_max_duration") or 3.0)
    if peak >= min_peak and duration <= max_duration:
        return True
    if "edge_clip" in hints and peak >= float(runtime.get("low_conf_standalone_min_peak") or 0.45):
        return True
    return False


def build_mask_timeline(
    video_path: str,
    events: list[dict],
    meta: dict,
    runtime: dict | None = None,
) -> dict:
    runtime = runtime or {}
    fps = float(meta["fps"])
    total_frames = int(meta.get("frames") or 0)
    width = int(meta["width"])
    height = int(meta["height"])
    detect_interval = max(1, int(runtime.get("interval") or 1))
    split_gap = int(runtime.get("timeline_split_gap_frames") or max(8, detect_interval * 4))
    max_interpolate_gap = int(runtime.get("timeline_max_interpolate_gap") or max(6, detect_interval * 3))
    extend_frames = int(runtime.get("timeline_extend_frames") or min(10, max(2, detect_interval * 2)))

    proposals: list[dict] = []
    all_entries: list[dict] = []
    debug_events: list[dict] = []

    for ev in events:
        tier = str(ev.get("tier") or "")
        if tier == "low_conf" and _include_low_conf_for_review(ev, runtime):
            pass
        elif tier not in {"auto", "review"}:
            continue
        traj = ev.get("trajectory") or []
        if not traj:
            continue
        source_event_id = str(ev.get("event_id"))
        groups = _split_trajectory(
            traj,
            max_gap_frames=split_gap,
            min_iou=float(runtime.get("timeline_split_min_iou") or 0.03),
            max_center_jump=float(runtime.get("timeline_split_center_jump") or 1.10),
            max_area_ratio=float(runtime.get("timeline_split_area_ratio") or 4.0),
        )
        for idx, group in enumerate(groups, start=1):
            proposal_id = f"mask_{source_event_id}_s{idx:03d}"
            is_edge_partial = "edge_partial_face_candidate" in set(ev.get("rule_hints") or [])
            entries, dbg = _interpolate_segment(
                group,
                proposal_id=proposal_id,
                source_event_id=source_event_id,
                source_tier=tier,
                fps=fps,
                total_frames=total_frames,
                width=width,
                height=height,
                extend_frames=0 if is_edge_partial else extend_frames,
                max_interpolate_gap=max_interpolate_gap,
                edge_aware=not is_edge_partial,
            )
            if not entries:
                continue
            frames = [int(e["frame"]) for e in entries]
            confs = [float(e["confidence"]) for e in entries if e.get("confidence") is not None]
            start_frame, end_frame = min(frames), max(frames)
            proposal = {
                "proposal_id": proposal_id,
                "event_id": proposal_id,
                "source_event_id": source_event_id,
                "source_tier": tier,
                "review_status": "auto_safe" if tier == "auto" else "pending",
                "requires_explicit_accept": tier != "auto" or _requires_explicit_accept(ev, meta),
                "track_id": group[0].get("track_id") or ev.get("track_id") or ev.get("primary_track_id"),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": round(start_frame / fps if fps > 0 else 0.0, 3),
                "end_time": round(end_frame / fps if fps > 0 else 0.0, 3),
                "duration_sec": round((end_frame - start_frame + 1) / fps if fps > 0 else 0.0, 3),
                "frame_count": len(set(frames)),
                "detection_count": len(group),
                "peak_confidence": round(max(confs), 4) if confs else ev.get("peak_confidence"),
                "avg_confidence": round(sum(confs) / len(confs), 4) if confs else ev.get("avg_confidence"),
                "rule_hints": ev.get("rule_hints") or [],
            }
            proposals.append(proposal)
            all_entries.extend(entries)
            debug_events.append({**proposal, **dbg})

    all_entries.sort(key=lambda e: (int(e["frame"]), str(e["proposal_id"])))
    proposals.sort(key=lambda p: (int(p["start_frame"]), str(p["proposal_id"])))

    return {
        "schema": "mask_timeline.v2",
        "video": os.path.abspath(video_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "decision_policy": "auto_safe_plus_explicit_review_accept",
        "render_contract": "entries are frozen; final render must not track/refine/interpolate",
        "runtime": {
            "split_gap_frames": split_gap,
            "max_interpolate_gap": max_interpolate_gap,
            "extend_frames": extend_frames,
            "expand": runtime.get("expand"),
            "mosaic_block": runtime.get("mosaic_block"),
        },
        "proposal_count": len(proposals),
        "entry_count": len(all_entries),
        "proposals": proposals,
        "entries": all_entries,
        "_debug": {
            "events": debug_events,
            "summary": {
                "auto_proposals": sum(1 for p in proposals if p["source_tier"] == "auto"),
                "review_proposals": sum(1 for p in proposals if p["source_tier"] == "review"),
                "low_conf_review_proposals": sum(1 for p in proposals if p["source_tier"] == "low_conf"),
                "multi_segment_sources": len({
                    p["source_event_id"]
                    for p in proposals
                    if sum(1 for q in proposals if q["source_event_id"] == p["source_event_id"]) > 1
                }),
            },
        },
    }


def save_mask_timeline(output_dir: str, timeline: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, MASK_TIMELINE_NAME)
    payload = {k: v for k, v in timeline.items() if k != "_debug"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    debug_path = os.path.join(output_dir, MASK_TIMELINE_DEBUG_NAME)
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(timeline.get("_debug", {}), f, ensure_ascii=False, indent=2)
    return path


def load_mask_timeline(output_dir: str) -> dict | None:
    path = os.path.join(output_dir, MASK_TIMELINE_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _read_frame(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def _apply_mosaic(frame: np.ndarray, bbox: list[float], expand: float = 0.20) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    box = _clip_bbox([x1 - bw * expand, y1 - bh * expand, x2 + bw * expand, y2 + bh * expand], w, h)
    ix1, iy1, ix2, iy2 = [int(round(v)) for v in box]
    if ix2 <= ix1 or iy2 <= iy1:
        return
    roi = frame[iy1:iy2, ix1:ix2]
    rh, rw = roi.shape[:2]
    if rh <= 0 or rw <= 0:
        return
    sw, sh = max(1, rw // 12), max(1, rh // 12)
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_NEAREST)
    frame[iy1:iy2, ix1:ix2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    border = max(8, int(round(min(w, h) * 0.012)))
    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 0, 255), border)


def _imwrite_jpg(path: str, img, quality: int = 88) -> bool:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def _build_preview_sheet(
    timeline: dict,
    proposal_id: str,
    *,
    expand: float = 0.20,
    max_samples: int = 4,
    target_w: int = 420,
) -> np.ndarray | None:
    video_path = timeline["video"]
    proposal = next(
        (p for p in timeline.get("proposals", []) if str(p.get("proposal_id")) == str(proposal_id)),
        None,
    )
    if proposal is None:
        return None

    entries = sorted(
        (
            e for e in timeline.get("entries", [])
            if str(e.get("proposal_id")) == str(proposal_id)
        ),
        key=lambda e: int(e["frame"]),
    )
    if not entries:
        return None

    sample_indices = np.linspace(0, len(entries) - 1, min(max_samples, len(entries)), dtype=int)
    sample_frames: list[dict] = []
    seen_frames: set[int] = set()
    for sample_i in sample_indices:
        entry = entries[int(sample_i)]
        frame_idx = int(entry["frame"])
        if frame_idx in seen_frames:
            continue
        seen_frames.add(frame_idx)
        sample_frames.append(entry)

    tiles = []
    cap = cv2.VideoCapture(video_path)
    try:
        for entry in sample_frames:
            frame = _read_frame(cap, int(entry["frame"]))
            if frame is None:
                continue
            _apply_mosaic(frame, entry["bbox"], expand=expand)
            label = f"{proposal_id}  {float(entry.get('timestamp') or 0.0):.2f}s"
            cv2.putText(
                frame,
                label,
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            h, w = frame.shape[:2]
            target_h = max(1, int(round(h * target_w / w)))
            tiles.append(cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()

    if not tiles:
        return None

    rows = []
    for i in range(0, len(tiles), 2):
        row_tiles = tiles[i:i + 2]
        if len(row_tiles) == 1:
            row_tiles.append(np.zeros_like(row_tiles[0]))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def generate_mask_review_contact_sheet(
    timeline: dict,
    review_dir: str,
    proposal_id: str,
    *,
    expand: float = 0.20,
    max_samples: int = 4,
    force: bool = False,
) -> str | None:
    sheet_dir = os.path.join(review_dir, "event_contact_sheet")
    os.makedirs(sheet_dir, exist_ok=True)
    path = os.path.join(sheet_dir, f"{proposal_id}.jpg")
    if os.path.isfile(path) and not force:
        return path

    sheet = _build_preview_sheet(
        timeline,
        proposal_id,
        expand=expand,
        max_samples=max_samples,
    )
    if sheet is None:
        return None
    return path if _imwrite_jpg(path, sheet) else None


def _main_review_filter(proposal: dict) -> tuple[bool, str]:
    """Return whether a proposal belongs in the default human-review queue."""
    hints = set(proposal.get("rule_hints") or [])
    tier = proposal.get("source_tier")
    duration = float(proposal.get("duration_sec") or 0.0)
    peak = float(proposal.get("peak_confidence") or 0.0)

    if "edge_partial_face_candidate" in hints:
        if duration >= MAIN_REVIEW_EDGE_MIN_DURATION:
            return True, "edge_partial_long"
        return False, "edge_partial_short_aggressive_fallback"

    if tier == "low_conf":
        if peak >= MAIN_REVIEW_LOW_CONF_MIN_PEAK:
            return True, "low_conf_high_peak"
        if "edge_clip" in hints and peak >= MAIN_REVIEW_LOW_CONF_EDGE_MIN_PEAK:
            return True, "low_conf_edge_clip"
        return False, "low_conf_below_main_review_threshold"

    return True, "ordinary_review"


def export_mask_review_pack(
    timeline: dict,
    review_dir: str,
    *,
    expand: float = 0.20,
    max_samples: int = 4,
    lazy_previews: bool = True,
) -> str:
    os.makedirs(review_dir, exist_ok=True)
    sheet_dir = os.path.join(review_dir, "event_contact_sheet")

    video_path = timeline["video"]
    entries_by_proposal: dict[str, list[dict]] = defaultdict(list)
    for entry in timeline.get("entries", []):
        entries_by_proposal[str(entry["proposal_id"])].append(entry)

    review_proposals = []
    suppressed = []
    for proposal in timeline.get("proposals", []):
        if proposal.get("source_tier") not in {"review", "low_conf"}:
            continue
        keep, reason = _main_review_filter(proposal)
        row = {**proposal, "review_filter_reason": reason}
        if keep:
            review_proposals.append(row)
        else:
            suppressed.append(row)

    pending = []
    for proposal in review_proposals:
        pid = proposal["proposal_id"]
        entries = entries_by_proposal.get(pid) or []
        if not entries:
            continue
        fname = f"{pid}.jpg"
        if not lazy_previews:
            os.makedirs(sheet_dir, exist_ok=True)
            generate_mask_review_contact_sheet(
                timeline,
                review_dir,
                pid,
                expand=expand,
                max_samples=max_samples,
                force=True,
            )
        pending.append({
            **proposal,
            "event_id": pid,
            "previews": None,
            "contact_sheet": os.path.join("event_contact_sheet", fname).replace("\\", "/"),
            "preview_mode": "lazy" if lazy_previews else "eager",
        })

    payload = {
        "schema": "mask_review_pack.v2",
        "video": os.path.abspath(video_path),
        "timeline": MASK_TIMELINE_NAME,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": len(pending),
        "suppressed_count": len(suppressed),
        "main_review_policy": {
            "edge_partial_min_duration_sec": MAIN_REVIEW_EDGE_MIN_DURATION,
            "low_conf_min_peak": MAIN_REVIEW_LOW_CONF_MIN_PEAK,
            "low_conf_edge_min_peak": MAIN_REVIEW_LOW_CONF_EDGE_MIN_PEAK,
            "lazy_previews": lazy_previews,
        },
        "events": pending,
    }
    path = os.path.join(review_dir, "pending_events.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    suppressed_path = os.path.join(review_dir, SUPPRESSED_REVIEW_NAME)
    with open(suppressed_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema": "mask_review_suppressed.v2",
            "video": os.path.abspath(video_path),
            "timeline": MASK_TIMELINE_NAME,
            "count": len(suppressed),
            "events": suppressed,
        }, f, ensure_ascii=False, indent=2)
    return path


def write_mask_confirmed_template(review_dir: str, pending_events: list[dict]) -> str:
    """Create a v2 decisions file and keep only decisions matching current proposals."""
    path = os.path.join(review_dir, "confirmed_events.json")
    pending_ids = {str(e["event_id"]) for e in pending_events}
    kept: dict[str, str] = {}
    if os.path.isfile(path):
        kept = {
            eid: status
            for eid, status in parse_review_decisions(path).items()
            if eid in pending_ids and status in (ACCEPT_STATUSES | {STATUS_REJECTED, STATUS_SKIPPED})
        }
    payload = {
        "schema": "mask_review_decisions.v2",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "decision_unit": "mask_timeline_proposal",
        "total_review_events": len(pending_ids),
        "decided_count": len(kept),
        **dict(sorted(kept.items())),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def select_render_entries(
    timeline: dict,
    decisions: dict[str, str],
) -> tuple[dict[int, list[list[float]]], dict[str, Any]]:
    proposals = {p["proposal_id"]: p for p in timeline.get("proposals", [])}
    selected: set[str] = set()
    partial: dict[str, str] = {}
    rejected = 0
    unreviewed = 0
    for pid, proposal in proposals.items():
        tier = proposal.get("source_tier")
        decision = decisions.get(pid)
        if tier == "auto":
            selected.add(pid)
        elif decision == STATUS_ACCEPTED:
            selected.add(pid)
        elif decision in {STATUS_ACCEPTED_FIRST_HALF, STATUS_ACCEPTED_SECOND_HALF}:
            selected.add(pid)
            partial[pid] = decision
        elif decision == STATUS_REJECTED:
            rejected += 1
        else:
            unreviewed += 1

    render: dict[int, list[list[float]]] = defaultdict(list)
    mid_by_pid = {
        pid: (float(proposal["start_frame"]) + float(proposal["end_frame"])) * 0.5
        for pid, proposal in proposals.items()
        if pid in partial
    }
    for entry in timeline.get("entries", []):
        pid = entry.get("proposal_id")
        if pid not in selected:
            continue
        decision = partial.get(pid)
        if decision == STATUS_ACCEPTED_FIRST_HALF and int(entry["frame"]) > mid_by_pid[pid]:
            continue
        if decision == STATUS_ACCEPTED_SECOND_HALF and int(entry["frame"]) < mid_by_pid[pid]:
            continue
        if pid in selected:
            render[int(entry["frame"])].append([float(v) for v in entry["bbox"]])

    stats = {
        "selected_proposals": len(selected),
        "auto_selected": sum(1 for pid in selected if proposals[pid].get("source_tier") == "auto"),
        "review_selected": sum(1 for pid in selected if proposals[pid].get("source_tier") in {"review", "low_conf"}),
        "review_partial_selected": len(partial),
        "review_rejected": rejected,
        "review_unreviewed_skipped": unreviewed,
        "render_frames": len(render),
    }
    return dict(render), stats
