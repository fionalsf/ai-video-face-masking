"""Behavior-level Event Merge Layer — independent of presence segmentation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from statistics import mean
from typing import Any

from event_builder import (
    FaceEvent,
    TIER_AUTO,
    TIER_LOW_CONF,
    TIER_REVIEW,
    _make_event,
    resolve_temporal_padding,
)
from event_merge import (
    DEFAULT_CENTER_CONTINUITY_RATIO,
    DEFAULT_CONF_IOU_RELAX,
    DEFAULT_MERGE_IOU_THRESHOLD,
    SEGMENTATION_EVENTS_NAME,
    _combine_trajectory,
    _detection_gap_sec,
    _event_dict,
    _merge_tier,
    bbox_continuity,
    load_presence_events,
    rebuild_events_from_tracked,
)
from event_quality import score_event_dict
from gap_analysis import (
    DEFAULT_HARD_SPLIT_GAP_SEC,
    EVENT_SEGMENT_MAP_NAME,
    analyze_gaps,
    presence_segment_id,
)
from video_meta import get_video_meta, sec_to_timecode

BEHAVIOR_LAYER_LABEL = "behavior_merge_layer_v1"
BEHAVIOR_EVENTS_NAME = "behavior_events.json"
MERGE_TRACE_NAME = "merge_trace.json"

DEFAULT_BEHAVIOR_GAP_SEC = 8.0
MIN_BEHAVIOR_GAP_SEC = 5.0
MAX_BEHAVIOR_GAP_SEC = 8.0
BEHAVIOR_SEMANTIC = "人在画面中的连续行为段"
TARGET_EVENT_RANGE = (30, 80)


def load_presence_segment_map(output_dir: str) -> dict[str, dict]:
    path = os.path.join(os.path.abspath(output_dir), EVENT_SEGMENT_MAP_NAME)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return {row["event_id"]: row for row in doc.get("events") or []}


def _mini_event_from_det(det: dict, track_id: int) -> dict:
    return {
        "track_id": track_id,
        "avg_confidence": det["conf"],
        "peak_confidence": det["conf"],
        "trajectory": [{
            "t": det["t"],
            "frame": det["frame"],
            "bbox": det["bbox"],
            "conf": det["conf"],
        }],
    }


def should_merge_detection_pair(
    prev: dict,
    curr: dict,
    track_id: int,
    *,
    behavior_gap_sec: float = DEFAULT_BEHAVIOR_GAP_SEC,
    iou_threshold: float = DEFAULT_MERGE_IOU_THRESHOLD,
    center_ratio: float = DEFAULT_CENTER_CONTINUITY_RATIO,
    conf_iou_relax: float = DEFAULT_CONF_IOU_RELAX,
    prev_presence_id: str | None = None,
    curr_presence_id: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    detection_gap = round(float(curr["t"]) - float(prev["t"]), 3)
    if detection_gap > behavior_gap_sec:
        return False, {
            "reason": "gap_exceeded",
            "detection_gap_sec": detection_gap,
            "behavior_gap_sec": behavior_gap_sec,
            "cross_presence": _is_cross_presence_ids(prev_presence_id, curr_presence_id),
        }

    cont_ok, cont_meta = bbox_continuity(
        _mini_event_from_det(prev, track_id),
        _mini_event_from_det(curr, track_id),
        iou_threshold=iou_threshold,
        center_ratio=center_ratio,
        conf_iou_relax=conf_iou_relax,
    )
    cross_presence = _is_cross_presence_ids(prev_presence_id, curr_presence_id)
    if not cont_ok:
        return False, {
            "reason": "bbox_discontinuity",
            "detection_gap_sec": detection_gap,
            "cross_presence": cross_presence,
            **cont_meta,
        }

    return True, {
        "reason": "behavior_merged",
        "detection_gap_sec": detection_gap,
        "cross_presence": cross_presence,
        "from_presence_segment_id": prev_presence_id,
        "to_presence_segment_id": curr_presence_id,
        **cont_meta,
    }


def _is_cross_presence_ids(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return a != b


def _presence_id_for_detection(
    det: dict,
    presence_groups: list[list[dict]],
    track_id: int,
) -> str | None:
    frame = int(det["frame"])
    for idx, group in enumerate(presence_groups):
        if group[0]["frame"] <= frame <= group[-1]["frame"]:
            for g in group:
                if int(g["frame"]) == frame:
                    return presence_segment_id(track_id, idx)
    return None


def _overlap_segmentation_ids(
    behavior: dict,
    segmentation_events: list[dict],
) -> list[str]:
    tid = int(behavior["track_id"])
    start = float(behavior["start_time"])
    end = float(behavior["end_time"])
    ids: list[str] = []
    for ev in segmentation_events:
        if int(ev["track_id"]) != tid:
            continue
        if float(ev["end_time"]) < start or float(ev["start_time"]) > end:
            continue
        ids.append(ev["event_id"])
    return ids


def build_behavior_events_from_tracked(
    tracked: list[dict],
    *,
    behavior_gap_sec: float = DEFAULT_BEHAVIOR_GAP_SEC,
    iou_threshold: float = DEFAULT_MERGE_IOU_THRESHOLD,
    center_ratio: float = DEFAULT_CENTER_CONTINUITY_RATIO,
    conf_iou_relax: float = DEFAULT_CONF_IOU_RELAX,
    fps: float = 30.0,
    frame_h: int = 1080,
    frame_w: int = 1920,
    total_frames: int | None = None,
    detect_interval: int | None = None,
    segmentation_events: list[dict] | None = None,
    hard_split_gap_sec: float = DEFAULT_HARD_SPLIT_GAP_SEC,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Rebuild behavior events from tracked detections (no presence boundary constraint)."""
    from rules import suggest_rule_hints

    behavior_gap_sec = max(MIN_BEHAVIOR_GAP_SEC, min(MAX_BEHAVIOR_GAP_SEC, behavior_gap_sec))
    gap_result = analyze_gaps(tracked, hard_split_gap_sec=hard_split_gap_sec, fps=fps)
    pre_pad, post_pad = resolve_temporal_padding(
        fps, detect_interval=detect_interval,
    )

    by_track: dict[int, list[dict]] = {}
    for d in tracked:
        by_track.setdefault(int(d["track_id"]), []).append(d)
    for tid in by_track:
        by_track[tid].sort(key=lambda x: int(x["frame"]))

    behavior_events: list[dict[str, Any]] = []
    edge_decisions: list[dict[str, Any]] = []
    cross_presence_traces: list[dict[str, Any]] = []
    behavior_num = 0
    cross_presence_edge_count = 0

    for track_id in sorted(by_track):
        seq = by_track[track_id]
        presence_groups = gap_result.presence_by_track.get(track_id, [seq])
        chunks: list[list[dict]] = []
        chunk_edges_list: list[list[dict[str, Any]]] = []
        current_chunk: list[dict] = [seq[0]]
        current_edges: list[dict[str, Any]] = []

        for det in seq[1:]:
            prev = current_chunk[-1]
            prev_pres = _presence_id_for_detection(prev, presence_groups, track_id)
            curr_pres = _presence_id_for_detection(det, presence_groups, track_id)
            ok, meta = should_merge_detection_pair(
                prev, det, track_id,
                behavior_gap_sec=behavior_gap_sec,
                iou_threshold=iou_threshold,
                center_ratio=center_ratio,
                conf_iou_relax=conf_iou_relax,
                prev_presence_id=prev_pres,
                curr_presence_id=curr_pres,
            )
            edge = {
                "track_id": track_id,
                "from_frame": int(prev["frame"]),
                "to_frame": int(det["frame"]),
                "merged": ok,
                **meta,
            }
            edge_decisions.append(edge)
            if ok:
                current_chunk.append(det)
                current_edges.append(meta)
                if meta.get("cross_presence"):
                    cross_presence_edge_count += 1
            else:
                chunks.append(current_chunk)
                chunk_edges_list.append(current_edges)
                current_chunk = [det]
                current_edges = []

        chunks.append(current_chunk)
        chunk_edges_list.append(current_edges)

        for chunk, edges in zip(chunks, chunk_edges_list):
            behavior_num += 1
            face_ev = _make_event(
                behavior_num,
                track_id,
                chunk,
                frame_h,
                frame_w,
                suggest_rule_hints,
                fps=fps,
                total_frames=total_frames,
                pre_padding_sec=pre_pad,
                post_padding_sec=post_pad,
            )
            pres_ids = sorted({
                pid for d in chunk
                if (pid := _presence_id_for_detection(d, presence_groups, track_id))
            })
            cross_presence = len(pres_ids) > 1
            source_seg_ids = (
                _overlap_segmentation_ids(face_ev.to_dict(), segmentation_events)
                if segmentation_events else []
            )
            be = _behavior_dict_from_face_event(
                face_ev,
                behavior_num,
                presence_segments_spanned=pres_ids,
                cross_presence_merge=cross_presence,
                source_segmentation_event_ids=source_seg_ids,
                merge_edges=edges,
            )
            behavior_events.append(be)
            if cross_presence:
                cross_presence_traces.append({
                    "behavior_event_id": be["behavior_event_id"],
                    "track_id": track_id,
                    "source_segmentation_event_ids": source_seg_ids,
                    "presence_segments_spanned": pres_ids,
                    "merge_edges": edges,
                })

    seg_count = len(segmentation_events) if segmentation_events else 0
    stats = {
        "source_mode": "track_native",
        "source_segmentation_count": seg_count,
        "behavior_event_count": len(behavior_events),
        "events_merged_away": max(0, seg_count - len(behavior_events)),
        "compression_ratio": round(seg_count / max(1, len(behavior_events)), 3) if seg_count else None,
        "behavior_gap_sec": behavior_gap_sec,
        "behavior_gap_range_sec": [MIN_BEHAVIOR_GAP_SEC, MAX_BEHAVIOR_GAP_SEC],
        "cross_presence_merge_count": len(cross_presence_traces),
        "cross_presence_edge_count": cross_presence_edge_count,
        "presence_boundary_hard_constraint": False,
        "track_count": len(by_track),
        "detection_count": len(tracked),
        "layer": BEHAVIOR_LAYER_LABEL,
        "semantic_definition": BEHAVIOR_SEMANTIC,
        "target_event_range": list(TARGET_EVENT_RANGE),
    }
    return behavior_events, edge_decisions, cross_presence_traces, stats


def _behavior_dict_from_face_event(
    face_ev: FaceEvent,
    behavior_num: int,
    *,
    presence_segments_spanned: list[str],
    cross_presence_merge: bool,
    source_segmentation_event_ids: list[str],
    merge_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    d = face_ev.to_dict()
    hints = list(d.get("rule_hints") or [])
    if len(source_segmentation_event_ids) > 1:
        hints.append(f"behavior_merge:{len(source_segmentation_event_ids)}")
    if cross_presence_merge:
        hints.append("cross_presence_merge")

    tier = d["tier"]
    review_status = (
        "confirmed_face" if tier == TIER_AUTO
        else "logged_only" if tier == TIER_LOW_CONF
        else "pending"
    )
    bevt_id = f"bevt_{behavior_num:04d}"
    return {
        "behavior_event_id": bevt_id,
        "event_id": bevt_id,
        "track_id": d["track_id"],
        "tier": tier,
        "review_status": review_status,
        "source_segmentation_event_ids": source_segmentation_event_ids,
        "source_event_count": len(source_segmentation_event_ids),
        "presence_segments_spanned": presence_segments_spanned,
        "cross_presence_merge": cross_presence_merge,
        "start_time": d["start_time"],
        "end_time": d["end_time"],
        "start_timecode": d["start_timecode"],
        "end_timecode": d["end_timecode"],
        "start_frame": d["start_frame"],
        "end_frame": d["end_frame"],
        "duration_sec": d["duration_sec"],
        "avg_confidence": d["avg_confidence"],
        "peak_confidence": d["peak_confidence"],
        "detection_count": d["detection_count"],
        "trajectory": d["trajectory"],
        "rule_hints": hints,
        "merge_edges": merge_edges,
    }


def enrich_with_presence_meta(
    events: list[FaceEvent | dict],
    segment_map: dict[str, dict],
) -> list[dict]:
    out: list[dict] = []
    for ev in events:
        d = dict(_event_dict(ev))
        meta = segment_map.get(d["event_id"], {})
        d["presence_segment_id"] = meta.get("presence_segment_id")
        d["presence_segment_index"] = meta.get("presence_segment_index")
        out.append(d)
    return out


def should_behavior_merge(
    prev: dict,
    curr: dict,
    *,
    behavior_gap_sec: float = DEFAULT_BEHAVIOR_GAP_SEC,
    iou_threshold: float = DEFAULT_MERGE_IOU_THRESHOLD,
    center_ratio: float = DEFAULT_CENTER_CONTINUITY_RATIO,
    conf_iou_relax: float = DEFAULT_CONF_IOU_RELAX,
) -> tuple[bool, dict[str, Any]]:
    if int(prev["track_id"]) != int(curr["track_id"]):
        return False, {"reason": "track_mismatch"}

    detection_gap = round(_detection_gap_sec(prev, curr), 3)
    if detection_gap > behavior_gap_sec:
        return False, {
            "reason": "gap_exceeded",
            "detection_gap_sec": detection_gap,
            "behavior_gap_sec": behavior_gap_sec,
            "cross_presence": _is_cross_presence(prev, curr),
        }

    cont_ok, cont_meta = bbox_continuity(
        prev, curr,
        iou_threshold=iou_threshold,
        center_ratio=center_ratio,
        conf_iou_relax=conf_iou_relax,
    )
    cross_presence = _is_cross_presence(prev, curr)
    if not cont_ok:
        return False, {
            "reason": "bbox_discontinuity",
            "detection_gap_sec": detection_gap,
            "cross_presence": cross_presence,
            **cont_meta,
        }

    return True, {
        "reason": "behavior_merged",
        "detection_gap_sec": detection_gap,
        "cross_presence": cross_presence,
        "from_presence_segment_id": prev.get("presence_segment_id"),
        "to_presence_segment_id": curr.get("presence_segment_id"),
        **cont_meta,
    }


def _is_cross_presence(prev: dict, curr: dict) -> bool:
    a = prev.get("presence_segment_id")
    b = curr.get("presence_segment_id")
    if not a or not b:
        return False
    return a != b


def _build_behavior_event(
    group: list[dict],
    behavior_num: int,
    merge_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    source_ids = [g["event_id"] for g in group]
    traj = _combine_trajectory(group)
    confs = [float(p["conf"]) for p in traj]
    peak = max(confs) if confs else 0.0
    avg_conf = round(mean(confs), 4) if confs else 0.0

    start_time = min(float(g["start_time"]) for g in group)
    end_time = max(float(g["end_time"]) for g in group)
    start_frame = min(int(g["start_frame"]) for g in group)
    end_frame = max(int(g["end_frame"]) for g in group)

    pres_ids = sorted({g.get("presence_segment_id") for g in group if g.get("presence_segment_id")})
    cross_presence = len(pres_ids) > 1

    hints: list[str] = []
    for g in group:
        for h in g.get("rule_hints") or []:
            if h not in hints:
                hints.append(h)
    if len(source_ids) > 1:
        hints.append(f"behavior_merge:{len(source_ids)}")
    if cross_presence:
        hints.append("cross_presence_merge")

    tier = _merge_tier(group)
    review_status = (
        "confirmed_face" if tier == TIER_AUTO
        else "logged_only" if tier == TIER_LOW_CONF
        else "pending"
    )

    return {
        "behavior_event_id": f"bevt_{behavior_num:04d}",
        "event_id": f"bevt_{behavior_num:04d}",
        "track_id": int(group[0]["track_id"]),
        "tier": tier,
        "review_status": review_status,
        "source_segmentation_event_ids": source_ids,
        "source_event_count": len(source_ids),
        "presence_segments_spanned": pres_ids,
        "cross_presence_merge": cross_presence,
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "start_timecode": sec_to_timecode(start_time),
        "end_timecode": sec_to_timecode(end_time),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "duration_sec": round(end_time - start_time, 3),
        "avg_confidence": avg_conf,
        "peak_confidence": round(peak, 4),
        "detection_count": len(traj),
        "trajectory": traj,
        "rule_hints": hints,
        "merge_edges": merge_edges,
    }


def merge_behavior_events(
    events: list[dict],
    *,
    behavior_gap_sec: float = DEFAULT_BEHAVIOR_GAP_SEC,
    iou_threshold: float = DEFAULT_MERGE_IOU_THRESHOLD,
    center_ratio: float = DEFAULT_CENTER_CONTINUITY_RATIO,
    conf_iou_relax: float = DEFAULT_CONF_IOU_RELAX,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    behavior_gap_sec = max(MIN_BEHAVIOR_GAP_SEC, min(MAX_BEHAVIOR_GAP_SEC, behavior_gap_sec))

    by_track: dict[int, list[dict]] = {}
    for ev in events:
        by_track.setdefault(int(ev["track_id"]), []).append(ev)
    for tid in by_track:
        by_track[tid].sort(key=lambda e: (e["start_time"], e["start_frame"]))

    behavior_events: list[dict[str, Any]] = []
    edge_decisions: list[dict[str, Any]] = []
    cross_presence_traces: list[dict[str, Any]] = []
    behavior_num = 0

    for track_id in sorted(by_track):
        track_events = by_track[track_id]
        group: list[dict] = []
        group_edges: list[dict[str, Any]] = []

        def flush_group() -> None:
            nonlocal behavior_num
            if not group:
                return
            behavior_num += 1
            be = _build_behavior_event(group, behavior_num, group_edges)
            behavior_events.append(be)
            if be["cross_presence_merge"]:
                cross_presence_traces.append({
                    "behavior_event_id": be["behavior_event_id"],
                    "track_id": track_id,
                    "source_segmentation_event_ids": be["source_segmentation_event_ids"],
                    "presence_segments_spanned": be["presence_segments_spanned"],
                    "merge_edges": list(group_edges),
                })
            group.clear()
            group_edges.clear()

        for ev in track_events:
            if not group:
                group.append(ev)
                continue
            ok, meta = should_behavior_merge(
                group[-1], ev,
                behavior_gap_sec=behavior_gap_sec,
                iou_threshold=iou_threshold,
                center_ratio=center_ratio,
                conf_iou_relax=conf_iou_relax,
            )
            edge = {
                "track_id": track_id,
                "from_event_id": group[-1]["event_id"],
                "to_event_id": ev["event_id"],
                "merged": ok,
                **meta,
            }
            edge_decisions.append(edge)
            if ok:
                group.append(ev)
                group_edges.append(meta)
            else:
                flush_group()
                group.append(ev)
        flush_group()

    stats = {
        "source_segmentation_count": len(events),
        "behavior_event_count": len(behavior_events),
        "events_merged_away": len(events) - len(behavior_events),
        "compression_ratio": round(len(events) / max(1, len(behavior_events)), 3),
        "behavior_gap_sec": behavior_gap_sec,
        "behavior_gap_range_sec": [MIN_BEHAVIOR_GAP_SEC, MAX_BEHAVIOR_GAP_SEC],
        "cross_presence_merge_count": len(cross_presence_traces),
        "presence_boundary_hard_constraint": False,
        "layer": BEHAVIOR_LAYER_LABEL,
        "semantic_definition": BEHAVIOR_SEMANTIC,
        "target_event_range": list(TARGET_EVENT_RANGE),
    }
    return behavior_events, edge_decisions, cross_presence_traces, stats


def _segmentation_event_dicts(events: list[FaceEvent | dict] | None) -> list[dict]:
    if not events:
        return []
    return [_event_dict(ev) for ev in events]


def _load_tracked(output_dir: str) -> list[dict]:
    path = os.path.join(os.path.abspath(output_dir), "tracked_detections.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_behavior_merge_layer(
    events: list[FaceEvent | dict] | None = None,
    *,
    output_dir: str | None = None,
    tracked: list[dict] | None = None,
    behavior_gap_sec: float = DEFAULT_BEHAVIOR_GAP_SEC,
    iou_threshold: float = DEFAULT_MERGE_IOU_THRESHOLD,
    center_ratio: float = DEFAULT_CENTER_CONTINUITY_RATIO,
    conf_iou_relax: float = DEFAULT_CONF_IOU_RELAX,
    video: str | None = None,
    fps: float | None = None,
    frame_h: int = 1080,
    frame_w: int = 1920,
    total_frames: int | None = None,
    detect_interval: int | None = None,
    segment_map: dict[str, dict] | None = None,
) -> dict[str, Any]:
    segmentation_payload = _segmentation_event_dicts(events)
    summary: dict[str, Any] = {}
    if output_dir:
        summary_path = os.path.join(os.path.abspath(output_dir), "detection_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)

    video = video or summary.get("video")
    fps = fps or (float(summary.get("fps") or 0) or None)
    detect_interval = detect_interval or int(summary.get("detect_interval") or 0) or None
    if video and os.path.isfile(video):
        meta = get_video_meta(video)
        frame_h, frame_w = meta["height"], meta["width"]
        total_frames = total_frames or meta["frames"]
        fps = fps or float(meta["fps"])

    if tracked is None and output_dir:
        tracked_path = os.path.join(os.path.abspath(output_dir), "tracked_detections.json")
        if os.path.isfile(tracked_path):
            tracked = _load_tracked(output_dir)

    if tracked:
        behavior_events, edge_decisions, cross_traces, stats = build_behavior_events_from_tracked(
            tracked,
            behavior_gap_sec=behavior_gap_sec,
            iou_threshold=iou_threshold,
            center_ratio=center_ratio,
            conf_iou_relax=conf_iou_relax,
            fps=float(fps or 30.0),
            frame_h=frame_h,
            frame_w=frame_w,
            total_frames=total_frames,
            detect_interval=detect_interval,
            segmentation_events=segmentation_payload or None,
        )
    elif segmentation_payload:
        if segment_map is None and output_dir:
            segment_map = load_presence_segment_map(output_dir)
        segment_map = segment_map or {}
        enriched = enrich_with_presence_meta(segmentation_payload, segment_map)
        behavior_events, edge_decisions, cross_traces, stats = merge_behavior_events(
            enriched,
            behavior_gap_sec=behavior_gap_sec,
            iou_threshold=iou_threshold,
            center_ratio=center_ratio,
            conf_iou_relax=conf_iou_relax,
        )
        stats["source_mode"] = "segmentation_merge_fallback"
    else:
        raise ValueError("behavior_merge_layer requires tracked detections or segmentation events")

    for be in behavior_events:
        be["event_quality_score"] = score_event_dict(be, fps=fps or 30.0)

    parameters = {
        "behavior_gap_sec": stats["behavior_gap_sec"],
        "behavior_gap_range_sec": stats["behavior_gap_range_sec"],
        "iou_threshold": iou_threshold,
        "center_continuity_ratio": center_ratio,
        "conf_iou_relax": conf_iou_relax,
        "confidence_as_hard_rule": False,
        "presence_boundary_hard_constraint": False,
        "source_mode": stats.get("source_mode", "track_native"),
    }

    behavior_doc = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layer": BEHAVIOR_LAYER_LABEL,
        "semantic_definition": BEHAVIOR_SEMANTIC,
        "video": video,
        "fps": fps,
        "parameters": parameters,
        "source_mode": stats.get("source_mode", "track_native"),
        "source_segmentation_count": stats.get("source_segmentation_count", 0),
        "behavior_event_count": stats["behavior_event_count"],
        "target_event_range": list(TARGET_EVENT_RANGE),
        "events": behavior_events,
    }

    trace_doc = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layer": BEHAVIOR_LAYER_LABEL,
        "semantic_definition": BEHAVIOR_SEMANTIC,
        "parameters": parameters,
        "source_mode": stats.get("source_mode", "track_native"),
        "source_segmentation_count": stats.get("source_segmentation_count", 0),
        "behavior_event_count": stats["behavior_event_count"],
        "cross_presence_merge_count": stats["cross_presence_merge_count"],
        "stats": stats,
        "cross_presence_merges": cross_traces,
        "edge_decisions": edge_decisions,
        "behavior_merge_chains": [
            {
                "behavior_event_id": be["behavior_event_id"],
                "track_id": be["track_id"],
                "source_segmentation_event_ids": be["source_segmentation_event_ids"],
                "cross_presence_merge": be["cross_presence_merge"],
                "presence_segments_spanned": be["presence_segments_spanned"],
            }
            for be in behavior_events
            if be.get("source_event_count", 0) > 1 or be.get("cross_presence_merge")
        ],
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        behavior_path = os.path.join(output_dir, BEHAVIOR_EVENTS_NAME)
        trace_path = os.path.join(output_dir, MERGE_TRACE_NAME)
        with open(behavior_path, "w", encoding="utf-8") as f:
            json.dump(behavior_doc, f, ensure_ascii=False, indent=2)
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_doc, f, ensure_ascii=False, indent=2)
        stats["behavior_events_path"] = behavior_path
        stats["merge_trace_path"] = trace_path

    stats["behavior_events"] = behavior_events
    stats["merge_trace"] = trace_doc
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="Behavior Merge Layer (presence-independent)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--behavior-gap", type=float, default=DEFAULT_BEHAVIOR_GAP_SEC)
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_MERGE_IOU_THRESHOLD)
    p.add_argument("--center-ratio", type=float, default=DEFAULT_CENTER_CONTINUITY_RATIO)
    p.add_argument("--conf-iou-relax", type=float, default=DEFAULT_CONF_IOU_RELAX)
    p.add_argument("--event-gap", type=float, default=1.0)
    p.add_argument("--rebuild-events", action="store_true")
    args = p.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    video, fps = None, None

    if args.rebuild_events:
        seg = rebuild_events_from_tracked(out_dir, event_gap=args.event_gap)
        events_payload = [e.to_dict() for e in seg]
    else:
        try:
            events_payload, doc = load_presence_events(out_dir)
            video = doc.get("video")
            fps = doc.get("fps")
        except FileNotFoundError:
            print("[info] rebuilding segmentation events", file=sys.stderr)
            seg = rebuild_events_from_tracked(out_dir, event_gap=args.event_gap)
            events_payload = [e.to_dict() for e in seg]

    summary_path = os.path.join(out_dir, "detection_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        video = video or s.get("video")
        fps = fps or (float(s.get("fps") or 0) or None)

    stats = run_behavior_merge_layer(
        events_payload,
        output_dir=out_dir,
        behavior_gap_sec=args.behavior_gap,
        iou_threshold=args.iou_threshold,
        center_ratio=args.center_ratio,
        conf_iou_relax=args.conf_iou_relax,
        video=video,
        fps=fps,
    )
    print(f"[done] {stats.get('behavior_events_path')}")
    print(f"[trace] {stats.get('merge_trace_path')}")
    print(
        f"  {stats['source_segmentation_count']} segmentation ->"
        f" {stats['behavior_event_count']} behavior events"
        f" | cross_presence={stats['cross_presence_merge_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
