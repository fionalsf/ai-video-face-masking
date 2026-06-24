"""Production Event Merge Layer — behavior-level events after Scheme C builder."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any

from event_builder import FaceEvent, TIER_AUTO, TIER_LOW_CONF, TIER_REVIEW, bbox_iou
from event_quality import score_event_dict
from gap_analysis import DEFAULT_HARD_SPLIT_GAP_SEC
from video_meta import sec_to_timecode

PRODUCTION_PIPELINE = "scheme_c + event_merge_layer"
MERGE_LAYER_LABEL = "event_merge_production_v1"
FINAL_EVENTS_NAME = "final_events.json"
DEBUG_REPORT_NAME = "event_debug_report.json"
SEGMENTATION_EVENTS_NAME = "segmentation_events.json"

DEFAULT_MERGE_GAP_SEC = 3.0
MIN_MERGE_GAP_SEC = 2.5
MAX_MERGE_GAP_SEC = 3.0
DEFAULT_MERGE_IOU_THRESHOLD = 0.2
DEFAULT_CENTER_CONTINUITY_RATIO = 1.5
DEFAULT_CONF_IOU_RELAX = 0.12
BEHAVIOR_SEMANTIC = "人脸连续出现的一段行为"

_TIER_RANK = {TIER_AUTO: 0, TIER_REVIEW: 1, TIER_LOW_CONF: 2}


@dataclass
class MergeEdgeDecision:
    track_id: int
    from_event_id: str
    to_event_id: str
    merged: bool
    event_gap_sec: float
    detection_gap_sec: float
    reason: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "from_event_id": self.from_event_id,
            "to_event_id": self.to_event_id,
            "merged": self.merged,
            "event_gap_sec": self.event_gap_sec,
            "detection_gap_sec": self.detection_gap_sec,
            "reason": self.reason,
            **self.meta,
        }


def _event_dict(ev: FaceEvent | dict) -> dict:
    if isinstance(ev, FaceEvent):
        return ev.to_dict()
    return ev


def _trajectory(ev: FaceEvent | dict) -> list[dict]:
    if isinstance(ev, FaceEvent):
        return ev.trajectory
    return ev.get("trajectory") or []


def _bbox_at_boundary(ev: FaceEvent | dict, *, end: bool) -> list[float] | None:
    traj = _trajectory(ev)
    if not traj:
        return None
    pt = traj[-1] if end else traj[0]
    bbox = pt.get("bbox")
    return list(bbox) if bbox else None


def _event_gap_sec(prev: FaceEvent | dict, curr: FaceEvent | dict) -> float:
    p = _event_dict(prev)
    c = _event_dict(curr)
    return max(0.0, float(c["start_time"]) - float(p["end_time"]))


def _detection_gap_sec(prev: FaceEvent | dict, curr: FaceEvent | dict) -> float:
    prev_traj = _trajectory(prev)
    curr_traj = _trajectory(curr)
    if not prev_traj or not curr_traj:
        return _event_gap_sec(prev, curr)
    return max(0.0, float(curr_traj[0]["t"]) - float(prev_traj[-1]["t"]))


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_diagonal(bbox: list[float]) -> float:
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    return math.hypot(w, h)


def center_continuous(
    bbox_a: list[float],
    bbox_b: list[float],
    *,
    ratio: float = DEFAULT_CENTER_CONTINUITY_RATIO,
) -> bool:
    cx_a, cy_a = _bbox_center(bbox_a)
    cx_b, cy_b = _bbox_center(bbox_b)
    dist = math.hypot(cx_a - cx_b, cy_a - cy_b)
    ref = max(_bbox_diagonal(bbox_a), _bbox_diagonal(bbox_b))
    return ref > 0 and dist <= ratio * ref


def effective_iou_threshold(
    base: float,
    conf_a: float,
    conf_b: float,
    *,
    max_relax: float = DEFAULT_CONF_IOU_RELAX,
) -> float:
    """Confidence relaxes IoU threshold only; never blocks merge by itself."""
    avg_conf = (conf_a + conf_b) / 2.0
    relax = max_relax * max(0.0, avg_conf - 0.5)
    return max(0.05, base - relax)


def bbox_continuity(
    prev: FaceEvent | dict,
    curr: FaceEvent | dict,
    *,
    iou_threshold: float = DEFAULT_MERGE_IOU_THRESHOLD,
    center_ratio: float = DEFAULT_CENTER_CONTINUITY_RATIO,
    conf_iou_relax: float = DEFAULT_CONF_IOU_RELAX,
) -> tuple[bool, dict[str, Any]]:
    bbox_a = _bbox_at_boundary(prev, end=True)
    bbox_b = _bbox_at_boundary(curr, end=False)
    if bbox_a is None or bbox_b is None:
        return False, {"reason": "missing_bbox", "iou": None, "center_continuous": False}

    iou = bbox_iou(bbox_a, bbox_b)
    center_ok = center_continuous(bbox_a, bbox_b, ratio=center_ratio)
    p = _event_dict(prev)
    c = _event_dict(curr)
    eff_iou = effective_iou_threshold(
        iou_threshold,
        float(p.get("avg_confidence") or p.get("peak_confidence") or 0.0),
        float(c.get("avg_confidence") or c.get("peak_confidence") or 0.0),
        max_relax=conf_iou_relax,
    )
    ok = iou >= eff_iou or center_ok
    return ok, {
        "iou": round(iou, 4),
        "effective_iou_threshold": round(eff_iou, 4),
        "center_continuous": center_ok,
        "bbox_continuity": ok,
    }


def should_semantic_merge(
    prev: FaceEvent | dict,
    curr: FaceEvent | dict,
    *,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
    iou_threshold: float = DEFAULT_MERGE_IOU_THRESHOLD,
    center_ratio: float = DEFAULT_CENTER_CONTINUITY_RATIO,
    conf_iou_relax: float = DEFAULT_CONF_IOU_RELAX,
) -> tuple[bool, dict[str, Any]]:
    p = _event_dict(prev)
    c = _event_dict(curr)
    if int(p["track_id"]) != int(c["track_id"]):
        return False, {"reason": "track_mismatch"}

    event_gap = round(_event_gap_sec(prev, curr), 2)
    detection_gap = round(_detection_gap_sec(prev, curr), 3)
    if event_gap > merge_gap_sec:
        return False, {
            "reason": "gap_exceeded",
            "event_gap_sec": event_gap,
            "detection_gap_sec": detection_gap,
            "merge_gap_sec": merge_gap_sec,
        }

    cont_ok, cont_meta = bbox_continuity(
        prev,
        curr,
        iou_threshold=iou_threshold,
        center_ratio=center_ratio,
        conf_iou_relax=conf_iou_relax,
    )
    if not cont_ok:
        return False, {
            "reason": "bbox_discontinuity",
            "event_gap_sec": event_gap,
            "detection_gap_sec": detection_gap,
            **cont_meta,
        }

    return True, {
        "reason": "merged",
        "event_gap_sec": event_gap,
        "detection_gap_sec": detection_gap,
        **cont_meta,
    }


def _merge_tier(events: list[FaceEvent | dict]) -> str:
    worst = TIER_AUTO
    for ev in events:
        d = _event_dict(ev)
        tier = d.get("tier", TIER_REVIEW)
        if _TIER_RANK.get(tier, 1) > _TIER_RANK.get(worst, 0):
            worst = tier
    return worst


def _combine_trajectory(events: list[FaceEvent | dict]) -> list[dict]:
    pts: list[dict] = []
    for ev in events:
        pts.extend(_trajectory(ev))
    pts.sort(key=lambda x: (int(x["frame"]), float(x["t"])))
    return pts


def _build_final_event(
    group: list[FaceEvent | dict],
    final_num: int,
    *,
    merge_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    source_ids = [_event_dict(ev)["event_id"] for ev in group]
    traj = _combine_trajectory(group)
    confs = [float(p["conf"]) for p in traj]
    peak = max(confs) if confs else 0.0
    avg_conf = round(mean(confs), 4) if confs else 0.0

    start_time = min(float(_event_dict(ev)["start_time"]) for ev in group)
    end_time = max(float(_event_dict(ev)["end_time"]) for ev in group)
    start_frame = min(int(_event_dict(ev)["start_frame"]) for ev in group)
    end_frame = max(int(_event_dict(ev)["end_frame"]) for ev in group)

    hints: list[str] = []
    for ev in group:
        for h in _event_dict(ev).get("rule_hints") or []:
            if h not in hints:
                hints.append(h)
    if len(source_ids) > 1:
        hints.append(f"behavior_merge:{len(source_ids)}")

    tier = _merge_tier(group)
    review_status = (
        "confirmed_face" if tier == TIER_AUTO
        else "logged_only" if tier == TIER_LOW_CONF
        else "pending"
    )

    return {
        "event_id": f"evt_{final_num:04d}",
        "track_id": int(_event_dict(group[0])["track_id"]),
        "tier": tier,
        "review_status": review_status,
        "source_event_ids": source_ids,
        "source_event_count": len(source_ids),
        "behavior_merged": len(source_ids) > 1,
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
        "merge_edges": merge_meta,
    }


def merge_presence_events(
    events: list[FaceEvent | dict],
    *,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
    iou_threshold: float = DEFAULT_MERGE_IOU_THRESHOLD,
    center_ratio: float = DEFAULT_CENTER_CONTINUITY_RATIO,
    conf_iou_relax: float = DEFAULT_CONF_IOU_RELAX,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[MergeEdgeDecision], dict[str, Any]]:
    by_track: dict[int, list[FaceEvent | dict]] = {}
    for ev in events:
        d = _event_dict(ev)
        by_track.setdefault(int(d["track_id"]), []).append(ev)
    for tid in by_track:
        by_track[tid].sort(key=lambda e: (_event_dict(e)["start_time"], _event_dict(e)["start_frame"]))

    final_events: list[dict[str, Any]] = []
    merge_map: list[dict[str, Any]] = []
    edge_decisions: list[MergeEdgeDecision] = []
    final_num = 0

    for track_id in sorted(by_track):
        track_events = by_track[track_id]
        group: list[FaceEvent | dict] = []
        group_edges: list[dict[str, Any]] = []

        def flush_group() -> None:
            nonlocal final_num
            if not group:
                return
            final_num += 1
            fe = _build_final_event(group, final_num, merge_meta=group_edges)
            final_events.append(fe)
            merge_map.append({
                "final_event_id": fe["event_id"],
                "track_id": track_id,
                "source_event_ids": fe["source_event_ids"],
                "source_event_count": fe["source_event_count"],
                "behavior_merged": fe["behavior_merged"],
            })
            group.clear()
            group_edges.clear()

        for ev in track_events:
            if not group:
                group.append(ev)
                continue
            ok, meta = should_semantic_merge(
                group[-1],
                ev,
                merge_gap_sec=merge_gap_sec,
                iou_threshold=iou_threshold,
                center_ratio=center_ratio,
                conf_iou_relax=conf_iou_relax,
            )
            p = _event_dict(group[-1])
            c = _event_dict(ev)
            edge_decisions.append(MergeEdgeDecision(
                track_id=track_id,
                from_event_id=p["event_id"],
                to_event_id=c["event_id"],
                merged=ok,
                event_gap_sec=float(meta.get("event_gap_sec", round(_event_gap_sec(group[-1], ev), 2))),
                detection_gap_sec=float(meta.get("detection_gap_sec", round(_detection_gap_sec(group[-1], ev), 3))),
                reason=str(meta.get("reason", "unknown")),
                meta={k: v for k, v in meta.items() if k not in ("reason", "event_gap_sec", "detection_gap_sec")},
            ))
            if ok:
                group.append(ev)
                group_edges.append(meta)
            else:
                flush_group()
                group.append(ev)
        flush_group()

    source_to_final = {
        sid: entry["final_event_id"]
        for entry in merge_map
        for sid in entry["source_event_ids"]
    }

    stats = {
        "source_event_count": len(events),
        "merged_event_count": len(final_events),
        "merge_group_count": sum(1 for m in merge_map if m["behavior_merged"]),
        "events_merged_away": len(events) - len(final_events),
        "compression_ratio": round(len(events) / max(1, len(final_events)), 3),
        "merge_gap_sec": merge_gap_sec,
        "merge_gap_range_sec": [MIN_MERGE_GAP_SEC, MAX_MERGE_GAP_SEC],
        "iou_threshold": iou_threshold,
        "center_continuity_ratio": center_ratio,
        "conf_iou_relax": conf_iou_relax,
        "confidence_as_hard_rule": False,
        "merge_layer": MERGE_LAYER_LABEL,
        "pipeline": PRODUCTION_PIPELINE,
        "semantic_definition": BEHAVIOR_SEMANTIC,
        "source_to_final": source_to_final,
    }
    return final_events, merge_map, edge_decisions, stats


def _build_debug_report(
    *,
    stats: dict[str, Any],
    edge_decisions: list[MergeEdgeDecision],
    merge_map: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    edges = [e.to_dict() for e in edge_decisions]
    merged_edges = [e for e in edges if e["merged"]]
    rejected_edges = [e for e in edges if not e["merged"]]
    reject_reasons: dict[str, int] = {}
    for e in rejected_edges:
        reject_reasons[e["reason"]] = reject_reasons.get(e["reason"], 0) + 1

    gap_buckets = {"<=2.5": 0, "2.5-3.0": 0, ">3.0": 0}
    for e in edges:
        g = e["event_gap_sec"]
        if g <= MIN_MERGE_GAP_SEC:
            gap_buckets["<=2.5"] += 1
        elif g <= MAX_MERGE_GAP_SEC:
            gap_buckets["2.5-3.0"] += 1
        else:
            gap_buckets[">3.0"] += 1

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": PRODUCTION_PIPELINE,
        "semantic_definition": BEHAVIOR_SEMANTIC,
        "parameters": parameters,
        "source_event_count": stats["source_event_count"],
        "final_event_count": stats["merged_event_count"],
        "merged_event_count": stats["merged_event_count"],
        "merge_group_count": stats["merge_group_count"],
        "events_merged_away": stats["events_merged_away"],
        "compression_ratio": stats["compression_ratio"],
        "edge_decision_total": len(edges),
        "edge_merged": len(merged_edges),
        "edge_rejected": len(rejected_edges),
        "reject_reason_counts": reject_reasons,
        "event_gap_histogram": gap_buckets,
        "merge_map": merge_map,
        "merge_groups": [m for m in merge_map if m["behavior_merged"]],
        "edge_decisions": edges,
    }


def run_production_merge(
    events: list[FaceEvent | dict],
    *,
    output_dir: str | None = None,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
    iou_threshold: float = DEFAULT_MERGE_IOU_THRESHOLD,
    center_ratio: float = DEFAULT_CENTER_CONTINUITY_RATIO,
    conf_iou_relax: float = DEFAULT_CONF_IOU_RELAX,
    video: str | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    merge_gap_sec = max(MIN_MERGE_GAP_SEC, min(MAX_MERGE_GAP_SEC, merge_gap_sec))

    final_events, merge_map, edge_decisions, stats = merge_presence_events(
        events,
        merge_gap_sec=merge_gap_sec,
        iou_threshold=iou_threshold,
        center_ratio=center_ratio,
        conf_iou_relax=conf_iou_relax,
    )

    for fe in final_events:
        fe["event_quality_score"] = score_event_dict(fe, fps=fps or 30.0)

    parameters = {
        "merge_gap_sec": merge_gap_sec,
        "merge_gap_range_sec": [MIN_MERGE_GAP_SEC, MAX_MERGE_GAP_SEC],
        "iou_threshold": iou_threshold,
        "center_continuity_ratio": center_ratio,
        "conf_iou_relax": conf_iou_relax,
        "confidence_as_hard_rule": False,
        "hard_split_gap_sec": DEFAULT_HARD_SPLIT_GAP_SEC,
    }

    quality_summary = {"high": 0, "medium": 0, "low": 0}
    for fe in final_events:
        q = fe["event_quality_score"]["quality"]
        quality_summary[q.lower()] = quality_summary.get(q.lower(), 0) + 1

    final_doc: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": PRODUCTION_PIPELINE,
        "merge_layer": MERGE_LAYER_LABEL,
        "semantic_definition": BEHAVIOR_SEMANTIC,
        "video": video,
        "fps": fps,
        "parameters": parameters,
        "source_event_count": stats["source_event_count"],
        "merged_event_count": stats["merged_event_count"],
        "merge_map": merge_map,
        "quality_summary": quality_summary,
        "events": final_events,
    }

    debug_doc = _build_debug_report(
        stats=stats,
        edge_decisions=edge_decisions,
        merge_map=merge_map,
        parameters=parameters,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, FINAL_EVENTS_NAME)
        debug_path = os.path.join(output_dir, DEBUG_REPORT_NAME)
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_doc, f, ensure_ascii=False, indent=2)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_doc, f, ensure_ascii=False, indent=2)
        stats["final_events_path"] = final_path
        stats["debug_report_path"] = debug_path

    stats["final_events"] = final_events
    stats["merge_map"] = merge_map
    stats["quality_summary"] = quality_summary
    return stats


def run_semantic_merge(*args, **kwargs) -> dict[str, Any]:
    """Backward-compatible alias for production merge."""
    return run_production_merge(*args, **kwargs)


def save_segmentation_events(
    events: list[FaceEvent | dict],
    output_dir: str,
    *,
    video: str | None = None,
    fps: float | None = None,
) -> str:
    path = os.path.join(os.path.abspath(output_dir), SEGMENTATION_EVENTS_NAME)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_stage": "scheme_c_event_builder",
        "video": video,
        "fps": fps,
        "event_count": len(events),
        "events": [_event_dict(e) for e in events],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def face_event_from_final(d: dict) -> FaceEvent:
    """Convert production final event dict to FaceEvent for legacy consumers."""
    return FaceEvent(
        event_id=d["event_id"],
        track_id=int(d["track_id"]),
        tier=d.get("tier", TIER_REVIEW),
        start_time=float(d["start_time"]),
        end_time=float(d["end_time"]),
        start_frame=int(d["start_frame"]),
        end_frame=int(d["end_frame"]),
        avg_confidence=float(d.get("avg_confidence") or 0.0),
        peak_confidence=float(d.get("peak_confidence") or 0.0),
        detection_count=int(d.get("detection_count") or 0),
        trajectory=list(d.get("trajectory") or []),
        rule_hints=list(d.get("rule_hints") or []),
    )


def load_final_events(output_dir: str) -> dict[str, Any]:
    path = os.path.join(os.path.abspath(output_dir), FINAL_EVENTS_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_presence_events(output_dir: str) -> tuple[list[dict], dict]:
    output_dir = os.path.abspath(output_dir)
    for name in (SEGMENTATION_EVENTS_NAME, "face_events.json", "event_preview.json"):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            events = doc.get("events") or []
            if events:
                return events, doc
    raise FileNotFoundError(
        f"No segmentation events in {output_dir} (need {SEGMENTATION_EVENTS_NAME} or face_events.json)"
    )


def rebuild_events_from_tracked(
    output_dir: str,
    *,
    event_gap: float = 1.0,
) -> list[FaceEvent]:
    from event_builder import build_events
    from video_meta import get_video_meta

    output_dir = os.path.abspath(output_dir)
    tracked_path = os.path.join(output_dir, "tracked_detections.json")
    summary_path = os.path.join(output_dir, "detection_summary.json")
    with open(tracked_path, encoding="utf-8") as f:
        tracked = json.load(f)
    summary: dict = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    fps = float(summary.get("fps") or 30.0)
    video = summary.get("video", "")
    frame_h, frame_w, total_frames = 1080, 1920, None
    if video and os.path.isfile(video):
        meta = get_video_meta(video)
        frame_h, frame_w = meta["height"], meta["width"]
        total_frames = meta["frames"]
    return build_events(
        tracked,
        gap_sec=event_gap,
        frame_h=frame_h,
        frame_w=frame_w,
        fps=fps,
        total_frames=total_frames,
        detect_interval=int(summary.get("detect_interval") or 0) or None,
        output_dir=output_dir,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Production Event Merge Layer (Scheme C -> final events)")
    p.add_argument("--output-dir", required=True, help="Detection output directory")
    p.add_argument("--merge-gap", type=float, default=DEFAULT_MERGE_GAP_SEC)
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_MERGE_IOU_THRESHOLD)
    p.add_argument("--center-ratio", type=float, default=DEFAULT_CENTER_CONTINUITY_RATIO)
    p.add_argument("--conf-iou-relax", type=float, default=DEFAULT_CONF_IOU_RELAX)
    p.add_argument("--event-gap", type=float, default=1.0)
    p.add_argument("--rebuild-events", action="store_true")
    args = p.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    video, fps = None, None
    events_payload: list[dict] = []

    if args.rebuild_events:
        events = rebuild_events_from_tracked(out_dir, event_gap=args.event_gap)
        events_payload = [e.to_dict() for e in events]
    else:
        try:
            events_payload, doc = load_presence_events(out_dir)
            video = doc.get("video")
            fps = doc.get("fps")
        except FileNotFoundError:
            print("[info] rebuilding segmentation events from tracked_detections.json")
            events = rebuild_events_from_tracked(out_dir, event_gap=args.event_gap)
            events_payload = [e.to_dict() for e in events]

    summary_path = os.path.join(out_dir, "detection_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        video = video or s.get("video")
        fps = fps or (float(s.get("fps") or 0) or None)

    stats = run_production_merge(
        events_payload,
        output_dir=out_dir,
        merge_gap_sec=args.merge_gap,
        iou_threshold=args.iou_threshold,
        center_ratio=args.center_ratio,
        conf_iou_relax=args.conf_iou_relax,
        video=video,
        fps=fps,
    )
    print(f"[done] {stats.get('final_events_path')}")
    print(f"[debug] {stats.get('debug_report_path')}")
    print(
        f"  {stats['source_event_count']} segmentation -> {stats['merged_event_count']} final"
        f" | merge_groups={stats['merge_group_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
