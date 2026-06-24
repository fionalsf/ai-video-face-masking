"""Benchmark evaluation: masking errors + merge/split alignment vs ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_IOU_MATCH = 0.30
DEFAULT_SAMPLE_STEP_SEC = 0.1


@dataclass
class TimeInterval:
    start: float
    end: float
    event_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def normalize_events(raw_events: list[dict]) -> list[TimeInterval]:
    out: list[TimeInterval] = []
    for ev in raw_events:
        if ev.get("should_mask") is False:
            continue
        start = float(ev.get("start_time", 0))
        end = float(ev.get("end_time", 0))
        if end <= start:
            continue
        eid = str(ev.get("gt_event_id") or ev.get("event_id") or ev.get("behavior_event_id") or "")
        out.append(TimeInterval(start, end, eid, dict(ev)))
    return out


def pred_events_from_doc(doc: dict | list) -> list[TimeInterval]:
    if isinstance(doc, list):
        events = doc
    else:
        events = doc.get("events") or []
    out: list[TimeInterval] = []
    for ev in events:
        start = float(ev.get("start_time", 0))
        end = float(ev.get("end_time", 0))
        if end <= start:
            continue
        if ev.get("review_status") == "logged_only" and ev.get("tier") == "low_conf":
            continue
        eid = str(ev.get("event_id") or ev.get("behavior_event_id") or "")
        out.append(TimeInterval(start, end, eid, dict(ev)))
    return out


def interval_iou(a: TimeInterval, b: TimeInterval) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: x[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        ls, le = merged[-1]
        if start <= le:
            merged[-1] = (ls, max(le, end))
        else:
            merged.append((start, end))
    return merged


def mask_timeline(
    intervals: list[TimeInterval],
    t_start: float,
    t_end: float,
    *,
    step_sec: float = DEFAULT_SAMPLE_STEP_SEC,
) -> tuple[list[float], list[bool]]:
    times: list[float] = []
    flags: list[bool] = []
    t = t_start
    spans = _merge_intervals([(iv.start, iv.end) for iv in intervals])
    while t <= t_end + 1e-9:
        masked = any(s <= t <= e for s, e in spans)
        times.append(round(t, 4))
        flags.append(masked)
        t += step_sec
    return times, flags


def compute_mask_errors(
    gt_events: list[TimeInterval],
    pred_events: list[TimeInterval],
    *,
    t_start: float,
    t_end: float,
    step_sec: float = DEFAULT_SAMPLE_STEP_SEC,
) -> dict[str, Any]:
    _, gt_mask = mask_timeline(gt_events, t_start, t_end, step_sec=step_sec)
    times, pred_mask = mask_timeline(pred_events, t_start, t_end, step_sec=step_sec)

    false_mask = 0.0
    miss_mask = 0.0
    gt_mask_sec = 0.0
    pred_mask_sec = 0.0
    for i, _t in enumerate(times):
        g = gt_mask[i]
        p = pred_mask[i]
        if g:
            gt_mask_sec += step_sec
        if p:
            pred_mask_sec += step_sec
        if p and not g:
            false_mask += step_sec
        if g and not p:
            miss_mask += step_sec

    eval_sec = max(step_sec, (len(times) * step_sec))
    return {
        "false_mask_sec": round(false_mask, 3),
        "miss_mask_sec": round(miss_mask, 3),
        "gt_mask_sec": round(gt_mask_sec, 3),
        "pred_mask_sec": round(pred_mask_sec, 3),
        "false_mask_ratio": round(false_mask / eval_sec, 4) if eval_sec else 0.0,
        "miss_mask_ratio": round(miss_mask / gt_mask_sec, 4) if gt_mask_sec else 0.0,
        "mask_iou": round(
            (pred_mask_sec - false_mask) / max(1e-9, gt_mask_sec + pred_mask_sec - false_mask - miss_mask + miss_mask),
            4,
        ) if (gt_mask_sec + pred_mask_sec) > 0 else 0.0,
        "sample_step_sec": step_sec,
        "sample_count": len(times),
    }


def _best_matches(
    gt: list[TimeInterval],
    pred: list[TimeInterval],
    *,
    iou_threshold: float = DEFAULT_IOU_MATCH,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    pairs: list[tuple[int, int, float]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            iou = interval_iou(g, p)
            if iou >= iou_threshold:
                pairs.append((gi, pi, iou))
    pairs.sort(key=lambda x: -x[2])

    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for gi, pi, iou in pairs:
        if gi in used_g or pi in used_p:
            continue
        matches.append((gi, pi, iou))
        used_g.add(gi)
        used_p.add(pi)

    unmatched_g = [i for i in range(len(gt)) if i not in used_g]
    unmatched_p = [i for i in range(len(pred)) if i not in used_p]
    return matches, unmatched_g, unmatched_p


def compute_merge_split_errors(
    gt_events: list[TimeInterval],
    pred_events: list[TimeInterval],
    *,
    iou_threshold: float = DEFAULT_IOU_MATCH,
    coverage_threshold: float = 0.25,
) -> dict[str, Any]:
    over_merge: list[dict[str, Any]] = []
    over_split: list[dict[str, Any]] = []

    for pi, pred in enumerate(pred_events):
        overlapping_gt: list[tuple[int, float]] = []
        for gi, gt in enumerate(gt_events):
            inter = max(0.0, min(pred.end, gt.end) - max(pred.start, gt.start))
            gt_len = gt.duration or 1e-9
            if inter / gt_len >= coverage_threshold:
                overlapping_gt.append((gi, interval_iou(gt, pred)))
        if len(overlapping_gt) > 1:
            over_merge.append({
                "pred_event_id": pred.event_id,
                "pred_interval": [pred.start, pred.end],
                "matched_gt_event_ids": [gt_events[gi].event_id for gi, _ in overlapping_gt],
                "matched_gt_count": len(overlapping_gt),
                "best_iou": round(max(i for _, i in overlapping_gt), 4),
            })

    for gi, gt in enumerate(gt_events):
        overlapping_pred: list[tuple[int, float]] = []
        for pi, pred in enumerate(pred_events):
            inter = max(0.0, min(pred.end, gt.end) - max(pred.start, gt.start))
            gt_len = gt.duration or 1e-9
            if inter / gt_len >= coverage_threshold:
                overlapping_pred.append((pi, interval_iou(gt, pred)))
        if len(overlapping_pred) > 1:
            over_split.append({
                "gt_event_id": gt.event_id,
                "gt_interval": [gt.start, gt.end],
                "matched_pred_event_ids": [pred_events[pi].event_id for pi, _ in overlapping_pred],
                "matched_pred_count": len(overlapping_pred),
                "best_iou": round(max(i for _, i in overlapping_pred), 4),
            })

    matches, unmatched_g, unmatched_p = _best_matches(
        gt_events, pred_events, iou_threshold=iou_threshold,
    )

    return {
        "over_merge_count": len(over_merge),
        "over_split_count": len(over_split),
        "over_merge_cases": over_merge,
        "over_split_cases": over_split,
        "matched_pairs": len(matches),
        "unmatched_gt_count": len(unmatched_g),
        "unmatched_pred_count": len(unmatched_p),
        "iou_match_threshold": iou_threshold,
    }


def review_cost_metrics(pred_events_raw: list[dict]) -> dict[str, Any]:
    total = len(pred_events_raw)
    review = sum(
        1 for e in pred_events_raw
        if e.get("tier") == "review" or e.get("review_status") == "pending"
    )
    auto = sum(1 for e in pred_events_raw if e.get("tier") == "auto")
    low = sum(1 for e in pred_events_raw if e.get("tier") == "low_conf")
    sec_per_review = 30.0
    return {
        "event_count": total,
        "review_event_count": review,
        "auto_event_count": auto,
        "low_conf_event_count": low,
        "estimated_review_minutes": round(review * sec_per_review / 60.0, 2),
        "review_ratio": round(review / total, 4) if total else 0.0,
    }


def evaluate_variant(
    gt_events: list[dict],
    pred_doc: dict | list,
    *,
    clip_start: float,
    clip_end: float,
    variant_name: str,
) -> dict[str, Any]:
    gt = normalize_events(gt_events)
    gt_clip = [g for g in gt if g.end > clip_start and g.start < clip_end]
    pred_all = pred_events_from_doc(pred_doc)
    pred_clip = [p for p in pred_all if p.end > clip_start and p.start < clip_end]

    pred_raw = pred_doc if isinstance(pred_doc, list) else (pred_doc.get("events") or [])
    pred_raw_clip = [
        e for e in pred_raw
        if float(e.get("end_time", 0)) > clip_start and float(e.get("start_time", 0)) < clip_end
    ]

    mask = compute_mask_errors(gt_clip, pred_clip, t_start=clip_start, t_end=clip_end)
    merge_split = compute_merge_split_errors(gt_clip, pred_clip)
    review = review_cost_metrics(pred_raw_clip)

    return {
        "variant": variant_name,
        "clip_range": [clip_start, clip_end],
        "mask_errors": mask,
        "merge_split_errors": merge_split,
        "review_cost": review,
        "objective_score": round(
            mask["false_mask_sec"] + mask["miss_mask_sec"] + review["estimated_review_minutes"] * 2.0,
            3,
        ),
    }


def aggregate_clip_metrics(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_clip:
        return {}
    keys_float = (
        "false_mask_sec", "miss_mask_sec", "over_merge_count", "over_split_count",
        "event_count", "review_event_count", "estimated_review_minutes", "objective_score",
    )
    agg: dict[str, float] = {k: 0.0 for k in keys_float}
    for row in per_clip:
        m = row.get("mask_errors") or {}
        ms = row.get("merge_split_errors") or {}
        r = row.get("review_cost") or {}
        agg["false_mask_sec"] += m.get("false_mask_sec", 0)
        agg["miss_mask_sec"] += m.get("miss_mask_sec", 0)
        agg["over_merge_count"] += ms.get("over_merge_count", 0)
        agg["over_split_count"] += ms.get("over_split_count", 0)
        agg["event_count"] += r.get("event_count", 0)
        agg["review_event_count"] += r.get("review_event_count", 0)
        agg["estimated_review_minutes"] += r.get("estimated_review_minutes", 0)
        agg["objective_score"] += row.get("objective_score", 0)
    n = len(per_clip)
    return {
        "clip_count": n,
        "total_false_mask_sec": round(agg["false_mask_sec"], 3),
        "total_miss_mask_sec": round(agg["miss_mask_sec"], 3),
        "total_over_merge": int(agg["over_merge_count"]),
        "total_over_split": int(agg["over_split_count"]),
        "total_events": int(agg["event_count"]),
        "total_review_events": int(agg["review_event_count"]),
        "total_estimated_review_minutes": round(agg["estimated_review_minutes"], 2),
        "mean_objective_score": round(agg["objective_score"] / n, 3),
    }
