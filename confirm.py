#!/usr/bin/env python3
"""Morning confirm: merge Review decisions into final.mp4."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import cv2

from event_builder import TIER_AUTO, TIER_LOW_CONF
from mask_timeline import load_mask_timeline, select_render_entries
from render import events_to_render, mux_audio, render_video
from video_meta import get_video_meta


def parse_args():
    p = argparse.ArgumentParser(description="合并 Review 确认结果 → final.mp4")
    p.add_argument("--output-dir", required=True, help="pipeline 输出目录 output/视频名")
    p.add_argument("--expand", type=float, default=None)
    p.add_argument("--mosaic-block", type=int, default=22)
    p.add_argument("--encoder", default="auto")
    p.add_argument("--mask-scale-divisor", type=int, default=8)
    p.add_argument("--filter-threads", type=int, default=1)
    p.add_argument("--final-name", default="final.mp4", help="Output video filename inside --output-dir.")
    p.add_argument("--no-audio", action="store_true", help="Skip audio mux for faster render tests.")
    return p.parse_args()


def parse_review_decisions(confirmed: dict) -> dict[str, str]:
    """Support review_ui flat map and legacy confirmed template."""
    meta_keys = {
        "events", "summary", "video", "review_dir", "started_at", "updated_at",
        "total_review_events", "decided_count", "schema", "decision_unit",
    }
    events = confirmed.get("events")
    if isinstance(events, list) and events and isinstance(events[0], dict):
        out: dict[str, str] = {}
        for e in events:
            eid = e.get("event_id")
            if not eid:
                continue
            status = e.get("status", "")
            if status == "confirmed_face":
                out[eid] = "accepted"
            elif status == "rejected_fp":
                out[eid] = "rejected"
            elif status:
                out[eid] = status
        return out
    return {
        str(k): str(v)
        for k, v in confirmed.items()
        if str(k) not in meta_keys
        and (str(k).startswith("bevt_") or str(k).startswith("evt_") or str(k).startswith("mask_"))
    }


def load_confirmed(review_dir: str) -> dict:
    path = os.path.join(review_dir, "confirmed_events.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 {path}，请先运行 Review UI")
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def load_face_events(out_dir: str) -> dict:
    path = os.path.join(out_dir, "face_events.json")
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def load_runtime(out_dir: str) -> dict:
    path = os.path.join(out_dir, "review_report.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    return report.get("runtime") or {}


def event_start(ev: dict) -> float:
    return float(ev.get("start_time", ev.get("start", 0.0)))


def event_end(ev: dict) -> float:
    return float(ev.get("end_time", ev.get("end", 0.0)))


def event_duration(ev: dict) -> float:
    return max(0.0, event_end(ev) - event_start(ev))


def bbox_at_time(ev: dict, target_t: float) -> list[float] | None:
    trajectory = ev.get("trajectory") or []
    if not trajectory:
        return None
    point = trajectory_point_at_time(ev, target_t)
    if point is None:
        return None
    bbox = point.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return [float(v) for v in bbox]


def trajectory_point_at_time(ev: dict, target_t: float) -> dict | None:
    trajectory = ev.get("trajectory") or []
    if not trajectory:
        return None
    return min(trajectory, key=lambda p: abs(float(p.get("t", 0.0)) - target_t))


def bbox_iou(a: list[float], b: list[float]) -> float:
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


def bbox_center_distance_ratio(a: list[float], b: list[float]) -> float:
    acx = (a[0] + a[2]) * 0.5
    acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5
    bcy = (b[1] + b[3]) * 0.5
    scale = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)
    return (((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / scale


def events_are_continuous(a: dict, b: dict, max_gap: float) -> bool:
    gap = max(event_start(a) - event_end(b), event_start(b) - event_end(a), 0.0)
    return gap <= max_gap


def low_conf_matches_anchor(low_ev: dict, anchor_ev: dict, max_gap: float) -> bool:
    if not events_are_continuous(low_ev, anchor_ev, max_gap):
        return False
    probe_t = min(max((event_start(low_ev) + event_end(low_ev)) * 0.5, event_start(anchor_ev)), event_end(anchor_ev))
    low_box = bbox_at_time(low_ev, probe_t)
    anchor_box = bbox_at_time(anchor_ev, probe_t)
    if not low_box or not anchor_box:
        return False
    return bbox_iou(low_box, anchor_box) >= 0.08 or bbox_center_distance_ratio(low_box, anchor_box) <= 0.85


def skin_ratio(crop) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mask_a = (y > 45) & (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 140)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_b = (v > 45) & (s > 18) & ((h <= 25) | (h >= 165))
    return float((mask_a | mask_b).mean())


def standalone_low_conf_candidate(ev: dict, cap, meta: dict, runtime: dict) -> bool:
    hints = set(ev.get("rule_hints") or [])
    if "edge_clip" not in hints:
        return False
    if float(ev.get("peak_confidence") or 0.0) < float(runtime.get("low_conf_standalone_min_peak") or 0.45):
        return False
    if event_duration(ev) > float(runtime.get("low_conf_standalone_max_duration") or 2.5):
        return False

    mid_t = (event_start(ev) + event_end(ev)) * 0.5
    point = trajectory_point_at_time(ev, mid_t)
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
    if not (0.0015 <= area_ratio <= float(runtime.get("low_conf_standalone_max_area") or 0.075)):
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
    return skin_ratio(crop) >= float(runtime.get("low_conf_standalone_min_skin") or 0.16)


def select_low_conf_events(
    low_conf_events: list[dict],
    accepted_events: list[dict],
    decisions: dict[str, str],
    runtime: dict,
    video_path: str | None = None,
    meta: dict | None = None,
) -> tuple[list[dict], int]:
    if bool(runtime.get("mask_lowconf", False)):
        return low_conf_events, 0

    max_gap = float(runtime.get("low_conf_bridge_gap") or 1.0)
    max_duration = float(runtime.get("low_conf_bridge_max_duration") or 3.0)
    min_peak = float(runtime.get("low_conf_bridge_min_peak") or 0.35)

    selected: list[dict] = []
    anchors = list(accepted_events)
    remaining = list(low_conf_events)
    cap = None
    if video_path and meta and bool(runtime.get("low_conf_standalone", True)):
        cap = cv2.VideoCapture(video_path)
    changed = True
    try:
        while changed:
            changed = False
            next_remaining = []
            for ev in remaining:
                eid = ev.get("event_id")
                decision = decisions.get(eid)
                if decision == "rejected":
                    continue
                explicit_accept = decision == "accepted"
                bridged = False
                if not explicit_accept:
                    if event_duration(ev) <= max_duration and float(ev.get("peak_confidence") or 0.0) >= min_peak:
                        bridged = any(low_conf_matches_anchor(ev, anchor, max_gap) for anchor in anchors)
                    if not bridged and not (cap is not None and standalone_low_conf_candidate(ev, cap, meta or {}, runtime)):
                        next_remaining.append(ev)
                        continue
                selected.append(ev)
                anchors.append(ev)
                changed = True
            remaining = next_remaining
    finally:
        if cap is not None:
            cap.release()

    return selected, len(selected)


def review_requires_explicit_accept(ev: dict, meta: dict | None = None) -> bool:
    hints = set(ev.get("rule_hints") or [])
    if hints & {"cross_presence_span", "low_conf_review_candidate"}:
        return True

    traj = ev.get("trajectory") or []
    if any(float(p.get("conf", 1.0)) < 0.55 for p in traj):
        return True

    if meta:
        width = int(meta.get("width") or 0)
        height = int(meta.get("height") or 0)
        if width > 0 and height > 0:
            for p in traj:
                bbox = p.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox]
                box_w = max(0.0, x2 - x1)
                box_h = max(0.0, y2 - y1)
                aspect = box_w / max(1.0, box_h)
                touches_edge = x1 <= 2 or y1 <= 12 or x2 >= width - 2 or y2 >= height - 2
                unusually_large = (box_w * box_h) / max(1, width * height) > 0.035
                if touches_edge and unusually_large:
                    return True
                if y1 <= 12 and box_w >= 180 and aspect >= 1.35:
                    return True
    return False


def run_confirm(args) -> int:
    confirm_started = time.perf_counter()
    out_dir = os.path.abspath(args.output_dir)
    review_dir = os.path.join(out_dir, "review")
    final = os.path.join(out_dir, args.final_name)
    report_path = os.path.join(out_dir, "review_report.json")

    confirmed = load_confirmed(review_dir)
    face_data = load_face_events(out_dir)
    runtime = load_runtime(out_dir)
    video_path = face_data["video"]
    meta = get_video_meta(video_path)

    decisions = parse_review_decisions(confirmed)
    mask_timeline = load_mask_timeline(out_dir)
    expand = args.expand if args.expand is not None else float(runtime.get("expand") or 0.18)
    if mask_timeline is not None:
        timeline_video = mask_timeline.get("video") or video_path
        if os.path.abspath(timeline_video) != os.path.abspath(video_path):
            video_path = timeline_video
            meta = get_video_meta(video_path)
        render, timeline_stats = select_render_entries(mask_timeline, decisions)
        print(
            "[timeline-confirm] "
            f"Auto={timeline_stats['auto_selected']} "
            f"Review={timeline_stats['review_selected']} "
            f"Rejected={timeline_stats['review_rejected']} "
            f"UnreviewedSkipped={timeline_stats['review_unreviewed_skipped']} "
            f"Frames={timeline_stats['render_frames']}"
        )
        if not render:
            shutil.copy2(video_path, final)
        else:
            render_started = time.perf_counter()
            tmp = final + ".tmp.mp4"
            render_video(
                video_path,
                tmp,
                render,
                meta,
                expand=expand,
                mosaic_block=args.mosaic_block,
                encoder=args.encoder,
                refine_face_boxes=False,
                mask_scale_divisor=args.mask_scale_divisor,
                filter_threads=args.filter_threads,
            )
            if args.no_audio:
                os.replace(tmp, final)
            else:
                mux_audio(tmp, video_path, final)
            timeline_stats["render_wall_sec"] = round(time.perf_counter() - render_started, 3)

        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
        else:
            report = {}
        report.update({
            "morning_confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "delivery_ready": True,
            "final_video": final,
            "confirm_mode": "mask_timeline_v2",
            "timeline_confirm_stats": timeline_stats,
            "confirm_total_sec": round(time.perf_counter() - confirm_started, 3),
            "review_summary": confirmed.get("summary", {}) if isinstance(confirmed, dict) else {},
        })
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[完成] 交付成片：{final}")
        return 0

    auto_events = [e for e in face_data["events"] if e.get("tier") == TIER_AUTO]
    all_low_conf_events = [e for e in face_data["events"] if e.get("tier") == TIER_LOW_CONF]

    review_confirmed = []
    review_rejected = 0
    for ev in face_data["events"]:
        if ev.get("tier") != "review":
            continue
        eid = ev["event_id"]
        decision = decisions.get(eid)
        if decision == "rejected":
            review_rejected += 1
            continue
        if decision is None and review_requires_explicit_accept(ev, meta):
            review_rejected += 1
            continue
        if decision in (None, "skipped"):
            decision = "accepted"
        if decision == "accepted":
            review_confirmed.append(ev)

    low_conf_events, low_conf_bridge_count = select_low_conf_events(
        all_low_conf_events,
        auto_events + review_confirmed,
        decisions,
        runtime,
        video_path=video_path,
        meta=meta,
    )
    all_mask_events = auto_events + review_confirmed + low_conf_events
    if low_conf_bridge_count:
        print(f"[bridge] low_conf continuous fragments added: {low_conf_bridge_count}")
    print(
        f"[确认] Auto={len(auto_events)} + LowConf={len(low_conf_events)} + Review确认={len(review_confirmed)}"
        f" (reject={review_rejected}) → 共 {len(all_mask_events)} 个打码 event"
    )

    if not all_mask_events:
        shutil.copy2(video_path, final)
    else:
        render = events_to_render(
            all_mask_events,
            meta["frames"],
            extend_frames=int(runtime.get("render_extend_frames") or 3),
            video_path=video_path,
            meta=meta,
            motion_compensate=bool(runtime.get("motion_compensate", False)),
            motion_max_gap=int(runtime.get("motion_max_gap") or 45),
            motion_singleton_frames=int(runtime.get("motion_singleton_frames") or 24),
            motion_min_points=int(runtime.get("motion_min_points") or 4),
            motion_anchor=float(runtime.get("motion_anchor") or 0.18),
        )
        tmp = final + ".tmp.mp4"
        render_video(video_path, tmp, render, meta, expand=expand,
                     mosaic_block=args.mosaic_block, encoder=args.encoder,
                     refine_face_boxes=bool(runtime.get("refine_face_boxes", True)),
                     mask_scale_divisor=args.mask_scale_divisor,
                     filter_threads=args.filter_threads)
        if args.no_audio:
            os.replace(tmp, final)
        else:
            mux_audio(tmp, video_path, final)

    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {}
    report.update({
        "morning_confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "delivery_ready": True,
        "final_video": final,
        "review_confirmed_count": len(review_confirmed),
        "low_conf_bridge_count": low_conf_bridge_count,
        "review_summary": confirmed.get("summary", {}),
    })
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[完成] 交付成片：{final}")
    return 0


def main():
    sys.exit(run_confirm(parse_args()))


if __name__ == "__main__":
    main()
