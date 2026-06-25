"""Build production behavior events from identity clusters + temporal continuity."""



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

from event_merge import center_continuous, DEFAULT_CENTER_CONTINUITY_RATIO

from event_builder import bbox_iou

from event_quality import score_event_dict

from gap_analysis import group_by_track, analyze_gaps, presence_segment_id

from identity_stitching import (

    IDENTITY_CLUSTERS_NAME,

    TARGET_EVENT_RANGE,

    load_identity_clusters,

)

from video_meta import get_video_meta, sec_to_timecode



BEHAVIOR_EVENTS_NAME = "behavior_events.json"

DEFAULT_BEHAVIOR_GAP_SEC = 8.0

BEHAVIOR_BUILDER_LAYER = "identity_behavior_builder_v1.2"  # V1.2 track_id拆分

BEHAVIOR_SEMANTIC = "identity cluster 内的时间连续行为段（track_id 变更处强制切段）"





def _det_bbox_continuous(prev: dict, curr: dict) -> bool:

    if bbox_iou(prev["bbox"], curr["bbox"]) >= 0.2:

        return True

    return center_continuous(prev["bbox"], curr["bbox"], ratio=DEFAULT_CENTER_CONTINUITY_RATIO)





def chunk_identity_detections(

    detections: list[dict],

    *,

    behavior_gap_sec: float = DEFAULT_BEHAVIOR_GAP_SEC,

) -> list[list[dict]]:

    """V1.2 track_id拆分：identity stitch 合并的多 track 在 track_id 变更处强制切段。"""

    if not detections:

        return []

    ordered = sorted(detections, key=lambda d: (float(d["t"]), int(d["frame"])))

    chunks: list[list[dict]] = [[ordered[0]]]

    for det in ordered[1:]:

        prev = chunks[-1][-1]

        gap = float(det["t"]) - float(prev["t"])

        track_changed = int(det["track_id"]) != int(prev["track_id"])

        if gap > behavior_gap_sec or not _det_bbox_continuous(prev, det) or track_changed:

            chunks.append([det])

        else:

            chunks[-1].append(det)

    return chunks





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





def build_identity_behavior_events(

    tracked: list[dict],

    identity_clusters: list[dict[str, Any]],

    *,

    output_dir: str | None = None,

    video: str | None = None,

    fps: float = 30.0,

    frame_h: int = 1080,

    frame_w: int = 1920,

    total_frames: int | None = None,

    detect_interval: int | None = None,

    behavior_gap_sec: float = DEFAULT_BEHAVIOR_GAP_SEC,

) -> dict[str, Any]:

    from rules import suggest_rule_hints



    behavior_gap_sec = max(5.0, min(8.0, behavior_gap_sec))

    pre_pad, post_pad = resolve_temporal_padding(fps, detect_interval=detect_interval)

    by_track = group_by_track(tracked)

    gap_result = analyze_gaps(tracked, fps=fps)



    behavior_events: list[dict[str, Any]] = []

    event_num = 0



    for cluster in identity_clusters:

        identity_id = cluster["identity_id"]

        track_ids = cluster["track_ids"]

        dets: list[dict] = []

        for tid in sorted(track_ids):

            dets.extend(by_track.get(int(tid), []))



        chunks = chunk_identity_detections(dets, behavior_gap_sec=behavior_gap_sec)

        cluster_track_ids = sorted(int(t) for t in track_ids)



        for chunk in chunks:

            event_num += 1

            primary_track = int(chunk[0]["track_id"])

            chunk_track_ids = sorted({int(d["track_id"]) for d in chunk})

            face_ev = _make_event(

                event_num,

                primary_track,

                chunk,

                frame_h,

                frame_w,

                suggest_rule_hints,

                fps=fps,

                total_frames=total_frames,

                pre_padding_sec=pre_pad,

                post_padding_sec=post_pad,

            )



            pres_ids: set[str] = set()

            for d in chunk:

                tid = int(d["track_id"])

                groups = gap_result.presence_by_track.get(tid, [])

                pid = _presence_id_for_detection(d, groups, tid)

                if pid:

                    pres_ids.add(pid)



            cross_track = len(chunk_track_ids) > 1

            cross_presence = len(pres_ids) > 1

            hints = list(face_ev.rule_hints)

            if len(cluster_track_ids) > 1:

                hints.append(f"identity_stitch:{len(cluster_track_ids)}")

            if cross_presence:

                hints.append("cross_presence_span")



            tier = face_ev.tier

            review_status = (

                "confirmed_face" if tier == TIER_AUTO

                else "logged_only" if tier == TIER_LOW_CONF

                else "pending"

            )

            bevt_id = f"bevt_{event_num:04d}"

            traj = [

                {

                    "t": d["t"],

                    "frame": d["frame"],

                    "bbox": d["bbox"],

                    "conf": d["conf"],

                    "track_id": int(d["track_id"]),

                }

                for d in chunk

            ]



            behavior_events.append({

                "behavior_event_id": bevt_id,

                "event_id": bevt_id,

                "identity_id": identity_id,

                "source_track_ids": chunk_track_ids,

                "primary_track_id": primary_track,

                "tier": tier,

                "review_status": review_status,

                "cross_track_merge": cross_track,

                "presence_segments_spanned": sorted(pres_ids),

                "cross_presence_span": cross_presence,

                "start_time": face_ev.start_time,

                "end_time": face_ev.end_time,

                "start_timecode": sec_to_timecode(face_ev.start_time),

                "end_timecode": sec_to_timecode(face_ev.end_time),

                "start_frame": face_ev.start_frame,

                "end_frame": face_ev.end_frame,

                "duration_sec": round(face_ev.end_time - face_ev.start_time, 3),

                "avg_confidence": face_ev.avg_confidence,

                "peak_confidence": face_ev.peak_confidence,

                "detection_count": face_ev.detection_count,

                "trajectory": traj,

                "rule_hints": hints,

                "semantic_unit": "identity_temporal_segment",

            })



    for ev in behavior_events:

        ev["event_quality_score"] = score_event_dict(ev, fps=fps)



    target_lo, target_hi = TARGET_EVENT_RANGE

    count = len(behavior_events)

    doc = {

        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),

        "layer": BEHAVIOR_BUILDER_LAYER,

        "semantic_definition": BEHAVIOR_SEMANTIC,

        "pipeline": "detection -> tracking -> presence -> identity_stitching -> identity_behavior_builder",

        "video": video,

        "fps": fps,

        "parameters": {

            "behavior_gap_sec": behavior_gap_sec,

            "event_unit": "identity_id",

            "track_id_as_event_unit": False,

            "split_on_track_id_change": True,  # V1.2 track_id拆分

        },

        "identity_cluster_count": len(identity_clusters),

        "behavior_event_count": count,

        "target_event_range": list(TARGET_EVENT_RANGE),

        "target_in_range": target_lo <= count <= target_hi,

        "events": behavior_events,

    }



    stats = {

        "behavior_event_count": count,

        "identity_cluster_count": len(identity_clusters),

        "target_in_range": doc["target_in_range"],

        "behavior_events": behavior_events,

    }



    if output_dir:

        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, BEHAVIOR_EVENTS_NAME)

        with open(path, "w", encoding="utf-8") as f:

            json.dump(doc, f, ensure_ascii=False, indent=2)

        stats["behavior_events_path"] = path



    return stats





def behavior_event_to_face_event(ev: dict) -> FaceEvent:

    """Adapter for render / review pack (trajectory retains per-det track_id)."""

    traj = [

        {"t": p["t"], "frame": p["frame"], "bbox": p["bbox"], "conf": p["conf"]}

        for p in (ev.get("trajectory") or [])

    ]

    return FaceEvent(

        event_id=ev["event_id"],

        track_id=int(ev.get("primary_track_id") or ev["source_track_ids"][0]),

        tier=ev.get("tier", TIER_REVIEW),

        start_time=float(ev["start_time"]),

        end_time=float(ev["end_time"]),

        start_frame=int(ev["start_frame"]),

        end_frame=int(ev["end_frame"]),

        avg_confidence=float(ev.get("avg_confidence") or 0),

        peak_confidence=float(ev.get("peak_confidence") or 0),

        detection_count=int(ev.get("detection_count") or len(traj)),

        trajectory=traj,

        rule_hints=list(ev.get("rule_hints") or []),

    )





def run_identity_behavior_pipeline(

    tracked: list[dict],

    *,

    output_dir: str,

    video: str | None = None,

    fps: float = 30.0,

    frame_h: int = 1080,

    frame_w: int = 1920,

    total_frames: int | None = None,

    detect_interval: int | None = None,

    identity_clusters: list[dict[str, Any]] | None = None,

    behavior_gap_sec: float = DEFAULT_BEHAVIOR_GAP_SEC,

) -> dict[str, Any]:

    if identity_clusters is None:

        identity_clusters, _doc = load_identity_clusters(output_dir)

    return build_identity_behavior_events(

        tracked,

        identity_clusters,

        output_dir=output_dir,

        video=video,

        fps=fps,

        frame_h=frame_h,

        frame_w=frame_w,

        total_frames=total_frames,

        detect_interval=detect_interval,

        behavior_gap_sec=behavior_gap_sec,

    )





def main() -> int:

    p = argparse.ArgumentParser(description="Identity-based behavior event builder")

    p.add_argument("--output-dir", required=True)

    p.add_argument("--behavior-gap", type=float, default=DEFAULT_BEHAVIOR_GAP_SEC)

    args = p.parse_args()



    out_dir = os.path.abspath(args.output_dir)

    tracked_path = os.path.join(out_dir, "tracked_detections.json")

    with open(tracked_path, encoding="utf-8") as f:

        tracked = json.load(f)



    summary: dict = {}

    summary_path = os.path.join(out_dir, "detection_summary.json")

    if os.path.isfile(summary_path):

        with open(summary_path, encoding="utf-8") as f:

            summary = json.load(f)



    video = summary.get("video")

    fps = float(summary.get("fps") or 30.0)

    frame_h, frame_w, total_frames = 1080, 1920, None

    if video and os.path.isfile(video):

        meta = get_video_meta(video)

        frame_h, frame_w = meta["height"], meta["width"]

        total_frames = meta["frames"]

        fps = float(meta["fps"])



    stats = run_identity_behavior_pipeline(

        tracked,

        output_dir=out_dir,

        video=video,

        fps=fps,

        frame_h=frame_h,

        frame_w=frame_w,

        total_frames=total_frames,

        detect_interval=int(summary.get("detect_interval") or 0) or None,

        behavior_gap_sec=args.behavior_gap,

    )

    print(f"[done] {stats.get('behavior_events_path')}")

    print(f"  {stats['behavior_event_count']} behavior events | target_in_range={stats['target_in_range']}")

    return 0





if __name__ == "__main__":

    raise SystemExit(main())

