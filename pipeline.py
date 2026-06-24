#!/usr/bin/env python3
"""Single-video pipeline: detect -> track -> presence -> identity stitch -> behavior events."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from audit_log import write_audit_log
from event_builder import TIER_AUTO, TIER_LOW_CONF, TIER_REVIEW, build_events, events_by_tier
from event_merge import save_segmentation_events
from export_events import export_review_pack, write_confirmed_events_template
from identity_behavior_builder import (
    behavior_event_to_face_event,
    build_identity_behavior_events,
)
from identity_stitching import run_identity_stitching
from low_conf_log import write_low_conf_stats
from render import render_masked_output
from tracker import run_detect_track
from video_meta import get_video_meta, safe_video_stem

PIPELINE_LABEL = "detection -> tracking -> presence -> identity_stitching -> identity_behavior_builder"


def parse_args():
    p = argparse.ArgumentParser(description="Identity-based face event pipeline (single video)")
    p.add_argument("-i", "--input", required=True, help="Input video")
    p.add_argument("-o", "--output-dir", default="output", help="Output root directory")
    p.add_argument("--interval", type=int, default=5, help="Detection stride (frames)")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--model", default="models/face.pt", help="YOLO-face weights")
    p.add_argument("--device", default="0", help="GPU id or cpu")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--event-gap", type=float, default=1.0, help="Presence segmentation gap (seconds)")
    p.add_argument("--stitch-gap", type=float, default=60.0, help="Temporal soft prior tau for identity graph (sec)")
    p.add_argument("--behavior-gap", type=float, default=8.0, help="Behavior split gap within identity (seconds)")
    p.add_argument("--expand", type=float, default=0.18)
    p.add_argument("--mosaic-block", type=int, default=22)
    p.add_argument("--encoder", default="auto")
    p.add_argument("--skip-render", action="store_true", help="Skip Auto-tier render (debug)")
    return p.parse_args()


def run_pipeline(args) -> int:
    video_path = os.path.abspath(args.input)
    if not os.path.isfile(video_path):
        print(f"[error] Video not found: {video_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.model):
        print(f"[error] Model not found: {args.model}", file=sys.stderr)
        return 1

    stem = safe_video_stem(video_path)
    out_dir = os.path.join(os.path.abspath(args.output_dir), stem)
    review_dir = os.path.join(out_dir, "review")
    os.makedirs(review_dir, exist_ok=True)

    meta = get_video_meta(video_path)
    print(
        f"[info] {stem} | {meta['width']}x{meta['height']} @ {meta['fps']:.2f}fps"
        f" | {meta['frames']} frames"
    )

    print("[1/5] Detection + Tracking...")
    detections = run_detect_track(
        video_path, args.model, meta,
        device=args.device, conf=args.conf, imgsz=args.imgsz, interval=args.interval,
    )
    with open(os.path.join(out_dir, "tracked_detections.json"), "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)

    print("[2/5] Presence segmentation (Scheme C, debug artifacts)...")
    builder_stats: dict = {}
    segmentation_events = build_events(
        detections, gap_sec=args.event_gap,
        frame_h=meta["height"], frame_w=meta["width"],
        merge_stats=builder_stats,
        fps=float(meta["fps"]),
        total_frames=meta["frames"],
        detect_interval=args.interval,
        output_dir=out_dir,
    )
    save_segmentation_events(
        segmentation_events, out_dir, video=video_path, fps=float(meta["fps"]),
    )
    print(f"      segmentation events (non-production): {len(segmentation_events)}")

    print("[3/5] Identity stitching...")
    stitch_stats = run_identity_stitching(
        detections,
        output_dir=out_dir,
        video=video_path,
        temporal_tau=args.stitch_gap,
    )
    print(
        f"      {stitch_stats['track_count']} tracks -> {stitch_stats['identity_cluster_count']} identities"
        f" | appearance={stitch_stats['appearance_method']}"
        f" | links={stitch_stats['linked_edge_count']}"
    )

    print("[4/5] Identity behavior event builder (production)...")
    behavior_stats = build_identity_behavior_events(
        detections,
        stitch_stats["clusters"],
        output_dir=out_dir,
        video=video_path,
        fps=float(meta["fps"]),
        frame_h=meta["height"],
        frame_w=meta["width"],
        total_frames=meta["frames"],
        detect_interval=args.interval,
        behavior_gap_sec=args.behavior_gap,
    )
    print(
        f"      behavior events: {behavior_stats['behavior_event_count']}"
        f" | target_30_80={behavior_stats['target_in_range']}"
    )

    events = [behavior_event_to_face_event(ev) for ev in behavior_stats["behavior_events"]]
    tiers = events_by_tier(events)
    auto_ev = tiers[TIER_AUTO]
    review_ev = tiers[TIER_REVIEW]
    low_ev = tiers[TIER_LOW_CONF]

    print(
        f"[5/5] Review/Render — Auto={len(auto_ev)} Review={len(review_ev)} LowConf={len(low_ev)}"
    )

    with open(os.path.join(out_dir, "face_events.json"), "w", encoding="utf-8") as f:
        json.dump({
            "video": video_path,
            "fps": meta["fps"],
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": PIPELINE_LABEL,
            "event_unit": "identity_id",
            "events": [e.to_dict() for e in events],
        }, f, ensure_ascii=False, indent=2)

    write_low_conf_stats(out_dir, video_path, low_ev)
    write_audit_log(out_dir, video_path, auto_ev)

    draft_path = os.path.join(out_dir, "masked_draft.mp4")
    if not args.skip_render:
        auto_dicts = [e.to_dict() for e in auto_ev]
        if auto_dicts:
            print(f"[Auto] Rendering {len(auto_ev)} events -> masked_draft.mp4")
            render_masked_output(
                video_path, draft_path, auto_dicts, meta,
                expand=args.expand, mosaic_block=args.mosaic_block, encoder=args.encoder,
            )
        else:
            import shutil
            shutil.copy2(video_path, draft_path)
            print("[Auto] No Auto events; copied source as masked_draft.mp4")
    else:
        print("[Auto] --skip-render: skipped")

    if review_ev:
        export_review_pack(video_path, review_dir, review_ev)
        write_confirmed_events_template(review_dir, video_path)
        print(f"[Review] {len(review_ev)} events pending -> {review_dir}")
    else:
        print("[Review] No events need review")

    report = {
        "video": video_path,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "events_total": len(events),
        "auto_masked": len(auto_ev),
        "pending_review": len(review_ev),
        "low_conf_logged": len(low_ev),
        "builder_stats": builder_stats,
        "stitch_stats": {
            "track_count": stitch_stats["track_count"],
            "identity_cluster_count": stitch_stats["identity_cluster_count"],
            "appearance_method": stitch_stats["appearance_method"],
        },
        "behavior_stats": {
            "behavior_event_count": behavior_stats["behavior_event_count"],
            "target_in_range": behavior_stats["target_in_range"],
        },
        "pipeline": PIPELINE_LABEL,
        "production_events": "behavior_events.json",
        "output_dir": out_dir,
        "masked_draft": draft_path,
        "review_dir": review_dir if review_ev else None,
        "delivery_ready": len(review_ev) == 0,
        "morning_action": "none" if not review_ev else f"streamlit run review_ui.py -- --review-dir {review_dir}",
    }
    report_path = os.path.join(out_dir, "review_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if report["delivery_ready"]:
        final_path = os.path.join(out_dir, "final.mp4")
        import shutil
        shutil.copy2(draft_path, final_path)
        print("[done] No Review events; final.mp4 ready")

    print(f"\n[done] Output: {out_dir}")
    print("  identity_clusters.json | behavior_events.json | review_report.json")
    if review_ev:
        print(f'  Morning: streamlit run review_ui.py -- --review-dir "{review_dir}"')
    return 0


def main():
    sys.exit(run_pipeline(parse_args()))


if __name__ == "__main__":
    main()
