"""Pipeline orchestrator — wires modules, no business logic."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

from core.event import Tier, events_by_tier
from modules.detect.detector import FaceDetector
from modules.event.builder import build_events
from modules.render.renderer import render_events
from modules.review.export import export_review_pack, filter_review_events, write_confirmed_events_template
from modules.scoring.scorer import score_events
from modules.track.tracker import ByteTracker
from utils.audit_log import write_audit_log
from utils.low_conf_log import write_low_conf_stats
from utils.video_meta import get_video_meta, safe_video_stem


def parse_args():
    p = argparse.ArgumentParser(description="Event pipeline (orchestrator)")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output-dir", default="output")
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--model", default="models/face.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--event-gap", type=float, default=1.0)
    p.add_argument("--expand", type=float, default=0.18)
    p.add_argument("--mosaic-block", type=int, default=22)
    p.add_argument("--encoder", default="auto")
    p.add_argument("--skip-render", action="store_true")
    return p.parse_args()


def run_pipeline(args) -> int:
    video_path = os.path.abspath(args.input)
    if not os.path.isfile(video_path):
        print(f"[error] video not found: {video_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.model):
        print(f"[error] model not found: {args.model}", file=sys.stderr)
        return 1

    stem = safe_video_stem(video_path)
    out_dir = os.path.join(os.path.abspath(args.output_dir), stem)
    review_dir = os.path.join(out_dir, "review")
    os.makedirs(review_dir, exist_ok=True)

    meta = get_video_meta(video_path)
    print(f"[info] {stem} | {meta['width']}x{meta['height']} @ {meta['fps']:.2f}fps | {meta['frames']} frames")

    # 1 detect
    detector = FaceDetector(args.model, args.device, args.conf, args.imgsz)
    try:
        detections = detector.detect_video_sparse(video_path, meta["fps"], meta["frames"], args.interval)
    finally:
        detector.close()
    print(f"[detect] {len(detections)} boxes")

    # 2 track
    tracked = ByteTracker().track_detections(detections)
    print(f"[track] {len(tracked)} tracked boxes")

    # 3 event
    events = build_events(tracked, gap_sec=args.event_gap)
    print(f"[event] {len(events)} events")

    # 4 scoring
    events = score_events(events, meta["width"], meta["height"])
    tiers = events_by_tier(events)
    auto_ev = tiers[Tier.AUTO.value]
    review_ev = tiers[Tier.REVIEW.value]
    low_ev = tiers[Tier.LOW_CONF.value]
    print(f"[scoring] auto={len(auto_ev)} review={len(review_ev)} low={len(low_ev)}")

    with open(os.path.join(out_dir, "face_events.json"), "w", encoding="utf-8") as f:
        json.dump({
            "video": video_path, "fps": meta["fps"],
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "events": [e.to_dict() for e in events],
        }, f, ensure_ascii=False, indent=2)

    write_low_conf_stats(out_dir, video_path, low_ev)
    write_audit_log(out_dir, video_path, auto_ev)

    draft_path = os.path.join(out_dir, "masked_draft.mp4")
    if not args.skip_render:
        if auto_ev:
            print(f"[render] auto {len(auto_ev)} events -> masked_draft.mp4")
            render_events(video_path, draft_path, auto_ev, meta,
                          expand=args.expand, mosaic_block=args.mosaic_block, encoder=args.encoder)
        else:
            shutil.copy2(video_path, draft_path)
            print("[render] no auto events, copied source")
    else:
        print("[render] skipped")

    if review_ev:
        export_review_pack(video_path, review_dir, review_ev)
        write_confirmed_events_template(review_dir, video_path)
        print(f"[review] {len(review_ev)} pending -> {review_dir}")
    else:
        print("[review] none")

    report = {
        "video": video_path,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "events_total": len(events),
        "auto_masked": len(auto_ev),
        "pending_review": len(review_ev),
        "low_conf_logged": len(low_ev),
        "output_dir": out_dir,
        "masked_draft": draft_path,
        "review_dir": review_dir if review_ev else None,
        "delivery_ready": len(review_ev) == 0,
        "morning_action": "none" if not review_ev else f"streamlit run review_ui.py -- --review-dir {review_dir}",
    }
    with open(os.path.join(out_dir, "review_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if report["delivery_ready"]:
        shutil.copy2(draft_path, os.path.join(out_dir, "final.mp4"))
        print("[done] final.mp4 ready (no review)")

    print(f"\n[done] {out_dir}")
    if review_ev:
        print(f'  streamlit run review_ui.py -- --review-dir "{review_dir}"')
    return 0


def main():
    sys.exit(run_pipeline(parse_args()))


if __name__ == "__main__":
    main()
