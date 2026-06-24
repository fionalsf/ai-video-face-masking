"""Merge Review confirmations into final.mp4."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

from core.event import Event, Tier
from core.io import load_json
from modules.render.renderer import events_to_render, mux_audio, render_video
from utils.video_meta import get_video_meta


def parse_args():
    p = argparse.ArgumentParser(description="Confirm Review decisions -> final.mp4")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--expand", type=float, default=0.18)
    p.add_argument("--mosaic-block", type=int, default=22)
    p.add_argument("--encoder", default="auto")
    return p.parse_args()


def run_confirm(args) -> int:
    out_dir = os.path.abspath(args.output_dir)
    review_dir = os.path.join(out_dir, "review")
    draft = os.path.join(out_dir, "masked_draft.mp4")
    final = os.path.join(out_dir, "final.mp4")
    report_path = os.path.join(out_dir, "review_report.json")

    if not os.path.isfile(draft):
        print(f"[error] missing masked_draft.mp4: {draft}", file=sys.stderr)
        return 1

    with open(os.path.join(review_dir, "confirmed_events.json"), encoding="utf-8") as f:
        confirmed = json.load(f)
    face_data = load_json(os.path.join(out_dir, "face_events.json"))
    video_path = face_data["video"]
    meta = get_video_meta(video_path)

    all_events = [Event.from_dict(e) for e in face_data["events"]]
    auto_events = [e for e in all_events if e.tier == Tier.AUTO.value]
    confirmed_map = {e["event_id"]: e for e in confirmed.get("events", []) if e.get("status")}

    review_confirmed: list[Event] = []
    for ev in all_events:
        if ev.tier != Tier.REVIEW.value:
            continue
        dec = confirmed_map.get(ev.event_id)
        if dec and dec.get("status") == "confirmed_face":
            review_confirmed.append(ev)

    mask_events = auto_events + review_confirmed
    print(f"[confirm] auto={len(auto_events)} + review_ok={len(review_confirmed)} -> {len(mask_events)} events")

    if not mask_events:
        shutil.copy2(draft, final)
    else:
        render = events_to_render(mask_events, meta["frames"])
        tmp = final + ".tmp.mp4"
        render_video(video_path, tmp, render, meta,
                     expand=args.expand, mosaic_block=args.mosaic_block, encoder=args.encoder)
        mux_audio(tmp, video_path, final)

    report = load_json(report_path) if os.path.isfile(report_path) else {}
    report.update({
        "morning_confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "delivery_ready": True,
        "final_video": final,
        "review_confirmed_count": len(review_confirmed),
        "review_summary": confirmed.get("summary", {}),
    })
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[confirm] final -> {final}")
    return 0


def main():
    sys.exit(run_confirm(parse_args()))


if __name__ == "__main__":
    main()
