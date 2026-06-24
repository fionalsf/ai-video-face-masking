#!/usr/bin/env python3
"""Morning confirm: merge Review decisions into final.mp4."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

from event_builder import TIER_AUTO
from render import events_to_render, mux_audio, render_video
from video_meta import get_video_meta


def parse_args():
    p = argparse.ArgumentParser(description="合并 Review 确认结果 → final.mp4")
    p.add_argument("--output-dir", required=True, help="pipeline 输出目录 output/视频名")
    p.add_argument("--expand", type=float, default=0.18)
    p.add_argument("--mosaic-block", type=int, default=22)
    p.add_argument("--encoder", default="auto")
    return p.parse_args()


def load_confirmed(review_dir: str) -> dict:
    path = os.path.join(review_dir, "confirmed_events.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 {path}，请先运行 Review UI")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_face_events(out_dir: str) -> dict:
    path = os.path.join(out_dir, "face_events.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_confirm(args) -> int:
    out_dir = os.path.abspath(args.output_dir)
    review_dir = os.path.join(out_dir, "review")
    draft = os.path.join(out_dir, "masked_draft.mp4")
    final = os.path.join(out_dir, "final.mp4")
    report_path = os.path.join(out_dir, "review_report.json")

    if not os.path.isfile(draft):
        print(f"[错误] 找不到 masked_draft.mp4：{draft}", file=sys.stderr)
        return 1

    confirmed = load_confirmed(review_dir)
    face_data = load_face_events(out_dir)
    video_path = face_data["video"]
    meta = get_video_meta(video_path)

    auto_events = [e for e in face_data["events"] if e.get("tier") == TIER_AUTO]
    confirmed_map = {e["event_id"]: e for e in confirmed.get("events", []) if e.get("status")}

    review_confirmed = []
    for ev in face_data["events"]:
        if ev.get("tier") != "review":
            continue
        dec = confirmed_map.get(ev["event_id"])
        if dec and dec.get("status") == "confirmed_face":
            review_confirmed.append(ev)

    all_mask_events = auto_events + review_confirmed
    print(f"[确认] Auto={len(auto_events)} + Review确认={len(review_confirmed)} → 共 {len(all_mask_events)} 个打码 event")

    if not all_mask_events:
        shutil.copy2(draft, final)
    else:
        render = events_to_render(all_mask_events, meta["frames"])
        tmp = final + ".tmp.mp4"
        render_video(video_path, tmp, render, meta, expand=args.expand,
                     mosaic_block=args.mosaic_block, encoder=args.encoder)
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
