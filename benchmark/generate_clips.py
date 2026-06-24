"""Generate standardized evaluation clip manifest (20-50 segments)."""

from __future__ import annotations

import argparse
import json
import os
import sys

from benchmark.ground_truth import CLIPS_SCHEMA_VERSION, generate_uniform_clips, save_json


def main() -> int:
    p = argparse.ArgumentParser(description="Generate benchmark clip manifest")
    p.add_argument("--video-id", required=True, help="Video stem e.g. DJI_20260511100755_0008_D")
    p.add_argument("--duration-sec", type=float, required=True)
    p.add_argument("--target-count", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    clips = generate_uniform_clips(args.duration_sec, target_count=args.target_count)
    doc = {
        "schema_version": CLIPS_SCHEMA_VERSION,
        "video_id": args.video_id,
        "duration_sec": args.duration_sec,
        "clip_count": len(clips),
        "target_count": args.target_count,
        "clips": clips,
    }
    out = args.out or os.path.join("benchmark", "clips", f"{args.video_id}_clips.json")
    save_json(out, doc)
    print(f"[clips] {os.path.abspath(out)} ({len(clips)} segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
