#!/usr/bin/env python3
"""python -m modules.render -i events.json -v video.mp4 -o out.mp4"""

from __future__ import annotations

import argparse
import sys

from core.io import load_events
from modules.render.renderer import render_events
from utils.video_meta import get_video_meta


def main() -> int:
    p = argparse.ArgumentParser(description="Render module")
    p.add_argument("-i", "--input", required=True, help="Scored events JSON")
    p.add_argument("-v", "--video", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--expand", type=float, default=0.18)
    p.add_argument("--mosaic-block", type=int, default=22)
    p.add_argument("--encoder", default="auto")
    args = p.parse_args()
    meta = get_video_meta(args.video)
    render_events(
        args.video, args.output, load_events(args.input), meta,
        expand=args.expand, mosaic_block=args.mosaic_block, encoder=args.encoder,
    )
    print(f"[render] -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
