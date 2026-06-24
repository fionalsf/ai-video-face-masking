#!/usr/bin/env python3
"""Draw timeline.json bboxes on full video for visual validation (no mosaic)."""

from __future__ import annotations

import argparse
import os
import sys

from timeline_overlay import DEFAULT_TIMELINE_PREVIEW_OUTPUT, render_timeline_overlay

DEFAULT_OUTPUT = DEFAULT_TIMELINE_PREVIEW_OUTPUT


def parse_args():
    p = argparse.ArgumentParser(description="Timeline debug overlay preview video")
    p.add_argument("--video", required=True, help="Source video path")
    p.add_argument("--timeline", required=True, help="timeline.json path")
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output mp4 path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument("--thickness", type=int, default=2, help="Bbox line thickness")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video = os.path.abspath(args.video)
    timeline = os.path.abspath(args.timeline)
    output = os.path.abspath(args.output)

    if not os.path.isfile(video):
        print(f"[error] Video not found: {video}", file=sys.stderr)
        return 1
    if not os.path.isfile(timeline):
        print(f"[error] Timeline not found: {timeline}", file=sys.stderr)
        return 1

    stats = render_timeline_overlay(
        video,
        timeline,
        output,
        thickness=args.thickness,
        label_style="preview",
        progress_desc="timeline preview",
    )
    print(f"[done] {stats['output']}")
    print(f"  accepted events: {stats['accepted_events']}")
    print(f"  timeline entries: {stats['timeline_entries']}")
    print(f"  frames with overlay: {stats['frames_with_overlay']} / {stats['total_frames']}")
    print(f"  entries drawn: {stats['entries_drawn']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
