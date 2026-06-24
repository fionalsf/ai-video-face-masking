#!/usr/bin/env python3
"""Timeline render executor: overlay debug or block pixelation mosaic (timeline.json only)."""

from __future__ import annotations

import argparse
import os
import sys

from timeline_overlay import (
    DEFAULT_OUTPUT_BY_MODE,
    MODE_FINAL,
    MODE_OVERLAY,
    MODE_PREVIEW,
    MOSAIC_LEVELS,
    PIXELATION_MODES,
    render_timeline_video,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render from timeline.json only. "
            "overlay=bbox debug; preview/final=block pixelation mosaic. "
            "No detection, tracking, or bbox recomputation."
        )
    )
    p.add_argument("--video", required=True, help="Source video path")
    p.add_argument("--timeline", required=True, help="timeline.json path")
    p.add_argument(
        "--output",
        default=None,
        help="Output mp4 (default by mode: overlay.mp4 / preview_mosaic.mp4 / final_mosaic.mp4)",
    )
    p.add_argument(
        "--mode",
        choices=[MODE_OVERLAY, MODE_PREVIEW, MODE_FINAL],
        default=MODE_OVERLAY,
        help="overlay=draw bboxes; preview/final=block pixelation mosaic",
    )
    p.add_argument(
        "--mosaic_level",
        choices=list(MOSAIC_LEVELS),
        default="high",
        help="Mosaic intensity for preview/final (low/medium/high/extreme)",
    )
    p.add_argument("--show_bbox", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--show_event_id", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--show_track_id", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--no-hud", action="store_true", help="Hide top-left frame HUD")
    p.add_argument("--thickness", type=int, default=2, help="Box line thickness when shown")
    p.add_argument("--start-sec", type=float, default=None, help="Clip start time in seconds")
    p.add_argument("--end-sec", type=float, default=None, help="Clip end time in seconds")
    return p.parse_args()


def resolve_debug_defaults(mode: str, value: bool | None) -> bool:
    if value is not None:
        return value
    return mode == MODE_OVERLAY


def main() -> int:
    args = parse_args()
    video = os.path.abspath(args.video)
    timeline = os.path.abspath(args.timeline)
    output = os.path.abspath(args.output or DEFAULT_OUTPUT_BY_MODE[args.mode])

    if not os.path.isfile(video):
        print(f"[error] Video not found: {video}", file=sys.stderr)
        return 1
    if not os.path.isfile(timeline):
        print(f"[error] Timeline not found: {timeline}", file=sys.stderr)
        return 1

    show_bbox = resolve_debug_defaults(args.mode, args.show_bbox)
    show_event_id = resolve_debug_defaults(args.mode, args.show_event_id)
    show_track_id = resolve_debug_defaults(args.mode, args.show_track_id)
    # Production deliverables (preview/final) must not burn debug HUD into the video.
    show_hud = False if args.no_hud or args.mode in (MODE_PREVIEW, MODE_FINAL) else True

    stats = render_timeline_video(
        video,
        timeline,
        output,
        mode=args.mode,
        show_bbox=show_bbox,
        show_event_id=show_event_id,
        show_track_id=show_track_id,
        show_hud=show_hud,
        thickness=args.thickness,
        mosaic_level=args.mosaic_level,
        progress_desc=f"{args.mode} render",
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )

    print(f"[done] {stats['output']}")
    print(f"  mode: {stats['mode']}")
    print("  source: timeline.json only (executor, no detection/tracking)")
    print(f"  accepted events: {stats['accepted_events']}")
    print(f"  timeline entries: {stats['timeline_entries']}")
    print(
        f"  frames with regions: {stats['frames_with_overlay']} / {stats['rendered_frames']}"
    )
    if stats.get("start_frame") is not None or stats.get("end_frame") is not None:
        print(
            f"  clip: frames {stats['start_frame']}-{stats['end_frame']} "
            f"({stats['rendered_frames']} rendered)"
        )
    if stats["mode"] in PIXELATION_MODES:
        print(f"  mosaic_level: {stats['mosaic_level']} (downscale={stats['mosaic_downscale']})")
        print(f"  expand_ratio: {stats['expand_ratio']}")
        print(f"  pixelation regions: {stats['mosaic_regions']}")
    print(f"  debug: bbox={show_bbox} event_id={show_event_id} track_id={show_track_id} hud={show_hud}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
