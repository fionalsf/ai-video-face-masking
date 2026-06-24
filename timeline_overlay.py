"""Draw overlays strictly from timeline.json entries (no detection/tracking)."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from video_meta import get_video_meta, sec_to_timecode

GREEN = (0, 255, 0)
LABEL_BG = (0, 200, 0)
LABEL_FG = (0, 0, 0)
HUD_BG = (0, 0, 0)
HUD_FG = (0, 255, 0)

DEFAULT_TIMELINE_PREVIEW_OUTPUT = os.path.join("output", "debug", "timeline_preview.mp4")
DEFAULT_OVERLAY_OUTPUT = os.path.join("output", "render", "overlay.mp4")
DEFAULT_PREVIEW_MOSAIC_OUTPUT = os.path.join("output", "render", "preview_mosaic.mp4")
DEFAULT_FINAL_MOSAIC_OUTPUT = os.path.join("output", "render", "final_mosaic.mp4")

MODE_OVERLAY = "overlay"
MODE_PREVIEW = "preview"
MODE_FINAL = "final"
RENDER_MODES = (MODE_OVERLAY, MODE_PREVIEW, MODE_FINAL)
PIXELATION_MODES = (MODE_PREVIEW, MODE_FINAL)

DEFAULT_OUTPUT_BY_MODE = {
    MODE_OVERLAY: DEFAULT_OVERLAY_OUTPUT,
    MODE_PREVIEW: DEFAULT_PREVIEW_MOSAIC_OUTPUT,
    MODE_FINAL: DEFAULT_FINAL_MOSAIC_OUTPUT,
}

MOSAIC_LEVEL_LOW = "low"
MOSAIC_LEVEL_MEDIUM = "medium"
MOSAIC_LEVEL_HIGH = "high"
MOSAIC_LEVEL_EXTREME = "extreme"
MOSAIC_LEVELS = (
    MOSAIC_LEVEL_LOW,
    MOSAIC_LEVEL_MEDIUM,
    MOSAIC_LEVEL_HIGH,
    MOSAIC_LEVEL_EXTREME,
)
MOSAIC_DOWNSCALE = {
    MOSAIC_LEVEL_LOW: 0.3,
    MOSAIC_LEVEL_MEDIUM: 0.15,
    MOSAIC_LEVEL_HIGH: 0.08,
    MOSAIC_LEVEL_EXTREME: 0.03,
}
DEFAULT_MOSAIC_LEVEL = MOSAIC_LEVEL_HIGH
DEFAULT_EXPAND_RATIO = 0.2


def load_timeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_by_frame(entries: list[dict]) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for entry in entries:
        by_frame[int(entry["frame"])].append(entry)
    return dict(by_frame)


def clip_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def expand_bbox(
    bbox: list[float],
    width: int,
    height: int,
    expand_ratio: float = DEFAULT_EXPAND_RATIO,
) -> list[float]:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x = bw * expand_ratio
    pad_y = bh * expand_ratio
    return [
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(width - 1), x2 + pad_x),
        min(float(height - 1), y2 + pad_y),
    ]


def mosaic_downscale_ratio(level: str) -> float:
    if level not in MOSAIC_DOWNSCALE:
        raise ValueError(f"Unsupported mosaic_level: {level}. Use: {', '.join(MOSAIC_LEVELS)}")
    return MOSAIC_DOWNSCALE[level]


def apply_pixelation_to_bbox(
    frame: np.ndarray,
    bbox: list[float],
    *,
    downscale: float = MOSAIC_DOWNSCALE[DEFAULT_MOSAIC_LEVEL],
    expand_ratio: float = DEFAULT_EXPAND_RATIO,
) -> None:
    """Block pixelation: expand bbox, downscale ROI, upscale with NEAREST only."""
    h, w = frame.shape[:2]
    expanded = expand_bbox(bbox, w, h, expand_ratio)
    x1, y1, x2, y2 = clip_bbox(expanded, w, h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 1 or rw < 1:
        return

    ratio = max(0.01, min(1.0, downscale))
    sw = max(1, int(round(rw * ratio)))
    sh = max(1, int(round(rh * ratio)))
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


def draw_label_block(
    frame: np.ndarray,
    lines: list[str],
    anchor_x: int,
    anchor_y: int,
    font_scale: float,
    line_h: int,
) -> None:
    pad_x, pad_y = 4, 3
    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        max_w = max(max_w, tw)
    block_h = pad_y * 2 + line_h * len(lines)
    block_w = max_w + pad_x * 2
    y_top = max(0, anchor_y - block_h)
    x_left = max(0, min(frame.shape[1] - block_w, anchor_x))
    cv2.rectangle(frame, (x_left, y_top), (x_left + block_w, y_top + block_h), LABEL_BG, -1)
    for i, line in enumerate(lines):
        y = y_top + pad_y + (i + 1) * line_h - 4
        cv2.putText(
            frame,
            line,
            (x_left + pad_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            LABEL_FG,
            1,
            cv2.LINE_AA,
        )


def entry_label_lines(
    entry: dict,
    fps: float,
    *,
    style: str = "render",
    show_event_id: bool = True,
    show_track_id: bool = True,
) -> list[str]:
    event_id = entry.get("event_id", "?")
    track_id = entry.get("track_id", "?")
    frame_idx = int(entry["frame"])
    ts = entry.get("timestamp")
    if ts is None:
        ts = frame_idx / fps if fps > 0 else 0.0
    ts = float(ts)

    lines: list[str] = []
    if show_event_id or show_track_id:
        parts: list[str] = []
        if show_event_id:
            parts.append(str(event_id))
        if show_track_id:
            parts.append(f"track={track_id}")
        lines.append("  ".join(parts))

    if style == "preview":
        lines.append(f"F:{frame_idx} | {sec_to_timecode(ts)}")
    elif not lines:
        lines.append(f"t={ts:.3f}s  {sec_to_timecode(ts)}")
    else:
        lines.append(f"t={ts:.3f}s  {sec_to_timecode(ts)}")
    return lines


def draw_entry_debug(
    frame: np.ndarray,
    entry: dict,
    fps: float,
    thickness: int,
    font_scale: float,
    line_h: int,
    label_offset: int,
    *,
    style: str = "render",
    show_bbox: bool = True,
    show_event_id: bool = True,
    show_track_id: bool = True,
) -> None:
    if not (show_bbox or show_event_id or show_track_id):
        return

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clip_bbox(entry["bbox"], w, h)
    if show_bbox:
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, thickness)
    if show_event_id or show_track_id:
        draw_label_block(
            frame,
            entry_label_lines(
                entry,
                fps,
                style=style,
                show_event_id=show_event_id,
                show_track_id=show_track_id,
            ),
            x1,
            y1 - label_offset,
            font_scale,
            line_h,
        )


def draw_entry_overlay(
    frame: np.ndarray,
    entry: dict,
    fps: float,
    thickness: int,
    font_scale: float,
    line_h: int,
    label_offset: int,
    *,
    style: str = "render",
    show_bbox: bool = True,
    show_event_id: bool = True,
    show_track_id: bool = True,
) -> None:
    draw_entry_debug(
        frame,
        entry,
        fps,
        thickness,
        font_scale,
        line_h,
        label_offset,
        style=style,
        show_bbox=show_bbox,
        show_event_id=show_event_id,
        show_track_id=show_track_id,
    )


def draw_frame_hud(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    overlay_count: int,
    font_scale: float,
) -> None:
    tc = sec_to_timecode(frame_idx / fps if fps > 0 else 0.0)
    text = f"Frame {frame_idx} | {tc} | overlays {overlay_count}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    pad = 6
    cv2.rectangle(frame, (0, 0), (tw + pad * 2, th + pad * 2), HUD_BG, -1)
    cv2.putText(
        frame,
        text,
        (pad, th + pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        HUD_FG,
        2,
        cv2.LINE_AA,
    )


def render_timeline_video(
    video_path: str,
    timeline_path: str,
    output_path: str,
    *,
    mode: str = MODE_OVERLAY,
    show_bbox: bool = True,
    show_event_id: bool = True,
    show_track_id: bool = True,
    show_hud: bool = True,
    thickness: int = 2,
    label_style: str = "render",
    mosaic_level: str = DEFAULT_MOSAIC_LEVEL,
    expand_ratio: float = DEFAULT_EXPAND_RATIO,
    progress_desc: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> dict:
    """Render video using timeline.json entries only (overlay or pixelation mosaic)."""
    if mode not in RENDER_MODES:
        raise ValueError(f"Unsupported mode: {mode}. Use: {', '.join(RENDER_MODES)}")

    apply_pixelation = mode in PIXELATION_MODES
    downscale = mosaic_downscale_ratio(mosaic_level) if apply_pixelation else 0.0

    timeline = load_timeline(timeline_path)
    entries = timeline.get("entries", [])
    by_frame = index_by_frame(entries)

    video_path = os.path.abspath(video_path)
    tl_video = timeline.get("video")
    if tl_video and os.path.abspath(tl_video) != video_path:
        print(f"[warn] timeline video differs: {tl_video}", file=sys.stderr)

    meta = get_video_meta(video_path)
    fps = float(timeline.get("fps") or meta["fps"])
    width, height = meta["width"], meta["height"]
    total_frames = meta["frames"]
    start_frame = max(0, int(round(start_sec * fps))) if start_sec is not None else 0
    end_frame = (
        min(total_frames - 1, int(round(end_sec * fps)))
        if end_sec is not None
        else total_frames - 1
    )
    if start_frame > end_frame:
        raise ValueError(f"Invalid range: start_sec={start_sec} end_sec={end_sec}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {output_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot open video: {video_path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    font_scale = max(0.45, min(0.75, height / 1200))
    line_h = int(round(16 * font_scale / 0.55))
    range_desc = (
        f"{sec_to_timecode(start_frame / fps)}-{sec_to_timecode(end_frame / fps)}"
        if start_sec is not None or end_sec is not None
        else None
    )
    desc = progress_desc or (f"{mode} render")
    if range_desc:
        desc = f"{desc} [{range_desc}]"

    frames_with_overlay = 0
    entries_processed = 0
    mosaic_regions = 0

    try:
        for frame_idx in tqdm(range(start_frame, end_frame + 1), desc=desc, unit="frame"):
            ok, frame = cap.read()
            if not ok:
                break

            overlays = by_frame.get(frame_idx, [])
            if overlays:
                frames_with_overlay += 1
                for i, entry in enumerate(overlays):
                    if apply_pixelation:
                        apply_pixelation_to_bbox(
                            frame,
                            entry["bbox"],
                            downscale=downscale,
                            expand_ratio=expand_ratio,
                        )
                        mosaic_regions += 1
                    draw_entry_debug(
                        frame,
                        entry,
                        fps,
                        thickness,
                        font_scale,
                        line_h,
                        label_offset=i * (line_h * 2 + 8),
                        style=label_style,
                        show_bbox=show_bbox,
                        show_event_id=show_event_id,
                        show_track_id=show_track_id,
                    )
                    entries_processed += 1

            if show_hud:
                draw_frame_hud(frame, frame_idx, fps, len(overlays), font_scale)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return {
        "video": video_path,
        "timeline": os.path.abspath(timeline_path),
        "output": os.path.abspath(output_path),
        "mode": mode,
        "mosaic_level": mosaic_level if apply_pixelation else None,
        "mosaic_downscale": downscale if apply_pixelation else None,
        "expand_ratio": expand_ratio if apply_pixelation else None,
        "fps": fps,
        "total_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "rendered_frames": end_frame - start_frame + 1,
        "accepted_events": timeline.get(
            "accepted_event_count", len(timeline.get("accepted_events", []))
        ),
        "timeline_entries": len(entries),
        "frames_with_overlay": frames_with_overlay,
        "entries_processed": entries_processed,
        "mosaic_regions": mosaic_regions,
    }


def render_timeline_overlay(
    video_path: str,
    timeline_path: str,
    output_path: str,
    *,
    thickness: int = 2,
    label_style: str = "render",
    progress_desc: str = "render overlay",
) -> dict:
    stats = render_timeline_video(
        video_path,
        timeline_path,
        output_path,
        mode=MODE_OVERLAY,
        show_bbox=True,
        show_event_id=True,
        show_track_id=True,
        show_hud=True,
        thickness=thickness,
        label_style=label_style,
        progress_desc=progress_desc,
    )
    stats["entries_drawn"] = stats["entries_processed"]
    return stats
