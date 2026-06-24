#!/usr/bin/env python3
"""Debug: contact-sheet grid per Event for visual continuity validation."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from event_builder import build_events
from export import draw_detections
from tracker import cpu_byte_track
from validate_track_event import load_phase1, per_frame_to_detections, records_to_sparse
from video_meta import get_video_meta, sec_to_timecode


def parse_args():
    p = argparse.ArgumentParser(description="Event contact-sheet visual validation")
    p.add_argument(
        "--detection-dir",
        required=True,
        help="Phase 1/2 output dir (detections.json + tracked_detections.json)",
    )
    p.add_argument("--video", help="Video path (default: detection_summary.json)")
    p.add_argument("--event-gap", type=float, default=1.0)
    p.add_argument(
        "--output-dir",
        help="Output root (default: <detection-dir>/event_contact_sheet)",
    )
    p.add_argument("--max-cells", type=int, default=9, help="Max frames per contact sheet")
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--cell-width", type=int, default=480)
    p.add_argument("--cell-height", type=int, default=270)
    p.add_argument("--gif-fps", type=float, default=4.0, help="GIF frame rate (3-5 recommended)")
    p.add_argument("--gif-min-duration", type=float, default=2.0, help="Min event duration (sec) for GIF")
    p.add_argument("--gif-max-width", type=int, default=640, help="GIF frame max width")
    return p.parse_args()


def load_tracked(detection_dir: str, fps: float) -> list[dict]:
    tracked_path = os.path.join(detection_dir, "tracked_detections.json")
    if os.path.isfile(tracked_path):
        with open(tracked_path, encoding="utf-8") as f:
            return json.load(f)

    _, records, _ = load_phase1(detection_dir, None)
    sparse = records_to_sparse(records)
    per_frame = cpu_byte_track(sparse)
    return per_frame_to_detections(per_frame, fps)


def sample_trajectory(trajectory: list[dict], max_cells: int) -> list[dict]:
    n = len(trajectory)
    if n <= max_cells:
        return list(trajectory)
    if max_cells <= 1:
        return [trajectory[0]]
    idx = [int(round(i * (n - 1) / (max_cells - 1))) for i in range(max_cells)]
    seen: set[int] = set()
    out: list[dict] = []
    for j in idx:
        if j not in seen:
            seen.add(j)
            out.append(trajectory[j])
    return out


def grid_shape(count: int, cols: int) -> tuple[int, int]:
    rows = (count + cols - 1) // cols
    return rows, cols


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if ok:
        return frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
    ok, frame = cap.read()
    return frame if ok else None


def render_cell(
    frame: np.ndarray,
    point: dict,
    cell_w: int,
    cell_h: int,
) -> np.ndarray:
    det = {"bbox": point["bbox"], "confidence": point["conf"]}
    annotated = draw_detections(frame, [det])
    h, w = annotated.shape[:2]
    scale = min(cell_w / w, (cell_h - 28) / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(annotated, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)
    x0 = (cell_w - nw) // 2
    y0 = (cell_h - 28 - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized

    tc = sec_to_timecode(point["t"])
    label = f"f{point['frame']:06d}  {tc}  conf={point['conf']:.3f}"
    cv2.rectangle(canvas, (0, cell_h - 26), (cell_w, cell_h), (40, 40, 40), -1)
    cv2.putText(
        canvas, label, (8, cell_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA,
    )
    return canvas


def build_contact_sheet(
    cap: cv2.VideoCapture,
    trajectory: list[dict],
    max_cells: int,
    grid_cols: int,
    cell_w: int,
    cell_h: int,
) -> np.ndarray | None:
    points = sample_trajectory(trajectory, max_cells)
    cells: list[np.ndarray] = []
    for pt in points:
        frame = read_frame(cap, int(pt["frame"]))
        if frame is None:
            continue
        cells.append(render_cell(frame, pt, cell_w, cell_h))
    if not cells:
        return None

    rows, cols = grid_shape(len(cells), grid_cols)
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    sheet[:] = (16, 16, 16)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        y1, x1 = r * cell_h, c * cell_w
        sheet[y1:y1 + cell_h, x1:x1 + cell_w] = cell
    return sheet


def nearest_trajectory_point(trajectory: list[dict], t: float) -> dict:
    return min(trajectory, key=lambda p: abs(p["t"] - t))


def sample_gif_times(start_t: float, end_t: float, gif_fps: float) -> list[float]:
    if end_t <= start_t:
        return [start_t]
    step = 1.0 / max(gif_fps, 0.1)
    times: list[float] = []
    t = start_t
    while t <= end_t + 1e-9:
        times.append(round(t, 3))
        t += step
    if times[-1] < end_t - 1e-9:
        times.append(end_t)
    return times


def annotate_gif_frame(frame: np.ndarray, point: dict) -> np.ndarray:
    det = {"bbox": point["bbox"], "confidence": point["conf"]}
    vis = draw_detections(frame, [det])
    tc = sec_to_timecode(point["t"])
    label = f"{point['event_id_label']}  f{point['frame']:06d}  {tc}  conf={point['conf']:.3f}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(vis, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return vis


def build_event_gif(
    cap: cv2.VideoCapture,
    trajectory: list[dict],
    fps: float,
    event_id: str,
    gif_fps: float,
    max_width: int,
) -> list[np.ndarray]:
    if not trajectory:
        return []
    start_t, end_t = trajectory[0]["t"], trajectory[-1]["t"]
    frames: list[np.ndarray] = []
    for t in sample_gif_times(start_t, end_t, gif_fps):
        pt = dict(nearest_trajectory_point(trajectory, t))
        pt["event_id_label"] = event_id
        frame_idx = int(round(t * fps))
        frame = read_frame(cap, frame_idx)
        if frame is None:
            continue
        vis = annotate_gif_frame(frame, pt)
        h, w = vis.shape[:2]
        if w > max_width:
            nh = max(1, int(h * max_width / w))
            vis = cv2.resize(vis, (max_width, nh), interpolation=cv2.INTER_AREA)
        frames.append(vis)
    return frames


def save_gif(frames: list[np.ndarray], path: str, gif_fps: float) -> None:
    from PIL import Image

    pil_frames = [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        for f in frames
    ]
    duration_ms = int(round(1000.0 / max(gif_fps, 0.1)))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def write_summary_html(out_dir: str, rows: list[dict], gif_rel_dir: str) -> str:
    path = os.path.join(out_dir, "summary.html")
    thumb_w = 320
    body_rows = []
    for row in rows:
        img = html.escape(row["image"])
        gif_cell = "—"
        if row.get("gif"):
            g = html.escape(row["gif"])
            gif_cell = f'<a href="{gif_rel_dir}/{g}" target="_blank"><img src="{gif_rel_dir}/{g}" width="160" alt="gif"></a>'
        body_rows.append(
            f"""<tr>
  <td><a href="{img}" target="_blank"><img src="{img}" width="{thumb_w}" alt="{html.escape(row['event_id'])}"></a></td>
  <td>{gif_cell}</td>
  <td><a href="{img}" target="_blank">{html.escape(row['event_id'])}</a></td>
  <td>{row['track_id']}</td>
  <td>{row['duration_sec']:.3f}s</td>
  <td>{row['peak_confidence']:.4f}</td>
  <td>{row['frame_count']}</td>
  <td>{html.escape(row['start_timecode'])} &rarr; {html.escape(row['end_timecode'])}</td>
</tr>"""
        )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Event Contact Sheets</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #111; color: #eee; }}
h1 {{ font-size: 1.2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #333; padding: 8px; vertical-align: middle; }}
th {{ background: #222; text-align: left; }}
tr:hover {{ background: #1a1a1a; }}
a {{ color: #8cf; }}
img {{ display: block; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Event Contact Sheet Summary ({len(rows)} events)</h1>
<p>Click thumbnail or Event ID to open full contact sheet.</p>
<table>
<thead>
<tr>
  <th>Contact Sheet</th>
  <th>GIF (&gt;2s)</th>
  <th>Event ID</th>
  <th>Track ID</th>
  <th>Duration</th>
  <th>Peak Conf</th>
  <th>Frame Count</th>
  <th>Time Range</th>
</tr>
</thead>
<tbody>
{''.join(body_rows)}
</tbody>
</table>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def run(args) -> int:
    detection_dir = os.path.abspath(args.detection_dir)
    out_dir = os.path.abspath(args.output_dir or os.path.join(detection_dir, "event_contact_sheet"))
    gif_dir = os.path.join(detection_dir, "event_gifs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    gif_rel_dir = os.path.relpath(gif_dir, out_dir).replace("\\", "/")

    summary_path = os.path.join(detection_dir, "detection_summary.json")
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    video_path = args.video or summary.get("video")
    if not video_path or not os.path.isfile(video_path):
        print(f"[error] Video not found: {video_path}", file=sys.stderr)
        return 1

    meta = get_video_meta(video_path)
    tracked = load_tracked(detection_dir, float(summary.get("fps") or meta["fps"]))
    events = build_events(
        tracked,
        gap_sec=args.event_gap,
        frame_h=meta["height"],
        frame_w=meta["width"],
        fps=float(summary.get("fps") or meta["fps"]),
        total_frames=meta["frames"],
        detect_interval=int(summary.get("detect_interval") or 0) or None,
    )

    print(f"[info] video: {video_path}")
    print(f"[info] events: {len(events)}")
    print(f"[info] output: {out_dir}")
    print(f"[info] gifs  : {gif_dir} (duration > {args.gif_min_duration}s @ {args.gif_fps}fps)")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Cannot open video: {video_path}", file=sys.stderr)
        return 1

    video_fps = float(summary.get("fps") or meta["fps"])
    html_rows: list[dict] = []
    written = 0
    gifs_written = 0
    for event in events:
        duration = round(event.end_time - event.start_time, 3)
        sheet = build_contact_sheet(
            cap,
            event.trajectory,
            args.max_cells,
            args.grid_cols,
            args.cell_width,
            args.cell_height,
        )
        if sheet is None:
            print(f"[warn] skip {event.event_id}: no frames")
            continue
        img_name = f"{event.event_id}.jpg"
        img_path = os.path.join(out_dir, img_name)
        cv2.imwrite(img_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1

        gif_name = None
        if duration > args.gif_min_duration:
            gif_frames = build_event_gif(
                cap, event.trajectory, video_fps, event.event_id,
                args.gif_fps, args.gif_max_width,
            )
            if gif_frames:
                gif_name = f"{event.event_id}.gif"
                save_gif(gif_frames, os.path.join(gif_dir, gif_name), args.gif_fps)
                gifs_written += 1

        html_rows.append({
            "event_id": event.event_id,
            "track_id": event.track_id,
            "duration_sec": duration,
            "peak_confidence": event.peak_confidence,
            "frame_count": event.detection_count,
            "start_timecode": sec_to_timecode(event.start_time),
            "end_timecode": sec_to_timecode(event.end_time),
            "image": img_name,
            "gif": gif_name,
        })

    cap.release()
    html_path = write_summary_html(out_dir, html_rows, gif_rel_dir)

    print()
    print("[done] Event contact sheets")
    print(f"  sheets : {written}")
    print(f"  gifs   : {gifs_written}")
    print(f"  folder : {out_dir}")
    print(f"  gifs   : {gif_dir}")
    print(f"  html   : {html_path}")
    return 0


def main():
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
