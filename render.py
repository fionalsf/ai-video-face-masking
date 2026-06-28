"""FFmpeg mosaic render from event trajectories (legacy pipeline).

Timeline-based render: use render_overlay.py — reads timeline.json only.
  overlay / preview / final modes, block pixelation only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


def resolve_encoder(encoder: str) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if encoder != "auto":
        return encoder
    if ffmpeg is None:
        return "libx264"
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30,
        )
        if b"h264_nvenc" in out.stdout:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def _bbox_to_cxcywh(bbox: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy, w, h


def _cxcywh_to_bbox(cx: float, cy: float, w: float, h: float) -> list[float]:
    hw = w / 2.0
    hh = h / 2.0
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def _ema_cxcywh_sequence(
    frame_to_cxcywh: dict[int, tuple[float, float, float, float]],
    alpha: float,
) -> dict[int, tuple[float, float, float, float]]:
    frames = sorted(frame_to_cxcywh.keys())
    if len(frames) <= 1:
        return frame_to_cxcywh
    result: dict[int, tuple[float, float, float, float]] = {}
    prev = frame_to_cxcywh[frames[0]]
    result[frames[0]] = prev
    for f in frames[1:]:
        cur = frame_to_cxcywh[f]
        smoothed = tuple(
            alpha * cur[i] + (1.0 - alpha) * prev[i]
            for i in range(4)
        )
        result[f] = smoothed
        prev = smoothed
    return result


def _postprocess_bbox_sequence(
    frame_to_bbox: dict[int, list[float]],
    *,
    enable_smoothing: bool = False,
    smoothing_alpha: float = 0.7,
) -> dict[int, list[float]]:
    if not enable_smoothing or len(frame_to_bbox) <= 1:
        return frame_to_bbox
    cxcywh = {f: _bbox_to_cxcywh(bbox) for f, bbox in frame_to_bbox.items()}
    smoothed = _ema_cxcywh_sequence(cxcywh, smoothing_alpha)
    return {f: _cxcywh_to_bbox(*v) for f, v in smoothed.items()}


def _build_event_render(
    traj: list[dict],
    total_frames: int,
    extend_frames: int,
) -> dict[int, list[float]]:
    event_render: dict[int, list[float]] = {}
    for pt in traj:
        event_render[pt["frame"]] = list(pt["bbox"])
    for i in range(len(traj) - 1):
        f0, b0 = traj[i]["frame"], np.asarray(traj[i]["bbox"], dtype=np.float64)
        f1, b1 = traj[i + 1]["frame"], np.asarray(traj[i + 1]["bbox"], dtype=np.float64)
        gap = f1 - f0
        if gap <= 1:
            continue
        for f in range(f0 + 1, f1):
            t = (f - f0) / gap
            event_render[f] = (b0 + (b1 - b0) * t).tolist()
    f_first = traj[0]["frame"]
    b_first = traj[0]["bbox"]
    for f in range(max(0, f_first - extend_frames), f_first):
        event_render[f] = list(b_first)
    f_last = traj[-1]["frame"]
    b_last = traj[-1]["bbox"]
    end = total_frames if total_frames > 0 else f_last + extend_frames + 1
    for f in range(f_last + 1, min(end, f_last + 1 + extend_frames)):
        event_render[f] = list(b_last)
    return event_render


def events_to_render(
    events: list[dict],
    total_frames: int,
    extend_frames: int = 3,
    enable_smoothing: bool = False,
    smoothing_alpha: float = 0.7,
) -> dict[int, list[list[float]]]:
    """Convert event list (with trajectory) to frame -> bboxes render dict."""
    render: dict[int, list] = defaultdict(list)

    for ev in events:
        traj = ev.get("trajectory") or []
        if not traj:
            continue
        event_render = _build_event_render(traj, total_frames, extend_frames)
        event_render = _postprocess_bbox_sequence(
            event_render,
            enable_smoothing=enable_smoothing,
            smoothing_alpha=smoothing_alpha,
        )
        for f, bbox in event_render.items():
            render[f].append(bbox)

    return dict(render)


def ffmpeg_frame_reader(path: str, w: int, h: int):
    ffmpeg = shutil.which("ffmpeg")
    frame_size = w * h * 3
    if ffmpeg is None:
        cap = cv2.VideoCapture(path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
        return

    cmd = [ffmpeg, "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_size * 4)
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


def render_video(
    input_video: str,
    output_video: str,
    render: dict[int, list[list[float]]],
    meta: dict,
    expand: float = 0.18,
    mosaic_block: int = 22,
    bitrate: str = "12M",
    encoder: str = "auto",
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("需要 ffmpeg 进行打码渲染")

    w, h = meta["width"], meta["height"]
    total = meta["frames"]
    mw, mh = max(2, w // 4), max(2, h // 4)
    sx, sy = mw / w, mh / h
    block = max(2, mosaic_block)
    dw, dh = max(1, w // block), max(1, h // block)
    enc = resolve_encoder(encoder)
    fc = (
        f"[0:v]format=yuv420p,split=2[base][p];"
        f"[p]scale={dw}:{dh}:flags=neighbor,scale={w}:{h}:flags=neighbor,format=yuva420p[pixf];"
        f"[1:v]scale={w}:{h}:flags=neighbor,format=gray[mask];"
        f"[pixf][mask]alphamerge[pixa];"
        f"[base][pixa]overlay=format=auto[out]"
    )
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-threads", "2", "-filter_threads", "1",
        "-i", input_video,
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{mw}x{mh}",
        "-r", meta["fps_str"], "-thread_queue_size", "64", "-i", "-",
        "-filter_complex", fc, "-map", "[out]", "-c:v", enc,
    ]
    if "nvenc" in enc:
        cmd += ["-preset", "p4", "-b:v", bitrate]
    else:
        cmd += ["-preset", "veryfast", "-b:v", bitrate]
    cmd.append(output_video)

    print(f"[渲染] ffmpeg + {enc}，遮罩帧 {len(render)}...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=mw * mh)
    last_det = (max(render) + 1) if render else 0
    cap = max(total, last_det) + 10 if total > 0 else 10**9
    frame_idx = 0
    pbar = tqdm(total=total if total > 0 else None, unit="f", desc="渲染")
    try:
        while frame_idx < cap:
            mask = np.zeros((mh, mw), np.uint8)
            for box in render.get(frame_idx, []):
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                ex1 = int(round((x1 - bw * expand) * sx))
                ey1 = int(round((y1 - bh * expand) * sy))
                ex2 = int(round((x2 + bw * expand) * sx))
                ey2 = int(round((y2 + bh * expand) * sy))
                ex1 = max(0, min(mw - 1, ex1))
                ey1 = max(0, min(mh - 1, ey1))
                ex2 = max(0, min(mw, ex2))
                ey2 = max(0, min(mh, ey2))
                if ex2 > ex1 and ey2 > ey1:
                    mask[ey1:ey2, ex1:ex2] = 255
            try:
                proc.stdin.write(mask.tobytes())
            except (BrokenPipeError, OSError):
                break
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        try:
            proc.stdin.close()
        except Exception:
            pass
        ret = proc.wait()
    if ret not in (0, None):
        raise RuntimeError(f"ffmpeg 渲染退出码 {ret}")


def mux_audio(tmp_video: str, input_video: str, output: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.move(tmp_video, output)
        return
    cmd = [
        ffmpeg, "-y", "-i", tmp_video, "-i", input_video,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "copy", "-shortest", output,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.isfile(tmp_video):
        os.remove(tmp_video)


def render_masked_output(
    input_video: str,
    output_video: str,
    events: list[dict],
    meta: dict,
    extend_frames: int = 3,
    enable_smoothing: bool = False,
    smoothing_alpha: float = 0.7,
    **kwargs,
) -> None:
    render = events_to_render(
        events,
        meta["frames"],
        extend_frames=extend_frames,
        enable_smoothing=enable_smoothing,
        smoothing_alpha=smoothing_alpha,
    )
    if not render:
        shutil.copy2(input_video, output_video)
        return
    tmp = output_video + ".tmp_noaudio.mp4"
    try:
        render_video(input_video, tmp, render, meta, **kwargs)
        mux_audio(tmp, input_video, output_video)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise
