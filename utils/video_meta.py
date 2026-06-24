"""Video metadata helpers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import cv2

from core.event import sec_to_timecode


def safe_video_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'[<>:"/\\|?*]', "_", stem) or "video"


def get_video_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    meta["fps_str"] = ffprobe_fps(path) or f"{meta['fps']:.6f}"
    return meta


def ffprobe_fps(path: str) -> str | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30,
        )
        s = out.stdout.decode(errors="ignore").strip()
        return s if s and not s.startswith("0") else None
    except Exception:
        return None


__all__ = ["safe_video_stem", "get_video_meta", "sec_to_timecode"]
