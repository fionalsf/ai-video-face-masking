"""FFmpeg mosaic render from event trajectories (legacy pipeline).

Timeline-based render: use render_overlay.py — reads timeline.json only.
  overlay / preview / final modes, block pixelation only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import OrderedDict, defaultdict

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


def _clip_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width - 1), float(x1)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 <= x1:
        x2 = min(float(width), x1 + 2.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 2.0)
    return [x1, y1, x2, y2]


def _edge_aware_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    """Expand boxes inward when a face is clipped by a frame edge."""
    x1, y1, x2, y2 = _clip_bbox(bbox, width, height)
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    edge = max(4.0, min(width, height) * 0.01)

    left = x1 <= edge
    right = x2 >= width - edge
    top = y1 <= edge
    bottom = y2 >= height - edge

    if left:
        x2 += bw * 0.35
    if right:
        x1 -= bw * 0.35
    if top:
        y2 += bh * 0.65
    if bottom:
        y1 -= bh * 0.45

    return _clip_bbox([x1, y1, x2, y2], width, height)


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


class _GrayFrameCache:
    def __init__(self, video_path: str, width: int, height: int, maxsize: int = 160):
        self.video_path = video_path
        self.width = width
        self.height = height
        self.maxsize = maxsize
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cap = cv2.VideoCapture(self.video_path)

    def read(self, frame_idx: int):
        frame_idx = int(frame_idx)
        if frame_idx in self.cache:
            gray = self.cache.pop(frame_idx)
            self.cache[frame_idx] = gray
            return gray
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.cache[frame_idx] = gray
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
        return gray

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _feature_points(gray: np.ndarray, bbox: list[float], max_points: int = 48):
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in _clip_bbox(bbox, w, h)]
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=3,
        blockSize=5,
        mask=mask,
    )
    if pts is not None and len(pts) >= 4:
        return pts.astype(np.float32)

    cx, cy, bw, bh = _bbox_to_cxcywh([x1, y1, x2, y2])
    fallback = [
        [cx, cy],
        [x1 + bw * 0.25, y1 + bh * 0.25],
        [x2 - bw * 0.25, y1 + bh * 0.25],
        [x1 + bw * 0.25, y2 - bh * 0.25],
        [x2 - bw * 0.25, y2 - bh * 0.25],
    ]
    return np.asarray(fallback, dtype=np.float32).reshape(-1, 1, 2)


def _track_bbox_step(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    bbox: list[float],
    points,
    *,
    min_points: int,
) -> tuple[list[float] | None, np.ndarray | None]:
    if points is None or len(points) < min_points:
        points = _feature_points(prev_gray, bbox)
    if points is None or len(points) < min_points:
        return None, None
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        points,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.03),
    )
    if next_pts is None or status is None:
        return None, None
    good = status.reshape(-1) == 1
    if int(np.sum(good)) < min_points:
        return None, None
    delta = next_pts[good].reshape(-1, 2) - points[good].reshape(-1, 2)
    dx, dy = np.median(delta, axis=0).tolist()
    shifted = [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]
    return shifted, next_pts[good].reshape(-1, 1, 2).astype(np.float32)


def _blend_bbox(a: list[float], b: list[float], alpha: float) -> list[float]:
    return [float(a[i]) * (1.0 - alpha) + float(b[i]) * alpha for i in range(4)]


def _linear_bbox(b0: list[float], b1: list[float], t: float) -> list[float]:
    a = np.asarray(b0, dtype=np.float64)
    b = np.asarray(b1, dtype=np.float64)
    return (a + (b - a) * t).tolist()


def _track_between_keyframes(
    frames: _GrayFrameCache,
    f0: int,
    b0: list[float],
    f1: int,
    b1: list[float],
    *,
    width: int,
    height: int,
    min_points: int,
    anchor: float,
) -> dict[int, list[float]] | None:
    if f1 <= f0 + 1:
        return {}
    prev_gray = frames.read(f0)
    if prev_gray is None:
        return None
    bbox = list(b0)
    points = _feature_points(prev_gray, bbox)
    out: dict[int, list[float]] = {}
    for f in range(f0 + 1, f1):
        cur_gray = frames.read(f)
        if cur_gray is None:
            return None
        tracked, points = _track_bbox_step(
            prev_gray,
            cur_gray,
            bbox,
            points,
            min_points=min_points,
        )
        if tracked is None:
            return None
        linear = _linear_bbox(b0, b1, (f - f0) / (f1 - f0))
        bbox = _clip_bbox(_blend_bbox(tracked, linear, anchor), width, height)
        out[f] = bbox
        prev_gray = cur_gray
        if points is None or len(points) < min_points * 2:
            points = _feature_points(prev_gray, bbox)
    return out


def _track_from_keyframe(
    frames: _GrayFrameCache,
    frame_idx: int,
    bbox: list[float],
    *,
    start: int,
    end: int,
    direction: int,
    width: int,
    height: int,
    min_points: int,
) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    prev_gray = frames.read(frame_idx)
    if prev_gray is None:
        return out
    cur_frame = frame_idx
    cur_bbox = list(bbox)
    points = _feature_points(prev_gray, cur_bbox)
    while True:
        nxt = cur_frame + direction
        if nxt < start or nxt > end:
            break
        cur_gray = frames.read(nxt)
        if cur_gray is None:
            break
        tracked, points = _track_bbox_step(
            prev_gray,
            cur_gray,
            cur_bbox,
            points,
            min_points=min_points,
        )
        if tracked is None:
            break
        cur_bbox = _clip_bbox(tracked, width, height)
        out[nxt] = cur_bbox
        cur_frame = nxt
        prev_gray = cur_gray
        if points is None or len(points) < min_points * 2:
            points = _feature_points(prev_gray, cur_bbox)
    return out


def _build_event_render(
    traj: list[dict],
    total_frames: int,
    extend_frames: int,
    *,
    frames: _GrayFrameCache | None = None,
    width: int | None = None,
    height: int | None = None,
    motion_compensate: bool = False,
    motion_max_gap: int = 45,
    motion_singleton_frames: int = 24,
    motion_min_points: int = 4,
    motion_anchor: float = 0.18,
    edge_aware: bool = True,
    max_interpolate_gap: int = 20,
    min_point_conf: float = 0.0,
) -> dict[int, list[float]]:
    event_render: dict[int, list[float]] = {}
    traj = sorted(traj, key=lambda x: int(x["frame"]))
    if min_point_conf > 0:
        traj = [
            pt for pt in traj
            if pt.get("conf") is None or float(pt.get("conf", 1.0)) >= min_point_conf
        ]
    if not traj:
        return event_render
    if width and height:
        for pt in traj:
            bbox = list(pt["bbox"])
            if edge_aware:
                bbox = _edge_aware_bbox(bbox, width, height)
            else:
                bbox = _clip_bbox(bbox, width, height)
            pt = pt.copy()
            pt["bbox"] = bbox
            event_render[int(pt["frame"])] = bbox
    else:
        for pt in traj:
            event_render[int(pt["frame"])] = list(pt["bbox"])
    normalized_traj = [
        {"frame": int(f), "bbox": list(b)}
        for f, b in sorted(event_render.items())
    ]
    traj = normalized_traj

    if (
        motion_compensate
        and frames is not None
        and width is not None
        and height is not None
        and len(traj) == 1
    ):
        f0 = traj[0]["frame"]
        b0 = traj[0]["bbox"]
        span = max(extend_frames, motion_singleton_frames)
        end = total_frames - 1 if total_frames > 0 else f0 + span
        start = max(0, f0 - span)
        end = min(end, f0 + span)
        event_render.update(
            _track_from_keyframe(
                frames,
                f0,
                b0,
                start=start,
                end=end,
                direction=-1,
                width=width,
                height=height,
                min_points=motion_min_points,
            )
        )
        event_render.update(
            _track_from_keyframe(
                frames,
                f0,
                b0,
                start=start,
                end=end,
                direction=1,
                width=width,
                height=height,
                min_points=motion_min_points,
            )
        )
        return event_render

    for pt in traj:
        event_render[pt["frame"]] = list(pt["bbox"])
    for i in range(len(traj) - 1):
        f0, b0 = traj[i]["frame"], list(traj[i]["bbox"])
        f1, b1 = traj[i + 1]["frame"], list(traj[i + 1]["bbox"])
        gap = f1 - f0
        if gap <= 1:
            continue
        if gap > max_interpolate_gap:
            left_end = min(f1, f0 + extend_frames + 1)
            for f in range(f0 + 1, left_end):
                event_render.setdefault(f, list(b0))
            right_start = max(f0 + 1, f1 - extend_frames)
            for f in range(right_start, f1):
                event_render.setdefault(f, list(b1))
            continue
        tracked = None
        if (
            motion_compensate
            and frames is not None
            and width is not None
            and height is not None
            and gap <= motion_max_gap
        ):
            tracked = _track_between_keyframes(
                frames,
                f0,
                b0,
                f1,
                b1,
                width=width,
                height=height,
                min_points=motion_min_points,
                anchor=motion_anchor,
            )
        if tracked is not None:
            event_render.update(tracked)
            continue
        b0_arr = np.asarray(b0, dtype=np.float64)
        b1_arr = np.asarray(b1, dtype=np.float64)
        for f in range(f0 + 1, f1):
            t = (f - f0) / gap
            event_render[f] = (b0_arr + (b1_arr - b0_arr) * t).tolist()
    f_first = traj[0]["frame"]
    b_first = traj[0]["bbox"]
    pre_start = max(0, f_first - extend_frames)
    if motion_compensate and frames is not None and width is not None and height is not None:
        event_render.update(
            _track_from_keyframe(
                frames,
                f_first,
                b_first,
                start=pre_start,
                end=f_first,
                direction=-1,
                width=width,
                height=height,
                min_points=motion_min_points,
            )
        )
    for f in range(pre_start, f_first):
        event_render.setdefault(f, list(b_first))
    f_last = traj[-1]["frame"]
    b_last = traj[-1]["bbox"]
    end = total_frames if total_frames > 0 else f_last + extend_frames + 1
    post_end = min(end - 1, f_last + extend_frames)
    if motion_compensate and frames is not None and width is not None and height is not None:
        event_render.update(
            _track_from_keyframe(
                frames,
                f_last,
                b_last,
                start=f_last,
                end=post_end,
                direction=1,
                width=width,
                height=height,
                min_points=motion_min_points,
            )
        )
    for f in range(f_last + 1, post_end + 1):
        event_render.setdefault(f, list(b_last))
    return event_render


def events_to_render(
    events: list[dict],
    total_frames: int,
    extend_frames: int = 3,
    enable_smoothing: bool = False,
    smoothing_alpha: float = 0.7,
    video_path: str | None = None,
    meta: dict | None = None,
    motion_compensate: bool = False,
    motion_max_gap: int = 45,
    motion_singleton_frames: int = 24,
    motion_min_points: int = 4,
    motion_anchor: float = 0.18,
    edge_aware: bool = True,
    max_interpolate_gap: int = 20,
) -> dict[int, list[list[float]]]:
    """Convert event list (with trajectory) to frame -> bboxes render dict."""
    render: dict[int, list] = defaultdict(list)
    frames = None
    width = height = None
    if motion_compensate or edge_aware:
        if meta is not None:
            width = int(meta["width"])
            height = int(meta["height"])
        if motion_compensate and video_path and width and height:
            frames = _GrayFrameCache(video_path, width, height)

    try:
        for ev in events:
            traj = ev.get("trajectory") or []
            if not traj:
                continue
            hints = set(ev.get("rule_hints") or [])
            is_edge_partial = "edge_partial_face_candidate" in hints
            event_render = _build_event_render(
                traj,
                total_frames,
                0 if is_edge_partial else extend_frames,
                frames=frames,
                width=width,
                height=height,
                motion_compensate=motion_compensate and not is_edge_partial,
                motion_max_gap=motion_max_gap,
                motion_singleton_frames=motion_singleton_frames,
                motion_min_points=motion_min_points,
                motion_anchor=motion_anchor,
                edge_aware=edge_aware and not is_edge_partial,
                max_interpolate_gap=max_interpolate_gap,
                min_point_conf=0.55 if ev.get("tier") in {"auto", "review"} and not is_edge_partial else 0.0,
            )
            event_render = _postprocess_bbox_sequence(
                event_render,
                enable_smoothing=enable_smoothing,
                smoothing_alpha=smoothing_alpha,
            )
            for f, bbox in event_render.items():
                render[f].append(bbox)
    finally:
        if frames is not None:
            frames.close()

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


def _edge_partial_face_boxes(frame: np.ndarray) -> list[list[float]]:
    h, w = frame.shape[:2]
    strip_w = max(72, int(w * 0.13))
    min_h = max(110, int(h * 0.16))
    min_area = max(6000, int(w * h * 0.008))
    boxes: list[list[float]] = []

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    y, cr, cb = cv2.split(ycrcb)
    hh, ss, vv = cv2.split(hsv)
    skin = (
        (cr > 133) & (cr < 180) &
        (cb > 77) & (cb < 135) &
        (ss > 18) & (vv > 45) &
        (y > 35)
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)

    for side, x0, x1 in (("left", 0, strip_w),):
        roi = skin[:, x0:x1]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bh < min_h or bh < bw * 1.1:
                continue
            touches_edge = bx <= 4 if side == "left" else bx + bw >= strip_w - 4
            if not touches_edge:
                continue
            if best is None or area > best[0]:
                best = (area, bx, by, bw, bh, contour)
        if best is not None:
            _, bx, by, bw, bh, contour = best
            component = np.zeros_like(roi)
            cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
            ys, xs = np.where(component > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue

            gx1 = x0 + float(np.percentile(xs, 2))
            gx2 = x0 + float(np.percentile(xs, 96))
            y_low = float(np.percentile(ys, 4))
            y_high = float(np.percentile(ys, 94))
            skin_w = max(24.0, gx2 - gx1)
            max_box_w = max(112.0, min(float(w) * 0.16, float(strip_w) * 1.22))
            max_box_h = max(220.0, min(float(h) * 0.58, max(skin_w * 2.45, 260.0)))

            row_counts = np.bincount(ys, minlength=h).astype(np.float32)
            if y_high - y_low > max_box_h:
                window = int(round(max_box_h))
                kernel_1d = np.ones(window, dtype=np.float32)
                scores = np.convolve(row_counts, kernel_1d, mode="valid")
                y_low = float(int(np.argmax(scores)))
                y_high = y_low + max_box_h

            pad_x = max(12.0, skin_w * 0.16)
            skin_h = y_high - y_low
            top_pad = max(72.0, skin_h * 0.42)
            bottom_pad = max(18.0, skin_h * 0.12)
            y1 = y_low - top_pad
            y2 = y_high + bottom_pad
            face_w = min(max_box_w, max(skin_w + pad_x * 2.0, 84.0))
            if side == "left":
                x1_box = 0.0
                x2_box = min(max(float(gx2) + pad_x, face_w), max_box_w)
            else:
                x1_box = max(float(gx1) - pad_x, float(w) - max_box_w)
                x2_box = float(w)
            boxes.append(_clip_bbox([x1_box, y1, x2_box, y2], w, h))
    return boxes


def _skin_mask(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    y, cr, cb = cv2.split(ycrcb)
    hh, ss, vv = cv2.split(hsv)
    mask_a = (y > 45) & (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 140)
    mask_b = (vv > 45) & (ss > 18) & ((hh <= 25) | (hh >= 165))
    return ((mask_a | mask_b).astype(np.uint8) * 255)


def _refine_face_bbox_with_skin(frame: np.ndarray, bbox: list[float]) -> list[float]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    rx1 = max(0, int(round(x1 - bw * 0.9)))
    ry1 = max(0, int(round(y1 - bh * 0.7)))
    rx2 = min(w, int(round(x2 + bw * 1.4)))
    ry2 = min(h, int(round(y2 + bh * 0.9)))
    if rx2 <= rx1 or ry2 <= ry1:
        return bbox

    roi = frame[ry1:ry2, rx1:rx2]
    skin = _skin_mask(roi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skin, 8)

    best = None
    orig_cx, orig_cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    for idx in range(1, count):
        sx, sy, sw, sh, area = stats[idx]
        if area < 120:
            continue
        aspect = sw / max(1, sh)
        if not (0.25 <= aspect <= 3.2):
            continue
        gx1, gy1 = rx1 + sx, ry1 + sy
        gx2, gy2 = gx1 + sw, gy1 + sh
        cx, cy = (gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5
        dist = abs(cx - orig_cx) / bw + abs(cy - orig_cy) / bh
        score = area / (1.0 + dist)
        if best is None or score > best[0]:
            best = (score, float(gx1), float(gy1), float(gx2), float(gy2))

    if best is None:
        return bbox

    _, gx1, gy1, gx2, gy2 = best
    skin_w, skin_h = max(1.0, gx2 - gx1), max(1.0, gy2 - gy1)
    # Do not let a packaging false-positive grow into a large review box.
    if (skin_w * skin_h) > (w * h * 0.085):
        return bbox
    pad = max(10.0, max(skin_w, skin_h) * 0.18)
    refined = [gx1 - pad, gy1 - pad * 0.8, gx2 + pad, gy2 + pad * 0.5]
    refined = _clip_bbox(refined, w, h)
    refined_w = max(1.0, refined[2] - refined[0])
    refined_h = max(1.0, refined[3] - refined[1])
    orig_area = max(1.0, bw * bh)
    refined_area = refined_w * refined_h
    if refined_area > orig_area * 2.35:
        return bbox
    if refined_w > bw * 2.1 or refined_h > bh * 1.85:
        return bbox
    return refined


def render_video(
    input_video: str,
    output_video: str,
    render: dict[int, list[list[float]]],
    meta: dict,
    expand: float = 0.18,
    mosaic_block: int = 22,
    bitrate: str = "12M",
    encoder: str = "auto",
    edge_partial_face: bool = False,
    refine_face_boxes: bool = False,
    mask_scale_divisor: int = 8,
    filter_threads: int = 1,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("需要 ffmpeg 进行打码渲染")

    w, h = meta["width"], meta["height"]
    total = meta["frames"]
    mask_scale_divisor = max(4, int(mask_scale_divisor or 8))
    mw, mh = max(2, w // mask_scale_divisor), max(2, h // mask_scale_divisor)
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
    filter_threads = max(1, int(filter_threads or 1))
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-threads", "2", "-filter_threads", str(filter_threads),
        "-i", input_video,
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{mw}x{mh}",
        "-r", meta["fps_str"], "-thread_queue_size", "64", "-i", "-",
        "-filter_complex", fc, "-map", "[out]",
    ]
    cmd += ["-c:v", enc]
    if "nvenc" in enc:
        cmd += ["-preset", "p1", "-b:v", bitrate]
    else:
        cmd += ["-preset", "veryfast", "-b:v", bitrate]
    cmd.append(output_video)

    print(f"[渲染] ffmpeg + {enc}，遮罩帧 {len(render)}...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=mw * mh)
    frame_iter = ffmpeg_frame_reader(input_video, w, h) if (edge_partial_face or refine_face_boxes) else None
    last_det = (max(render) + 1) if render else 0
    cap = max(total, last_det) + 10 if total > 0 else 10**9
    frame_idx = 0
    zero_mask_bytes = bytes(mw * mh)
    pbar = tqdm(total=total if total > 0 else None, unit="f", desc="渲染")
    try:
        while frame_idx < cap:
            boxes = list(render.get(frame_idx, []))
            if frame_iter is not None:
                try:
                    frame = next(frame_iter)
                except StopIteration:
                    frame = None
                if frame is not None:
                    if refine_face_boxes and boxes:
                        boxes = [_refine_face_bbox_with_skin(frame, box) for box in boxes]
                    if edge_partial_face:
                        boxes.extend(_edge_partial_face_boxes(frame))
            if boxes:
                mask = np.zeros((mh, mw), np.uint8)
                for box in boxes:
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
                payload = mask.tobytes()
            else:
                payload = zero_mask_bytes
            try:
                proc.stdin.write(payload)
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
    motion_compensate: bool = False,
    motion_max_gap: int = 45,
    motion_singleton_frames: int = 24,
    motion_min_points: int = 4,
    motion_anchor: float = 0.18,
    edge_aware: bool = True,
    **kwargs,
) -> None:
    render = events_to_render(
        events,
        meta["frames"],
        extend_frames=extend_frames,
        enable_smoothing=enable_smoothing,
        smoothing_alpha=smoothing_alpha,
        video_path=input_video,
        meta=meta,
        motion_compensate=motion_compensate,
        motion_max_gap=motion_max_gap,
        motion_singleton_frames=motion_singleton_frames,
        motion_min_points=motion_min_points,
        motion_anchor=motion_anchor,
        edge_aware=edge_aware,
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
