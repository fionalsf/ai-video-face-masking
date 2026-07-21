"""YOLO-face sparse detect + ByteTrack."""

from __future__ import annotations

from collections import defaultdict
import shutil
import subprocess
from types import SimpleNamespace
import time

import cv2
import numpy as np
from tqdm import tqdm

from detect_backends import DetectionBackend, create_detection_backend


def resolve_device(device: str) -> str:
    import torch
    if device != "cpu" and not torch.cuda.is_available():
        print("[detect] CUDA unavailable; falling back to CPU.")
        return "cpu"
    return device


def gpu_sparse_detect(
    backend: DetectionBackend,
    video_path: str,
    meta: dict,
    interval: int,
    batch_size: int = 4,
    decode_backend: str = "opencv",
    imgsz: int = 1280,
) -> dict[int, list[tuple[list[float], float]]]:
    """Return sparse[frame_idx] = [(bbox, conf), ...]."""
    sparse: dict[int, list] = defaultdict(list)
    interval = max(1, interval)
    batch_size = max(1, int(batch_size))
    total = (meta["frames"] + interval - 1) // interval if meta["frames"] > 0 else None
    predict_sec = 0.0

    def consume_batch(
        frames: list[np.ndarray],
        frame_indices: list[int],
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        nonlocal predict_sec
        if not frames:
            return
        started = time.perf_counter()
        try:
            results = backend.predict_batch(frames)
        except (MemoryError, RuntimeError) as exc:
            if len(frames) <= 1:
                raise
            mid = len(frames) // 2
            print(f"[detect] batch fallback {len(frames)} -> {mid}+{len(frames) - mid}: {type(exc).__name__}")
            consume_batch(frames[:mid], frame_indices[:mid], scale_x, scale_y)
            consume_batch(frames[mid:], frame_indices[mid:], scale_x, scale_y)
            return
        predict_sec += time.perf_counter() - started
        for frame_idx, detections in zip(frame_indices, results):
            for bbox, c in detections:
                x1, y1, x2, y2 = bbox
                mapped = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                sparse[int(frame_idx)].append((mapped, float(c)))

    def read_ffmpeg_cuda() -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found on PATH")
        src_w, src_h = int(meta["width"]), int(meta["height"])
        resize_scale = (
            1.0
            if decode_backend == "cuda-full"
            else min(1.0, float(imgsz) / max(src_w, src_h))
        )
        out_w = max(2, int(round(src_w * resize_scale / 2.0)) * 2)
        out_h = max(2, int(round(src_h * resize_scale / 2.0)) * 2)
        frame_bytes = out_w * out_h * 3
        vf = f"select=not(mod(n\\,{interval})),scale={out_w}:{out_h}:flags=bilinear"
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda", "-c:v", "hevc_cuvid", "-i", video_path,
            "-an", "-vf", vf, "-fps_mode", "vfr",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None
        frames: list[np.ndarray] = []
        frame_indices: list[int] = []
        try:
            for frame_idx in range(0, max(0, int(meta["frames"])), interval):
                raw = proc.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    break
                frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w, 3))
                frame_indices.append(frame_idx)
                if len(frames) >= batch_size:
                    consume_batch(frames, frame_indices, src_w / out_w, src_h / out_h)
                    pbar.update(len(frames))
                    frames.clear()
                    frame_indices.clear()
            if frames:
                consume_batch(frames, frame_indices, src_w / out_w, src_h / out_h)
                pbar.update(len(frames))
            proc.stdout.close()
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            returncode = proc.wait()
            if returncode != 0:
                raise RuntimeError(f"ffmpeg CUDA decode failed ({returncode}): {stderr.strip()}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    pbar = tqdm(total=total, unit="f", desc="GPU detect")
    try:
        if decode_backend in {"cuda", "cuda-full"}:
            read_ffmpeg_cuda()
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video: {video_path}")
            frames: list[np.ndarray] = []
            frame_indices: list[int] = []
            try:
                frame_idx = 0
                while True:
                    if frame_idx % interval == 0:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frames.append(frame)
                        frame_indices.append(frame_idx)
                        if len(frames) >= batch_size:
                            consume_batch(frames, frame_indices)
                            pbar.update(len(frames))
                            frames.clear()
                            frame_indices.clear()
                    else:
                        ok = cap.grab()
                        if not ok:
                            break
                    frame_idx += 1
                if frames:
                    consume_batch(frames, frame_indices)
                    pbar.update(len(frames))
            finally:
                cap.release()
    finally:
        pbar.close()
    reader_name = "ffmpeg-cuda-hevc" if decode_backend in {"cuda", "cuda-full"} else "opencv-grab"
    print(f"[detect timing] backend={backend.name} reader={reader_name} predict={predict_sec:.1f}s")
    return sparse


def cpu_byte_track(
    sparse_dets: dict[int, list[tuple[list[float], float]]],
) -> dict[int, list[tuple[int, list[float], float]]]:
    """Return per_frame[frame_idx] = [(track_id, bbox, conf), ...]."""
    from ultralytics.trackers.byte_tracker import BYTETracker

    class _TrackResults:
        def __init__(self, xyxy, conf, cls):
            self._xyxy = np.asarray(xyxy, dtype=np.float32)
            self.conf = np.asarray(conf, dtype=np.float32)
            self.cls = np.asarray(cls, dtype=np.float32)
            w = self._xyxy[:, 2] - self._xyxy[:, 0]
            h = self._xyxy[:, 3] - self._xyxy[:, 1]
            self.xywh = np.stack(
                [self._xyxy[:, 0] + w / 2, self._xyxy[:, 1] + h / 2, w, h], axis=1
            ).astype(np.float32)

        def __len__(self):
            return len(self.conf)

        def __getitem__(self, mask):
            mask = np.asarray(mask)
            return _TrackResults(self._xyxy[mask], self.conf[mask], self.cls[mask])

    if not sparse_dets:
        return {}

    targs = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    tracker = BYTETracker(targs)
    per_frame: dict[int, list] = {}

    started = time.perf_counter()
    for frame_idx in sorted(sparse_dets):
        items = sparse_dets[frame_idx]
        boxes = [b for b, _ in items]
        confs = [c for _, c in items]
        if not boxes:
            continue
        res = _TrackResults(boxes, confs, [0.0] * len(confs))
        tracks = tracker.update(res, img=None)
        out = []
        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                row = t.tolist() if hasattr(t, "tolist") else list(t)
                if len(row) < 6:
                    continue
                x1, y1, x2, y2, tid, score = row[:6]
                out.append((int(tid), [x1, y1, x2, y2], float(score)))
        per_frame[frame_idx] = out
    print(f"[track timing] bytetrack={time.perf_counter() - started:.1f}s")
    return per_frame


def run_detect_track(
    video_path: str,
    model_path: str,
    meta: dict,
    device: str = "0",
    conf: float = 0.35,
    imgsz: int = 1280,
    interval: int = 5,
    batch_size: int = 4,
    infer_backend: str = "torch",
    onnx_model_path: str | None = None,
    decode_backend: str = "opencv",
) -> list[dict]:
    """Full detect+track -> flat detection list."""
    device = resolve_device(device)
    backend = create_detection_backend(infer_backend, model_path, onnx_model_path, device, conf, imgsz)
    fps = meta["fps"] or 25.0

    print(
        f"[1/2] sparse detect "
        f"(backend={backend.name}, interval={interval}, conf={conf}, batch={batch_size})..."
    )
    sparse = gpu_sparse_detect(
        backend, video_path, meta, interval,
        batch_size=batch_size, decode_backend=decode_backend, imgsz=imgsz,
    )
    n_raw = sum(len(v) for v in sparse.values())
    print(f"[2/2] CPU ByteTrack ({len(sparse)} frames, {n_raw} boxes)...")
    per_frame = cpu_byte_track(sparse)
    backend.close()

    detections = []
    for frame_idx in sorted(per_frame):
        t_sec = frame_idx / fps
        for track_id, bbox, c in per_frame[frame_idx]:
            detections.append({
                "frame": frame_idx,
                "t": round(t_sec, 3),
                "track_id": track_id,
                "bbox": [round(v, 1) for v in bbox],
                "conf": round(c, 4),
            })
    return detections
