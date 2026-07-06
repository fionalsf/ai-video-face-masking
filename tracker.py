"""YOLO-face sparse detect + ByteTrack."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
import time

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def resolve_device(device: str) -> str:
    import torch
    if device != "cpu" and not torch.cuda.is_available():
        print("[提示] 未检测到 CUDA，改用 CPU。")
        return "cpu"
    return device


def gpu_sparse_detect(
    model: YOLO,
    video_path: str,
    meta: dict,
    device: str,
    conf: float,
    imgsz: int,
    interval: int,
    batch_size: int = 4,
) -> dict[int, list[tuple[list[float], float]]]:
    """Return sparse[frame_idx] = [(bbox, conf), ...]."""
    sparse: dict[int, list] = defaultdict(list)
    interval = max(1, interval)
    batch_size = max(1, int(batch_size))
    total = (meta["frames"] + interval - 1) // interval if meta["frames"] > 0 else None
    predict_sec = 0.0

    def consume_batch(frames: list[np.ndarray], frame_indices: list[int]) -> None:
        nonlocal predict_sec
        if not frames:
            return
        started = time.perf_counter()
        try:
            results = model.predict(
                source=frames,
                stream=False,
                conf=conf,
                imgsz=imgsz,
                device=device,
                half=device != "cpu",
                verbose=False,
            )
        except (MemoryError, RuntimeError) as exc:
            if len(frames) <= 1:
                raise
            mid = len(frames) // 2
            print(f"[detect] batch fallback {len(frames)} -> {mid}+{len(frames) - mid}: {type(exc).__name__}")
            consume_batch(frames[:mid], frame_indices[:mid])
            consume_batch(frames[mid:], frame_indices[mid:])
            return
        predict_sec += time.perf_counter() - started
        for frame_idx, r in zip(frame_indices, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
            for j in range(len(xyxy)):
                c = float(confs[j]) if confs is not None else 1.0
                sparse[int(frame_idx)].append((xyxy[j].tolist(), c))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    pbar = tqdm(total=total, unit="f", desc="GPU detect")
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
        pbar.close()
        cap.release()
    print(f"[detect timing] reader=opencv-grab predict={predict_sec:.1f}s")
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
) -> list[dict]:
    """Full detect+track -> flat detection list."""
    device = resolve_device(device)
    model = YOLO(model_path)
    fps = meta["fps"] or 25.0

    print(f"[1/2] GPU 稀疏检测（interval={interval}, conf={conf}, batch={batch_size}）...")
    sparse = gpu_sparse_detect(model, video_path, meta, device, conf, imgsz, interval, batch_size=batch_size)
    n_raw = sum(len(v) for v in sparse.values())
    print(f"[2/2] CPU ByteTrack（{len(sparse)} 帧, {n_raw} 框）...")
    per_frame = cpu_byte_track(sparse)
    del model

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
