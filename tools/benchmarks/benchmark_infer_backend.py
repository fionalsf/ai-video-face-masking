#!/usr/bin/env python3
"""Benchmark detector backends without running the full pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from detect_backends import create_detection_backend


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark sparse face detector inference backends")
    p.add_argument("-i", "--input", required=True, help="Input video")
    p.add_argument("--backend", choices=["torch", "onnx", "tensorrt"], default="onnx")
    p.add_argument("--model", default="models/face.pt")
    p.add_argument("--onnx-model", default="models/face.onnx")
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--interval", type=int, default=2)
    p.add_argument("--samples", type=int, default=80, help="Number of sparse frames to benchmark")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--warmup", type=int, default=2)
    return p.parse_args()


def read_sparse_frames(video_path: str, interval: int, samples: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frames = []
    frame_idx = 0
    try:
        while len(frames) < samples:
            if frame_idx % interval == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            else:
                if not cap.grab():
                    break
            frame_idx += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("No frames read")
    return frames


def chunks(items, size: int):
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main() -> int:
    args = parse_args()
    frames = read_sparse_frames(args.input, max(1, args.interval), max(1, args.samples))
    backend = create_detection_backend(
        args.backend,
        args.model,
        args.onnx_model,
        args.device,
        args.conf,
        args.imgsz,
    )

    provider = getattr(backend, "provider", backend.name)
    print(
        f"[bench] backend={backend.name} provider={provider} "
        f"samples={len(frames)} batch={args.batch} imgsz={args.imgsz}"
    )

    warmup_frames = frames[: min(len(frames), max(0, args.warmup))]
    if warmup_frames:
        backend.predict_batch(warmup_frames)

    started = time.perf_counter()
    boxes = 0
    for batch in chunks(frames, args.batch):
        result = backend.predict_batch(batch)
        boxes += sum(len(x) for x in result)
    elapsed = time.perf_counter() - started
    backend.close()

    fps = len(frames) / elapsed if elapsed > 0 else 0.0
    print(f"[bench] elapsed={elapsed:.3f}s sparse_fps={fps:.2f} boxes={boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
