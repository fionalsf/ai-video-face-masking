#!/usr/bin/env python3
"""Phase 1: YOLO-face detection validation only (no tracking / event / render)."""

from __future__ import annotations

import argparse
import os
import sys

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from export import (
    build_detection_summary,
    export_review_images,
    frame_timestamp,
    write_phase1_outputs,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 1 — YOLO-face detection validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", dest="video", help="Input video path")
    p.add_argument("-i", "--input", dest="video", help=argparse.SUPPRESS)
    p.add_argument(
        "--output-dir",
        default="output/detection",
        help="Output root; results go to <output-dir>/<video_stem>/",
    )
    p.add_argument("--interval", type=int, default=5, help="Detection stride (frames)")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    p.add_argument("--model", default="models/face.pt", help="YOLO-face weights")
    p.add_argument("--device", default="0", help="GPU id or cpu")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference size")
    return p.parse_args()


def get_video_meta(path: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, width, height


def parse_yolo_results(result, conf_threshold: float) -> list[dict]:
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "confidence": round(conf, 4),
        })
    return detections


def run_detection(args) -> int:
    if not args.video:
        print("[error] --video is required", file=sys.stderr)
        print("Usage: python detect.py --video xxx.mp4", file=sys.stderr)
        return 1

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        print(f"[error] Video not found: {video_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.model):
        print(f"[error] Model not found: {args.model}", file=sys.stderr)
        return 1

    fps, total_frames, width, height = get_video_meta(video_path)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    interval = max(1, args.interval)

    out_dir = os.path.join(os.path.abspath(args.output_dir), video_stem)
    review_dir = os.path.join(out_dir, "review_images")
    os.makedirs(review_dir, exist_ok=True)

    print(f"[info] video: {video_path}")
    print(f"[info] {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    print(f"[info] stride={interval}, conf={args.conf}, imgsz={args.imgsz}")
    print(f"[info] output: {out_dir}")

    model = YOLO(args.model)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Cannot open video: {video_path}", file=sys.stderr)
        return 1

    frame_records: list[dict] = []
    all_confs: list[float] = []
    sampled_frames = 0
    frame_idx = 0
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="detect", unit="frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % interval == 0:
            result = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            detections = parse_yolo_results(result, args.conf)
            ts_sec = frame_timestamp(frame_idx, fps)

            frame_records.append({
                "frame": frame_idx,
                "timestamp_sec": ts_sec,
                "detections": detections,
            })
            if detections:
                all_confs.extend(d["confidence"] for d in detections)
                export_review_images(review_dir, frame_idx, frame, detections)

            sampled_frames += 1

        frame_idx += 1
        pbar.update(1)
        if total_frames > 0 and frame_idx >= total_frames:
            break

    pbar.close()
    cap.release()

    if total_frames <= 0:
        total_frames = frame_idx

    summary = build_detection_summary(
        video_path=video_path,
        fps=fps,
        total_frames=total_frames,
        sampled_frames=sampled_frames,
        all_confs=all_confs,
        interval=interval,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
    )
    write_phase1_outputs(out_dir, frame_records, summary)

    print()
    print("[done] Phase 1 detection validation")
    print(f"  detections.json       : {os.path.join(out_dir, 'detections.json')}")
    print(f"  detection_summary.json: {os.path.join(out_dir, 'detection_summary.json')}")
    print(f"  review_images/        : {len(all_confs)} images")
    print(f"  sampled_frames        : {summary['sampled_frames']}")
    print(f"  total_detections      : {summary['total_detections']}")
    print(f"  avg_confidence        : {summary['avg_confidence']}")
    print(f"  detections_per_minute : {summary['detections_per_minute']}")
    return 0


def main():
    args = parse_args()
    sys.exit(run_detection(args))


if __name__ == "__main__":
    main()
