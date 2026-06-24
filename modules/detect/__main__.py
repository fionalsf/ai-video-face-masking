#!/usr/bin/env python3
"""python -m modules.detect -i video.mp4 -o detections.json"""

from __future__ import annotations

import argparse
import sys

from core.io import save_detections
from modules.detect.detector import FaceDetector
from utils.video_meta import get_video_meta


def main() -> int:
    p = argparse.ArgumentParser(description="Detect module")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--model", default="models/face.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--interval", type=int, default=5)
    args = p.parse_args()
    meta = get_video_meta(args.input)
    det = FaceDetector(args.model, args.device, args.conf, args.imgsz)
    try:
        if args.input.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            import cv2
            dets = det.detect_frame(cv2.imread(args.input), 0, 0.0)
        else:
            dets = det.detect_video_sparse(args.input, meta["fps"], meta["frames"], args.interval)
    finally:
        det.close()
    save_detections(args.output, dets, video=args.input, fps=meta["fps"], interval=args.interval)
    print(f"[detect] {len(dets)} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
