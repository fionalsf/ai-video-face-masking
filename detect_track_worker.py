#!/usr/bin/env python3
"""Isolated detect+track worker.

The main pipeline can run this script as a subprocess so model runtimes
release GPU/CPU memory before later review-pack stages start.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description="Run face detect+track in an isolated process")
    p.add_argument("--config", required=True, help="JSON config written by pipeline.py")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    from tracker import run_detect_track

    started = time.perf_counter()
    detections = run_detect_track(
        cfg["video_path"],
        cfg["model_path"],
        cfg["meta"],
        device=cfg.get("device", "0"),
        conf=float(cfg.get("conf", 0.35)),
        imgsz=int(cfg.get("imgsz", 1280)),
        interval=int(cfg.get("interval", 5)),
        batch_size=int(cfg.get("batch_size", 4)),
        infer_backend=cfg.get("infer_backend", "torch"),
        onnx_model_path=cfg.get("onnx_model_path"),
        decode_backend=cfg.get("decode_backend", "opencv"),
    )

    output_path = cfg["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)

    report_path = cfg.get("report_path")
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "backend": cfg.get("infer_backend", "torch"),
                    "decode_backend": cfg.get("decode_backend", "opencv"),
                    "detections": len(detections),
                    "wall_sec": round(time.perf_counter() - started, 3),
                    "output_path": output_path,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
