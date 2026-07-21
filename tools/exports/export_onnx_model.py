#!/usr/bin/env python3
"""Export the face detector to ONNX for accelerated inference tests."""

from __future__ import annotations

import argparse
import os
import shutil

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Export Ultralytics face detector to ONNX")
    p.add_argument("--model", default="models/face.pt", help="Input .pt model")
    p.add_argument("--output", default="models/face.onnx", help="Output .onnx model")
    p.add_argument("--imgsz", type=int, default=1280, help="Export image size")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset")
    p.add_argument("--dynamic", action="store_true", help="Export dynamic batch axes")
    p.add_argument("--simplify", action="store_true", help="Run ONNX simplifier during export")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = os.path.abspath(args.model)
    output_path = os.path.abspath(args.output)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    model = YOLO(model_path)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    exported_path = os.path.abspath(str(exported))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if exported_path != output_path:
        shutil.move(exported_path, output_path)
    print(f"[export] ONNX model ready: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
