#!/usr/bin/env python3
"""python -m modules.track -i detections.json -o tracked.json"""

from __future__ import annotations

import argparse
import sys

from core.io import load_detections, save_tracked
from modules.track.tracker import ByteTracker


def main() -> int:
    p = argparse.ArgumentParser(description="Track module")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()
    tracked = ByteTracker().track_detections(load_detections(args.input))
    save_tracked(args.output, tracked)
    print(f"[track] {len(tracked)} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
