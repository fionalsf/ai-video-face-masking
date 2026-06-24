#!/usr/bin/env python3
"""python -m modules.event -i tracked.json -o events.json"""

from __future__ import annotations

import argparse
import sys

from core.io import load_tracked, save_events
from modules.event.builder import build_events


def main() -> int:
    p = argparse.ArgumentParser(description="Event module")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--event-gap", type=float, default=1.0)
    args = p.parse_args()
    events = build_events(load_tracked(args.input), gap_sec=args.event_gap)
    save_events(args.output, events)
    print(f"[event] {len(events)} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
