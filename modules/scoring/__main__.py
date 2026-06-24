#!/usr/bin/env python3
"""python -m modules.scoring -i events.json -o scored.json --width 1920 --height 1080"""

from __future__ import annotations

import argparse
import sys

from core.io import load_events, save_events
from core.event import events_by_tier
from modules.scoring.scorer import score_events


def main() -> int:
    p = argparse.ArgumentParser(description="Scoring module")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    args = p.parse_args()
    scored = score_events(load_events(args.input), args.width, args.height)
    tiers = events_by_tier(scored)
    save_events(args.output, scored)
    print(f"[scoring] auto={len(tiers['auto'])} review={len(tiers['review'])} low={len(tiers['low_conf'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
