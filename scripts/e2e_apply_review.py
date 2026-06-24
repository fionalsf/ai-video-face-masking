#!/usr/bin/env python3
"""Apply batch review decisions for E2E validation (peak >= 0.75 -> accepted)."""

from __future__ import annotations

import json
import os
import sys

from review_stats import STATUS_ACCEPTED, STATUS_REJECTED, update_review_report


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "output/detection/DJI_20260511100755_0008_D"
    out = os.path.abspath(out)
    with open(os.path.join(out, "event_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    events = summary["events"]
    video = summary.get("video", "")

    decisions: dict[str, str] = {}
    for e in events:
        peak = float(e.get("peak_confidence") or 0)
        decisions[e["event_id"]] = STATUS_ACCEPTED if peak >= 0.75 else STATUS_REJECTED

    with open(os.path.join(out, "confirmed_events.json"), "w", encoding="utf-8") as f:
        json.dump(dict(sorted(decisions.items())), f, ensure_ascii=False, indent=2)

    report = update_review_report(out, video, events, decisions)
    with open(os.path.join(out, "review_result.json"), "w", encoding="utf-8") as f:
        json.dump({"decisions": decisions, "events_reviewed": len(decisions)}, f, ensure_ascii=False, indent=2)

    acc = sum(1 for v in decisions.values() if v == STATUS_ACCEPTED)
    rej = sum(1 for v in decisions.values() if v == STATUS_REJECTED)
    print(f"review: accepted={acc} rejected={rej} accept_rate={report['accept_rate']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
