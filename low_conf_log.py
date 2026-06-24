"""Low confidence event statistics (log only)."""

from __future__ import annotations

import json
import os
import time

from event_builder import FaceEvent


def write_low_conf_stats(out_dir: str, video_path: str, low_events: list[FaceEvent]) -> str:
    path = os.path.join(out_dir, "low_conf_stats.json")
    payload = {
        "video": os.path.abspath(video_path),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "threshold": "< 0.75 peak_confidence",
        "count": len(low_events),
        "events": [
            {
                "event_id": e.event_id,
                "track_id": e.track_id,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "peak_confidence": e.peak_confidence,
                "avg_confidence": e.avg_confidence,
                "detection_count": e.detection_count,
                "rule_hints": e.rule_hints,
            }
            for e in low_events
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
