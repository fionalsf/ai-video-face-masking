"""Audit log for Auto-tier events."""

from __future__ import annotations

import json
import os
import time

from core.event import Event


def write_audit_log(out_dir: str, video_path: str, auto_events: list[Event]) -> str:
    path = os.path.join(out_dir, "audit.log")
    lines = [
        "# audit.log - Auto tier masked events",
        f"# video: {os.path.abspath(video_path)}",
        f"# generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"# count: {len(auto_events)}",
        "",
    ]
    for ev in auto_events:
        lines.append(
            f"{ev.event_id}\ttrack={ev.track_id}\t"
            f"{ev.start_time:.3f}-{ev.end_time:.3f}\t"
            f"peak={ev.peak_confidence:.4f}\tavg={ev.avg_confidence:.4f}\t"
            f"frames={ev.start_frame}-{ev.end_frame}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    json_path = os.path.join(out_dir, "audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "video": os.path.abspath(video_path),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "events": [e.to_dict() for e in auto_events],
        }, f, ensure_ascii=False, indent=2)
    return path
