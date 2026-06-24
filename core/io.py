"""JSON I/O at module boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from core.event import Detection, Event, TrackedDetection


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_detections(path: str | Path) -> list[Detection]:
    data = load_json(path)
    items = data.get("detections", [])
    return [Detection.from_dict(d) for d in items]


def save_detections(path: str | Path, detections: list[Detection], **meta) -> None:
    save_json(path, {"detections": [d.to_dict() for d in detections], **meta})


def load_tracked(path: str | Path) -> list[TrackedDetection]:
    data = load_json(path)
    items = data.get("tracked", data.get("detections", []))
    return [TrackedDetection.from_dict(d) for d in items]


def save_tracked(path: str | Path, tracked: list[TrackedDetection], **meta) -> None:
    save_json(path, {"tracked": [t.to_dict() for t in tracked], **meta})


def load_events(path: str | Path) -> list[Event]:
    data = load_json(path)
    return [Event.from_dict(e) for e in data.get("events", [])]


def save_events(path: str | Path, events: list[Event], **meta) -> None:
    save_json(path, {"events": [e.to_dict() for e in events], **meta})
