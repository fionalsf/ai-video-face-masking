"""One-off: rebuild face_events.json from behavior_events.json (no pipeline code changes)."""
import json
import os
import sys
import time

from identity_behavior_builder import behavior_event_to_face_event

PIPELINE_LABEL = (
    "detection -> tracking -> presence -> identity_stitching -> identity_behavior_builder_v1.2"
)


def main() -> int:
    out_dir = os.path.abspath(sys.argv[1])
    beh_path = os.path.join(out_dir, "behavior_events.json")
    face_path = os.path.join(out_dir, "face_events.json")

    with open(beh_path, encoding="utf-8") as f:
        beh = json.load(f)

    old_count = None
    if os.path.isfile(face_path):
        with open(face_path, encoding="utf-8") as f:
            old_count = len(json.load(f).get("events", []))

    events = [behavior_event_to_face_event(ev) for ev in beh["events"]]
    video = beh.get("video")
    if not video and os.path.isfile(face_path):
        with open(face_path, encoding="utf-8") as f:
            video = json.load(f).get("video")

    doc = {
        "video": video,
        "fps": beh.get("fps"),
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": PIPELINE_LABEL,
        "event_unit": "identity_id",
        "source": "behavior_events.json",
        "behavior_event_count": beh.get("behavior_event_count"),
        "events": [e.to_dict() for e in events],
    }
    with open(face_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    auto = sum(1 for e in events if e.tier == "auto")
    review = sum(1 for e in events if e.tier == "review")
    low = sum(1 for e in events if e.tier == "low_conf")
    print(f"[refresh] face_events {old_count} -> {len(events)} (auto={auto} review={review} low={low})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
