"""Merge tracked detections into Events (no scoring)."""

from __future__ import annotations

from collections import defaultdict

from core.event import Event, TrackPoint, TrackedDetection


def build_events(tracked: list[TrackedDetection], gap_sec: float = 1.0) -> list[Event]:
    by_track: dict[int, list[TrackedDetection]] = defaultdict(list)
    for d in tracked:
        by_track[d.track_id].append(d)
    for tid in by_track:
        by_track[tid].sort(key=lambda x: x.frame)

    events: list[Event] = []
    counter = 0
    for track_id in sorted(by_track):
        seq = by_track[track_id]
        chunk: list[TrackedDetection] = []
        for det in seq:
            if chunk and (det.t - chunk[-1].t) >= gap_sec:
                counter += 1
                events.append(_chunk_to_event(counter, track_id, chunk))
                chunk = []
            chunk.append(det)
        if chunk:
            counter += 1
            events.append(_chunk_to_event(counter, track_id, chunk))
    return events


def _chunk_to_event(num: int, track_id: int, chunk: list[TrackedDetection]) -> Event:
    traj = [
        TrackPoint(frame=d.frame, t=d.t, bbox=list(d.bbox), conf=d.conf)
        for d in chunk
    ]
    return Event(
        event_id=f"evt_{num:04d}",
        track_id=track_id,
        start_frame=chunk[0].frame,
        end_frame=chunk[-1].frame,
        start_time=chunk[0].t,
        end_time=chunk[-1].t,
        trajectory=traj,
    )
