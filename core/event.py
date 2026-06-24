"""Core Event data model ? the single contract between all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from statistics import mean


class Tier(StrEnum):
    AUTO = "auto"
    REVIEW = "review"
    LOW_CONF = "low_conf"


@dataclass
class Detection:
    frame: int
    t: float
    bbox: list[float]
    conf: float

    def to_dict(self) -> dict:
        return {"frame": self.frame, "t": self.t, "bbox": self.bbox, "conf": self.conf}

    @classmethod
    def from_dict(cls, d: dict) -> Detection:
        return cls(
            frame=int(d["frame"]),
            t=float(d["t"]),
            bbox=list(d["bbox"]),
            conf=float(d["conf"]),
        )


@dataclass
class TrackedDetection(Detection):
    track_id: int = 0

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["track_id"] = self.track_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrackedDetection:
        return cls(
            frame=int(d["frame"]),
            t=float(d["t"]),
            bbox=list(d["bbox"]),
            conf=float(d["conf"]),
            track_id=int(d["track_id"]),
        )


@dataclass
class TrackPoint:
    frame: int
    t: float
    bbox: list[float]
    conf: float

    def to_dict(self) -> dict:
        return {"frame": self.frame, "t": self.t, "bbox": self.bbox, "conf": self.conf}

    @classmethod
    def from_dict(cls, d: dict) -> TrackPoint:
        return cls(
            frame=int(d["frame"]),
            t=float(d["t"]),
            bbox=list(d["bbox"]),
            conf=float(d["conf"]),
        )


@dataclass
class Event:
    event_id: str
    track_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    trajectory: list[TrackPoint] = field(default_factory=list)
    peak_confidence: float | None = None
    avg_confidence: float | None = None
    tier: str | None = None
    rule_hints: list[str] = field(default_factory=list)
    review_status: str = "pending"

    @property
    def duration_sec(self) -> float:
        return round(self.end_time - self.start_time, 3)

    @property
    def detection_count(self) -> int:
        return len(self.trajectory)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_timecode": sec_to_timecode(self.start_time),
            "end_timecode": sec_to_timecode(self.end_time),
            "duration_sec": self.duration_sec,
            "trajectory": [p.to_dict() for p in self.trajectory],
            "peak_confidence": self.peak_confidence,
            "avg_confidence": self.avg_confidence,
            "detection_count": self.detection_count,
            "tier": self.tier,
            "rule_hints": list(self.rule_hints),
            "review_status": self.review_status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        traj = [TrackPoint.from_dict(p) for p in d.get("trajectory", [])]
        return cls(
            event_id=d["event_id"],
            track_id=int(d["track_id"]),
            start_frame=int(d["start_frame"]),
            end_frame=int(d["end_frame"]),
            start_time=float(d["start_time"]),
            end_time=float(d["end_time"]),
            trajectory=traj,
            peak_confidence=d.get("peak_confidence"),
            avg_confidence=d.get("avg_confidence"),
            tier=d.get("tier"),
            rule_hints=list(d.get("rule_hints") or []),
            review_status=d.get("review_status", "pending"),
        )

    def compute_conf_stats(self) -> None:
        if not self.trajectory:
            return
        confs = [p.conf for p in self.trajectory]
        self.peak_confidence = round(max(confs), 4)
        self.avg_confidence = round(mean(confs), 4)


def sec_to_timecode(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{int((sec % 1) * 1000):03d}"


def events_by_tier(events: list[Event]) -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = {t.value: [] for t in Tier}
    for e in events:
        key = e.tier or Tier.LOW_CONF.value
        if key in out:
            out[key].append(e)
        else:
            out[Tier.LOW_CONF.value].append(e)
    return out
