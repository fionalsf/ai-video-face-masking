"""One-shot script to create modular architecture files."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
created: list[str] = []


def w(rel: str, content: str) -> str:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    created.append(str(p.relative_to(ROOT)))
    return str(p)


# modules/__init__.py
w("modules/__init__.py", "")

# 1. detect
w("modules/detect/__init__.py", 'from modules.detect.detector import FaceDetector\n\n__all__ = ["FaceDetector"]\n')
w(
    "modules/detect/detector.py",
    '''"""Face detection via ultralytics YOLO."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from core.event import Detection


def resolve_device(device: str) -> str:
    import torch
    if device != "cpu" and not torch.cuda.is_available():
        print("[提示] 未检测到 CUDA，改用 CPU。")
        return "cpu"
    return device


class FaceDetector:
    def __init__(
        self,
        model_path: str,
        device: str = "0",
        conf: float = 0.35,
        imgsz: int = 1280,
    ):
        self.device = resolve_device(device)
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_path)

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        fps: float = 25.0,
    ) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=self.device != "cpu",
            verbose=False,
        )
        out: list[Detection] = []
        t_sec = frame_idx / fps if fps > 0 else 0.0
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
            for j in range(len(xyxy)):
                c = float(confs[j]) if confs is not None else 1.0
                out.append(Detection(
                    frame=frame_idx,
                    t=round(t_sec, 3),
                    bbox=[round(v, 1) for v in xyxy[j].tolist()],
                    conf=round(c, 4),
                ))
        return out

    def detect_video_sparse(
        self,
        video_path: str,
        meta: dict,
        interval: int = 5,
    ) -> list[Detection]:
        interval = max(1, interval)
        fps = meta.get("fps") or 25.0
        detections: list[Detection] = []
        results = self.model.predict(
            source=video_path,
            stream=True,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=self.device != "cpu",
            vid_stride=interval,
            verbose=False,
        )
        total = (meta["frames"] + interval - 1) // interval if meta["frames"] > 0 else None
        for proc_idx, r in enumerate(tqdm(results, total=total, unit="f", desc="GPU detect")):
            frame_idx = proc_idx * interval
            t_sec = frame_idx / fps
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
            for j in range(len(xyxy)):
                c = float(confs[j]) if confs is not None else 1.0
                detections.append(Detection(
                    frame=frame_idx,
                    t=round(t_sec, 3),
                    bbox=[round(v, 1) for v in xyxy[j].tolist()],
                    conf=round(c, 4),
                ))
        return detections
''',
)

# 2. track
w("modules/track/__init__.py", 'from modules.track.tracker import ByteTracker\n\n__all__ = ["ByteTracker"]\n')
w(
    "modules/track/tracker.py",
    '''"""ByteTrack wrapper for tracked detections."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from core.event import Detection, TrackedDetection


class _TrackResults:
    def __init__(self, xyxy, conf, cls):
        self._xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)
        w = self._xyxy[:, 2] - self._xyxy[:, 0]
        h = self._xyxy[:, 3] - self._xyxy[:, 1]
        self.xywh = np.stack(
            [self._xyxy[:, 0] + w / 2, self._xyxy[:, 1] + h / 2, w, h], axis=1
        ).astype(np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        mask = np.asarray(mask)
        return _TrackResults(self._xyxy[mask], self.conf[mask], self.cls[mask])


class ByteTracker:
    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        from ultralytics.trackers.byte_tracker import BYTETracker

        targs = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
        )
        self._tracker = BYTETracker(targs)

    def track_detections(self, detections: list[Detection]) -> list[TrackedDetection]:
        if not detections:
            return []

        by_frame: dict[int, list[Detection]] = defaultdict(list)
        for d in detections:
            by_frame[d.frame].append(d)

        tracked: list[TrackedDetection] = []
        for frame_idx in sorted(by_frame):
            items = by_frame[frame_idx]
            boxes = [d.bbox for d in items]
            confs = [d.conf for d in items]
            if not boxes:
                continue
            res = _TrackResults(boxes, confs, [0.0] * len(confs))
            tracks = self._tracker.update(res, img=None)
            if tracks is None or len(tracks) == 0:
                continue
            for t in tracks:
                row = t.tolist() if hasattr(t, "tolist") else list(t)
                if len(row) < 6:
                    continue
                x1, y1, x2, y2, tid, score = row[:6]
                tracked.append(TrackedDetection(
                    frame=frame_idx,
                    t=items[0].t,
                    bbox=[round(v, 1) for v in [x1, y1, x2, y2]],
                    conf=round(float(score), 4),
                    track_id=int(tid),
                ))
        return tracked
''',
)

# 3. event builder
w("modules/event/__init__.py", 'from modules.event.builder import build_events\n\n__all__ = ["build_events"]\n')
w(
    "modules/event/builder.py",
    '''"""Build events from tracked detections (no scoring/tier)."""

from __future__ import annotations

from core.event import Event, TrackPoint, TrackedDetection


def build_events(tracked: list[TrackedDetection], gap_sec: float = 1.0) -> list[Event]:
    by_track: dict[int, list[TrackedDetection]] = {}
    for d in tracked:
        by_track.setdefault(d.track_id, []).append(d)
    for tid in by_track:
        by_track[tid].sort(key=lambda x: x.frame)

    events: list[Event] = []
    evt_counter = 0

    for track_id in sorted(by_track):
        seq = by_track[track_id]
        chunk: list[TrackedDetection] = []
        for det in seq:
            if chunk and (det.t - chunk[-1].t) >= gap_sec:
                evt_counter += 1
                events.append(_make_event(evt_counter, track_id, chunk))
                chunk = []
            chunk.append(det)
        if chunk:
            evt_counter += 1
            events.append(_make_event(evt_counter, track_id, chunk))

    return events


def _make_event(evt_num: int, track_id: int, chunk: list[TrackedDetection]) -> Event:
    trajectory = [
        TrackPoint(frame=d.frame, t=d.t, bbox=d.bbox, conf=d.conf)
        for d in chunk
    ]
    return Event(
        event_id=f"evt_{evt_num:04d}",
        track_id=track_id,
        start_frame=chunk[0].frame,
        end_frame=chunk[-1].frame,
        start_time=chunk[0].t,
        end_time=chunk[-1].t,
        trajectory=trajectory,
    )
''',
)

w(
    "modules/event/__main__.py",
    '''"""CLI: load tracked json, output events json."""

from __future__ import annotations

import argparse
import sys

from core.io import load_tracked, save_events
from modules.event.builder import build_events


def parse_args():
    p = argparse.ArgumentParser(description="Build events from tracked detections")
    p.add_argument("-i", "--input", required=True, help="Tracked JSON path")
    p.add_argument("-o", "--output", required=True, help="Events JSON path")
    p.add_argument("--event-gap", type=float, default=1.0, help="Event split gap (seconds)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tracked = load_tracked(args.input)
    events = build_events(tracked, gap_sec=args.event_gap)
    save_events(args.output, events)
    print(f"[done] {len(events)} events -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''',
)

# 5-7. scoring
w("modules/scoring/__init__.py", 'from modules.scoring.scorer import score_events\n\n__all__ = ["score_events"]\n')
w(
    "modules/scoring/rules.py",
    '''"""Event-level rule hints (force downgrade to Review)."""

from __future__ import annotations


def box_cy_ratio(bbox: list[float], frame_h: int) -> float:
    y1, y2 = bbox[1], bbox[3]
    return ((y1 + y2) / 2.0) / max(1.0, frame_h)


def box_touches_edge(bbox: list[float], frame_w: int, frame_h: int, margin: float = 5.0) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= margin or y1 <= margin or x2 >= frame_w - margin or y2 >= frame_h - margin


def suggest_rule_hints(
    chunk: list[dict],
    frame_h: int,
    frame_w: int,
    peak_conf: float,
    workzone_cy_ratio: float = 0.60,
) -> list[str]:
    hints = []
    peak_det = max(chunk, key=lambda d: d["conf"])
    bbox = peak_det["bbox"]

    if box_cy_ratio(bbox, frame_h) > workzone_cy_ratio:
        hints.append("workzone_low")
    if box_touches_edge(bbox, frame_w, frame_h) and peak_conf < 0.85:
        hints.append("edge_clip")
    w = max(1.0, bbox[2] - bbox[0])
    h = max(1.0, bbox[3] - bbox[1])
    ar = w / h
    if ar < 0.45 or ar > 2.0:
        hints.append("aspect_ratio")
    return hints
''',
)

w(
    "modules/scoring/scorer.py",
    '''"""Score events: confidence stats, tier, rule hints."""

from __future__ import annotations

from core.event import Event, Tier
from modules.scoring.rules import suggest_rule_hints

AUTO_THRESHOLD = 0.85
REVIEW_MIN = 0.75


def _classify_tier(peak_conf: float) -> str:
    if peak_conf >= AUTO_THRESHOLD:
        return Tier.AUTO.value
    if peak_conf >= REVIEW_MIN:
        return Tier.REVIEW.value
    return Tier.LOW_CONF.value


def _review_status(tier: str) -> str:
    if tier == Tier.AUTO.value:
        return "confirmed_face"
    if tier == Tier.REVIEW.value:
        return "pending"
    return "logged_only"


def score_events(events: list[Event], frame_w: int, frame_h: int) -> list[Event]:
    for ev in events:
        ev.compute_conf_stats()
        peak = ev.peak_confidence or 0.0
        tier = _classify_tier(peak)
        chunk = [p.to_dict() for p in ev.trajectory]
        hints = suggest_rule_hints(chunk, frame_h, frame_w, peak)
        ev.rule_hints = hints
        if hints and tier == Tier.AUTO.value:
            tier = Tier.REVIEW.value
        ev.tier = tier
        ev.review_status = _review_status(tier)
    return events
''',
)

w(
    "modules/scoring/__main__.py",
    '''"""CLI: load events json, score and save."""

from __future__ import annotations

import argparse
import sys

from core.io import load_events, save_events
from modules.scoring.scorer import score_events


def parse_args():
    p = argparse.ArgumentParser(description="Score events (tier + rule hints)")
    p.add_argument("-i", "--input", required=True, help="Events JSON path")
    p.add_argument("-o", "--output", required=True, help="Scored events JSON path")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    events = load_events(args.input)
    events = score_events(events, frame_w=args.width, frame_h=args.height)
    save_events(args.output, events)
    print(f"[done] scored {len(events)} events -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''',
)

print(f"Created {len(created)} files so far")
