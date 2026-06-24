"""ByteTrack wrapper: Detection in -> TrackedDetection out."""

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
    def __init__(self):
        from ultralytics.trackers.byte_tracker import BYTETracker
        targs = SimpleNamespace(
            track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
            track_buffer=30, match_thresh=0.8, fuse_score=True,
        )
        self._tracker = BYTETracker(targs)

    def track_frame(self, frame_idx: int, t: float, detections: list[Detection]) -> list[TrackedDetection]:
        if not detections:
            return []
        res = _TrackResults(
            [d.bbox for d in detections], [d.conf for d in detections], [0.0] * len(detections),
        )
        tracks = self._tracker.update(res, img=None)
        out: list[TrackedDetection] = []
        if tracks is not None and len(tracks) > 0:
            for tr in tracks:
                row = tr.tolist() if hasattr(tr, "tolist") else list(tr)
                if len(row) < 6:
                    continue
                x1, y1, x2, y2, tid, score = row[:6]
                out.append(TrackedDetection(
                    frame=frame_idx, t=t,
                    bbox=[round(float(v), 1) for v in (x1, y1, x2, y2)],
                    conf=round(float(score), 4), track_id=int(tid),
                ))
        return out

    def track_detections(self, detections: list[Detection]) -> list[TrackedDetection]:
        by_frame: dict[int, list[Detection]] = defaultdict(list)
        for d in detections:
            by_frame[d.frame].append(d)
        tracked: list[TrackedDetection] = []
        for frame_idx in sorted(by_frame):
            tracked.extend(self.track_frame(frame_idx, by_frame[frame_idx][0].t, by_frame[frame_idx]))
        return tracked
