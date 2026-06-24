"""Build Face Events from tracked detections + tier classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from video_meta import sec_to_timecode

from gap_analysis import (
    DEFAULT_HARD_SPLIT_GAP_SEC,
    analyze_gaps,
    presence_segment_id,
    save_gap_analysis_debug,
    split_presence_segments,
)


TIER_AUTO = "auto"
TIER_REVIEW = "review"
TIER_LOW_CONF = "low_conf"

AUTO_THRESHOLD = 0.85
REVIEW_MIN = 0.75
HIGH_CONF_RECOVERY = 0.75

SCHEME_C_LABEL = "scheme_c_gap_semantic_layer"

DEFAULT_PRE_PADDING_SEC = 0.25
DEFAULT_POST_PADDING_SEC = 0.4


def resolve_temporal_padding(
    fps: float,
    *,
    detect_interval: int | None = None,
    pre_padding_sec: float | None = None,
    post_padding_sec: float | None = None,
) -> tuple[float, float]:
    """Resolve pre/post padding; auto from detect_interval when available."""
    if detect_interval is not None and detect_interval > 0 and fps > 0:
        auto_sec = (detect_interval * 2) / fps
        pre = pre_padding_sec if pre_padding_sec is not None else auto_sec
        post = post_padding_sec if post_padding_sec is not None else auto_sec
    else:
        pre = pre_padding_sec if pre_padding_sec is not None else DEFAULT_PRE_PADDING_SEC
        post = post_padding_sec if post_padding_sec is not None else DEFAULT_POST_PADDING_SEC
    return pre, post


def padded_event_bounds(
    first_det_time: float,
    last_det_time: float,
    first_det_frame: int,
    last_det_frame: int,
    *,
    pre_padding_sec: float,
    post_padding_sec: float,
    fps: float,
    total_frames: int | None,
) -> tuple[float, float, int, int]:
    """Expand event window beyond first/last detection (visual coverage interval)."""
    start_time = max(0.0, first_det_time - pre_padding_sec)
    end_time = last_det_time + post_padding_sec
    if total_frames is not None and total_frames > 0 and fps > 0:
        end_time = min(end_time, (total_frames - 1) / fps)

    if fps > 0:
        start_frame = max(0, int(round(start_time * fps)))
        end_frame = int(round(end_time * fps))
    else:
        start_frame = first_det_frame
        end_frame = last_det_frame

    if total_frames is not None and total_frames > 0:
        end_frame = min(total_frames - 1, end_frame)
        start_frame = min(start_frame, total_frames - 1)

    if fps > 0:
        start_time = start_frame / fps
        end_time = end_frame / fps

    return round(start_time, 3), round(end_time, 3), start_frame, end_frame


@dataclass
class FaceEvent:
    event_id: str
    track_id: int
    tier: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    avg_confidence: float
    peak_confidence: float
    detection_count: int
    trajectory: list[dict] = field(default_factory=list)
    rule_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "tier": self.tier,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_timecode": sec_to_timecode(self.start_time),
            "end_timecode": sec_to_timecode(self.end_time),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_sec": round(self.end_time - self.start_time, 3),
            "avg_confidence": self.avg_confidence,
            "peak_confidence": self.peak_confidence,
            "detection_count": self.detection_count,
            "trajectory": self.trajectory,
            "rule_hints": self.rule_hints,
            "review_status": self._default_review_status(),
        }

    def _default_review_status(self) -> str:
        if self.tier == TIER_AUTO:
            return "confirmed_face"
        if self.tier == TIER_REVIEW:
            return "pending"
        return "logged_only"


def classify_tier(peak_conf: float) -> str:
    if peak_conf >= AUTO_THRESHOLD:
        return TIER_AUTO
    if peak_conf >= REVIEW_MIN:
        return TIER_REVIEW
    return TIER_LOW_CONF


def bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def should_merge_detections(
    prev: dict,
    curr: dict,
    gap_sec: float,
    *,
    hard_split_gap_sec: float = DEFAULT_HARD_SPLIT_GAP_SEC,
) -> bool:
    """Scheme C: time-only merge on same track; hard split when gap > hard_split_gap_sec.

    confidence / IoU do not participate in merge decisions.
    """
    _ = gap_sec  # retained for API compatibility with legacy stats callers
    gap = curr["t"] - prev["t"]
    return gap <= hard_split_gap_sec


def _build_chunks_legacy(seq: list[dict], gap_sec: float) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    chunk: list[dict] = []
    for det in seq:
        if chunk and (det["t"] - chunk[-1]["t"]) >= gap_sec:
            chunks.append(chunk)
            chunk = []
        chunk.append(det)
    if chunk:
        chunks.append(chunk)
    return chunks


def _build_chunks_merged(
    seq: list[dict],
    gap_sec: float,
    *,
    hard_split_gap_sec: float = DEFAULT_HARD_SPLIT_GAP_SEC,
) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    chunk: list[dict] = []
    for det in seq:
        if chunk and not should_merge_detections(
            chunk[-1], det, gap_sec, hard_split_gap_sec=hard_split_gap_sec
        ):
            chunks.append(chunk)
            chunk = []
        chunk.append(det)
    if chunk:
        chunks.append(chunk)
    return chunks


def _count_single_frame_chunks(chunks: list[list[dict]]) -> int:
    return sum(1 for c in chunks if len(c) == 1)


def _singleton_rescue_direction(
    prev: dict | None,
    curr: dict,
    nxt: dict | None,
    gap_sec: float,
    *,
    hard_split_gap_sec: float = DEFAULT_HARD_SPLIT_GAP_SEC,
) -> str | None:
    """Time-only singleton rescue (same hard split as should_merge_detections)."""
    if prev and should_merge_detections(
        prev, curr, gap_sec, hard_split_gap_sec=hard_split_gap_sec
    ):
        return "prev"
    if nxt and should_merge_detections(
        curr, nxt, gap_sec, hard_split_gap_sec=hard_split_gap_sec
    ):
        return "next"
    return None


def _absorb_singleton_chunks(
    chunks: list[dict],
    gap_sec: float,
    *,
    hard_split_gap_sec: float = DEFAULT_HARD_SPLIT_GAP_SEC,
) -> tuple[list[list[dict]], set[int]]:
    """Merge 1-detection chunks into neighbors when possible.

    Returns merged chunks and indices of orphan singletons (low_conf_event).
    """
    if not chunks:
        return [], set()

    work = [list(c) for c in chunks]
    changed = True
    while changed:
        changed = False
        next_pass: list[list[dict]] = []
        i = 0
        while i < len(work):
            chunk = work[i]
            if len(chunk) == 1:
                prev_det = next_pass[-1][-1] if next_pass else None
                next_det = work[i + 1][0] if i + 1 < len(work) else None
                direction = _singleton_rescue_direction(
                    prev_det,
                    chunk[0],
                    next_det,
                    gap_sec,
                    hard_split_gap_sec=hard_split_gap_sec,
                )
                if direction == "prev" and next_pass:
                    next_pass[-1].extend(chunk)
                    changed = True
                    i += 1
                    continue
                if direction == "next" and i + 1 < len(work):
                    next_pass.append(chunk + work[i + 1])
                    changed = True
                    i += 2
                    continue
            next_pass.append(chunk)
            i += 1
        work = next_pass

    orphan_indices = {i for i, c in enumerate(work) if len(c) == 1}
    return work, orphan_indices


def build_events(
    detections: list[dict],
    gap_sec: float = 1.0,
    frame_h: int = 1080,
    frame_w: int = 1920,
    merge_stats: dict[str, Any] | None = None,
    *,
    fps: float = 30.0,
    total_frames: int | None = None,
    detect_interval: int | None = None,
    pre_padding_sec: float | None = None,
    post_padding_sec: float | None = None,
    hard_split_gap_sec: float = DEFAULT_HARD_SPLIT_GAP_SEC,
    output_dir: str | None = None,
) -> list[FaceEvent]:
    from rules import suggest_rule_hints

    pre_pad, post_pad = resolve_temporal_padding(
        fps,
        detect_interval=detect_interval,
        pre_padding_sec=pre_padding_sec,
        post_padding_sec=post_padding_sec,
    )

    gap_result = analyze_gaps(
        detections,
        hard_split_gap_sec=hard_split_gap_sec,
        fps=fps,
    )

    by_track: dict[int, list[dict]] = {}
    for d in detections:
        by_track.setdefault(d["track_id"], []).append(d)
    for tid in by_track:
        by_track[tid].sort(key=lambda x: x["frame"])

    original_count = sum(len(_build_chunks_legacy(seq, gap_sec)) for seq in by_track.values())

    flat_chunks: list[tuple[int, list[dict], bool, str, int]] = []
    excluded_fragments: list[dict] = []
    single_before = 0

    for track_id in sorted(by_track):
        presence_segments = gap_result.presence_by_track.get(track_id, [])
        for seg_idx, segment in enumerate(presence_segments):
            if not segment:
                continue
            chunks = _build_chunks_merged(
                segment, gap_sec, hard_split_gap_sec=hard_split_gap_sec
            )
            single_before += _count_single_frame_chunks(chunks)
            chunks, orphan_idx = _absorb_singleton_chunks(
                chunks, gap_sec, hard_split_gap_sec=hard_split_gap_sec
            )
            pres_id = presence_segment_id(track_id, seg_idx)
            for cidx, chunk in enumerate(chunks):
                is_orphan = cidx in orphan_idx
                if is_orphan:
                    excluded_fragments.append({
                        "track_id": track_id,
                        "presence_segment_id": pres_id,
                        "time": chunk[0]["t"],
                        "frame": chunk[0]["frame"],
                        "confidence": chunk[0]["conf"],
                        "bbox": chunk[0]["bbox"],
                        "reason": "singleton_fragment",
                    })
                flat_chunks.append((track_id, chunk, is_orphan, pres_id, seg_idx))

    events: list[FaceEvent] = []
    event_bindings: list[dict[str, Any]] = []
    for evt_num, (track_id, chunk, is_orphan, pres_id, seg_idx) in enumerate(
        flat_chunks, start=1
    ):
        ev = _make_event(
            evt_num,
            track_id,
            chunk,
            frame_h,
            frame_w,
            suggest_rule_hints,
            force_low_conf=is_orphan,
            fps=fps,
            total_frames=total_frames,
            pre_padding_sec=pre_pad,
            post_padding_sec=post_pad,
        )
        events.append(ev)
        event_bindings.append({
            "event_id": ev.event_id,
            "track_id": track_id,
            "presence_segment_id": pres_id,
            "presence_segment_index": seg_idx,
            "detection_count": ev.detection_count,
            "start_time": ev.start_time,
            "end_time": ev.end_time,
            "start_frame": ev.start_frame,
            "end_frame": ev.end_frame,
        })

    single_after = sum(1 for e in events if e.detection_count == 1)
    merged_count = len(events)

    if output_dir:
        save_gap_analysis_debug(output_dir, gap_result, event_bindings)

    if merge_stats is not None:
        merge_stats.clear()
        merge_stats.update({
            "original_event_count": original_count,
            "merged_event_count": merged_count,
            "single_frame_event_count_before": single_before,
            "single_frame_event_count_after": single_after,
            "merge_ratio": round(original_count / max(1, merged_count), 3),
            "merge_strategy": SCHEME_C_LABEL,
            "gap_semantic_layer": True,
            "event_gap_sec": gap_sec,
            "hard_split_gap_sec": hard_split_gap_sec,
            "confidence_merge_enabled": False,
            "presence_segment_total": gap_result.presence_segment_total,
            "absence_segment_total": gap_result.absence_segment_total,
            "pre_padding_sec": pre_pad,
            "post_padding_sec": post_pad,
            "detect_interval": detect_interval,
            "padding_frames_auto": detect_interval * 2 if detect_interval else None,
            "excluded_singleton_fragments": len(excluded_fragments),
            "singleton_fragment_events": len(excluded_fragments),
            "low_conf_fragments": excluded_fragments,
        })

    return events


def _make_event(
    evt_num: int,
    track_id: int,
    chunk: list[dict],
    frame_h: int,
    frame_w: int,
    suggest_fn,
    force_low_conf: bool = False,
    *,
    fps: float = 30.0,
    total_frames: int | None = None,
    pre_padding_sec: float = DEFAULT_PRE_PADDING_SEC,
    post_padding_sec: float = DEFAULT_POST_PADDING_SEC,
) -> FaceEvent:
    confs = [d["conf"] for d in chunk]
    peak = max(confs)
    tier = classify_tier(peak)
    hints = suggest_fn(chunk, frame_h, frame_w, peak)
    if hints and tier == TIER_AUTO:
        tier = TIER_REVIEW
    if force_low_conf:
        hints = list(hints) + ["singleton_fragment"]
        tier = TIER_LOW_CONF

    trajectory = [
        {
            "t": d["t"],
            "frame": d["frame"],
            "bbox": d["bbox"],
            "conf": d["conf"],
        }
        for d in chunk
    ]
    start_time, end_time, start_frame, end_frame = padded_event_bounds(
        chunk[0]["t"],
        chunk[-1]["t"],
        chunk[0]["frame"],
        chunk[-1]["frame"],
        pre_padding_sec=pre_padding_sec,
        post_padding_sec=post_padding_sec,
        fps=fps,
        total_frames=total_frames,
    )
    return FaceEvent(
        event_id=f"evt_{evt_num:04d}",
        track_id=track_id,
        tier=tier,
        start_time=start_time,
        end_time=end_time,
        start_frame=start_frame,
        end_frame=end_frame,
        avg_confidence=round(mean(confs), 4),
        peak_confidence=round(peak, 4),
        detection_count=len(chunk),
        trajectory=trajectory,
        rule_hints=hints,
    )


def events_by_tier(events: list[FaceEvent]) -> dict[str, list[FaceEvent]]:
    out = {TIER_AUTO: [], TIER_REVIEW: [], TIER_LOW_CONF: []}
    for e in events:
        out[e.tier].append(e)
    return out
