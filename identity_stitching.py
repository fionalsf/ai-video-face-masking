"""Pipeline core: cross-track identity stitching (before behavior event build)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

import cv2

from appearance_embedder import AppearanceEmbedder
from gap_analysis import group_by_track
from identity_graph_matching import (
    DEFAULT_ASSIGNMENT_SCORE_MIN,
    DEFAULT_TEMPORAL_TAU_SEC,
    HSV_ASSIGNMENT_GAP_MAX_SEC,
    build_cluster_rows,
    global_identity_matching,
)

IDENTITY_CLUSTERS_NAME = "identity_clusters.json"
IDENTITY_GRAPH_NAME = "identity_graph.json"
TRACK_GRAPH_NAME = "track_graph.json"
STITCHING_LAYER = "identity_stitching_v2_graph"
TARGET_EVENT_RANGE = (30, 80)


@dataclass
class TrackProfile:
    track_id: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    detection_count: int
    peak_conf: float
    avg_conf: float
    first_bbox: list[float]
    last_bbox: list[float]
    first_center: tuple[float, float]
    last_center: tuple[float, float]
    exit_velocity: tuple[float, float] = (0.0, 0.0)
    entry_velocity: tuple[float, float] = (0.0, 0.0)
    embedding: list[float] = field(default_factory=list)
    sample_frame: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_sec": round(self.end_time - self.start_time, 3),
            "detection_count": self.detection_count,
            "peak_confidence": round(self.peak_conf, 4),
            "avg_confidence": round(self.avg_conf, 4),
            "first_bbox": [round(v, 1) for v in self.first_bbox],
            "last_bbox": [round(v, 1) for v in self.last_bbox],
            "sample_frame": self.sample_frame,
            "exit_velocity": [round(v, 3) for v in self.exit_velocity],
            "entry_velocity": [round(v, 3) for v in self.entry_velocity],
            "has_embedding": bool(self.embedding),
        }


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_diagonal(bbox: list[float]) -> float:
    return math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])


def _segment_velocity(a: dict, b: dict) -> tuple[float, float]:
    dt = float(b["t"]) - float(a["t"])
    if dt <= 0:
        return (0.0, 0.0)
    cx_a, cy_a = _bbox_center(a["bbox"])
    cx_b, cy_b = _bbox_center(b["bbox"])
    return ((cx_b - cx_a) / dt, (cy_b - cy_a) / dt)


def build_track_profiles(
    tracked: list[dict],
    *,
    use_detection_embeddings: bool = True,
) -> dict[int, TrackProfile]:
    by_track = group_by_track(tracked)
    profiles: dict[int, TrackProfile] = {}
    for track_id, seq in by_track.items():
        peak = max(seq, key=lambda d: float(d["conf"]))
        confs = [float(d["conf"]) for d in seq]
        first, last = seq[0], seq[-1]
        fb, lb = list(first["bbox"]), list(last["bbox"])
        exit_v = _segment_velocity(seq[-2], seq[-1]) if len(seq) >= 2 else (0.0, 0.0)
        entry_v = _segment_velocity(seq[0], seq[1]) if len(seq) >= 2 else (0.0, 0.0)
        profile = TrackProfile(
            track_id=int(track_id),
            start_time=float(first["t"]),
            end_time=float(last["t"]),
            start_frame=int(first["frame"]),
            end_frame=int(last["frame"]),
            detection_count=len(seq),
            peak_conf=float(peak["conf"]),
            avg_conf=mean(confs),
            first_bbox=fb,
            last_bbox=lb,
            first_center=_bbox_center(fb),
            last_center=_bbox_center(lb),
            exit_velocity=exit_v,
            entry_velocity=entry_v,
            sample_frame=int(peak["frame"]),
        )
        if use_detection_embeddings:
            embedding = peak.get("embedding")
            if isinstance(embedding, list) and embedding:
                profile.embedding = [float(v) for v in embedding]
        profiles[int(track_id)] = profile
    return profiles


def _crop_bbox(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def enrich_embeddings(
    profiles: dict[int, TrackProfile],
    tracked: list[dict],
    video_path: str | None,
    embedder: AppearanceEmbedder,
) -> None:
    if not video_path or not os.path.isfile(video_path):
        return
    by_track = group_by_track(tracked)
    frame_to_bbox: dict[int, list[tuple[int, list[float]]]] = {}
    for tid, prof in profiles.items():
        if prof.embedding:
            continue
        seq = by_track[tid]
        peak = max(seq, key=lambda d: float(d["conf"]))
        frame = int(peak["frame"])
        frame_to_bbox.setdefault(frame, []).append((tid, list(peak["bbox"])))
    if not frame_to_bbox:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    for frame_idx in sorted(frame_to_bbox):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        for tid, bbox in frame_to_bbox[frame_idx]:
            crop = _crop_bbox(frame, bbox)
            if crop is None:
                continue
            vec = embedder.embed(crop)
            profiles[tid].embedding = vec.tolist()
    cap.release()


def run_identity_stitching(
    tracked: list[dict],
    *,
    output_dir: str | None = None,
    video: str | None = None,
    temporal_tau: float = DEFAULT_TEMPORAL_TAU_SEC,
    assignment_score_min: float = DEFAULT_ASSIGNMENT_SCORE_MIN,
    enrich_appearance: bool = True,
    arcface_model: str | None = None,
    # legacy CLI alias
    max_gap_sec: float | None = None,
    appearance_min: float | None = None,
) -> dict[str, Any]:
    if max_gap_sec is not None:
        temporal_tau = max_gap_sec
    _ = appearance_min  # soft scoring only; kept for API compat

    stage_times: dict[str, float] = {}
    embedder = AppearanceEmbedder(model_path=arcface_model)
    started = time.perf_counter()
    profiles = build_track_profiles(
        tracked,
        use_detection_embeddings=enrich_appearance and not embedder.is_arcface,
    )
    stage_times["profile_build_sec"] = round(time.perf_counter() - started, 3)
    started = time.perf_counter()
    if enrich_appearance:
        enrich_embeddings(profiles, tracked, video, embedder)
    stage_times["embedding_enrich_sec"] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    all_edges, merge_decisions, clusters = global_identity_matching(
        profiles,
        embedder,
        temporal_tau=temporal_tau,
        assignment_score_min=assignment_score_min,
    )
    stage_times["graph_matching_sec"] = round(time.perf_counter() - started, 3)
    started = time.perf_counter()
    assigned_edges = [e for e in all_edges if e.get("assigned")]
    cluster_rows = build_cluster_rows(clusters, profiles, assigned_edges, merge_decisions)
    stage_times["cluster_rows_sec"] = round(time.perf_counter() - started, 3)

    parameters = {
        "assignment_method": "hungarian_global_matching",
        "temporal_tau_sec": temporal_tau,
        "temporal_prior": "soft_exponential_decay",
        "assignment_score_min": assignment_score_min,
        "max_successor_candidates": 25,
        "mutual_best_predecessor": True,
        "hsv_fallback_filters": {
            "spatial_min": 0.45,
            "motion_min": 0.30,
            "gap_max_sec": HSV_ASSIGNMENT_GAP_MAX_SEC,
        },
        "appearance_method": embedder.method,
        "appearance_enrich_enabled": enrich_appearance,
        "feature_weights": {
            "appearance": 0.55,
            "motion": 0.15,
            "temporal_soft": 0.15,
            "spatial": 0.10,
            "iou_weak": 0.05,
        },
        **embedder.info(),
    }

    graph_doc = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layer": STITCHING_LAYER,
        "video": video,
        "parameters": parameters,
        "track_count": len(profiles),
        "edge_count": len(all_edges),
        "assigned_edge_count": len(assigned_edges),
        "nodes": [profiles[t].to_dict() for t in sorted(profiles)],
        "edges": all_edges,
        "assigned_edges": assigned_edges,
        "merge_decisions": merge_decisions,
    }

    clusters_doc = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layer": STITCHING_LAYER,
        "video": video,
        "parameters": parameters,
        "track_count": len(profiles),
        "identity_cluster_count": len(cluster_rows),
        "assigned_edge_count": len(assigned_edges),
        "target_event_range": list(TARGET_EVENT_RANGE),
        "clusters": cluster_rows,
        "merge_decisions_summary": {
            "total_candidates_evaluated": len(merge_decisions),
            "accepted_merges": len(assigned_edges),
            "rejected_below_threshold": sum(
                1 for d in merge_decisions if not d.get("accepted")
            ),
        },
    }

    stats = {
        "track_count": len(profiles),
        "identity_cluster_count": len(cluster_rows),
        "linked_edge_count": len(assigned_edges),
        "edge_count": len(all_edges),
        "appearance_method": embedder.method,
        "appearance_enrich_enabled": enrich_appearance,
        "profile_embeddings": sum(1 for p in profiles.values() if p.embedding),
        "stage_times": stage_times,
        "clusters": cluster_rows,
        "cluster_map": {row["identity_id"]: row["track_ids"] for row in cluster_rows},
        "parameters": parameters,
        "merge_decisions": merge_decisions,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        graph_path = os.path.join(output_dir, IDENTITY_GRAPH_NAME)
        id_path = os.path.join(output_dir, IDENTITY_CLUSTERS_NAME)
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph_doc, f, ensure_ascii=False, indent=2)
        with open(id_path, "w", encoding="utf-8") as f:
            json.dump(clusters_doc, f, ensure_ascii=False, indent=2)
        stats["identity_graph_path"] = graph_path
        stats["identity_clusters_path"] = id_path
        stats["track_graph_path"] = graph_path

    return stats


def load_identity_clusters(output_dir: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = os.path.join(os.path.abspath(output_dir), IDENTITY_CLUSTERS_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc.get("clusters") or [], doc


def main() -> int:
    p = argparse.ArgumentParser(description="Identity stitching (pipeline core stage)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--video", default=None)
    p.add_argument("--stitch-gap", type=float, default=DEFAULT_TEMPORAL_TAU_SEC,
                   help="Temporal soft prior tau (seconds, exponential decay)")
    p.add_argument("--assignment-min", type=float, default=DEFAULT_ASSIGNMENT_SCORE_MIN)
    p.add_argument("--appearance-min", type=float, default=None, help="Deprecated; unused in graph matching")
    p.add_argument("--fast-identity", action="store_true", help="Skip full-video appearance embedding enrichment.")
    p.add_argument("--arcface-model", default=None)
    args = p.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    tracked_path = os.path.join(out_dir, "tracked_detections.json")
    if not os.path.isfile(tracked_path):
        print(f"[error] Missing {tracked_path}", file=sys.stderr)
        return 1
    with open(tracked_path, encoding="utf-8") as f:
        tracked = json.load(f)
    summary_path = os.path.join(out_dir, "detection_summary.json")
    video = args.video
    if not video and os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            video = json.load(f).get("video")

    stats = run_identity_stitching(
        tracked,
        output_dir=out_dir,
        video=video,
        temporal_tau=args.stitch_gap,
        assignment_score_min=args.assignment_min,
        enrich_appearance=not args.fast_identity,
        arcface_model=args.arcface_model,
    )
    print(f"[graph] {stats.get('identity_graph_path')}")
    print(f"[clusters] {stats.get('identity_clusters_path')}")
    print(
        f"  {stats['track_count']} tracks -> {stats['identity_cluster_count']} identities"
        f" | assigned={stats['linked_edge_count']}/{stats.get('edge_count', '?')} edges"
        f" | appearance={stats['appearance_method']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
