"""FROZEN v1 greedy-chain identity stitching — benchmark baseline only, do not tune."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from appearance_embedder import AppearanceEmbedder
from event_builder import bbox_iou
from event_merge import center_continuous, DEFAULT_CENTER_CONTINUITY_RATIO
from identity_stitching import TrackProfile, build_track_profiles, enrich_embeddings

V1_LABEL = "identity_stitching_v1_greedy_frozen"
V1_STITCH_TEMPORAL_MAX_SEC = 60.0
V1_APPEARANCE_MIN = 0.72
V1_STITCH_SCORE_MIN = 0.68
V1_MOTION_MAX_CENTER_ERR_RATIO = 2.5


def _spatial_score(prev: TrackProfile, nxt: TrackProfile) -> tuple[float, dict[str, Any]]:
    iou = bbox_iou(prev.last_bbox, nxt.first_bbox)
    center_ok = center_continuous(
        prev.last_bbox, nxt.first_bbox, ratio=DEFAULT_CENTER_CONTINUITY_RATIO,
    )
    dist = math.hypot(prev.last_center[0] - nxt.first_center[0], prev.last_center[1] - nxt.first_center[1])
    ref = max(
        math.hypot(prev.last_bbox[2] - prev.last_bbox[0], prev.last_bbox[3] - prev.last_bbox[1]),
        math.hypot(nxt.first_bbox[2] - nxt.first_bbox[0], nxt.first_bbox[3] - nxt.first_bbox[1]),
    )
    norm_dist = dist / ref if ref > 0 else 999.0
    spatial = 1.0 if iou >= 0.05 or center_ok else max(0.0, 1.0 - norm_dist / 3.0)
    return spatial, {
        "iou": round(iou, 4),
        "center_continuous": center_ok,
        "center_distance_ratio": round(norm_dist, 3),
        "spatial_score": round(spatial, 4),
    }


def _motion_score(prev: TrackProfile, nxt: TrackProfile, *, gap_sec: float) -> tuple[float, dict[str, Any]]:
    v_out, v_in = prev.exit_velocity, nxt.entry_velocity
    speed_out = math.hypot(*v_out)
    speed_in = math.hypot(*v_in)
    vel_score = 0.5
    if speed_out > 1e-3 and speed_in > 1e-3:
        vel_score = max(0.0, min(1.0, (v_out[0] * v_in[0] + v_out[1] * v_in[1]) / (speed_out * speed_in)))
    pred_cx = prev.last_center[0] + v_out[0] * gap_sec
    pred_cy = prev.last_center[1] + v_out[1] * gap_sec
    err = math.hypot(pred_cx - nxt.first_center[0], pred_cy - nxt.first_center[1])
    ref = max(
        math.hypot(prev.last_bbox[2] - prev.last_bbox[0], prev.last_bbox[3] - prev.last_bbox[1]),
        math.hypot(nxt.first_bbox[2] - nxt.first_bbox[0], nxt.first_bbox[3] - nxt.first_bbox[1]),
    )
    pos_score = max(0.0, 1.0 - err / (ref * V1_MOTION_MAX_CENTER_ERR_RATIO)) if ref > 0 else 0.0
    combined = 0.55 * pos_score + 0.45 * vel_score
    ok = pos_score >= 0.25 or vel_score >= 0.6
    return combined, {
        "motion_score": round(combined, 4),
        "motion_ok": ok,
        "position_prediction_score": round(pos_score, 4),
        "velocity_consistency_score": round(vel_score, 4),
    }


def _score_link(prev: TrackProfile, nxt: TrackProfile, embedder: AppearanceEmbedder) -> tuple[float, dict[str, Any]]:
    if nxt.start_time < prev.end_time - 0.05:
        return 0.0, {"linked": False, "reason": "temporal_overlap"}
    gap = float(nxt.start_time) - float(prev.end_time)
    if gap > V1_STITCH_TEMPORAL_MAX_SEC:
        return 0.0, {"linked": False, "reason": "gap_exceeded", "temporal_gap_sec": round(gap, 3)}

    temporal_score = 1.0 - (gap / V1_STITCH_TEMPORAL_MAX_SEC)
    spatial, s_meta = _spatial_score(prev, nxt)
    motion, m_meta = _motion_score(prev, nxt, gap_sec=gap)

    if not prev.embedding or not nxt.embedding:
        return 0.0, {"linked": False, "reason": "appearance_unavailable"}

    appearance = embedder.similarity(
        np.array(prev.embedding, dtype=np.float32),
        np.array(nxt.embedding, dtype=np.float32),
    )
    spatial_ok = bool(s_meta.get("center_continuous")) or float(s_meta.get("iou") or 0) >= 0.08
    if not spatial_ok or not m_meta.get("motion_ok") or appearance < V1_APPEARANCE_MIN:
        return 0.0, {
            "linked": False,
            "appearance_score": round(appearance, 4),
            "temporal_gap_sec": round(gap, 3),
            **s_meta, **m_meta,
        }

    combined = 0.25 * temporal_score + 0.25 * spatial + 0.40 * appearance + 0.10 * motion
    linked = combined >= V1_STITCH_SCORE_MIN
    return combined, {
        "linked": linked,
        "combined_score": round(combined, 4),
        "appearance_score": round(appearance, 4),
        "temporal_gap_sec": round(gap, 3),
        "temporal_score": round(temporal_score, 4),
        **s_meta, **m_meta,
    }


class _UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {i: i for i in items}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for item in self.parent:
            out.setdefault(self.find(item), []).append(item)
        return out


def _split_overlapping(clusters: dict[int, list[int]], profiles: dict[int, TrackProfile]) -> dict[int, list[int]]:
    cleaned: dict[int, list[int]] = {}
    next_root = 10_000
    for track_ids in clusters.values():
        members = sorted(track_ids, key=lambda tid: profiles[tid].start_time)
        groups: list[list[int]] = []
        for tid in members:
            prof = profiles[tid]
            placed = False
            for group in groups:
                group_end = max(profiles[g].end_time for g in group)
                if prof.start_time >= group_end - 0.05:
                    group.append(tid)
                    placed = True
                    break
            if not placed:
                groups.append([tid])
        for group in groups:
            cleaned[next_root] = group
            next_root += 1
    return cleaned


def run_stitching_v1_frozen(
    tracked: list[dict],
    *,
    video: str | None,
    embedder: AppearanceEmbedder | None = None,
) -> dict[str, Any]:
    embedder = embedder or AppearanceEmbedder()
    profiles = build_track_profiles(tracked)
    enrich_embeddings(profiles, tracked, video, embedder)

    track_ids = sorted(profiles)
    edges: list[dict[str, Any]] = []
    uf = _UnionFind(track_ids)
    ordered = sorted(profiles.values(), key=lambda p: (p.start_time, p.track_id))
    best_successor: dict[int, tuple[float, int]] = {}

    for i, prev in enumerate(ordered):
        for nxt in ordered[i + 1:]:
            if nxt.start_time < prev.end_time - 0.05:
                continue
            if nxt.start_time - prev.end_time > V1_STITCH_TEMPORAL_MAX_SEC:
                break
            score, meta = _score_link(prev, nxt, embedder)
            meta_copy = {k: v for k, v in meta.items() if k != "linked"}
            edges.append({
                "from_track_id": prev.track_id,
                "to_track_id": nxt.track_id,
                "combined_score": meta.get("combined_score", round(score, 4)),
                "linked": False,
                **meta_copy,
            })
            if not meta.get("linked"):
                continue
            cur = best_successor.get(prev.track_id)
            if cur is None or meta["combined_score"] > cur[0]:
                best_successor[prev.track_id] = (meta["combined_score"], nxt.track_id)

    for prev_id, (_score, next_id) in best_successor.items():
        for edge in edges:
            if edge["from_track_id"] == prev_id and edge["to_track_id"] == next_id:
                edge["linked"] = True
                edge["stitch_role"] = "greedy_successor"
                break
        uf.union(prev_id, next_id)

    clusters = _split_overlapping(uf.clusters(), profiles)
    cluster_rows: list[dict[str, Any]] = []
    linked = {(e["from_track_id"], e["to_track_id"]): e for e in edges if e.get("linked")}
    for idx, track_ids in enumerate(
        sorted(clusters.values(), key=lambda g: min(profiles[t].start_time for t in g)),
        start=1,
    ):
        ordered_ids = sorted(track_ids, key=lambda tid: profiles[tid].start_time)
        chain = []
        for a, b in zip(ordered_ids, ordered_ids[1:]):
            if (a, b) in linked:
                chain.append(linked[(a, b)])
        cluster_rows.append({
            "identity_id": f"id_{idx:04d}",
            "track_ids": ordered_ids,
            "track_count": len(ordered_ids),
            "start_time": round(min(profiles[t].start_time for t in ordered_ids), 3),
            "end_time": round(max(profiles[t].end_time for t in ordered_ids), 3),
            "stitch_chain": chain,
        })

    assigned = [e for e in edges if e.get("linked")]
    return {
        "layer": V1_LABEL,
        "clusters": cluster_rows,
        "edges": edges,
        "assigned_edges": assigned,
        "linked_edge_count": len(assigned),
        "appearance_method": embedder.method,
    }
