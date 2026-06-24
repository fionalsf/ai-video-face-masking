"""Global graph-based identity reconstruction (Hungarian matching on track segments)."""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from appearance_embedder import AppearanceEmbedder
from event_builder import bbox_iou

# Combined score weights (appearance primary; IoU weak)
W_APPEARANCE = 0.55
W_MOTION = 0.15
W_TEMPORAL = 0.15
W_SPATIAL = 0.10
W_IOU = 0.05

DEFAULT_TEMPORAL_TAU_SEC = 60.0
DEFAULT_ASSIGNMENT_SCORE_MIN = 0.72
MAX_EDGE_GAP_COMPUTE_SEC = 600.0
MAX_SUCCESSOR_CANDIDATES = 25
DEFAULT_HSV_APPEARANCE_ASSIGN_MIN = 0.88
DEFAULT_ARCFACE_APPEARANCE_ASSIGN_MIN = 0.45
BIG_COST = 1e6


class TrackSegment(Protocol):
    track_id: int
    start_time: float
    end_time: float
    first_bbox: list[float]
    last_bbox: list[float]
    first_center: tuple[float, float]
    last_center: tuple[float, float]
    exit_velocity: tuple[float, float]
    entry_velocity: tuple[float, float]
    embedding: list[float]


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_diagonal(bbox: list[float]) -> float:
    return math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])


def temporal_soft_score(gap_sec: float, *, tau: float = DEFAULT_TEMPORAL_TAU_SEC) -> float:
    """Soft prior; no hard cutoff (tau controls decay rate)."""
    if gap_sec < 0:
        return 0.0
    return math.exp(-gap_sec / max(tau, 1.0))


def spatial_continuity_score(prev: TrackSegment, nxt: TrackSegment) -> tuple[float, dict[str, Any]]:
    dist = math.hypot(prev.last_center[0] - nxt.first_center[0], prev.last_center[1] - nxt.first_center[1])
    ref = max(_bbox_diagonal(prev.last_bbox), _bbox_diagonal(nxt.first_bbox), 1.0)
    d_prev = _bbox_diagonal(prev.last_bbox)
    d_next = _bbox_diagonal(nxt.first_bbox)
    scale_ratio = min(d_prev, d_next) / max(d_prev, d_next) if max(d_prev, d_next) > 0 else 0.0
    center_score = max(0.0, 1.0 - dist / (ref * 3.0))
    spatial = 0.70 * center_score + 0.30 * scale_ratio
    return spatial, {
        "center_distance_ratio": round(dist / ref, 4),
        "scale_ratio": round(scale_ratio, 4),
        "center_score": round(center_score, 4),
        "spatial_score": round(spatial, 4),
    }


def motion_consistency_score(
    prev: TrackSegment,
    nxt: TrackSegment,
    *,
    gap_sec: float,
) -> tuple[float, dict[str, Any]]:
    v_out = prev.exit_velocity
    v_in = nxt.entry_velocity
    speed_out = math.hypot(*v_out)
    speed_in = math.hypot(*v_in)

    vel_score = 0.5
    if speed_out > 1e-3 and speed_in > 1e-3:
        dot = v_out[0] * v_in[0] + v_out[1] * v_in[1]
        vel_score = max(0.0, min(1.0, dot / (speed_out * speed_in)))

    pred_cx = prev.last_center[0] + v_out[0] * gap_sec
    pred_cy = prev.last_center[1] + v_out[1] * gap_sec
    err = math.hypot(pred_cx - nxt.first_center[0], pred_cy - nxt.first_center[1])
    ref = max(_bbox_diagonal(prev.last_bbox), _bbox_diagonal(nxt.first_bbox), 1.0)
    pos_score = max(0.0, 1.0 - err / (ref * 2.5))

    combined = 0.55 * pos_score + 0.45 * vel_score
    return combined, {
        "motion_score": round(combined, 4),
        "position_prediction_score": round(pos_score, 4),
        "velocity_consistency_score": round(vel_score, 4),
        "predicted_center_error_ratio": round(err / ref, 4),
    }


def iou_weak_score(prev: TrackSegment, nxt: TrackSegment) -> tuple[float, dict[str, Any]]:
    iou = bbox_iou(prev.last_bbox, nxt.first_bbox)
    weak = min(1.0, iou / 0.30)
    return weak, {"iou": round(iou, 4), "iou_weak_score": round(weak, 4)}


def compute_edge_features(
    prev: TrackSegment,
    nxt: TrackSegment,
    embedder: AppearanceEmbedder,
    *,
    temporal_tau: float = DEFAULT_TEMPORAL_TAU_SEC,
) -> dict[str, Any] | None:
    gap = float(nxt.start_time) - float(prev.end_time)
    if gap < -0.05:
        return None
    if gap > MAX_EDGE_GAP_COMPUTE_SEC:
        return None

    if not prev.embedding or not nxt.embedding:
        return None

    appearance = embedder.similarity(
        np.array(prev.embedding, dtype=np.float32),
        np.array(nxt.embedding, dtype=np.float32),
    )
    temporal = temporal_soft_score(gap, tau=temporal_tau)
    spatial, spatial_meta = spatial_continuity_score(prev, nxt)
    motion, motion_meta = motion_consistency_score(prev, nxt, gap_sec=max(gap, 0.0))
    iou_w, iou_meta = iou_weak_score(prev, nxt)

    combined = (
        W_APPEARANCE * appearance
        + W_MOTION * motion
        + W_TEMPORAL * temporal
        + W_SPATIAL * spatial
        + W_IOU * iou_w
    )

    contributions = [
        ("appearance", W_APPEARANCE * appearance, round(appearance, 4)),
        ("motion", W_MOTION * motion, round(motion, 4)),
        ("temporal_soft", W_TEMPORAL * temporal, round(temporal, 4)),
        ("spatial", W_SPATIAL * spatial, round(spatial, 4)),
        ("iou_weak", W_IOU * iou_w, round(iou_w, 4)),
    ]
    contributions.sort(key=lambda x: x[1], reverse=True)
    top_k = [
        {"feature": name, "weighted": round(w, 4), "raw": raw}
        for name, w, raw in contributions
    ]

    return {
        "from_track_id": prev.track_id,
        "to_track_id": nxt.track_id,
        "temporal_gap_sec": round(gap, 3),
        "appearance_score": round(appearance, 4),
        "appearance_method": embedder.method,
        "temporal_soft_score": round(temporal, 4),
        "combined_score": round(combined, 4),
        "top_features": top_k[:3],
        "feature_scores": {
            "appearance": round(appearance, 4),
            "motion": motion_meta["motion_score"],
            "temporal_soft": round(temporal, 4),
            "spatial": spatial_meta["spatial_score"],
            "iou_weak": iou_meta["iou_weak_score"],
        },
        "feature_weights": {
            "appearance": W_APPEARANCE,
            "motion": W_MOTION,
            "temporal_soft": W_TEMPORAL,
            "spatial": W_SPATIAL,
            "iou_weak": W_IOU,
        },
        **spatial_meta,
        **motion_meta,
        **iou_meta,
    }


def _hungarian_assignment(cost: np.ndarray) -> np.ndarray:
    try:
        from lap import lapjv
        _, assign, _ = lapjv(cost, extend_cost=True)
        return assign
    except ImportError:
        pass
    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(cost)
        assign = np.full(cost.shape[0], -1, dtype=np.int32)
        for r, c in zip(row, col):
            assign[r] = c
        return assign
    except ImportError:
        raise RuntimeError("Need lap or scipy for global identity matching")


def _is_mutual_best_predecessor(
    i: int,
    j: int,
    weight: np.ndarray,
    n: int,
) -> bool:
    """Row j's highest-weight predecessor should be row i."""
    best_i = -1
    best_w = -1.0
    for k in range(n):
        if k == j:
            continue
        w = float(weight[k, j])
        if w > best_w:
            best_w = w
            best_i = k
    return best_i == i


def global_identity_matching(
    profiles: dict[int, Any],
    embedder: AppearanceEmbedder,
    *,
    temporal_tau: float = DEFAULT_TEMPORAL_TAU_SEC,
    assignment_score_min: float = DEFAULT_ASSIGNMENT_SCORE_MIN,
    appearance_assign_min: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[int]]]:
    """
    Build temporal-local edge graph, run Hungarian max-weight matching,
    partition into identity clusters with appearance post-filter.
    """
    if appearance_assign_min is None:
        appearance_assign_min = (
            DEFAULT_ARCFACE_APPEARANCE_ASSIGN_MIN if embedder.is_arcface
            else DEFAULT_HSV_APPEARANCE_ASSIGN_MIN
        )

    track_ids = sorted(profiles)
    n = len(track_ids)
    all_edges: list[dict[str, Any]] = []
    cost = np.full((n, n), BIG_COST, dtype=np.float64)
    weight = np.zeros((n, n), dtype=np.float64)

    ordered = [profiles[tid] for tid in track_ids]
    for i, prev in enumerate(ordered):
        candidates: list[tuple[int, dict[str, Any]]] = []
        for j, nxt in enumerate(ordered):
            if i == j:
                continue
            if nxt.start_time < prev.end_time - 0.05:
                continue
            if nxt.start_time - prev.end_time > MAX_EDGE_GAP_COMPUTE_SEC:
                break
            feats = compute_edge_features(prev, nxt, embedder, temporal_tau=temporal_tau)
            if feats is None:
                continue
            candidates.append((j, feats))
        candidates.sort(key=lambda x: x[1]["combined_score"], reverse=True)
        for j, feats in candidates[:MAX_SUCCESSOR_CANDIDATES]:
            w = float(feats["combined_score"])
            weight[i, j] = w
            cost[i, j] = BIG_COST - w * 1e4
            all_edges.append({**feats, "assigned": False})

    assign = _hungarian_assignment(cost)
    assigned_edges: list[dict[str, Any]] = []
    merge_decisions: list[dict[str, Any]] = []

    for i in range(n):
        j = int(assign[i])
        if j < 0 or j >= n or i == j:
            continue
        w = float(weight[i, j])
        prev_tid, nxt_tid = track_ids[i], track_ids[j]
        edge_record = next(
            (e for e in all_edges if e["from_track_id"] == prev_tid and e["to_track_id"] == nxt_tid),
            None,
        )
        if edge_record is None:
            continue
        app = float(edge_record.get("appearance_score") or 0)
        spatial = float(edge_record.get("spatial_score") or 0)
        motion = float(edge_record.get("motion_score") or 0)
        gap = float(edge_record.get("temporal_gap_sec") or 999)

        if embedder.is_arcface:
            accepted = (
                w >= assignment_score_min
                and app >= appearance_assign_min
                and _is_mutual_best_predecessor(i, j, weight, n)
            )
        else:
            # HSV lacks discrimination; lean on spatial/motion/temporal locality
            accepted = (
                w >= assignment_score_min
                and spatial >= 0.45
                and motion >= 0.30
                and gap <= 90.0
                and _is_mutual_best_predecessor(i, j, weight, n)
            )
        edge_record["assigned"] = accepted
        edge_record["assignment_method"] = "hungarian_global_matching"
        decision = {
            "from_track_id": prev_tid,
            "to_track_id": nxt_tid,
            "accepted": accepted,
            "assignment_method": "hungarian_global_matching",
            "combined_score": round(w, 4),
            "assignment_score_min": assignment_score_min,
            "appearance_assign_min": appearance_assign_min,
            "top_features": edge_record.get("top_features", []),
            "feature_scores": edge_record.get("feature_scores", {}),
            "temporal_gap_sec": edge_record.get("temporal_gap_sec"),
            "reason": (
                "global_optimal_match" if accepted
                else "below_combined_threshold" if w < assignment_score_min
                else "not_mutual_best_predecessor" if not _is_mutual_best_predecessor(i, j, weight, n)
                else "hsv_spatial_motion_gap_filter" if not embedder.is_arcface
                else "below_appearance_threshold"
            ),
        }
        merge_decisions.append(decision)
        if accepted:
            assigned_edges.append(edge_record)

    # Union-find on assigned edges
    parent = {tid: tid for tid in track_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in assigned_edges:
        union(e["from_track_id"], e["to_track_id"])

    raw_clusters: dict[int, list[int]] = {}
    for tid in track_ids:
        raw_clusters.setdefault(find(tid), []).append(tid)

    clusters = _split_overlapping_clusters(raw_clusters, profiles)
    return all_edges, merge_decisions, clusters


def _split_overlapping_clusters(
    clusters: dict[int, list[int]],
    profiles: dict[int, Any],
) -> dict[int, list[int]]:
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


def order_tracks_in_cluster(
    track_ids: list[int],
    assigned_edges: list[dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Topological order within cluster using assigned successor edges."""
    tid_set = set(track_ids)
    succ: dict[int, int] = {}
    edge_map: dict[tuple[int, int], dict] = {}
    for e in assigned_edges:
        a, b = e["from_track_id"], e["to_track_id"]
        if a in tid_set and b in tid_set:
            succ[a] = b
            edge_map[(a, b)] = e

    preds = set(succ.values())
    heads = [t for t in track_ids if t not in preds]
    if not heads:
        heads = [min(track_ids, key=lambda t: t)]

    ordered: list[int] = []
    chain_edges: list[dict] = []
    visited: set[int] = set()
    for head in sorted(heads):
        cur = head
        while cur is not None and cur not in visited and cur in tid_set:
            visited.add(cur)
            ordered.append(cur)
            nxt = succ.get(cur)
            if nxt is not None and (cur, nxt) in edge_map:
                chain_edges.append(edge_map[(cur, nxt)])
            cur = nxt if nxt in tid_set and nxt not in visited else None

    for tid in sorted(track_ids):
        if tid not in visited:
            ordered.append(tid)
    return ordered, chain_edges


def build_cluster_rows(
    clusters: dict[int, list[int]],
    profiles: dict[int, Any],
    assigned_edges: list[dict[str, Any]],
    merge_decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted = [d for d in merge_decisions if d.get("accepted")]
    rows: list[dict[str, Any]] = []
    for idx, track_ids in enumerate(
        sorted(clusters.values(), key=lambda g: min(profiles[t].start_time for t in g)),
        start=1,
    ):
        ordered, chain = order_tracks_in_cluster(track_ids, assigned_edges)
        starts = [profiles[t].start_time for t in ordered]
        ends = [profiles[t].end_time for t in ordered]
        cluster_decisions = [
            d for d in accepted
            if d["from_track_id"] in track_ids and d["to_track_id"] in track_ids
        ]
        rows.append({
            "identity_id": f"id_{idx:04d}",
            "track_ids": ordered,
            "track_count": len(ordered),
            "start_time": round(min(starts), 3),
            "end_time": round(max(ends), 3),
            "duration_sec": round(max(ends) - min(starts), 3),
            "stitch_chain": chain,
            "merge_decisions": cluster_decisions,
            "merge_count": len(cluster_decisions),
        })
    return rows
