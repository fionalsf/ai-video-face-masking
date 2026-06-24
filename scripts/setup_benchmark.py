"""Create/fix benchmark framework files (UTF-8). Run from repo root: python scripts/setup_benchmark.py"""

from __future__ import annotations

import json
import os
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_maybe_utf16(path: str) -> str:
    raw = open(path, "rb").read()
    if b"\x00" in raw:
        for enc in ("utf-16-le", "utf-16", "utf-16-be"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
    return raw.decode("utf-8")


def write_utf8(rel_path: str, content: str) -> None:
    path = os.path.join(ROOT, rel_path.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content.rstrip() + "\n")
    print("wrote", rel_path)


def fix_existing(rel_path: str) -> None:
    path = os.path.join(ROOT, rel_path.replace("/", os.sep))
    if not os.path.isfile(path):
        return
    text = read_maybe_utf16(path)
    write_utf8(rel_path, text)


V1_FROZEN = textwrap.dedent(
    '''
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
    '''
).strip("\n")


COMPARE_STITCHING = textwrap.dedent(
    '''
    """Compare v1_greedy vs v2_graph on the same video + ground truth."""

    from __future__ import annotations

    import argparse
    import json
    import os
    import sys
    import time

    from benchmark.ground_truth import (
        GT_SCHEMA_VERSION,
        clips_from_ground_truth,
        gt_events_for_clip,
        load_clips,
        load_ground_truth,
        save_json,
    )
    from benchmark.metrics import aggregate_clip_metrics, evaluate_variant
    from benchmark.report import render_report
    from benchmark.run_variant import run_variant_pipeline


    def _resolve_video(output_dir: str, video: str | None) -> str:
        if video and os.path.isfile(video):
            return os.path.abspath(video)
        summary = os.path.join(output_dir, "detection_summary.json")
        if os.path.isfile(summary):
            with open(summary, encoding="utf-8") as f:
                v = json.load(f).get("video")
            if v and os.path.isfile(v):
                return os.path.abspath(v)
        raise FileNotFoundError("Video path required (--video) or detection_summary.json with valid video")


    def _evaluate_variant_clips(
        variant_name: str,
        pred_doc: dict,
        gt: dict,
        clips: list[dict],
    ) -> dict:
        per_clip = []
        for clip in clips:
            gt_events = gt_events_for_clip(gt, clip)
            row = evaluate_variant(
                gt_events,
                pred_doc,
                clip_start=float(clip["start_time"]),
                clip_end=float(clip["end_time"]),
                variant_name=variant_name,
            )
            row["clip_id"] = clip.get("clip_id")
            per_clip.append(row)
        return {
            "variant": variant_name,
            "per_clip": per_clip,
            "aggregate": aggregate_clip_metrics(per_clip),
        }


    def run_comparison(
        *,
        output_dir: str,
        gt_path: str,
        video: str | None = None,
        clips_path: str | None = None,
        benchmark_out: str | None = None,
        write_artifacts: bool = True,
    ) -> dict:
        output_dir = os.path.abspath(output_dir)
        gt = load_ground_truth(gt_path)
        video_path = _resolve_video(output_dir, video or gt.get("video"))
        clips = load_clips(clips_path) if clips_path else clips_from_ground_truth(gt)

        benchmark_out = benchmark_out or os.path.join(output_dir, "benchmark")
        os.makedirs(benchmark_out, exist_ok=True)

        results: dict = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video": video_path,
            "output_dir": output_dir,
            "ground_truth": os.path.abspath(gt_path),
            "clips_path": os.path.abspath(clips_path) if clips_path else None,
            "objective": "minimize masking error + review cost",
            "clip_count": len(clips),
            "frozen_parameters": True,
            "variants": {},
        }

        for variant in ("v1_greedy", "v2_graph"):
            pred = run_variant_pipeline(
                variant,
                output_dir=output_dir,
                video=video_path,
                write_artifacts=write_artifacts,
            )
            results["variants"][variant] = _evaluate_variant_clips(variant, pred, gt, clips)
            results["variants"][variant]["pipeline"] = {
                "identity_cluster_count": pred.get("identity_cluster_count"),
                "behavior_event_count": pred.get("behavior_event_count"),
                "stitching_layer": pred.get("stitching_layer"),
            }

        v1_score = results["variants"]["v1_greedy"]["aggregate"].get("mean_objective_score", 0)
        v2_score = results["variants"]["v2_graph"]["aggregate"].get("mean_objective_score", 0)
        results["recommended_variant"] = "v2_graph" if v2_score <= v1_score else "v1_greedy"

        a1 = results["variants"]["v1_greedy"]["aggregate"]
        a2 = results["variants"]["v2_graph"]["aggregate"]
        results["delta"] = {
            k: round(a2.get(k, 0) - a1.get(k, 0), 3) if isinstance(a1.get(k), float) else a2.get(k, 0) - a1.get(k, 0)
            for k in a1
        }

        prov = gt.get("provenance") or {}
        if prov.get("type") == "silver_bootstrap":
            results["notes"] = (
                "- Ground truth is silver bootstrap (not human verified); replace with annotated GT before decisions.\n"
                "- Stitching thresholds / temporal tau / IoU weights / graph strategy are frozen during benchmark."
            )
        else:
            results["notes"] = "- Stitching parameters frozen during benchmark period."

        metrics_path = os.path.join(benchmark_out, "metrics.json")
        report_path = os.path.join(benchmark_out, "report.md")
        save_json(metrics_path, results)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(render_report(results))
        results["metrics_path"] = metrics_path
        results["report_path"] = report_path
        return results


    def main() -> int:
        p = argparse.ArgumentParser(description="Benchmark v1_greedy vs v2_graph stitching")
        p.add_argument("--output-dir", required=True, help="Detection output dir with tracked_detections.json")
        p.add_argument("--gt", required=True, help="Ground truth JSON path")
        p.add_argument("--video", default=None)
        p.add_argument("--clips", default=None, help="Clips manifest JSON (optional)")
        p.add_argument("--benchmark-out", default=None)
        p.add_argument("--no-artifacts", action="store_true")
        args = p.parse_args()

        try:
            res = run_comparison(
                output_dir=args.output_dir,
                gt_path=args.gt,
                video=args.video,
                clips_path=args.clips,
                benchmark_out=args.benchmark_out,
                write_artifacts=not args.no_artifacts,
            )
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1

        print(f"[metrics] {res['metrics_path']}")
        print(f"[report] {res['report_path']}")
        print(f"  recommended={res['recommended_variant']} clips={res['clip_count']}")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    '''
).strip("\n")


GENERATE_CLIPS = textwrap.dedent(
    '''
    """Generate standardized evaluation clip manifest (20-50 segments)."""

    from __future__ import annotations

    import argparse
    import json
    import os
    import sys

    from benchmark.ground_truth import CLIPS_SCHEMA_VERSION, generate_uniform_clips, save_json


    def main() -> int:
        p = argparse.ArgumentParser(description="Generate benchmark clip manifest")
        p.add_argument("--video-id", required=True, help="Video stem e.g. DJI_20260511100755_0008_D")
        p.add_argument("--duration-sec", type=float, required=True)
        p.add_argument("--target-count", type=int, default=30)
        p.add_argument("--out", default=None)
        args = p.parse_args()

        clips = generate_uniform_clips(args.duration_sec, target_count=args.target_count)
        doc = {
            "schema_version": CLIPS_SCHEMA_VERSION,
            "video_id": args.video_id,
            "duration_sec": args.duration_sec,
            "clip_count": len(clips),
            "target_count": args.target_count,
            "clips": clips,
        }
        out = args.out or os.path.join("benchmark", "clips", f"{args.video_id}_clips.json")
        save_json(out, doc)
        print(f"[clips] {os.path.abspath(out)} ({len(clips)} segments)")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    '''
).strip("\n")


BOOTSTRAP_GT = textwrap.dedent(
    '''
    """Bootstrap silver ground truth from existing events JSON (for pipeline testing only)."""

    from __future__ import annotations

    import argparse
    import json
    import os
    import sys

    from benchmark.ground_truth import GT_SCHEMA_VERSION, save_json


    def events_to_gt(events_doc: dict, *, video: str, source: str) -> dict:
        events = events_doc.get("events") or events_doc
        if isinstance(events, dict):
            events = []
        gt_events = []
        for i, ev in enumerate(events, start=1):
            gt_events.append({
                "gt_event_id": str(ev.get("gt_event_id") or ev.get("event_id") or ev.get("behavior_event_id") or f"gt_{i:04d}"),
                "start_time": float(ev["start_time"]),
                "end_time": float(ev["end_time"]),
                "should_mask": ev.get("should_mask", True),
                "label": ev.get("label", "face_presence"),
            })
        duration = float(events_doc.get("duration_sec") or 0)
        if duration <= 0 and gt_events:
            duration = max(float(e["end_time"]) for e in gt_events)
        return {
            "schema_version": GT_SCHEMA_VERSION,
            "video": video,
            "duration_sec": duration,
            "provenance": {
                "type": "silver_bootstrap",
                "source_file": os.path.abspath(source),
                "warning": "Not human verified — for benchmark plumbing only",
            },
            "events": gt_events,
            "clips": [],
        }


    def main() -> int:
        p = argparse.ArgumentParser(description="Bootstrap silver GT from events JSON")
        p.add_argument("--events", required=True)
        p.add_argument("--video", required=True)
        p.add_argument("--out", required=True)
        args = p.parse_args()

        with open(args.events, encoding="utf-8") as f:
            doc = json.load(f)
        gt = events_to_gt(doc, video=args.video, source=args.events)
        save_json(args.out, gt)
        print(f"[gt] {os.path.abspath(args.out)} events={len(gt['events'])}")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    '''
).strip("\n")


def main() -> None:
    for rel in (
        "benchmark/metrics.py",
        "benchmark/ground_truth.py",
        "benchmark/report.py",
        "benchmark/run_variant.py",
        "benchmark/__init__.py",
        "benchmark/pipelines/__init__.py",
    ):
        fix_existing(rel)

    write_utf8("benchmark/pipelines/stitching_v1_frozen.py", V1_FROZEN)
    write_utf8(
        "benchmark/pipelines/stitching_v2_frozen.py",
        textwrap.dedent(
            '''
            """FROZEN v2 graph identity stitching — production defaults only."""

            from __future__ import annotations

            from typing import Any

            from identity_stitching import run_identity_stitching

            V2_LABEL = "identity_stitching_v2_graph_frozen"


            def run_stitching_v2_frozen(
                tracked: list[dict],
                *,
                video: str | None,
                output_dir: str | None = None,
            ) -> dict[str, Any]:
                stats = run_identity_stitching(tracked, output_dir=output_dir, video=video)
                stats["layer"] = V2_LABEL
                return stats
            '''
        ).strip("\n"),
    )

    # Fix linked_edge_count in run_variant
    rv_path = os.path.join(ROOT, "benchmark", "run_variant.py")
    rv = read_maybe_utf16(rv_path)
    rv = rv.replace(
        '"linked_edge_count": len(stitch.get("assigned_edges") or []),',
        '"linked_edge_count": stitch.get("linked_edge_count") or len(stitch.get("assigned_edges") or []),',
    )
    write_utf8("benchmark/run_variant.py", rv)

    write_utf8("benchmark/compare_stitching.py", COMPARE_STITCHING)
    write_utf8("benchmark/generate_clips.py", GENERATE_CLIPS)
    write_utf8("benchmark/bootstrap_gt.py", BOOTSTRAP_GT)

    video_id = "DJI_20260511100755_0008_D"
    duration = 634.634
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from benchmark.ground_truth import CLIPS_SCHEMA_VERSION, GT_SCHEMA_VERSION, generate_uniform_clips

    clips = generate_uniform_clips(duration, target_count=30)
    clips_doc = {
        "schema_version": CLIPS_SCHEMA_VERSION,
        "video_id": video_id,
        "duration_sec": duration,
        "clip_count": len(clips),
        "target_count": 30,
        "clips": clips,
    }
    clips_path = os.path.join(ROOT, "benchmark", "clips", f"{video_id}_clips.json")
    os.makedirs(os.path.dirname(clips_path), exist_ok=True)
    with open(clips_path, "w", encoding="utf-8") as f:
        json.dump(clips_doc, f, ensure_ascii=False, indent=2)
    print("wrote", clips_path)

    video_path = os.path.join(ROOT, f"{video_id}.MP4")
    gt_template = {
        "schema_version": GT_SCHEMA_VERSION,
        "video": video_path,
        "duration_sec": duration,
        "provenance": {"type": "human_annotation_template", "events_annotated": 0},
        "events": [],
        "clips_ref": f"benchmark/clips/{video_id}_clips.json",
    }
    gt_dir = os.path.join(ROOT, "benchmark", "gt")
    os.makedirs(gt_dir, exist_ok=True)
    template_path = os.path.join(gt_dir, f"{video_id}_gt_template.json")
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(gt_template, f, ensure_ascii=False, indent=2)
    print("wrote", template_path)

    events_path = os.path.join(ROOT, "output", "detection", video_id, "final_events.json")
    if os.path.isfile(events_path):
        with open(events_path, encoding="utf-8") as f:
            ev_doc = json.load(f)
        from benchmark.bootstrap_gt import events_to_gt

        bootstrap = events_to_gt(ev_doc, video=video_path, source=events_path)
        bootstrap_path = os.path.join(gt_dir, f"{video_id}_gt_bootstrap.json")
        with open(bootstrap_path, "w", encoding="utf-8") as f:
            json.dump(bootstrap, f, ensure_ascii=False, indent=2)
        print("wrote", bootstrap_path)


if __name__ == "__main__":
    main()
