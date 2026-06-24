"""Compare v1_greedy vs v2_graph on the same video + ground truth."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from benchmark.ground_truth import (
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
