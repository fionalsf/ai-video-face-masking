#!/usr/bin/env python3
"""Single-video pipeline: detect -> track -> presence -> identity stitch -> behavior events."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from audit_log import write_audit_log
from edge_review_candidates import build_edge_review_candidates
from event_builder import TIER_AUTO, TIER_LOW_CONF, TIER_REVIEW, build_events, events_by_tier
from event_merge import save_segmentation_events
from export_events import export_review_pack, write_confirmed_events_template
from identity_behavior_builder import (
    behavior_event_to_face_event,
    build_identity_behavior_events,
)
from identity_stitching import run_identity_stitching
from low_conf_log import write_low_conf_stats
from low_conf_promote import promote_low_conf_review_candidates
from mask_timeline import (
    build_mask_timeline,
    export_mask_review_pack,
    save_mask_timeline,
    write_mask_confirmed_template,
)
from render import render_masked_output
from tracker import run_detect_track
from video_meta import get_video_meta, safe_video_stem

PIPELINE_LABEL = "detection -> tracking -> presence -> identity_stitching -> identity_behavior_builder"

MODE_PRESETS = {
    "legacy": {
        "interval": 5,
        "conf": 0.35,
        "imgsz": 1280,
        "expand": 0.18,
        "mosaic_block": 22,
        "render_extend_frames": 3,
        "mask_review": False,
        "motion_compensate": False,
        "motion_max_gap": 45,
        "motion_singleton_frames": 12,
        "edge_partial_face": False,
        "edge_review_scan": False,
        "low_conf_standalone": True,
        "fast_identity": False,
        "timeline_review": True,
    },
    "preview": {
        "interval": 5,
        "conf": 0.25,
        "imgsz": 960,
        "expand": 0.28,
        "mosaic_block": 18,
        "render_extend_frames": 10,
        "mask_review": True,
        "motion_compensate": True,
        "motion_max_gap": 60,
        "motion_singleton_frames": 30,
        "edge_partial_face": True,
        "edge_review_scan": True,
        "low_conf_standalone": True,
        "fast_identity": False,
        "timeline_review": True,
    },
    "production": {
        "interval": 2,
        "conf": 0.25,
        "imgsz": 1280,
        "expand": 0.25,
        "mosaic_block": 22,
        "render_extend_frames": 12,
        "mask_review": True,
        "motion_compensate": True,
        "motion_max_gap": 45,
        "motion_singleton_frames": 30,
        "edge_partial_face": True,
        "edge_review_scan": True,
        "low_conf_standalone": True,
        "fast_identity": False,
        "timeline_review": True,
    },
    "fast_review": {
        "interval": 3,
        "conf": 0.23,
        "imgsz": 1280,
        "expand": 0.26,
        "mosaic_block": 22,
        "render_extend_frames": 14,
        "mask_review": True,
        "motion_compensate": True,
        "motion_max_gap": 60,
        "motion_singleton_frames": 36,
        "edge_partial_face": True,
        "edge_review_scan": False,
        "low_conf_standalone": False,
        "fast_identity": True,
        "timeline_review": True,
    },
    "privacy": {
        "interval": 1,
        "conf": 0.20,
        "imgsz": 1536,
        "expand": 0.35,
        "mosaic_block": 18,
        "render_extend_frames": 18,
        "mask_review": True,
        "motion_compensate": True,
        "motion_max_gap": 45,
        "motion_singleton_frames": 45,
        "edge_partial_face": True,
        "edge_review_scan": True,
        "low_conf_standalone": True,
        "fast_identity": False,
        "timeline_review": True,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Identity-based face event pipeline (single video)")
    p.add_argument("-i", "--input", required=True, help="Input video")
    p.add_argument("-o", "--output-dir", default="output", help="Output root directory")
    p.add_argument(
        "--mode",
        choices=sorted(MODE_PRESETS),
        default="legacy",
        help="Runtime preset: legacy keeps v1 behavior; preview/production improve recall.",
    )
    p.add_argument("--interval", type=int, default=None, help="Detection stride (frames)")
    p.add_argument("--conf", type=float, default=None, help="Detection confidence threshold")
    p.add_argument("--model", default="models/face.pt", help="YOLO-face weights")
    p.add_argument(
        "--infer-backend",
        choices=["auto", "torch", "onnx", "tensorrt"],
        default="torch",
        help="Detection inference backend. torch/auto keep stable v3 behavior; onnx/tensorrt are explicit acceleration tests.",
    )
    p.add_argument("--onnx-model", default="models/face.onnx", help="ONNX model path for --infer-backend onnx/auto")
    p.add_argument("--device", default="0", help="GPU id or cpu")
    p.add_argument(
        "--decode-backend",
        choices=["opencv", "cuda", "cuda-full"],
        default="opencv",
        help="Video decode backend; cuda uses NVIDIA HEVC hardware decode with pre-scaling, cuda-full preserves source resolution.",
    )
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--detect-batch", type=int, default=4, help="GPU detection batch size; auto-falls back on memory errors.")
    p.add_argument(
        "--detect-isolated",
        action="store_true",
        help="Run detect+track in a subprocess so inference runtimes release memory before review stages.",
    )
    p.add_argument("--event-gap", type=float, default=1.0, help="Presence segmentation gap (seconds)")
    p.add_argument("--stitch-gap", type=float, default=60.0, help="Temporal soft prior tau for identity graph (sec)")
    p.add_argument("--behavior-gap", type=float, default=8.0, help="Behavior split gap within identity (seconds)")
    p.add_argument("--expand", type=float, default=None)
    p.add_argument("--mosaic-block", type=int, default=None)
    p.add_argument("--mask-scale-divisor", type=int, default=8, help="Mask pipe resolution divisor for ffmpeg overlay; larger is faster.")
    p.add_argument("--filter-threads", type=int, default=1, help="FFmpeg filter thread count for mosaic overlay render.")
    p.add_argument("--render-extend-frames", type=int, default=None)
    p.add_argument("--motion-comp", dest="motion_compensate", action="store_true", help="Use optical-flow motion compensation while rendering masks.")
    p.add_argument("--no-motion-comp", dest="motion_compensate", action="store_false", help="Disable optical-flow motion compensation.")
    p.set_defaults(motion_compensate=None)
    p.add_argument("--motion-max-gap", type=int, default=None, help="Max keyframe gap for optical-flow interpolation.")
    p.add_argument("--motion-singleton-frames", type=int, default=None, help="Frames to track before/after one-point events.")
    p.add_argument("--motion-min-points", type=int, default=4, help="Minimum optical-flow points needed to accept tracking.")
    p.add_argument("--motion-anchor", type=float, default=0.18, help="Blend optical-flow boxes back toward detection keyframes.")
    p.add_argument("--edge-partial-face", dest="edge_partial_face", action="store_true", help="Mask large partial skin-face regions touching left/right frame edges.")
    p.add_argument("--no-edge-partial-face", dest="edge_partial_face", action="store_false", help="Disable partial edge-face fallback.")
    p.set_defaults(edge_partial_face=None)
    p.add_argument("--edge-review-scan", dest="edge_review_scan", action="store_true", help="Scan video for extra edge partial-face Review candidates.")
    p.add_argument("--no-edge-review-scan", dest="edge_review_scan", action="store_false", help="Skip extra edge partial-face Review candidate scan.")
    p.set_defaults(edge_review_scan=None)
    p.add_argument("--low-conf-standalone", dest="low_conf_standalone", action="store_true", help="Promote standalone low-confidence edge clips into Review.")
    p.add_argument("--no-low-conf-standalone", dest="low_conf_standalone", action="store_false", help="Skip standalone low-confidence promotion scan.")
    p.set_defaults(low_conf_standalone=None)
    p.add_argument("--fast-identity", dest="fast_identity", action="store_true", help="Skip full-video appearance embedding enrichment for faster review generation.")
    p.add_argument("--full-identity", dest="fast_identity", action="store_false", help="Use full appearance embedding enrichment during identity stitching.")
    p.set_defaults(fast_identity=None)
    p.add_argument("--mask-review", action="store_true", help="Render Review-tier events too.")
    p.add_argument("--mask-lowconf", action="store_true", help="Render LowConf-tier events too.")
    p.add_argument("--reuse-tracks", action="store_true", help="Reuse tracked_detections.json when present.")
    p.add_argument("--no-review-pack", action="store_true", help="Skip exporting review thumbnails.")
    p.add_argument("--encoder", default="auto")
    p.add_argument("--skip-render", action="store_true", help="Skip Auto-tier render (debug)")
    p.add_argument("--review-only", action="store_true", help="Export review pack and metadata without rendering masked_draft.mp4.")
    return p.parse_args()


def resolve_runtime_args(args) -> dict:
    preset = MODE_PRESETS[args.mode]
    return {
        "interval": args.interval if args.interval is not None else preset["interval"],
        "detect_batch": max(1, int(args.detect_batch or 1)),
        "conf": args.conf if args.conf is not None else preset["conf"],
        "imgsz": args.imgsz if args.imgsz is not None else preset["imgsz"],
        "expand": args.expand if args.expand is not None else preset["expand"],
        "mosaic_block": args.mosaic_block if args.mosaic_block is not None else preset["mosaic_block"],
        "mask_scale_divisor": max(4, int(args.mask_scale_divisor or 8)),
        "filter_threads": max(1, int(args.filter_threads or 1)),
        "render_extend_frames": (
            args.render_extend_frames
            if args.render_extend_frames is not None
            else preset["render_extend_frames"]
        ),
        "mask_review": args.mask_review or preset["mask_review"],
        "mask_lowconf": args.mask_lowconf,
        "motion_compensate": (
            args.motion_compensate
            if args.motion_compensate is not None
            else preset["motion_compensate"]
        ),
        "motion_max_gap": args.motion_max_gap if args.motion_max_gap is not None else preset["motion_max_gap"],
        "motion_singleton_frames": (
            args.motion_singleton_frames
            if args.motion_singleton_frames is not None
            else preset["motion_singleton_frames"]
        ),
        "motion_min_points": args.motion_min_points,
        "motion_anchor": args.motion_anchor,
        "edge_partial_face": (
            args.edge_partial_face
            if args.edge_partial_face is not None
            else preset["edge_partial_face"]
        ),
        "edge_review_scan": (
            args.edge_review_scan
            if args.edge_review_scan is not None
            else preset.get("edge_review_scan", preset["edge_partial_face"])
        ),
        "low_conf_standalone": (
            args.low_conf_standalone
            if args.low_conf_standalone is not None
            else preset.get("low_conf_standalone", True)
        ),
        "fast_identity": (
            args.fast_identity
            if args.fast_identity is not None
            else preset.get("fast_identity", False)
        ),
        "timeline_review": bool(preset.get("timeline_review", True)),
    }


def load_or_run_detect_track(args, runtime: dict, video_path: str, meta: dict, out_dir: str) -> list[dict]:
    tracked_path = os.path.join(out_dir, "tracked_detections.json")
    if args.reuse_tracks and os.path.isfile(tracked_path):
        print(f"[cache] Reusing tracked detections: {tracked_path}")
        with open(tracked_path, encoding="utf-8") as f:
            return json.load(f)

    if args.detect_isolated:
        worker_cfg_path = os.path.join(out_dir, "detect_worker_config.json")
        worker_report_path = os.path.join(out_dir, "detect_worker_report.json")
        worker_cfg = {
            "video_path": video_path,
            "model_path": os.path.abspath(args.model),
            "meta": meta,
            "device": args.device,
            "conf": runtime["conf"],
            "imgsz": runtime["imgsz"],
            "interval": runtime["interval"],
            "batch_size": runtime["detect_batch"],
            "infer_backend": args.infer_backend,
            "decode_backend": args.decode_backend,
            "onnx_model_path": os.path.abspath(args.onnx_model) if args.onnx_model else None,
            "output_path": tracked_path,
            "report_path": worker_report_path,
        }
        with open(worker_cfg_path, "w", encoding="utf-8") as f:
            json.dump(worker_cfg, f, ensure_ascii=False, indent=2)
        worker_script = os.path.join(os.path.dirname(__file__), "detect_track_worker.py")
        print(f"[detect] isolated worker -> {tracked_path}")
        completed = subprocess.run(
            [sys.executable, worker_script, "--config", worker_cfg_path],
            cwd=os.path.dirname(__file__),
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Detect worker failed with exit code {completed.returncode}")
        with open(tracked_path, encoding="utf-8") as f:
            return json.load(f)

    detections = run_detect_track(
        video_path,
        args.model,
        meta,
        device=args.device,
        conf=runtime["conf"],
        imgsz=runtime["imgsz"],
        interval=runtime["interval"],
        batch_size=runtime["detect_batch"],
        infer_backend=args.infer_backend,
        onnx_model_path=args.onnx_model,
        decode_backend=args.decode_backend,
    )
    with open(tracked_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)
    return detections


def select_render_events(auto_ev, review_ev, low_ev, runtime: dict) -> list:
    selected = list(auto_ev)
    if runtime["mask_review"]:
        selected.extend(review_ev)
    if runtime["mask_lowconf"]:
        selected.extend(low_ev)
    return selected


def run_pipeline(args) -> int:
    pipeline_started = time.perf_counter()
    stage_times: dict[str, float] = {}

    def mark_stage(name: str, started: float) -> None:
        stage_times[name] = round(time.perf_counter() - started, 3)

    video_path = os.path.abspath(args.input)
    if not os.path.isfile(video_path):
        print(f"[error] Video not found: {video_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.model):
        print(f"[error] Model not found: {args.model}", file=sys.stderr)
        return 1

    stem = safe_video_stem(video_path)
    out_dir = os.path.join(os.path.abspath(args.output_dir), stem)
    review_dir = os.path.join(out_dir, "review")
    os.makedirs(review_dir, exist_ok=True)
    runtime = resolve_runtime_args(args)

    meta = get_video_meta(video_path)
    print(
        f"[info] {stem} | {meta['width']}x{meta['height']} @ {meta['fps']:.2f}fps"
        f" | {meta['frames']} frames"
    )
    print(
        f"[mode] {args.mode} | interval={runtime['interval']} conf={runtime['conf']} "
        f"imgsz={runtime['imgsz']} detect_batch={runtime['detect_batch']} "
        f"infer_backend={args.infer_backend} detect_isolated={args.detect_isolated} "
        f"mask_review={runtime['mask_review']} "
        f"mask_lowconf={runtime['mask_lowconf']} motion_comp={runtime['motion_compensate']} "
        f"edge_partial={runtime['edge_partial_face']} fast_identity={runtime['fast_identity']}"
    )

    print("[1/5] Detection + Tracking...")
    stage_started = time.perf_counter()
    detections = load_or_run_detect_track(args, runtime, video_path, meta, out_dir)
    mark_stage("detect_track_sec", stage_started)

    print("[2/5] Presence segmentation (Scheme C, debug artifacts)...")
    stage_started = time.perf_counter()
    builder_stats: dict = {}
    segmentation_events = build_events(
        detections, gap_sec=args.event_gap,
        frame_h=meta["height"], frame_w=meta["width"],
        merge_stats=builder_stats,
        fps=float(meta["fps"]),
        total_frames=meta["frames"],
        detect_interval=runtime["interval"],
        output_dir=out_dir,
    )
    save_segmentation_events(
        segmentation_events, out_dir, video=video_path, fps=float(meta["fps"]),
    )
    print(f"      segmentation events (non-production): {len(segmentation_events)}")
    mark_stage("presence_segmentation_sec", stage_started)

    print("[3/5] Identity stitching...")
    stage_started = time.perf_counter()
    stitch_stats = run_identity_stitching(
        detections,
        output_dir=out_dir,
        video=video_path,
        temporal_tau=args.stitch_gap,
        enrich_appearance=not runtime["fast_identity"],
    )
    print(
        f"      {stitch_stats['track_count']} tracks -> {stitch_stats['identity_cluster_count']} identities"
        f" | appearance={stitch_stats['appearance_method']}"
        f" | links={stitch_stats['linked_edge_count']}"
    )
    mark_stage("identity_stitching_sec", stage_started)

    print("[4/5] Identity behavior event builder (production)...")
    stage_started = time.perf_counter()
    behavior_stats = build_identity_behavior_events(
        detections,
        stitch_stats["clusters"],
        output_dir=out_dir,
        video=video_path,
        fps=float(meta["fps"]),
        frame_h=meta["height"],
        frame_w=meta["width"],
        total_frames=meta["frames"],
        detect_interval=runtime["interval"],
        behavior_gap_sec=args.behavior_gap,
    )
    print(
        f"      behavior events: {behavior_stats['behavior_event_count']}"
        f" | target_30_80={behavior_stats['target_in_range']}"
    )
    mark_stage("behavior_builder_sec", stage_started)

    stage_started = time.perf_counter()
    events = [behavior_event_to_face_event(ev) for ev in behavior_stats["behavior_events"]]
    edge_review_ev = []
    if runtime["edge_partial_face"] and runtime.get("edge_review_scan", True):
        edge_review_ev = build_edge_review_candidates(
            video_path,
            meta,
            events,
            stride=max(1, runtime["interval"]),
            min_hits=2,
        )
        if edge_review_ev:
            print(f"      edge partial-face review candidates: {len(edge_review_ev)}")
            events.extend(edge_review_ev)

    low_conf_promoted = promote_low_conf_review_candidates(video_path, events, meta, runtime)
    if low_conf_promoted:
        print(f"      low-confidence face review candidates: {low_conf_promoted}")
    mark_stage("candidate_promotion_sec", stage_started)

    tiers = events_by_tier(events)
    auto_ev = tiers[TIER_AUTO]
    review_ev = tiers[TIER_REVIEW]
    low_ev = tiers[TIER_LOW_CONF]

    print(
        f"[5/5] Review/Render — Auto={len(auto_ev)} Review={len(review_ev)} LowConf={len(low_ev)}"
    )

    render_ev = select_render_events(auto_ev, review_ev, low_ev, runtime)

    with open(os.path.join(out_dir, "face_events.json"), "w", encoding="utf-8") as f:
        json.dump({
            "video": video_path,
            "fps": meta["fps"],
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": PIPELINE_LABEL,
            "event_unit": "identity_id",
            "events": [e.to_dict() for e in events],
        }, f, ensure_ascii=False, indent=2)

    event_dicts = [e.to_dict() for e in events]
    stage_started = time.perf_counter()
    mask_timeline = build_mask_timeline(video_path, event_dicts, meta, runtime)
    mask_timeline_path = save_mask_timeline(out_dir, mask_timeline)
    mark_stage("mask_timeline_sec", stage_started)
    timeline_review_pending = sum(
        1 for p in mask_timeline.get("proposals", [])
        if p.get("source_tier") != TIER_AUTO
    )
    print(
        f"[Timeline] frozen mask proposals: {mask_timeline['proposal_count']} "
        f"review_pending={timeline_review_pending} entries={mask_timeline['entry_count']} "
        f"-> {mask_timeline_path}"
    )

    write_low_conf_stats(out_dir, video_path, low_ev)
    write_audit_log(out_dir, video_path, auto_ev)

    if timeline_review_pending and not args.no_review_pack:
        stage_started = time.perf_counter()
        if runtime.get("timeline_review", True):
            export_mask_review_pack(
                mask_timeline,
                review_dir,
                expand=float(runtime.get("expand") or 0.20),
            )
            review_pending_path = os.path.join(review_dir, "pending_events.json")
            with open(review_pending_path, encoding="utf-8") as f:
                review_pending = json.load(f)
            write_mask_confirmed_template(review_dir, review_pending.get("events") or [])
        else:
            export_review_pack(video_path, review_dir, review_ev)
            write_confirmed_events_template(review_dir, video_path)
        print(f"[Review] {timeline_review_pending} mask proposals pending -> {review_dir}")
        mark_stage("review_pack_sec", stage_started)
    elif timeline_review_pending:
        print(f"[Review] {timeline_review_pending} mask proposals pending; review pack skipped")
    else:
        print("[Review] No events need review")

    draft_path = os.path.join(out_dir, "masked_draft.mp4")
    if not args.skip_render and not args.review_only:
        stage_started = time.perf_counter()
        render_dicts = [e.to_dict() for e in render_ev]
        if render_dicts:
            print(
                f"[Render] Rendering {len(render_ev)} events "
                f"(Auto={len(auto_ev)}, Review={'on' if runtime['mask_review'] else 'off'}, "
                f"LowConf={'on' if runtime['mask_lowconf'] else 'off'}) -> masked_draft.mp4"
            )
            render_masked_output(
                video_path, draft_path, render_dicts, meta,
                expand=runtime["expand"],
                mosaic_block=runtime["mosaic_block"],
                mask_scale_divisor=runtime["mask_scale_divisor"],
                filter_threads=runtime["filter_threads"],
                extend_frames=runtime["render_extend_frames"],
                motion_compensate=runtime["motion_compensate"],
                motion_max_gap=runtime["motion_max_gap"],
                motion_singleton_frames=runtime["motion_singleton_frames"],
                motion_min_points=runtime["motion_min_points"],
                motion_anchor=runtime["motion_anchor"],
                refine_face_boxes=bool(runtime.get("refine_face_boxes", True)),
                encoder=args.encoder,
            )
        else:
            import shutil
            shutil.copy2(video_path, draft_path)
            print("[Render] No selected events; copied source as masked_draft.mp4")
        mark_stage("draft_render_sec", stage_started)
    else:
        reason = "--review-only" if args.review_only else "--skip-render"
        print(f"[Render] {reason}: skipped")

    report = {
        "video": video_path,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "runtime": runtime,
        "infer_backend": args.infer_backend,
        "onnx_model": args.onnx_model,
        "detect_isolated": args.detect_isolated,
        "decode_backend": args.decode_backend,
        "stage_times": {
            **stage_times,
            "total_pipeline_sec": round(time.perf_counter() - pipeline_started, 3),
        },
        "events_total": len(events),
        "auto_masked": len(auto_ev),
        "pending_review": len(review_ev),
        "pending_mask_review": timeline_review_pending,
        "low_conf_logged": len(low_ev),
        "rendered_events": len(render_ev),
        "rendered_review": runtime["mask_review"],
        "rendered_low_conf": runtime["mask_lowconf"],
        "edge_review_candidates": len(edge_review_ev),
        "low_conf_review_candidates": low_conf_promoted,
        "builder_stats": builder_stats,
        "stitch_stats": {
            "track_count": stitch_stats["track_count"],
            "identity_cluster_count": stitch_stats["identity_cluster_count"],
            "appearance_method": stitch_stats["appearance_method"],
            "appearance_enrich_enabled": stitch_stats.get("appearance_enrich_enabled"),
            "profile_embeddings": stitch_stats.get("profile_embeddings"),
            "stage_times": stitch_stats.get("stage_times"),
        },
        "behavior_stats": {
            "behavior_event_count": behavior_stats["behavior_event_count"],
            "target_in_range": behavior_stats["target_in_range"],
        },
        "pipeline": PIPELINE_LABEL,
        "production_events": "behavior_events.json",
        "output_dir": out_dir,
        "masked_draft": draft_path,
        "review_dir": review_dir if timeline_review_pending else None,
        "mask_timeline": mask_timeline_path,
        "review_unit": "mask_timeline_proposal" if runtime.get("timeline_review", True) else "face_event",
        "delivery_ready": timeline_review_pending == 0,
        "morning_action": "none" if not timeline_review_pending else f"streamlit run review_ui.py -- --review-dir {review_dir}",
    }
    report_path = os.path.join(out_dir, "review_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if report["delivery_ready"] and os.path.isfile(draft_path):
        final_path = os.path.join(out_dir, "final.mp4")
        import shutil
        shutil.copy2(draft_path, final_path)
        print("[done] No Review events; final.mp4 ready")
    elif timeline_review_pending:
        print("[done] Review events pending; use masked_draft.mp4 for preview and run confirm.py after review")

    print(f"\n[done] Output: {out_dir}")
    print("  identity_clusters.json | behavior_events.json | review_report.json")
    if timeline_review_pending:
        print(f'  Morning: streamlit run review_ui.py -- --review-dir "{review_dir}"')
    return 0


def main():
    sys.exit(run_pipeline(parse_args()))


if __name__ == "__main__":
    main()
