#!/usr/bin/env python3
"""Execute and validate a short hybrid smart-render prototype.

Untouched GOP-aligned regions are stream-copied from the source. Regions that
need masking are taken from an existing full-render reference and encoded back
to the source codec. This isolates the codec/timestamp/concat risk before the
production mask renderer is changed to render individual regions directly.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smart_render import (
    Interval,
    analyze_output,
    find_tool,
    get_stream_info,
    load_render_from_output,
    probe_video_packets,
)


def run(cmd: list[str], *, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    print("[cmd]", shlex.join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {shlex.join(cmd)}\n{proc.stderr.strip()}"
        )
    if not quiet and proc.stderr.strip():
        print(proc.stderr.strip())
    return proc


def crop_plan(
    plan: list[dict[str, Any]], sample_start: int, sample_end_exclusive: int
) -> list[dict[str, Any]]:
    """Intersect a whole-video inclusive-frame plan with a sample window."""
    cropped: list[dict[str, Any]] = []
    for segment in plan:
        start = max(sample_start, int(segment["start_frame"]))
        end = min(sample_end_exclusive - 1, int(segment["end_frame"]))
        if start <= end:
            cropped.append({"type": segment["type"], "start_frame": start, "end_frame": end})
    return cropped


def localize_render(
    render: dict[int, list[list[float]]], start_frame: int, end_frame: int
) -> dict[int, list[list[float]]]:
    """Translate global timeline frame keys into segment-local frame keys."""
    return {
        frame - start_frame: boxes
        for frame, boxes in render.items()
        if start_frame <= frame <= end_frame and boxes
    }


def frame_time(frame: int, fps: float) -> str:
    return f"{frame / fps:.9f}"


def segment_duration(segment: dict[str, Any], fps: float) -> str:
    frames = int(segment["end_frame"]) - int(segment["start_frame"]) + 1
    return f"{frames / fps:.9f}"


def expected_mux_frames(
    planned_frames: int,
    fps: float,
    sample_start_sec: float,
    source_audio_duration_sec: float | None,
    *,
    no_audio: bool,
) -> int:
    """Account for final ``-shortest`` when source audio ends before video."""
    if no_audio or not source_audio_duration_sec or fps <= 0:
        return planned_frames
    available_audio_sec = max(0.0, source_audio_duration_sec - sample_start_sec)
    audio_frames = max(0, int(round(available_audio_sec * fps)))
    return min(planned_frames, audio_frames)


def choose_hevc_encoder(requested: str) -> str:
    if requested != "auto":
        return requested
    ffmpeg = find_tool("ffmpeg")
    encoders = run([ffmpeg, "-hide_banner", "-encoders"], quiet=True)
    listing = encoders.stdout + encoders.stderr
    return "hevc_nvenc" if "hevc_nvenc" in listing else "libx265"


def encoder_args(encoder: str, bitrate: str) -> list[str]:
    if encoder == "hevc_nvenc":
        return [
            "-c:v", encoder, "-preset", "p1", "-profile:v", "main",
            "-pix_fmt", "yuv420p", "-b:v", bitrate, "-g", "30",
        ]
    if encoder == "libx265":
        return [
            "-c:v", encoder, "-preset", "ultrafast", "-profile:v", "main",
            "-pix_fmt", "yuv420p", "-b:v", bitrate, "-g", "30",
            "-x265-params", "open-gop=0:repeat-headers=1",
        ]
    raise ValueError(f"Unsupported prototype encoder: {encoder}")


def validate_packets(path: str) -> dict[str, Any]:
    packets = probe_video_packets(path, probe_work_dir=os.path.dirname(path))
    pts = [float(p["pts_time"]) for p in packets if p.get("pts_time") not in (None, "N/A")]
    dts = [float(p["dts_time"]) for p in packets if p.get("dts_time") not in (None, "N/A")]
    return {
        "packet_count": len(packets),
        "pts_monotonic": all(b > a for a, b in zip(pts, pts[1:])),
        "dts_monotonic": all(b > a for a, b in zip(dts, dts[1:])),
        "first_pts": pts[0] if pts else None,
        "last_pts": pts[-1] if pts else None,
        "first_dts": dts[0] if dts else None,
        "last_dts": dts[-1] if dts else None,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    analysis = analyze_output(args.output_dir, video_path=args.video)
    source = os.path.abspath(analysis["video_path"])
    rendered = None
    render: dict[int, list[list[float]]] = {}
    if args.direct_mask_render:
        render, _, _ = load_render_from_output(args.output_dir)
    else:
        rendered = os.path.abspath(args.rendered_video or os.path.join(args.output_dir, "final.mp4"))
    output_dir = os.path.abspath(args.prototype_dir)
    segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    fps = float(analysis["summary"]["fps"])
    total_frames = int(analysis["summary"]["total_frames"])
    sample_start = int(args.start_frame)
    sample_frames = int(round(args.duration * fps))
    sample_end = min(total_frames, sample_start + sample_frames)
    if sample_start < 0 or sample_start >= sample_end:
        raise ValueError("Invalid sample frame window")

    keyframes = {
        max(0, int(round(float(t) * fps))) for t in analysis["keyframes_sec"]
    }
    if sample_start not in keyframes or (sample_end not in keyframes and sample_end != total_frames):
        raise ValueError(
            f"Sample boundaries must be source keyframes; got {sample_start}..{sample_end}"
        )

    plan = crop_plan(analysis["plan"], sample_start, sample_end)
    if not plan or not {s["type"] for s in plan}.issuperset({"copy", "render"}):
        raise ValueError("Sample must contain both copy and render segments")

    ffmpeg = find_tool("ffmpeg")
    encoder = choose_hevc_encoder(args.encoder)
    fps_str = analysis["stream_info"]["video"].get("avg_frame_rate") or f"{fps:.9f}"
    segment_paths: list[str] = []

    for index, segment in enumerate(plan):
        segment_path = os.path.join(segments_dir, f"{index:03d}_{segment['type']}.ts")
        if args.reuse_segments and os.path.isfile(segment_path) and os.path.getsize(segment_path) > 0:
            print(f"[reuse] {segment_path}")
            segment_paths.append(segment_path)
            continue
        start = int(segment["start_frame"])
        duration = segment_duration(segment, fps)
        if segment["type"] == "copy":
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-ss", frame_time(start, fps), "-t", duration, "-i", source,
                "-map", "0:v:0", "-an", "-c:v", "copy",
                "-bsf:v", "hevc_mp4toannexb", "-f", "mpegts", segment_path,
            ]
        else:
            end_exclusive = int(segment["end_frame"]) + 1
            if args.direct_mask_render:
                from render import render_video

                input_segment = os.path.join(segments_dir, f"{index:03d}_render_input.mp4")
                run([
                    ffmpeg, "-y", "-loglevel", "error",
                    "-ss", frame_time(start, fps), "-t", duration, "-i", source,
                    "-map", "0:v:0", "-an", "-c:v", "copy",
                    "-avoid_negative_ts", "make_zero", input_segment,
                ])
                local_render = localize_render(render, start, int(segment["end_frame"]))
                segment_meta = {
                    "width": int(analysis["stream_info"]["video"]["width"]),
                    "height": int(analysis["stream_info"]["video"]["height"]),
                    "frames": end_exclusive - start,
                    "fps_str": fps_str,
                }
                render_video(
                    input_segment,
                    segment_path,
                    local_render,
                    segment_meta,
                    expand=args.expand,
                    mosaic_block=args.mosaic_block,
                    bitrate=args.bitrate,
                    encoder=encoder,
                    edge_partial_face=False,
                    refine_face_boxes=False,
                    mask_scale_divisor=args.mask_scale_divisor,
                    filter_threads=args.filter_threads,
                    limit_output_frames=True,
                )
                if not args.keep_render_inputs and os.path.isfile(input_segment):
                    os.remove(input_segment)
            else:
                # Frame-number trimming keeps the reference aligned even if its
                # container duration differs slightly from the source.
                vf = f"trim=start_frame={start}:end_frame={end_exclusive},setpts=PTS-STARTPTS"
                cmd = [
                    ffmpeg, "-y", "-loglevel", "error", "-i", rendered,
                    "-map", "0:v:0", "-an", "-vf", vf, "-r", fps_str,
                    *encoder_args(encoder, args.bitrate), "-f", "mpegts", segment_path,
                ]
                run(cmd)
        if segment["type"] == "copy":
            run(cmd)
        segment_paths.append(segment_path)

    concat_file = os.path.join(output_dir, "concat.txt")
    with open(concat_file, "w", encoding="utf-8", newline="\n") as f:
        for path in segment_paths:
            f.write("file '" + path.replace("'", "'\\''") + "'\n")

    video_ts = os.path.join(output_dir, "smart_video.ts")
    run([
        ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", concat_file, "-map", "0:v:0", "-an", "-c:v", "copy", video_ts,
    ])

    sample_duration = (sample_end - sample_start) / fps
    source_sample = None
    source_info = None
    if not args.no_source_sample:
        source_sample = os.path.join(output_dir, "source_sample.mp4")
        run([
            ffmpeg, "-y", "-loglevel", "error", "-ss", frame_time(sample_start, fps),
            "-t", f"{sample_duration:.9f}", "-i", source,
            "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
            "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", source_sample,
        ])
        source_info = get_stream_info(source_sample, probe_work_dir=output_dir)

    output = os.path.abspath(args.output_file or os.path.join(output_dir, "smart_render_sample.mp4"))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    mux_cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", video_ts]
    if args.no_audio:
        mux_cmd += ["-map", "0:v:0", "-an", "-c", "copy"]
    else:
        mux_cmd += [
            "-ss", frame_time(sample_start, fps), "-t", f"{sample_duration:.9f}", "-i", source,
            "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-shortest",
        ]
    mux_cmd += ["-avoid_negative_ts", "make_zero", "-movflags", "+faststart", output]
    run(mux_cmd)

    decode_checked = not args.skip_full_decode
    decode = None
    if decode_checked:
        decode = run([ffmpeg, "-v", "error", "-i", output, "-f", "null", "-"], quiet=True)
    output_info = get_stream_info(output, probe_work_dir=output_dir)
    frame_sec = 1.0 / fps
    video_duration = float(output_info["video"].get("duration") or output_info["duration_sec"] or 0)
    audio_duration = float((output_info.get("audio") or {}).get("duration") or 0)
    expected_frames = sample_end - sample_start
    source_audio = analysis["stream_info"].get("audio") or {}
    source_audio_duration = float(source_audio.get("duration") or 0.0) or None
    expected_output_frames = expected_mux_frames(
        expected_frames,
        fps,
        sample_start / fps,
        source_audio_duration,
        no_audio=args.no_audio,
    )
    expected_output_duration = expected_output_frames / fps
    output_frames = int(output_info["total_frames"])
    packet_validation = validate_packets(output)
    report = {
        "source": source,
        "render_mode": "direct-mask" if args.direct_mask_render else "full-render-reference",
        "rendered_reference": rendered,
        "output": output,
        "source_sample": source_sample,
        "encoder": encoder,
        "fps": fps,
        "sample": {
            "start_frame": sample_start,
            "end_frame_exclusive": sample_end,
            "start_sec": sample_start / fps,
            "duration_sec": sample_duration,
            "expected_frames": expected_frames,
            "expected_mux_frames": expected_output_frames,
        },
        "segments": plan,
        "segment_count": len(plan),
        "copy_segment_count": sum(s["type"] == "copy" for s in plan),
        "render_segment_count": sum(s["type"] == "render" for s in plan),
        "source_sample_info": source_info,
        "output_info": output_info,
        "validation": {
            "output_frames": output_frames,
            "frame_count_match": output_frames == expected_output_frames,
            "duration_delta_sec": abs(video_duration - expected_output_duration),
            "duration_within_one_frame": abs(video_duration - expected_output_duration) <= frame_sec + 1e-6,
            "av_drift_sec": abs(video_duration - audio_duration) if audio_duration else None,
            "av_drift_within_one_frame": (
                abs(video_duration - audio_duration) <= frame_sec + 1e-6 if audio_duration else True
            ),
            "decode_checked": decode_checked,
            "decode_errors": decode.stderr.strip() if decode else "",
            "decode_ok": not decode.stderr.strip() if decode else True,
            **packet_validation,
        },
        "elapsed_sec": time.perf_counter() - started,
    }
    report["validation"]["passed"] = all([
        report["validation"]["frame_count_match"],
        report["validation"]["duration_within_one_frame"],
        report["validation"]["av_drift_within_one_frame"],
        report["validation"]["decode_ok"],
        report["validation"]["pts_monotonic"],
        report["validation"]["dts_monotonic"],
    ])
    report_path = os.path.join(output_dir, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = report_path
    if args.cleanup_work:
        if os.path.isdir(segments_dir):
            shutil.rmtree(segments_dir)
        for temporary in (concat_file, video_ts):
            if os.path.isfile(temporary):
                os.remove(temporary)
    return report


def render_reviewed_output_smart(
    output_dir: str,
    output_file: str,
    *,
    video: str | None = None,
    encoder: str = "auto",
    bitrate: str = "12M",
    expand: float = 0.18,
    mosaic_block: int = 22,
    mask_scale_divisor: int = 8,
    filter_threads: int = 1,
    no_audio: bool = False,
    work_dir: str | None = None,
    cleanup_work: bool = True,
    full_decode_validation: bool = False,
) -> dict[str, Any]:
    """Render a full reviewed timeline with the validated hybrid executor."""
    analysis = analyze_output(output_dir, video_path=video)
    codec = str(analysis["stream_info"]["video"].get("codec_name") or "")
    if codec != "hevc":
        raise ValueError(f"Smart render currently supports HEVC sources only, got {codec!r}")
    if encoder not in {"auto", "hevc_nvenc", "libx265"}:
        raise ValueError(f"Smart render requires an HEVC encoder, got {encoder!r}")
    fps = float(analysis["summary"]["fps"])
    total_frames = int(analysis["summary"]["total_frames"])
    ns = argparse.Namespace(
        output_dir=output_dir,
        rendered_video=None,
        video=video,
        prototype_dir=work_dir or os.path.join(output_dir, "smart_render_work"),
        output_file=output_file,
        start_frame=0,
        duration=total_frames / fps,
        encoder=encoder,
        bitrate=bitrate,
        reuse_segments=False,
        direct_mask_render=True,
        expand=expand,
        mosaic_block=mosaic_block,
        mask_scale_divisor=mask_scale_divisor,
        filter_threads=filter_threads,
        keep_render_inputs=False,
        no_source_sample=True,
        no_audio=no_audio,
        cleanup_work=cleanup_work,
        skip_full_decode=not full_decode_validation,
    )
    return execute(ns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short hybrid smart-render prototype")
    parser.add_argument("--output-dir", required=True, help="Reviewed pipeline output directory")
    parser.add_argument("--rendered-video", default=None, help="Existing full-render reference")
    parser.add_argument("--video", default=None, help="Override source video")
    parser.add_argument("--prototype-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=3300)
    parser.add_argument("--duration", type=float, default=60.06)
    parser.add_argument("--encoder", choices=["auto", "hevc_nvenc", "libx265"], default="auto")
    parser.add_argument("--bitrate", default="12M")
    parser.add_argument("--reuse-segments", action="store_true")
    parser.add_argument("--direct-mask-render", action="store_true")
    parser.add_argument("--expand", type=float, default=0.18)
    parser.add_argument("--mosaic-block", type=int, default=22)
    parser.add_argument("--mask-scale-divisor", type=int, default=8)
    parser.add_argument("--filter-threads", type=int, default=1)
    parser.add_argument("--keep-render-inputs", action="store_true")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--no-source-sample", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--cleanup-work", action="store_true")
    parser.add_argument("--skip-full-decode", action="store_true")
    return parser


def main() -> int:
    report = execute(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
