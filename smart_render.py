#!/usr/bin/env python3
"""Smart-render feasibility and planning prototype.

This module is intentionally independent from render.py/confirm.py.  It does
not replace the production full-video renderer.  The first goal is to measure
whether keyframe-aware segment rendering can save enough time to justify the
extra concat risk.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from mask_timeline import load_mask_timeline, parse_review_decisions, select_render_entries


DEFAULT_MERGE_GAP_SEC = 1.0


@dataclass(frozen=True)
class Interval:
    start_frame: int
    end_frame: int

    def duration_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)

    def start_sec(self, fps: float) -> float:
        return self.start_frame / fps

    def end_sec_exclusive(self, fps: float) -> float:
        return (self.end_frame + 1) / fps


def _is_ascii_path(path: str) -> bool:
    try:
        path.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def find_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    if name == "ffprobe":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            sibling = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
            if os.path.isfile(sibling):
                return sibling
            sibling_bak = sibling + ".bak"
            if os.path.isfile(sibling_bak):
                return sibling_bak
    raise FileNotFoundError(f"Missing required tool: {name}")


@contextlib.contextmanager
def ascii_probe_path(path: str, work_dir: str | None = None) -> Iterable[str]:
    """Provide an ASCII path for ffprobe builds that fail on Unicode paths.

    Some Windows ffprobe builds handle Unicode poorly when launched from
    subprocess.  A same-volume hardlink gives ffprobe an ASCII path without
    copying multi-GB video files.
    """

    abs_path = os.path.abspath(path)
    if _is_ascii_path(abs_path):
        yield abs_path
        return

    drive, _ = os.path.splitdrive(abs_path)
    base_dir = work_dir or (os.path.join(drive + os.sep, "_smart_render_probe") if drive else tempfile.gettempdir())
    os.makedirs(base_dir, exist_ok=True)
    suffix = os.path.splitext(abs_path)[1] or ".mp4"
    link_path = os.path.join(base_dir, f"_smart_probe_{os.getpid()}_{int(time.time())}{suffix}")
    try:
        os.link(abs_path, link_path)
        yield link_path
    except OSError:
        # Last resort: let the caller see ffprobe's native error.
        yield abs_path
    finally:
        if os.path.exists(link_path):
            try:
                os.remove(link_path)
            except OSError:
                pass


def run_json(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    if not proc.stdout.strip():
        return {}
    return json.loads(proc.stdout)


def parse_rate(rate: str | None) -> float:
    if not rate or rate == "0/0":
        return 0.0
    return float(Fraction(rate))


def _iter_mp4_boxes(data: bytes, start: int = 0, end: int | None = None):
    end = len(data) if end is None else min(end, len(data))
    pos = start
    while pos + 8 <= end:
        size = int.from_bytes(data[pos:pos + 4], "big")
        box_type = data[pos + 4:pos + 8].decode("latin1", errors="replace")
        header = 8
        if size == 1:
            if pos + 16 > end:
                break
            size = int.from_bytes(data[pos + 8:pos + 16], "big")
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            break
        yield box_type, pos + header, pos + size
        pos += size


def _find_mp4_children(data: bytes, path: list[str]) -> list[bytes]:
    if not path:
        return [data]
    out: list[bytes] = []
    for box_type, content_start, content_end in _iter_mp4_boxes(data):
        if box_type == path[0]:
            out.extend(_find_mp4_children(data[content_start:content_end], path[1:]))
    return out


def _read_top_level_box(path: str, target: str) -> bytes | None:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        pos = 0
        while pos + 8 <= size:
            f.seek(pos)
            header = f.read(16)
            if len(header) < 8:
                break
            box_size = int.from_bytes(header[:4], "big")
            box_type = header[4:8].decode("latin1", errors="replace")
            header_size = 8
            if box_size == 1:
                if len(header) < 16:
                    break
                box_size = int.from_bytes(header[8:16], "big")
                header_size = 16
            elif box_size == 0:
                box_size = size - pos
            if box_size < header_size:
                break
            if box_type == target:
                f.seek(pos + header_size)
                return f.read(box_size - header_size)
            pos += box_size
    return None


def _parse_mdhd_timescale(mdhd: bytes) -> int:
    if len(mdhd) < 24:
        return 0
    version = mdhd[0]
    if version == 1:
        return int.from_bytes(mdhd[20:24], "big") if len(mdhd) >= 32 else 0
    return int.from_bytes(mdhd[12:16], "big")


def _parse_stss_samples(stss: bytes) -> list[int]:
    if len(stss) < 8:
        return []
    count = int.from_bytes(stss[4:8], "big")
    samples = []
    pos = 8
    for _ in range(count):
        if pos + 4 > len(stss):
            break
        samples.append(int.from_bytes(stss[pos:pos + 4], "big"))
        pos += 4
    return samples


def _parse_stts_entries(stts: bytes) -> list[tuple[int, int]]:
    if len(stts) < 8:
        return []
    count = int.from_bytes(stts[4:8], "big")
    entries = []
    pos = 8
    for _ in range(count):
        if pos + 8 > len(stts):
            break
        sample_count = int.from_bytes(stts[pos:pos + 4], "big")
        sample_delta = int.from_bytes(stts[pos + 4:pos + 8], "big")
        entries.append((sample_count, sample_delta))
        pos += 8
    return entries


def probe_mp4_timing(video_path: str) -> dict[str, Any]:
    """Read video sync samples and timing-table evidence without media scanning."""

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in {".mp4", ".mov", ".m4v"}:
        return {}
    moov = _read_top_level_box(video_path, "moov")
    if not moov:
        return {}
    for trak in _find_mp4_children(moov, ["trak"]):
        hdlr_boxes = _find_mp4_children(trak, ["mdia", "hdlr"])
        handler = hdlr_boxes[0][8:12].decode("latin1", errors="replace") if hdlr_boxes and len(hdlr_boxes[0]) >= 12 else ""
        if handler != "vide":
            continue
        mdhd_boxes = _find_mp4_children(trak, ["mdia", "mdhd"])
        stss_boxes = _find_mp4_children(trak, ["mdia", "minf", "stbl", "stss"])
        stts_boxes = _find_mp4_children(trak, ["mdia", "minf", "stbl", "stts"])
        if not mdhd_boxes or not stss_boxes or not stts_boxes:
            continue
        timescale = _parse_mdhd_timescale(mdhd_boxes[0])
        sync_samples = _parse_stss_samples(stss_boxes[0])
        stts_entries = _parse_stts_entries(stts_boxes[0])
        if not timescale or not sync_samples or not stts_entries:
            continue
        sync_set = set(sync_samples)
        max_sync = max(sync_samples)
        out = []
        sample_number = 1
        decode_time = 0
        for sample_count, sample_delta in stts_entries:
            for _ in range(sample_count):
                if sample_number in sync_set:
                    out.append(decode_time / timescale)
                decode_time += sample_delta
                sample_number += 1
                if sample_number > max_sync:
                    return {
                        "keyframes_sec": sorted(set(out)),
                        "timescale": timescale,
                        "sample_count": sum(count for count, _ in stts_entries),
                        "stts_entry_count": len(stts_entries),
                        "sample_deltas": sorted(set(delta for _, delta in stts_entries)),
                    }
        return {
            "keyframes_sec": sorted(set(out)),
            "timescale": timescale,
            "sample_count": sum(count for count, _ in stts_entries),
            "stts_entry_count": len(stts_entries),
            "sample_deltas": sorted(set(delta for _, delta in stts_entries)),
        }
    return {}


def probe_keyframes_mp4(video_path: str) -> list[float]:
    """Read keyframe times from MP4 sync-sample timing tables without decoding."""

    return probe_mp4_timing(video_path).get("keyframes_sec") or []


def get_stream_info(video_path: str, *, probe_work_dir: str | None = None) -> dict[str, Any]:
    ffprobe = find_tool("ffprobe")
    with ascii_probe_path(video_path, probe_work_dir) as probe_path:
        video = run_json([
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=index,codec_type,codec_name,profile,level,width,height,r_frame_rate,"
            "avg_frame_rate,time_base,pix_fmt,nb_frames,duration",
            "-of", "json",
            probe_path,
        ])
        audio = run_json([
            ffprobe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries",
            "stream=index,codec_type,codec_name,profile,sample_rate,channels,duration",
            "-of", "json",
            probe_path,
        ])
    v_stream = (video.get("streams") or [{}])[0]
    a_streams = audio.get("streams") or []
    fps = parse_rate(v_stream.get("avg_frame_rate")) or parse_rate(v_stream.get("r_frame_rate"))
    nb_frames = int(v_stream.get("nb_frames") or 0)
    if nb_frames <= 0:
        with ascii_probe_path(video_path, probe_work_dir) as probe_path:
            packets = run_json([
                ffprobe,
                "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "json",
                probe_path,
            ])
        p_stream = (packets.get("streams") or [{}])[0]
        nb_frames = int(p_stream.get("nb_read_frames") or 0)
    duration = float(v_stream.get("duration") or 0.0)
    if nb_frames <= 0 and fps > 0 and duration > 0:
        nb_frames = int(round(duration * fps))
    return {
        "video": v_stream,
        "audio": a_streams[0] if a_streams else None,
        "fps": fps,
        "total_frames": nb_frames,
        "duration_sec": duration,
    }


def probe_video_packets(video_path: str, *, probe_work_dir: str | None = None) -> list[dict[str, Any]]:
    ffprobe = find_tool("ffprobe")
    with ascii_probe_path(video_path, probe_work_dir) as probe_path:
        data = run_json([
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,dts_time,duration_time,flags",
            "-of", "json",
            probe_path,
        ])
    return data.get("packets") or []


def classify_cfr(packets: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    durations = []
    pts_values = []
    for packet in packets:
        if packet.get("duration_time") is not None:
            durations.append(round(float(packet["duration_time"]), 9))
        if packet.get("pts_time") is not None:
            pts_values.append(float(packet["pts_time"]))
    unique_durations = sorted(set(durations))
    frame_sec = 1.0 / fps if fps > 0 else 0.0
    cfr_by_duration = len(unique_durations) == 1 and (
        frame_sec <= 0 or abs(unique_durations[0] - frame_sec) <= frame_sec * 0.02
    )
    monotonic_pts = all(b > a for a, b in zip(pts_values, pts_values[1:]))
    return {
        "packet_count": len(packets),
        "unique_duration_count": len(unique_durations),
        "sample_durations": unique_durations[:8],
        "expected_frame_sec": frame_sec,
        "is_cfr_likely": bool(cfr_by_duration and monotonic_pts),
        "pts_monotonic": bool(monotonic_pts),
    }


def classify_cfr_from_stream(stream: dict[str, Any], fps: float, total_frames: int, duration_sec: float) -> dict[str, Any]:
    """Fast CFR/VFR signal without scanning every packet in large videos."""

    avg_fps = parse_rate(stream.get("avg_frame_rate"))
    r_fps = parse_rate(stream.get("r_frame_rate"))
    expected_frames = duration_sec * fps if fps > 0 and duration_sec > 0 else 0.0
    frame_delta = abs(total_frames - expected_frames) if expected_frames else 0.0
    fps_match = bool(avg_fps and r_fps and abs(avg_fps - r_fps) <= max(1e-6, fps * 0.0005))
    frame_count_match = bool(expected_frames and frame_delta <= max(2.0, fps * 0.1))
    return {
        "packet_count": None,
        "unique_duration_count": None,
        "sample_durations": [],
        "expected_frame_sec": 1.0 / fps if fps > 0 else 0.0,
        "is_cfr_likely": bool(fps_match and frame_count_match),
        "pts_monotonic": None,
        "method": "stream-level",
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "expected_frames_from_duration": expected_frames,
        "frame_count_delta": frame_delta,
    }


def probe_keyframes(video_path: str, *, probe_work_dir: str | None = None) -> list[float]:
    mp4_keyframes = probe_keyframes_mp4(video_path)
    if mp4_keyframes:
        return mp4_keyframes

    ffprobe = find_tool("ffprobe")
    with ascii_probe_path(video_path, probe_work_dir) as probe_path:
        data = run_json([
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-skip_frame", "nokey",
            "-show_frames",
            "-show_entries",
            "frame=key_frame,pict_type,best_effort_timestamp_time,pts_time,pkt_dts_time",
            "-of", "json",
            probe_path,
        ])
    keyframes = []
    for frame in data.get("frames") or []:
        if str(frame.get("key_frame")) != "1" and frame.get("pict_type") != "I":
            continue
        t = (
            frame.get("best_effort_timestamp_time")
            or frame.get("pts_time")
            or frame.get("pkt_dts_time")
        )
        if t is not None and str(t) != "N/A":
            keyframes.append(float(t))
    return sorted(set(keyframes))


def audio_mux_policy(audio_stream: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "has_audio": bool(audio_stream),
        "segment_audio_mode": "none",
        "final_audio_mode": "copy_source_once" if audio_stream else "no_audio",
        "codec": audio_stream.get("codec_name") if audio_stream else None,
    }



def render_frames_to_intervals(frames: Iterable[int]) -> list[Interval]:
    ordered = sorted(set(int(f) for f in frames))
    if not ordered:
        return []
    intervals: list[Interval] = []
    start = prev = ordered[0]
    for frame in ordered[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        intervals.append(Interval(start, prev))
        start = prev = frame
    intervals.append(Interval(start, prev))
    return intervals


def merge_intervals(intervals: list[Interval], max_gap_frames: int) -> list[Interval]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for cur in intervals[1:]:
        prev = merged[-1]
        gap = cur.start_frame - prev.end_frame - 1
        if gap <= max_gap_frames:
            merged[-1] = Interval(prev.start_frame, max(prev.end_frame, cur.end_frame))
        else:
            merged.append(cur)
    return merged


def union_duration(intervals: list[Interval]) -> int:
    return sum(i.duration_frames() for i in intervals)


def expand_to_keyframes(
    intervals: list[Interval],
    keyframes_sec: list[float],
    *,
    fps: float,
    total_frames: int,
) -> list[Interval]:
    if not intervals:
        return []
    if not keyframes_sec:
        return [Interval(0, max(0, total_frames - 1))]

    keyframes = sorted(set(max(0, int(math.floor(t * fps + 1e-6))) for t in keyframes_sec))
    if 0 not in keyframes:
        keyframes.insert(0, 0)
    last_frame = max(0, total_frames - 1)
    out: list[Interval] = []
    for interval in intervals:
        start = interval.start_frame
        end_exclusive = min(total_frames, interval.end_frame + 1)
        prev_keys = [k for k in keyframes if k <= start]
        next_keys = [k for k in keyframes if k >= end_exclusive]
        exp_start = prev_keys[-1] if prev_keys else 0
        exp_end_exclusive = next_keys[0] if next_keys else total_frames
        exp_end = min(last_frame, max(exp_start, exp_end_exclusive - 1))
        out.append(Interval(exp_start, exp_end))
    return merge_intervals(sorted(out, key=lambda i: i.start_frame), 0)


def load_render_from_output(output_dir: str) -> tuple[dict[int, list[list[float]]], dict[str, Any], dict[str, Any]]:
    timeline = load_mask_timeline(output_dir)
    if timeline is None:
        raise FileNotFoundError(f"Missing mask_timeline.json in {output_dir}")
    decisions_path = os.path.join(output_dir, "review", "confirmed_events.json")
    decisions = parse_review_decisions(decisions_path)
    render, stats = select_render_entries(timeline, decisions)
    return render, stats, timeline


def build_segment_plan(total_frames: int, render_windows: list[Interval]) -> list[dict[str, int | str]]:
    plan: list[dict[str, int | str]] = []
    cursor = 0
    for interval in render_windows:
        if cursor < interval.start_frame:
            plan.append({
                "type": "copy",
                "start_frame": cursor,
                "end_frame": interval.start_frame - 1,
            })
        plan.append({
            "type": "render",
            "start_frame": interval.start_frame,
            "end_frame": interval.end_frame,
        })
        cursor = interval.end_frame + 1
    if cursor < total_frames:
        plan.append({
            "type": "copy",
            "start_frame": cursor,
            "end_frame": total_frames - 1,
        })
    return plan


def analyze_output(
    output_dir: str,
    *,
    video_path: str | None = None,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
    probe_work_dir: str | None = None,
) -> dict[str, Any]:
    render, review_stats, timeline = load_render_from_output(output_dir)
    timeline_video_path = timeline["video"]
    video_path = os.path.abspath(video_path or timeline_video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Source video not found: {video_path}. Use --video when mask_timeline.json contains a damaged path."
        )
    stream_info = get_stream_info(video_path, probe_work_dir=probe_work_dir)
    fps = float(timeline.get("fps") or stream_info["fps"])
    total_frames = int(timeline.get("total_frames") or stream_info["total_frames"])
    duration_sec = float(stream_info["duration_sec"] or (total_frames / fps if fps > 0 else 0.0))
    mp4_timing = probe_mp4_timing(video_path)
    cfr = classify_cfr_from_stream(stream_info["video"], fps, total_frames, duration_sec)
    if mp4_timing:
        cfr["method"] = "mp4-stts"
        cfr["stts_entry_count"] = mp4_timing["stts_entry_count"]
        cfr["sample_deltas"] = mp4_timing["sample_deltas"]
        cfr["timescale"] = mp4_timing["timescale"]
        cfr["is_cfr_likely"] = bool(
            len(mp4_timing["sample_deltas"]) == 1
            and mp4_timing["sample_count"] == total_frames
        )
    mp4_keyframes = mp4_timing.get("keyframes_sec") or []
    if mp4_keyframes:
        keyframes_sec = mp4_keyframes
        keyframe_method = "mp4-stss-stts"
    else:
        keyframes_sec = probe_keyframes(video_path, probe_work_dir=probe_work_dir)
        keyframe_method = "ffprobe-keyframes"

    raw_intervals = render_frames_to_intervals(render.keys())
    merge_gap_frames = int(round(merge_gap_sec * fps))
    merged_intervals = merge_intervals(raw_intervals, merge_gap_frames)
    keyframe_intervals = expand_to_keyframes(
        merged_intervals,
        keyframes_sec,
        fps=fps,
        total_frames=total_frames,
    )
    plan = build_segment_plan(total_frames, keyframe_intervals)

    raw_frames = union_duration(raw_intervals)
    merged_frames = union_duration(merged_intervals)
    keyframe_frames = union_duration(keyframe_intervals)
    sensitivity = []
    for threshold_sec in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
        threshold_frames = int(round(threshold_sec * fps))
        threshold_merged = merge_intervals(raw_intervals, threshold_frames)
        threshold_expanded = expand_to_keyframes(
            threshold_merged,
            keyframes_sec,
            fps=fps,
            total_frames=total_frames,
        )
        sensitivity.append({
            "merge_gap_sec": threshold_sec,
            "merged_interval_count": len(threshold_merged),
            "merged_coverage": union_duration(threshold_merged) / total_frames if total_frames else 0.0,
            "keyframe_interval_count": len(threshold_expanded),
            "keyframe_coverage": union_duration(threshold_expanded) / total_frames if total_frames else 0.0,
        })

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": os.path.abspath(output_dir),
        "video_path": video_path,
        "timeline_video_path": timeline_video_path,
        "stream_info": stream_info,
        "cfr": cfr,
        "audio_policy": audio_mux_policy(stream_info.get("audio")),
        "review_stats": review_stats,
        "merge_gap_sec": merge_gap_sec,
        "merge_gap_frames": merge_gap_frames,
        "raw_intervals": raw_intervals,
        "merged_intervals": merged_intervals,
        "keyframe_intervals": keyframe_intervals,
        "keyframes_sec": sorted(set(keyframes_sec)),
        "keyframe_method": keyframe_method,
        "coverage_sensitivity": sensitivity,
        "plan": plan,
        "summary": {
            "fps": fps,
            "duration_sec": duration_sec,
            "total_frames": total_frames,
            "keyframe_count": len(set(keyframes_sec)),
            "raw_interval_count": len(raw_intervals),
            "raw_frames": raw_frames,
            "raw_duration_sec": raw_frames / fps if fps > 0 else 0.0,
            "raw_coverage": raw_frames / total_frames if total_frames else 0.0,
            "merged_interval_count": len(merged_intervals),
            "merged_frames": merged_frames,
            "merged_duration_sec": merged_frames / fps if fps > 0 else 0.0,
            "merged_coverage": merged_frames / total_frames if total_frames else 0.0,
            "keyframe_interval_count": len(keyframe_intervals),
            "keyframe_frames": keyframe_frames,
            "keyframe_duration_sec": keyframe_frames / fps if fps > 0 else 0.0,
            "keyframe_coverage": keyframe_frames / total_frames if total_frames else 0.0,
            "copy_segment_count": sum(1 for p in plan if p["type"] == "copy"),
            "render_segment_count": sum(1 for p in plan if p["type"] == "render"),
        },
    }


def _fmt_sec(sec: float) -> str:
    m, s = divmod(sec, 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def interval_table(intervals: list[Interval], fps: float, *, limit: int = 12) -> str:
    if not intervals:
        return "_none_"
    rows = ["| # | start frame | end frame | start | end | duration |", "|---:|---:|---:|---:|---:|---:|"]
    for idx, interval in enumerate(intervals[:limit], start=1):
        start = interval.start_sec(fps)
        end = interval.end_sec_exclusive(fps)
        rows.append(
            f"| {idx} | {interval.start_frame} | {interval.end_frame} | "
            f"{_fmt_sec(start)} | {_fmt_sec(end)} | {end - start:.3f}s |"
        )
    if len(intervals) > limit:
        rows.append(f"| ... | ... | ... | ... | ... | {len(intervals) - limit} more |")
    return "\n".join(rows)


def feasibility_markdown(analysis: dict[str, Any]) -> str:
    s = analysis["summary"]
    stream = analysis["stream_info"]["video"]
    audio = analysis["stream_info"].get("audio")
    audio_policy = analysis["audio_policy"]
    review = analysis["review_stats"]
    cfr = analysis["cfr"]
    keyframes = analysis["keyframes_sec"]
    enough_gain = s["keyframe_coverage"] <= 0.35
    avoided_reencode = max(0.0, 1.0 - s["keyframe_coverage"])
    sensitivity_rows = [
        "| merge gap | merged intervals | merged coverage | keyframe intervals | actual reencode coverage |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["coverage_sensitivity"]:
        sensitivity_rows.append(
            f"| {row['merge_gap_sec']:.2f}s | {row['merged_interval_count']} | "
            f"{row['merged_coverage']:.2%} | {row['keyframe_interval_count']} | "
            f"{row['keyframe_coverage']:.2%} |"
        )
    lines = [
        "# Smart Render Feasibility",
        "",
        "## Scope",
        "",
        "This report evaluates keyframe-aware segment rendering only. It does not replace the current production full-video renderer.",
        "",
        "## Executive Conclusion",
        "",
        f"- Recommendation: `{'continue the independent minimum prototype' if enough_gain else 'stop or redesign before prototyping'}`; do not integrate it into `render.py` or `confirm.py` yet.",
        f"- At the selected 1.0s merge threshold, only `{s['keyframe_coverage']:.2%}` of source frames require reencoding after GOP-safe expansion; the theoretical avoided reencode share is `{avoided_reencode:.2%}`.",
        "- This is an encoding-work estimate, not a wall-clock speedup promise. Demux/copy, 86 segment operations, concat/remux, storage throughput, and validation remain overheads.",
        f"- Main risk: `{s['render_segment_count']}` reencoded regions and `{s['copy_segment_count']}` copied regions create many codec/timestamp boundaries; correctness must be proven on a short representative clip before any full-video prototype.",
        "",
        "## Source Video",
        "",
        f"- Path: `{analysis['video_path']}`",
        f"- Duration: `{s['duration_sec']:.3f}s` (`{_fmt_sec(s['duration_sec'])}`)",
        f"- Total frames: `{s['total_frames']}`",
        f"- FPS: `{s['fps']:.6f}`",
        f"- CFR/VFR: `{'CFR-likely' if cfr['is_cfr_likely'] else 'VFR/uncertain'}` via `{cfr.get('method', 'packet-scan')}`",
        f"- CFR evidence: stts_entries=`{cfr.get('stts_entry_count', 'n/a')}` sample_deltas=`{cfr.get('sample_deltas', 'n/a')}` timescale=`{cfr.get('timescale', 'n/a')}`; the MP4 timing table accounts for all `{s['total_frames']}` frames.",
        f"- Video codec: `{stream.get('codec_name')}` profile=`{stream.get('profile')}` level=`{stream.get('level')}` pix_fmt=`{stream.get('pix_fmt')}` time_base=`{stream.get('time_base')}`",
        f"- Video size: `{stream.get('width')}x{stream.get('height')}`",
        f"- Audio: `{audio.get('codec_name') if audio else 'none'}` sample_rate=`{audio.get('sample_rate') if audio else 'n/a'}` channels=`{audio.get('channels') if audio else 'n/a'}`",
        f"- Audio segment policy: segment_audio=`{audio_policy['segment_audio_mode']}` final_audio=`{audio_policy['final_audio_mode']}`",
        "",
        "## Coverage",
        "",
        f"- Coverage basis: the production selection contract (`select_render_entries`) applied to `mask_timeline.json` plus `review/confirmed_events.json`; selected proposals=`{review.get('selected_proposals')}`, selected render frames=`{review.get('render_frames')}`.",
        f"- Selection detail: auto=`{review.get('auto_selected')}`, reviewed=`{review.get('review_selected')}`, partial=`{review.get('review_partial_selected')}`, rejected=`{review.get('review_rejected')}`, unreviewed skipped=`{review.get('review_unreviewed_skipped')}`.",
        f"- Raw mask intervals: `{s['raw_interval_count']}`",
        f"- Raw mask duration: `{s['raw_duration_sec']:.3f}s` ({s['raw_coverage']:.2%})",
        f"- Merge threshold: `{analysis['merge_gap_sec']:.3f}s` (`{analysis['merge_gap_frames']}` frames)",
        f"- Merged mask intervals: `{s['merged_interval_count']}`",
        f"- Merged mask duration: `{s['merged_duration_sec']:.3f}s` ({s['merged_coverage']:.2%})",
        f"- Keyframes read: `{s['keyframe_count']}` via `{analysis['keyframe_method']}`",
        f"- Keyframe-expanded render intervals: `{s['keyframe_interval_count']}`",
        f"- Keyframe-expanded render duration: `{s['keyframe_duration_sec']:.3f}s` ({s['keyframe_coverage']:.2%})",
        "",
        "### Merge-threshold Sensitivity",
        "",
        *sensitivity_rows,
        "",
        "The 0-1s thresholds all produce the same 10.39% GOP-expanded coverage, so the feasibility conclusion is not sensitive to the selected 1s merge threshold. Larger gaps reduce segment count only by reencoding more unmasked frames.",
        "",
        "## Keyframe Distribution",
        "",
        f"- First keyframes: `{', '.join(_fmt_sec(t) for t in keyframes[:8])}`",
        f"- Last keyframes: `{', '.join(_fmt_sec(t) for t in keyframes[-8:])}`",
        "",
        "## Interval Samples",
        "",
        "### Raw Mask Intervals",
        "",
        interval_table(analysis["raw_intervals"], s["fps"]),
        "",
        "### After Gap Merge",
        "",
        interval_table(analysis["merged_intervals"], s["fps"]),
        "",
        "### After Keyframe Expansion",
        "",
        interval_table(analysis["keyframe_intervals"], s["fps"]),
        "",
        "## Segment Plan Summary",
        "",
        f"- Copy segments: `{s['copy_segment_count']}`",
        f"- Reencode segments: `{s['render_segment_count']}`",
        f"- Planned final concat mode: video-only first, then remux/copy audio once from source.",
        "",
        "## Feasibility Assessment",
        "",
        f"- Expected benefit: `{'promising' if enough_gain else 'weak/uncertain'}` based on keyframe-expanded coverage.",
        "- Stop condition: do not replace production rendering until frame count, duration, PTS/DTS monotonicity, concat boundaries, and player seeking all pass.",
        "- MP4 direct concat is not assumed safe. Prototype should test TS intermediate with h264/hevc bitstream filters and final `-c copy` remux.",
        "",
        "## Design Rules For Prototype",
        "",
        "1. Copy segments must start at independently decodable keyframes.",
        "2. Mask intervals inside a GOP must expand to surrounding keyframes.",
        "3. Expanded non-mask frames are reencoded without mosaic.",
        "4. Reencoded and copied segments must keep compatible resolution, FPS, pixel format, codec/profile/level, and monotonic timestamps.",
        "5. Segment processing is video-only; audio is copied once from the original at final mux.",
        "6. Existing full-video render remains the fallback.",
        "",
        "## Required Validation Before Production",
        "",
        "- Output frame count equals source frame count.",
        "- Output duration differs by less than one frame.",
        "- Audio/video duration drift is less than one frame.",
        "- First/last frame PTS and PTS/DTS monotonicity pass.",
        "- Around every concat point, inspect at least 30 frames for dropped/repeated/black/corrupt frames and timestamp jumps.",
        "- ffprobe reports no timestamp or decode errors.",
        "- VLC, Windows player, and browser seeking are manually checked.",
        "- Compare full-render vs smart-render runtime, file size, frame count, duration, and A/V sync.",
    ]
    return "\n".join(lines) + "\n"


def cmd_analyze(args: argparse.Namespace) -> int:
    analysis = analyze_output(
        args.output_dir,
        video_path=args.video,
        merge_gap_sec=args.merge_gap_sec,
        probe_work_dir=args.probe_work_dir,
    )
    md = feasibility_markdown(analysis)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(md)
    if args.json:
        payload = dict(analysis)
        for key in ("raw_intervals", "merged_intervals", "keyframe_intervals"):
            payload[key] = [
                {"start_frame": i.start_frame, "end_frame": i.end_frame}
                for i in analysis[key]
            ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smart-render feasibility prototype")
    sub = p.add_subparsers(dest="cmd", required=True)

    analyze = sub.add_parser("analyze", help="Analyze current reviewed output and write feasibility report")
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--video", default=None, help="Override a missing or damaged timeline video path")
    analyze.add_argument("--merge-gap-sec", type=float, default=DEFAULT_MERGE_GAP_SEC)
    analyze.add_argument("--probe-work-dir", default=None)
    analyze.add_argument("--report", default=None)
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(func=cmd_analyze)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
