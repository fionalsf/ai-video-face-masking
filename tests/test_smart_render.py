#!/usr/bin/env python3
"""Minimal tests for smart_render planning.

These tests intentionally focus on frame math and safety boundaries.  Optional
ffmpeg integration fixtures can be added later without changing production
rendering.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smart_render import (
    Interval,
    audio_mux_policy,
    build_segment_plan,
    expand_to_keyframes,
    merge_intervals,
    render_frames_to_intervals,
    union_duration,
)
from tools.benchmarks.smart_render_prototype import crop_plan, expected_mux_frames, localize_render


class SmartRenderPlanTests(unittest.TestCase):
    def test_interval_inside_gop_expands_to_surrounding_keyframes(self):
        fps = 30.0
        intervals = [Interval(45, 60)]
        keyframes = [0.0, 2.0, 4.0]
        expanded = expand_to_keyframes(intervals, keyframes, fps=fps, total_frames=180)
        self.assertEqual(expanded, [Interval(0, 119)])

    def test_multiple_adjacent_intervals_merge_before_keyframe_expansion(self):
        intervals = [Interval(30, 35), Interval(38, 44), Interval(80, 90)]
        merged = merge_intervals(intervals, max_gap_frames=5)
        self.assertEqual(merged, [Interval(30, 44), Interval(80, 90)])

    def test_far_apart_intervals_stay_separate(self):
        intervals = [Interval(30, 35), Interval(120, 140)]
        merged = merge_intervals(intervals, max_gap_frames=10)
        self.assertEqual(merged, intervals)

    def test_near_video_start_uses_zero_as_safe_start(self):
        expanded = expand_to_keyframes(
            [Interval(3, 12)],
            [0.0, 1.0, 2.0],
            fps=30.0,
            total_frames=90,
        )
        self.assertEqual(expanded, [Interval(0, 29)])

    def test_near_video_end_uses_total_frames_as_safe_end(self):
        expanded = expand_to_keyframes(
            [Interval(82, 88)],
            [0.0, 1.0, 2.0],
            fps=30.0,
            total_frames=90,
        )
        self.assertEqual(expanded, [Interval(60, 89)])

    def test_sparse_render_frames_build_expected_intervals(self):
        frames = [0, 1, 2, 10, 11, 50]
        self.assertEqual(
            render_frames_to_intervals(frames),
            [Interval(0, 2), Interval(10, 11), Interval(50, 50)],
        )

    def test_segment_plan_covers_whole_video_without_overlap(self):
        plan = build_segment_plan(100, [Interval(10, 19), Interval(40, 49)])
        self.assertEqual(
            plan,
            [
                {"type": "copy", "start_frame": 0, "end_frame": 9},
                {"type": "render", "start_frame": 10, "end_frame": 19},
                {"type": "copy", "start_frame": 20, "end_frame": 39},
                {"type": "render", "start_frame": 40, "end_frame": 49},
                {"type": "copy", "start_frame": 50, "end_frame": 99},
            ],
        )

    def test_2997_fps_frame_math_remains_frame_based(self):
        fps = 30000 / 1001
        expanded = expand_to_keyframes(
            [Interval(45, 60)],
            [0.0, 2.002, 4.004],
            fps=fps,
            total_frames=300,
        )
        self.assertEqual(expanded[0].start_frame, 0)
        self.assertGreaterEqual(expanded[0].end_frame, 118)
        self.assertTrue(math.isclose(fps, 29.97002997002997))

    def test_5994_fps_frame_math_remains_frame_based(self):
        fps = 60000 / 1001
        expanded = expand_to_keyframes(
            [Interval(150, 170)],
            [0.0, 2.002, 4.004],
            fps=fps,
            total_frames=360,
        )
        self.assertLessEqual(expanded[0].start_frame, 120)
        self.assertGreaterEqual(expanded[0].end_frame, 239)
        self.assertTrue(math.isclose(fps, 59.94005994005994))

    def test_no_audio_policy_keeps_segments_video_only(self):
        policy = audio_mux_policy(None)
        self.assertEqual(policy["segment_audio_mode"], "none")
        self.assertEqual(policy["final_audio_mode"], "no_audio")

    def test_aac_audio_policy_copies_source_once_at_final_mux(self):
        policy = audio_mux_policy({"codec_name": "aac", "sample_rate": "48000", "channels": 2})
        self.assertEqual(policy["segment_audio_mode"], "none")
        self.assertEqual(policy["final_audio_mode"], "copy_source_once")
        self.assertEqual(policy["codec"], "aac")

    def test_union_duration_is_inclusive_frame_count(self):
        self.assertEqual(union_duration([Interval(0, 0), Interval(10, 19)]), 11)

    def test_crop_plan_keeps_global_frame_coordinates(self):
        plan = [
            {"type": "copy", "start_frame": 0, "end_frame": 19},
            {"type": "render", "start_frame": 20, "end_frame": 29},
            {"type": "copy", "start_frame": 30, "end_frame": 49},
        ]
        self.assertEqual(
            crop_plan(plan, 10, 40),
            [
                {"type": "copy", "start_frame": 10, "end_frame": 19},
                {"type": "render", "start_frame": 20, "end_frame": 29},
                {"type": "copy", "start_frame": 30, "end_frame": 39},
            ],
        )

    def test_localize_render_translates_only_frames_inside_segment(self):
        render = {9: [[0, 0, 1, 1]], 10: [[1, 1, 2, 2]], 12: [[2, 2, 3, 3]], 13: []}
        self.assertEqual(
            localize_render(render, 10, 12),
            {0: [[1, 1, 2, 2]], 2: [[2, 2, 3, 3]]},
        )

    def test_expected_mux_frames_accounts_for_short_source_audio(self):
        fps = 30000 / 1001
        self.assertEqual(
            expected_mux_frames(66697, fps, 0.0, 2225.365333, no_audio=False),
            66694,
        )
        self.assertEqual(
            expected_mux_frames(66697, fps, 0.0, 2225.365333, no_audio=True),
            66697,
        )


if __name__ == "__main__":
    unittest.main()
