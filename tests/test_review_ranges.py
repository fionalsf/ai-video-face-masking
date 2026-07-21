#!/usr/bin/env python3
"""Regression tests for range-based review decisions."""

from __future__ import annotations

import unittest

from confirm import parse_review_decisions
from mask_timeline import select_render_entries


class ReviewRangeTests(unittest.TestCase):
    def test_nested_review_document_preserves_ranges(self) -> None:
        decisions = parse_review_decisions({
            "events": [{
                "event_id": "proposal_1",
                "status": "accepted_ranges",
                "ranges": [[1.0, 1.5], [2.0, 2.25]],
            }],
        })

        self.assertEqual(
            decisions["proposal_1"],
            {"status": "accepted_ranges", "ranges": [[1.0, 1.5], [2.0, 2.25]]},
        )

    def test_only_frames_inside_accepted_ranges_are_rendered(self) -> None:
        timeline = {
            "proposals": [{
                "proposal_id": "proposal_1",
                "source_tier": "review",
                "start_frame": 0,
                "end_frame": 90,
            }],
            "entries": [
                {"proposal_id": "proposal_1", "frame": 15, "timestamp": 0.5, "bbox": [1, 2, 3, 4]},
                {"proposal_id": "proposal_1", "frame": 30, "timestamp": 1.0, "bbox": [1, 2, 3, 4]},
                {"proposal_id": "proposal_1", "frame": 45, "timestamp": 1.5, "bbox": [1, 2, 3, 4]},
                {"proposal_id": "proposal_1", "frame": 60, "timestamp": 2.0, "bbox": [1, 2, 3, 4]},
            ],
        }
        decisions = {
            "proposal_1": {"status": "accepted_ranges", "ranges": [[1.0, 1.5]]},
        }

        render, stats = select_render_entries(timeline, decisions)

        self.assertEqual(sorted(render), [30, 45])
        self.assertEqual(stats["review_range_selected"], 1)
        self.assertEqual(stats["render_frames"], 2)


if __name__ == "__main__":
    unittest.main()
