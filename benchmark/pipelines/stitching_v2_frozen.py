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
