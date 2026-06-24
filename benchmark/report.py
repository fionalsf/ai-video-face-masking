"""Generate markdown benchmark report from metrics.json."""

from __future__ import annotations

from typing import Any


def render_report(metrics: dict[str, Any]) -> str:
    video = metrics.get("video", "")
    gt_path = metrics.get("ground_truth", "")
    objective = metrics.get("objective", "minimize masking error + review cost")
    lines = [
        "# Stitching Benchmark Report",
        "",
        f"- **Video**: `{video}`",
        f"- **Ground truth**: `{gt_path}`",
        f"- **Objective**: {objective}",
        f"- **Clips evaluated**: {metrics.get('clip_count', 0)}",
        "",
        "## Summary",
        "",
        "| Metric | v1_greedy | v2_graph | Δ (v2 - v1) |",
        "|--------|-----------|----------|-------------|",
    ]

    v1 = metrics.get("variants", {}).get("v1_greedy", {}).get("aggregate", {})
    v2 = metrics.get("variants", {}).get("v2_graph", {}).get("aggregate", {})
    delta = metrics.get("delta", {})

    rows = [
        ("False Mask (sec)", "total_false_mask_sec"),
        ("Miss Mask (sec)", "total_miss_mask_sec"),
        ("Over Merge", "total_over_merge"),
        ("Over Split", "total_over_split"),
        ("Events", "total_events"),
        ("Review Events", "total_review_events"),
        ("Est. Review (min)", "total_estimated_review_minutes"),
        ("Objective Score ↓", "mean_objective_score"),
    ]
    for label, key in rows:
        a = v1.get(key, 0)
        b = v2.get(key, 0)
        d = delta.get(key, b - a if isinstance(a, (int, float)) else "")
        lines.append(f"| {label} | {a} | {b} | {d} |")

    winner = metrics.get("recommended_variant")
    lines.extend([
        "",
        f"**Recommended variant (lower objective score)**: `{winner}`",
        "",
        "## Error Definitions",
        "",
        "- **False Mask**: predicted mask where GT says no mask",
        "- **Miss Mask**: GT mask not covered by prediction",
        "- **Over Merge**: one predicted event spans multiple GT events",
        "- **Over Split**: one GT event split across multiple predicted events",
        "",
        "## Notes",
        "",
        metrics.get("notes", "- Stitching parameters frozen during benchmark period."),
        "",
    ])
    return "\n".join(lines)
