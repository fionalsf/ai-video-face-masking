"""Drawing helpers for review previews."""

from __future__ import annotations

import cv2
import numpy as np


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        conf = det.get("confidence", det.get("conf", 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return vis
