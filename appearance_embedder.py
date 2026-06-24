"""Face appearance embeddings for cross-track identity stitching."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

DEFAULT_ARCFACE_MODEL = os.path.join("models", "arcface_r50.onnx")
ARCFACE_INPUT_SIZE = (112, 112)
EMBEDDING_DIM = 512


class AppearanceEmbedder:
    """ArcFace ONNX when available; HSV histogram fallback otherwise."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = os.path.abspath(model_path or DEFAULT_ARCFACE_MODEL)
        self.method = "hsv_histogram"
        self._session = None
        self._input_name: str | None = None
        self._try_load_arcface()

    def _try_load_arcface(self) -> None:
        if not os.path.isfile(self.model_path):
            return
        try:
            import onnxruntime as ort
        except ImportError:
            return
        try:
            self._session = ort.InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self.method = "arcface"
        except Exception:
            self._session = None

    @property
    def is_arcface(self) -> bool:
        return self._session is not None

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(EMBEDDING_DIM if self.is_arcface else 96, dtype=np.float32)
        if self.is_arcface:
            return self._embed_arcface(crop_bgr)
        return self._embed_hsv(crop_bgr)

    def _embed_arcface(self, crop_bgr: np.ndarray) -> np.ndarray:
        assert self._session is not None and self._input_name is not None
        face = cv2.resize(crop_bgr, ARCFACE_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))[np.newaxis, ...]
        out = self._session.run(None, {self._input_name: face})[0]
        vec = out.flatten().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _embed_hsv(crop_bgr: np.ndarray, bins: int = 32) -> np.ndarray:
        small = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        parts = []
        for ch, hi in ((0, 180), (1, 256), (2, 256)):
            h = cv2.calcHist([hsv], [ch], None, [bins], [0, hi]).astype(np.float32).flatten()
            total = float(h.sum())
            if total > 0:
                h /= total
            parts.append(h)
        return np.concatenate(parts)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        if self.is_arcface:
            return float(max(0.0, min(1.0, np.dot(a, b))))
        bc = float(np.sum(np.sqrt(a * b)))
        return max(0.0, min(1.0, bc))

    def info(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "model_path": self.model_path if self.is_arcface else None,
            "embedding_dim": EMBEDDING_DIM if self.is_arcface else 96,
        }
