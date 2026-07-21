"""Detection inference backends for sparse face detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any

import cv2
import numpy as np


_DLL_DIR_HANDLES: list[Any] = []


def _add_torch_cuda_dll_dir() -> None:
    """Let ONNX Runtime find the CUDA/cuDNN DLLs bundled with PyTorch on Windows."""
    if os.name != "nt":
        return
    try:
        import torch
    except Exception:
        return
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(torch_lib):
        return
    if hasattr(os, "add_dll_directory"):
        handle = os.add_dll_directory(torch_lib)
        _DLL_DIR_HANDLES.append(handle)
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")


def _add_tensorrt_dll_dir() -> None:
    """Let ONNX Runtime find TensorRT DLLs installed by the pip TensorRT package."""
    if os.name != "nt":
        return
    try:
        import tensorrt  # noqa: F401
    except Exception:
        return
    import site

    candidates: list[str] = []
    for base in site.getsitepackages():
        candidates.append(os.path.join(base, "tensorrt_libs"))
    for dll_dir in candidates:
        if not os.path.isdir(dll_dir):
            continue
        if hasattr(os, "add_dll_directory"):
            handle = os.add_dll_directory(dll_dir)
            _DLL_DIR_HANDLES.append(handle)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")


class DetectionBackend(ABC):
    name = "base"

    @abstractmethod
    def predict_batch(self, frames: list[np.ndarray]) -> list[list[tuple[list[float], float]]]:
        """Return per-frame detections as [(xyxy, conf), ...]."""

    def close(self) -> None:
        pass


class TorchYoloBackend(DetectionBackend):
    name = "torch"

    def __init__(self, model_path: str, device: str, conf: float, imgsz: int):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def predict_batch(self, frames: list[np.ndarray]) -> list[list[tuple[list[float], float]]]:
        results = self.model.predict(
            source=frames,
            stream=False,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=self.device != "cpu",
            verbose=False,
        )
        out: list[list[tuple[list[float], float]]] = []
        for r in results:
            frame_out: list[tuple[list[float], float]] = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
                for j in range(len(xyxy)):
                    c = float(confs[j]) if confs is not None else 1.0
                    frame_out.append((xyxy[j].tolist(), c))
            out.append(frame_out)
        return out

    def close(self) -> None:
        del self.model


class OnnxRuntimeYoloBackend(DetectionBackend):
    name = "onnx"

    def __init__(self, model_path: str, device: str, conf: float, imgsz: int, provider: str | None = None):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        _add_torch_cuda_dll_dir()
        _add_tensorrt_dll_dir()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime/onnxruntime-gpu is not installed in this Python environment") from exc

        providers = ort.get_available_providers()
        requested = provider or ("CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider")
        if requested not in providers:
            raise RuntimeError(
                f"ONNX Runtime provider {requested} is unavailable; available providers: {providers}"
            )
        provider_spec: list[Any]
        cuda_provider = (
            "CUDAExecutionProvider",
            {
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
            },
        )
        if requested == "CUDAExecutionProvider":
            provider_spec = [cuda_provider]
        elif requested == "TensorrtExecutionProvider":
            trt_fp16 = os.environ.get("FACE_MASK_TRT_FP16", "False")
            trt_precision = "fp16" if trt_fp16.lower() == "true" else "fp32"
            provider_spec = [
                (
                    requested,
                    {
                        "trt_fp16_enable": "True" if trt_precision == "fp16" else "False",
                        "trt_engine_cache_enable": "True",
                        "trt_engine_cache_path": os.path.join(
                            os.path.dirname(model_path), ".trt_cache", trt_precision
                        ),
                    },
                ),
                cuda_provider,
            ]
        else:
            provider_spec = [requested]
        if requested != "CPUExecutionProvider":
            provider_spec.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=provider_spec)
        if hasattr(self.session, "disable_fallback"):
            self.session.disable_fallback()
        active_providers = self.session.get_providers()
        if requested != "CPUExecutionProvider" and requested not in active_providers:
            raise RuntimeError(
                f"ONNX Runtime requested {requested}, but active providers are {active_providers}. "
                "GPU/TensorRT dependencies are likely missing from PATH."
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.conf = conf
        self.imgsz = imgsz
        self.provider = active_providers[0] if active_providers else requested

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        h, w = image.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - nw) / 2.0
        pad_y = (self.imgsz - nh) / 2.0
        canvas[int(round(pad_y)):int(round(pad_y)) + nh, int(round(pad_x)):int(round(pad_x)) + nw] = resized
        chw = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return chw, scale, pad_x, pad_y

    def predict_batch(self, frames: list[np.ndarray]) -> list[list[tuple[list[float], float]]]:
        tensors = []
        transforms = []
        for frame in frames:
            tensor, scale, pad_x, pad_y = self._letterbox(frame)
            tensors.append(tensor)
            transforms.append((frame.shape[1], frame.shape[0], scale, pad_x, pad_y))
        batch = np.ascontiguousarray(np.stack(tensors, axis=0))
        outputs = self.session.run(self.output_names, {self.input_name: batch})
        return _decode_yolo_outputs(outputs[0], transforms, self.conf)

    def close(self) -> None:
        if hasattr(self, "session"):
            del self.session


def _decode_yolo_outputs(
    raw: Any,
    transforms: list[tuple[int, int, float, float, float]],
    conf: float,
) -> list[list[tuple[list[float], float]]]:
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[1] < arr.shape[2]:
        arr = np.transpose(arr, (0, 2, 1))
    if arr.ndim != 3:
        raise RuntimeError(f"Unsupported ONNX output shape: {arr.shape}")

    per_frame: list[list[tuple[list[float], float]]] = []
    for batch_idx, preds in enumerate(arr):
        frame_w, frame_h, scale, pad_x, pad_y = transforms[batch_idx]
        if preds.shape[1] < 5:
            raise RuntimeError(f"Unsupported ONNX prediction shape: {preds.shape}")
        if preds.shape[1] == 5:
            scores = preds[:, 4]
        else:
            class_scores = preds[:, 4:]
            scores = class_scores.max(axis=1)
        keep = scores >= conf
        preds = preds[keep]
        scores = scores[keep]
        if len(preds) == 0:
            per_frame.append([])
            continue
        boxes = preds[:, :4].astype(np.float32)
        boxes = _xywh_to_xyxy(boxes)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h - 1)
        keep_idx = _nms(boxes, scores.astype(np.float32), iou_thresh=0.70)
        frame_out = [(boxes[i].tolist(), float(scores[i])) for i in keep_idx]
        per_frame.append(frame_out)
    return per_frame


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


def create_detection_backend(
    backend: str,
    model_path: str,
    onnx_model_path: str | None,
    device: str,
    conf: float,
    imgsz: int,
) -> DetectionBackend:
    backend = (backend or "torch").lower()
    if backend == "torch":
        return TorchYoloBackend(model_path, device, conf, imgsz)
    if backend == "onnx":
        return OnnxRuntimeYoloBackend(onnx_model_path or model_path, device, conf, imgsz)
    if backend == "tensorrt":
        engine_path = onnx_model_path or model_path
        ort_backend = OnnxRuntimeYoloBackend(
            engine_path,
            device,
            conf,
            imgsz,
            provider="TensorrtExecutionProvider",
        )
        ort_backend.name = "tensorrt"
        return ort_backend
    if backend == "auto":
        # ONNX Runtime CUDA is available on some machines but not necessarily faster
        # or numerically identical, so auto keeps the proven production backend.
        return TorchYoloBackend(model_path, device, conf, imgsz)
    raise ValueError(f"Unknown inference backend: {backend}")
