"""YOLO wrapper: frame in -> bbox + conf out."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from core.event import Detection


def resolve_device(device: str) -> str:
    import torch
    if device != "cpu" and not torch.cuda.is_available():
        return "cpu"
    return device


class FaceDetector:
    def __init__(self, model_path: str, device: str = "0", conf: float = 0.35, imgsz: int = 1280):
        self.device = resolve_device(device)
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_path)

    def detect_frame(self, frame: np.ndarray, frame_idx: int = 0, t: float = 0.0) -> list[Detection]:
        results = self.model.predict(
            source=frame, conf=self.conf, imgsz=self.imgsz,
            device=self.device, half=self.device != "cpu", verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        r = results[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
        out: list[Detection] = []
        for j in range(len(xyxy)):
            c = float(confs[j]) if confs is not None else 1.0
            out.append(Detection(
                frame=frame_idx, t=t,
                bbox=[round(float(v), 1) for v in xyxy[j].tolist()],
                conf=round(c, 4),
            ))
        return out

    def detect_video_sparse(self, video_path: str, fps: float, total_frames: int, interval: int = 5) -> list[Detection]:
        interval = max(1, interval)
        results = self.model.predict(
            source=video_path, stream=True, conf=self.conf, imgsz=self.imgsz,
            device=self.device, half=self.device != "cpu", vid_stride=interval, verbose=False,
        )
        n_steps = (total_frames + interval - 1) // interval if total_frames > 0 else None
        flat: list[Detection] = []
        for proc_idx, r in enumerate(tqdm(results, total=n_steps, unit="f", desc="detect")):
            frame_idx = proc_idx * interval
            t_sec = round(frame_idx / (fps or 25.0), 3)
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
            for j in range(len(xyxy)):
                c = float(confs[j]) if confs is not None else 1.0
                flat.append(Detection(
                    frame=frame_idx, t=t_sec,
                    bbox=[round(float(v), 1) for v in xyxy[j].tolist()],
                    conf=round(c, 4),
                ))
        return flat

    def close(self) -> None:
        del self.model
