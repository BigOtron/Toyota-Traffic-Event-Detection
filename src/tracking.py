"""Detection + tracking: YOLOv8 + ByteTrack on every STRIDE-th frame.

Output is a DataFrame with the same columns as tools/track.py CSVs:
time, track_id, class, x, y, w, h  (x, y = bottom-centre of the box, video pixels).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "yolov8s.pt"
STRIDE = 3  # 29.97 fps -> ~10 fps
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def _new_model():
    """A fresh YOLO model for every video.

    Reusing one model and only resetting `model.predictor` is NOT enough in
    Ultralytics 8.3.x: each `track(persist=True)` on a new predictor registers
    the tracker callbacks again, so on the N-th video the tracker is updated
    N times per frame and tracks break. Loading yolov8s takes ~1 s.
    """
    import torch
    from ultralytics import YOLO
    torch.manual_seed(0)
    return YOLO(str(WEIGHTS))


ALIGN_TIMES = (0.0, 3.0, 6.0, 9.0, 12.0)  # seconds; frames kept for camera alignment


def track_video(video_path: str, stride: int = STRIDE, max_seconds: float | None = None,
                progress=None) -> tuple[pd.DataFrame, list[np.ndarray], float]:
    """Returns (tracks, frames_for_alignment, fps).

    max_seconds: stop after this much video (used by the web demo).
    progress: optional callback(fraction_done) for the web demo.
    """
    model = _new_model()  # fresh model = fresh tracker, no leftover callbacks
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    if max_seconds is not None:
        n_total = min(n_total, int(max_seconds * fps))
    keep = {int(round(t * fps / stride)) * stride for t in ALIGN_TIMES}
    rows, frames, i = [], [], 0
    while i < n_total and cap.grab():
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if i in keep:
                frames.append(frame.copy())
            r = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=list(CLASSES),
                            verbose=False)[0]  # default imgsz=640, same as the tuning CSVs
            if r.boxes.id is not None:
                for (x, y, w, h), tid, c in zip(r.boxes.xywh.tolist(), r.boxes.id.tolist(), r.boxes.cls.tolist()):
                    rows.append([round(i / fps, 2), int(tid), CLASSES[int(c)], x, y + h / 2, w, h])
            if progress is not None and i % (stride * 30) == 0:
                progress(i / n_total)
        i += 1
    cap.release()
    df = pd.DataFrame(rows, columns=["time", "track_id", "class", "x", "y", "w", "h"])
    return df, frames, fps
