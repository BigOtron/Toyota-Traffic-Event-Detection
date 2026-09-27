"""Full Part A pipeline in one call; used by solution.py, the renderer and the web demo."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .jaywalking import detect_jaywalking, on_road_mask, person_points
from .scene import Scene
from .tracking import track_video


@dataclass
class Result:
    events: list[list]
    tracks: pd.DataFrame            # all detections, video pixels
    persons: pd.DataFrame           # person points with zone features and `on_road`
    H: np.ndarray                   # video pixels -> reference pixels
    inliers: int
    fps: float
    duration: float
    size: tuple[int, int] = (0, 0)  # (width, height)
    first_frame: np.ndarray | None = field(default=None, repr=False)


def analyze(video_path: str, max_seconds: float | None = None, progress=None) -> Result:
    scene = Scene()
    tracks, frames, fps = track_video(video_path, max_seconds=max_seconds, progress=progress)
    cap = cv2.VideoCapture(video_path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    duration = n / fps if n > 0 else float(tracks.time.max() if len(tracks) else 0.0)
    if max_seconds is not None:
        duration = min(duration, max_seconds)
    if not frames:
        return Result([], tracks, pd.DataFrame(), np.eye(3), 0, fps, duration, size)

    H, inliers = scene.best_homography(frames)
    persons = person_points(tracks, H, scene)
    persons["on_road"] = on_road_mask(persons) if len(persons) else []
    events = detect_jaywalking(persons, duration)
    return Result(events, tracks, persons, H, inliers, fps, duration, size, frames[0])


def save_debug(res: Result, name: str, out: Path = Path("debug")) -> None:
    out.mkdir(exist_ok=True)
    res.tracks.to_csv(out / f"{name}.csv", index=False)
    if res.first_frame is not None:
        cv2.imwrite(str(out / f"{name}.png"), res.first_frame)
    (out / f"{name}_H.txt").write_text(f"inliers={res.inliers}\n{res.H.tolist()}\n")


def analyze_from_tracks(video_path: str, tracks: pd.DataFrame) -> Result:
    """Same as analyze(), but reuses saved tracks (CSV from tools/track.py or debug/) — no YOLO."""
    from .tracking import ALIGN_TIMES

    scene = Scene()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = []
    for t in ALIGN_TIMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps / 3)) * 3)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    duration = n / fps if n > 0 else float(tracks.time.max())
    H, inliers = scene.best_homography(frames)
    persons = person_points(tracks, H, scene)
    persons["on_road"] = on_road_mask(persons) if len(persons) else []
    events = detect_jaywalking(persons, duration)
    return Result(events, tracks, persons, H, inliers, fps, duration, size, frames[0] if frames else None)
