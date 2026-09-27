"""
Run YOLO + ByteTrack on one video, save trajectories and a picture of them.

Usage:
    python track.py samples/video1.mp4

Output:
    tracks/video1.csv  - time, track_id, class, x, y, w, h  (x, y = bottom-center of box)
    tracks/video1.png  - all trajectories drawn on the first frame
"""
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

STRIDE = 3  # process every 3rd frame (25 fps -> ~8 fps)
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {"person": (0, 0, 255), "bicycle": (0, 255, 255), "motorcycle": (0, 255, 255),
          "car": (0, 255, 0), "bus": (255, 0, 0), "truck": (255, 0, 255)}


def track_video(video_path: str) -> None:
    name = Path(video_path).stem
    out_dir = Path("tracks")
    out_dir.mkdir(exist_ok=True)

    model = YOLO("yolov8s.pt")  # downloads automatically on first run
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows, i, t0 = [], 0, time.time()
    while cap.grab():
        if i % STRIDE == 0:
            _, frame = cap.retrieve()
            r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                            classes=list(CLASSES), verbose=False)[0]
            if r.boxes.id is not None:
                for box, tid, c in zip(r.boxes.xywh.tolist(), r.boxes.id.tolist(), r.boxes.cls.tolist()):
                    x, y, w, h = box
                    rows.append([round(i / fps, 2), int(tid), CLASSES[int(c)],
                                 round(x, 1), round(y + h / 2, 1), round(w, 1), round(h, 1)])
        i += 1
    cap.release()

    df = pd.DataFrame(rows, columns=["time", "track_id", "class", "x", "y", "w", "h"])
    df.to_csv(out_dir / f"{name}.csv", index=False)

    # draw trajectories
    img = first_frame.copy()
    for tid, g in df.groupby("track_id"):
        if len(g) < 5:
            continue
        cls = g["class"].mode()[0]
        pts = g[["x", "y"]].to_numpy().astype(int)
        cv2.polylines(img, [pts], False, COLORS[cls], 2)
        cv2.circle(img, tuple(pts[-1]), 5, COLORS[cls], -1)  # dot = where the object ended
    cv2.imwrite(str(out_dir / f"{name}.png"), img)

    print(f"{name}: {i} frames, {df['track_id'].nunique()} tracks, "
          f"{time.time() - t0:.0f} s processing, video {i / fps:.0f} s")


if __name__ == "__main__":
    track_video(sys.argv[1])
