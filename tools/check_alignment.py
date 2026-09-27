"""Check camera alignment for frames or videos, without YOLO (a few seconds per file).

    python tools/check_alignment.py frames/*.png
    python tools/check_alignment.py samples/*.MP4

For each file prints the number of inliers (good: > 100; 0 = failed) and how far
four road points move. Saves align_<name>.jpg: the frame warped onto the reference,
50/50 blend. Road markings must not look doubled.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scene import REF_SIZE, Scene  # noqa: E402

scene = Scene()
ref = cv2.resize(cv2.imread(str(Path(__file__).resolve().parent.parent / "configs" / "reference.jpg")), REF_SIZE)
pts = np.float32([[500, 1500], [2000, 1100], [3500, 1000], [1500, 2000]]).reshape(-1, 1, 2)
for f in sys.argv[1:]:
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(f)
        frames = []
        for t in (0, 3, 6, 9, 12):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
        cap.release()
    else:
        frames = [cv2.imread(f)]
    if not frames or frames[0] is None:
        print(f"{f}: cannot read")
        continue
    H, inl = scene.best_homography(frames)
    best = max(frames, key=lambda fr: scene.homography_with_score(fr)[1])
    s = best.shape[1] / REF_SIZE[0]
    shift = cv2.perspectiveTransform(pts * s, H).reshape(-1, 2) - pts.reshape(-1, 2)
    print(f"{Path(f).name}: size={best.shape[1]}x{best.shape[0]} inliers={inl} "
          f"shift(px)={np.round(shift).astype(int).tolist()}" + ("  <-- FAILED" if inl == 0 else ""))
    warped = cv2.warpPerspective(best, H, REF_SIZE)
    cv2.imwrite(f"align_{Path(f).stem}.jpg", cv2.resize(cv2.addWeighted(warped, 0.5, ref, 0.5, 0), (1920, 1080)))
