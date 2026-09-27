"""Draw the scene zones on a frame (aligned to the reference) to check them by eye.

    python tools/draw_zones.py frame.png zones_check.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scene import REF_SIZE, Scene  # noqa: E402

scene = Scene()
frame = cv2.imread(sys.argv[1])
img = cv2.warpPerspective(frame, scene.homography(frame), REF_SIZE)
over = img.copy()
cv2.fillPoly(over, [scene.road.astype(np.int32)], (0, 140, 255))
for p in scene.islands:
    cv2.fillPoly(over, [p.astype(np.int32)], (0, 200, 0))
for p in scene.crosswalks:
    cv2.fillPoly(over, [p.astype(np.int32)], (255, 0, 255))
cv2.polylines(over, [scene.far_side.astype(np.int32)], True, (255, 255, 0), 4)
out = cv2.addWeighted(over, 0.4, img, 0.6, 0)
cv2.imwrite(sys.argv[2], cv2.resize(out, (1920, 1080)))
