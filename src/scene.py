"""Scene layout (zones) and camera alignment to the reference frame.

All zones in configs/scene.json are in pixel coordinates of the 3840x2160
reference frame. The camera shifts slightly between videos (up to ~100 px),
so every video is aligned to the reference with a homography computed from
SIFT matches on the static lower part of the frame (road markings, curbs).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
REF_SIZE = (3840, 2160)   # coordinate system of scene.json
WORK_WIDTH = 1920         # alignment runs at this width


class Scene:
    def __init__(self, config_path: Path = CONFIG_DIR / "scene.json"):
        cfg = json.loads(Path(config_path).read_text())
        poly = lambda p: np.asarray(p, np.float32).reshape(-1, 1, 2)
        self.road = poly(cfg["road"])
        self.far_side = poly(cfg["far_side"])
        self.crosswalks = [poly(p) for p in cfg["crosswalks"].values()]
        self.islands = [poly(p) for p in cfg["islands"].values()]
        self.reference = cv2.imread(str(CONFIG_DIR / cfg["reference_frame"]), cv2.IMREAD_GRAYSCALE)

    # ---- zone tests (signed distance in px: >0 inside, <0 outside) ----
    @staticmethod
    def _dist(poly, x, y) -> float:
        return cv2.pointPolygonTest(poly, (float(x), float(y)), True)

    def zone_features(self, x: float, y: float) -> tuple[float, float, float, bool]:
        d_road = self._dist(self.road, x, y)
        d_cw = max(self._dist(c, x, y) for c in self.crosswalks)
        d_isl = max(self._dist(c, x, y) for c in self.islands)
        far = self._dist(self.far_side, x, y) > 0
        return d_road, d_cw, d_isl, far

    # ---- alignment ----
    def homography(self, frame_bgr: np.ndarray) -> np.ndarray:
        """3x3 matrix mapping pixels of `frame_bgr` to reference (3840x2160) pixels."""
        return self.homography_with_score(frame_bgr)[0]

    def best_homography(self, frames: list[np.ndarray]) -> tuple[np.ndarray, int]:
        """Align several frames and keep the one with the most RANSAC inliers."""
        results = [self.homography_with_score(f) for f in frames if f is not None]
        return max(results, key=lambda r: r[1]) if results else (np.eye(3), 0)

    def homography_with_score(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """(H, number of inliers). inliers == 0 means alignment failed (scaling only)."""
        clahe = cv2.createCLAHE(3.0, (8, 8))
        h, w = frame_bgr.shape[:2]
        s_vid = WORK_WIDTH / w
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        a = clahe.apply(cv2.resize(gray, (WORK_WIDTH, round(h * s_vid)), interpolation=cv2.INTER_AREA))
        b = clahe.apply(self.reference)
        s_ref = b.shape[1] / REF_SIZE[0]

        def mask(img):  # static lower part only (no trees, buildings or sky)
            m = np.zeros_like(img)
            m[int(img.shape[0] * 0.40):, :] = 255
            return m

        sift = cv2.SIFT_create(8000)
        k1, d1 = sift.detectAndCompute(a, mask(a))
        k2, d2 = sift.detectAndCompute(b, mask(b))
        identity = np.diag([1 / s_ref, 1 / s_ref, 1.0]) @ np.diag([s_vid, s_vid, 1.0])
        if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
            return identity, 0
        good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.8 * n.distance]
        if len(good) < 30:
            return identity, 0
        p1 = np.float32([k1[m.queryIdx].pt for m in good])
        p2 = np.float32([k2[m.trainIdx].pt for m in good])
        H_small, inliers = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        if H_small is None or inliers.sum() < 25:
            return identity, 0
        # video px -> small video px -> small ref px -> reference px
        H = np.diag([1 / s_ref, 1 / s_ref, 1.0]) @ H_small @ np.diag([s_vid, s_vid, 1.0])
        return H, int(inliers.sum())


def to_reference(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return xy.reshape(0, 2)
    return cv2.perspectiveTransform(xy.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
