"""Run the Part A rules on saved tracks (CSV from tools/track.py) and score them.

Lets you tune rules in seconds without re-running YOLO.

    python tools/eval_from_csv.py --tracks tracks/ --frames frames/ --gt my_labels.json

tracks/<VIDEO>.csv  : output of tools/track.py for that video
frames/<VIDEO>.png  : one frame of that video (for camera alignment)
Video names are taken from the GT keys, e.g. "C3896.MP4" -> C3896.csv / C3896.png.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.jaywalking import detect_jaywalking, person_points  # noqa: E402
from src.scene import Scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="predictions_from_csv.json")
    a = ap.parse_args()

    gt = json.loads(Path(a.gt).read_text())
    scene = Scene()
    videos = {}
    for name, info in gt.items():
        stem = Path(name).stem
        tracks = pd.read_csv(Path(a.tracks) / f"{stem}.csv")
        H = scene.homography(cv2.imread(str(Path(a.frames) / f"{stem}.png")))
        events = detect_jaywalking(person_points(tracks, H, scene), info["duration"])
        videos[name] = {"events": events}
        print(f"{name}\n  GT  : {[e[:2] for e in info['events']]}\n  PRED: {[e[:2] for e in events]}")
    Path(a.out).write_text(json.dumps({"team": "dev", "videos": videos}, indent=1))

    sys.path.insert(0, str(ROOT))
    import evaluate as E
    r = E.evaluate_part_a(gt, videos)
    print(f"\nScore A = {r['score_a']:.3f}")
    for thr, m in r["micro"].items():
        print(f"  tIoU {thr}: TP={m['tp']} FP={m['fp']} FN={m['fn']}  F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
