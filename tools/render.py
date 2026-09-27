"""Render results for the website: annotated video, timeline and events for each video.

    python tools/render.py samples/*.MP4 --gt my_labels.json --out site/results
    python tools/render.py samples/*.MP4 --tracks tracks/ --gt my_labels.json   # reuse saved tracks, no YOLO

For every video writes to --out:
    <name>_annotated.mp4   1280 px, ~10 fps, H.264 (red box = pedestrian on the road outside a crossing)
    <name>_timeline.png    detected events vs. our labels
    <name>_events.json     events, duration, alignment inliers
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import analyze, analyze_from_tracks  # noqa: E402
from src.viz import render_video, timeline_png  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="site/results")
    ap.add_argument("--gt", default=None, help="labels json (optional)")
    ap.add_argument("--tracks", default=None, help="folder with <name>.csv tracks to skip YOLO")
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads(Path(a.gt).read_text()) if a.gt else {}
    for v in a.videos:
        name = Path(v).stem
        csv = Path(a.tracks) / f"{name}.csv" if a.tracks else None
        res = analyze_from_tracks(v, pd.read_csv(csv)) if csv and csv.exists() else analyze(v)
        print(f"[{name}] {len(res.events)} events, inliers={res.inliers}")
        g = gt.get(Path(v).name, {}).get("events") if gt else None
        timeline_png(res.events, res.duration, str(out / f"{name}_timeline.png"), gt=g, title=name)
        (out / f"{name}_events.json").write_text(json.dumps(
            {"video": Path(v).name, "duration": round(res.duration, 2), "inliers": res.inliers,
             "events": res.events, "labels": g}, indent=1))
        render_video(v, res, str(out / f"{name}_annotated.mp4"), width=a.width,
                     progress=lambda f: print(f"  rendering {f:5.0%}", end="\r"))
        print(f"  -> {out / (name + '_annotated.mp4')}")


if __name__ == "__main__":
    main()
