---
title: Traffic Event Detection
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.28.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# Traffic event detection — v1.0

Detects **jaywalking** (a pedestrian on the carriageway outside a crossing) in videos from one
fixed CCTV camera and returns `[start_sec, end_sec, "jaywalking"]` segments.
Part B (accident anticipation) is the default zero-risk estimator.

**Score A = 0.690** on our own labels of the 4 sample videos (13 events), full run with the
unchanged `run_submission.py` + `evaluate.py`. Runtime ≈ 1.0–1.6× video length on a laptop GPU.

## Run (the two commands)

```bash
pip install -r requirements.txt       # Python 3.10+; weights/yolov8s.pt is included
python run_submission.py --videos <folder with .mp4> --out predictions.json --team Developers
python evaluate.py --pred predictions.json --gt my_labels.json
```

`run_submission.py` and `evaluate.py` are the unchanged starter-kit files.
`predictions_samples.json` is the output of this exact command on the 4 sample videos.

**Weights:** `weights/yolov8s.pt` (22 MB, Ultralytics COCO release v8.3.0) is committed to the repo;
there is no download step. No internet is needed at run time (`lapx` for ByteTrack is in `requirements.txt`).
Versions are pinned (`ultralytics==8.3.40`, `torch==2.7.1` with CUDA 12.6); the pipeline runs on GPU
when one is visible and falls back to CPU otherwise.

## Pipeline

| Step | Module | Learned / rule |
|---|---|---|
| 1. YOLOv8s detection (person, bicycle, car, motorcycle, bus, truck) on every 3rd frame | `src/tracking.py` | learned (COCO weights, no fine-tuning) |
| 2. ByteTrack tracking; a fresh model per video | `src/tracking.py` | rule |
| 3. Camera alignment: SIFT + RANSAC homography to `configs/reference.jpg`, best of 5 frames | `src/scene.py` | rule |
| 4. Scene zones (road, 3 crosswalks, 5 islands, far side) in reference pixels | `configs/scene.json` | hand-drawn |
| 5. Jaywalking: on road, off crossings/islands, not a rider, walking ≥1 s and ≥250 px | `src/jaywalking.py` | rule |
| 6. Segments: pad 1.5 s, merge gaps < 4 s, drop < 2 s | `src/jaywalking.py` | rule |

`src/pipeline.py` runs steps 1–6 in one call and is shared by `solution.py`, the renderer and the web demo.

## Repository

```
solution.py              entry point for the organizers' harness (detect_events, RiskEstimator)
src/                     tracking, scene/alignment, jaywalking rule, pipeline, drawing
configs/                 scene.json (zones) + reference.jpg (reference view)
weights/yolov8s.pt       detector weights
my_labels.json           our labels of the 4 sample videos (dev set)
predictions_samples.json our predictions on the 4 sample videos
tools/track.py           tracks → CSV + trajectory picture
tools/eval_from_csv.py   score the rules on saved tracks in seconds (no YOLO)
tools/check_alignment.py check camera alignment for frames or videos
tools/draw_zones.py      draw zones on a frame
tools/render.py          annotated videos + timelines for the website
tools/make_eda.py        EDA figures for the website
app.py, site/            team website and live demo (Gradio, Hugging Face Spaces)
```

## Development workflow

```bash
python tools/track.py samples/C3896.MP4                                         # tracks/C3896.csv
python tools/eval_from_csv.py --tracks tracks/ --frames frames/ --gt my_labels.json
python tools/render.py samples/*.MP4 --tracks tracks/ --gt my_labels.json --out site/results
python tools/make_eda.py --tracks tracks/ --frames frames/ --out site/eda
pip install gradio && python app.py                                             # website locally
```

Set `TRAFFIC_DEBUG=1` to save tracks, the first frame and the homography of every video to `debug/`.

## Determinism

`torch.manual_seed(0)`; YOLO inference and ByteTrack are deterministic on the same machine. Two runs give the same `predictions.json`.

## Team — Developers (`8F408C31`)

| Member | Role | Who did what |
|---|---|---|
| Member 1 (captain) | Detection, tracking, camera alignment | TODO |
| Member 2 | Scene zones, labelling, rules and evaluation | TODO |
| Member 3 | Website, demo, packaging | TODO |

Links, profiles and previous projects are on the team website (`site/team.json`).

## Models, data and licences

- **YOLOv8s** — Ultralytics, COCO-pretrained weights, AGPL-3.0.
- **ByteTrack** — via Ultralytics (original: MIT).
- **Data** — only the 4 sample videos provided by the organizers, labelled by our team (`my_labels.json`). No external datasets.
- Other libraries: OpenCV (Apache-2.0), NumPy, pandas, matplotlib, Plotly, Gradio.

## Attribution

- Detector and tracker: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0) and its
  built-in ByteTrack configuration (`bytetrack.yaml`; ByteTrack by Zhang et al., MIT).
- `run_submission.py`, `evaluate.py` and the `solution.py` interface come from the WIUT Hackathon 2026 starter kit.
- All other code (`src/`, `tools/`, `app.py`) was written by the team.
