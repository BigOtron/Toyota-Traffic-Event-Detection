"""Team website + live demo (Gradio). Runs on Hugging Face Spaces (CPU) or locally:

    python app.py            -> http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from src.pipeline import analyze
from src.viz import render_video, timeline_plotly

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
MAX_SECONDS = 120          # demo processes at most the first 2 minutes
MAX_UPLOAD = "500mb"


def _read(p: Path, default=""):
    return p.read_text(encoding="utf-8") if p.exists() else default


TEAM = json.loads(_read(SITE / "team.json", "{}") or "{}")
METRICS = json.loads(_read(SITE / "metrics.json", "{}") or "{}")
EDA = json.loads(_read(SITE / "eda" / "overview.json", "[]") or "[]")
RESULTS = sorted(p.stem.replace("_events", "") for p in (SITE / "results").glob("*_events.json"))


# ------------------------------------------------------------------ demo
def run_demo(video, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload an .mp4 first.")
    progress(0.0, desc="Detecting and tracking road users…")
    res = analyze(video, max_seconds=MAX_SECONDS,
                  progress=lambda f: progress(0.05 + 0.6 * f, desc="Detecting and tracking road users…"))
    out = Path(tempfile.mkdtemp()) / "annotated.mp4"
    render_video(video, res, str(out), width=960, max_seconds=MAX_SECONDS,
                 progress=lambda f: progress(0.65 + 0.33 * f, desc="Rendering annotated video…"))
    table = pd.DataFrame([{"start, s": a, "end, s": b, "event": lab} for a, b, lab in res.events],
                         columns=["start, s", "end, s", "event"])
    status = (f"Processed {res.duration:.0f} s of video · {len(res.events)} event(s) · "
              f"camera alignment {'OK' if res.inliers else 'FAILED (zones may be off)'} "
              f"({res.inliers} matches)")
    js = Path(out.parent) / "events.json"
    js.write_text(json.dumps({"events": res.events, "duration": res.duration}, indent=1))
    return status, table, timeline_plotly(res.events, res.duration), str(out), str(js)


# ------------------------------------------------------------------ results
def show_result(name):
    if not name:
        return None, None, None, ""
    ev = json.loads(_read(SITE / "results" / f"{name}_events.json"))
    vid = SITE / "results" / f"{name}_annotated.mp4"
    rows = [{"source": "detected", "start, s": a, "end, s": b} for a, b, *_ in ev["events"]]
    rows += [{"source": "our label", "start, s": a, "end, s": b} for a, b, *_ in (ev.get("labels") or [])]
    note = "" if vid.exists() else "_Annotated video not rendered yet: run `python tools/render.py`._"
    return (str(vid) if vid.exists() else None,
            timeline_plotly(ev["events"], ev["duration"], ev.get("labels")),
            pd.DataFrame(rows), note)


def metrics_md():
    pc = METRICS.get("part_a", {}).get("per_class", {}).get("jaywalking")
    if not pc:
        return ""
    rows = "\n".join(f"| {t} | {pc[t]['f1']:.3f} | {pc[t]['precision']:.2f} | {pc[t]['recall']:.2f} | "
                     f"{pc[t]['tp']} / {pc[t]['fp']} / {pc[t]['fn']} |" for t in ("0.3", "0.5", "0.7"))
    return (f"**Score A = {METRICS['part_a']['score_a']:.3f}** on our own labels of the 4 sample videos "
            f"(13 jaywalking events, official `evaluate.py`).\n\n"
            "| tIoU | F1 | precision | recall | TP / FP / FN |\n|---|---|---|---|---|\n" + rows)


# ------------------------------------------------------------------ static pages
PIPELINE_HTML = """
<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;margin:8px 0 16px">
""" + "".join(
    f"""<div style="flex:1 1 150px;border:1px solid #d6d5cf;border-radius:10px;padding:10px 12px">
<div style="font-size:12px;color:#52514e">{k}. {kind}</div>
<div style="font-weight:600;margin:2px 0 4px">{title}</div>
<div style="font-size:13px;color:#52514e">{text}</div></div>"""
    for k, (kind, title, text) in enumerate([
        ("learned", "YOLOv8s detector", "person, bicycle, car, motorcycle, bus, truck; every 3rd frame"),
        ("rule", "ByteTrack", "links detections into tracks (Kalman filter + IoU)"),
        ("rule", "Camera alignment", "SIFT + RANSAC homography to one reference frame"),
        ("rule", "Scene zones", "road, 3 crosswalks, 5 islands drawn once"),
        ("rule", "Jaywalking rule", "on road, off crossings, walking ≥1 s and ≥250 px"),
        ("rule", "Segments", "pad 1.5 s, merge gaps < 4 s, drop < 2 s"),
    ], 1)) + "</div>"

APPROACH_MD = """
### Problem
Report every traffic event in a video from one fixed CCTV camera as `[start_sec, end_sec, label]`.
We have 4 unlabeled sample videos (4K, 29.97 fps, 2–6 min each) and label them ourselves.

### Why this design
* The core classes are rules on **trajectories + scene layout**, so we start from a detector and a tracker.
* In our samples only **jaywalking** occurs (13 events). The metric is a macro F1 over classes that
  appear in the test set *or in our predictions*, so predicting classes we cannot validate would lower
  the score. We ship one class that we measured.
* The camera moves between recordings (up to ~110 px). All zones are defined once on a reference
  frame and every video is aligned to it automatically.

### Models and data
* **YOLOv8s** (Ultralytics, COCO weights, AGPL-3.0), **ByteTrack** (via Ultralytics, MIT). No fine-tuning.
* **Data:** the 4 sample videos only, labelled by us (`my_labels.json`). No external datasets.
* **Learned:** detector. **Rule-based:** tracking association, alignment, zones, the jaywalking rule, segment post-processing.

### Reproduce
```bash
pip install -r requirements.txt           # weights/yolov8s.pt is in the repository
python run_submission.py --videos samples/ --out predictions.json
python evaluate.py --pred predictions.json --gt my_labels.json
```
"""

EDA_MD = """
### What the camera sees
"""


def eda_table():
    if not EDA:
        return pd.DataFrame()
    return pd.DataFrame([{
        "video": e["video"], "resolution": e["resolution"], "duration, s": e["duration_s"],
        "brightness (0–255)": e["brightness"], "camera shift, px": e["camera_shift_px"],
        "pedestrian tracks": e["tracks"]["person"], "car tracks": e["tracks"]["car"],
        "bus + truck tracks": e["tracks"]["bus"] + e["tracks"]["truck"]} for e in EDA])


def team_md():
    if not TEAM:
        return "Fill in `site/team.json`."
    parts = [f"## {TEAM.get('team_name', '')}"]
    for m in TEAM.get("members", []):
        links = " · ".join(f"[{k}]({m[k]})" for k in ("github", "linkedin", "portfolio") if m.get(k))
        parts.append(f"### {m['name']}\n**{m['role']}**\n\n{m.get('did', '')}\n\n"
                     f"Proud of: {m.get('projects', '')}\n\n{links}")
    lk = TEAM.get("links", {})
    parts.append("### Links\n" + "\n".join(f"* [{k.replace('_', ' ')}]({v})" for k, v in lk.items()))
    return "\n\n".join(parts)


# ------------------------------------------------------------------ layout
CSS = ".gradio-container{max-width:1100px!important;margin:auto} img{border-radius:8px}"

with gr.Blocks(title="Traffic events — jaywalking detector") as demo:
    gr.Markdown(f"# 🚦 Traffic event detection\n{TEAM.get('team_name', '')} · fixed CCTV camera · "
                "pedestrians on the road outside a crossing")
    with gr.Tabs():
        with gr.Tab("Live demo"):
            gr.Markdown(
                f"Upload an **.mp4** from this camera (up to {MAX_UPLOAD}; only the first "
                f"**{MAX_SECONDS // 60} minutes** are processed; 1080p is enough). Runs on CPU: "
                "a 1-minute clip takes a few minutes. **Red box** = pedestrian on the road outside a crossing, "
                "green = other pedestrians, blue = vehicles, magenta = crosswalks, green areas = islands.")
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Video(label="Your video (.mp4)", sources=["upload"])
                    examples = sorted((SITE / "examples").glob("*.mp4"))
                    if examples:
                        gr.Examples([[str(p)] for p in examples], inputs=inp, label="Or try a sample clip")
                    btn = gr.Button("Detect events", variant="primary")
                with gr.Column(scale=1):
                    status = gr.Markdown()
                    out_vid = gr.Video(label="Annotated video")
            out_plot = gr.Plot(label="Event timeline (hover for times)")
            with gr.Row():
                out_tab = gr.Dataframe(label="Events", interactive=False, headers=["start, s", "end, s", "event"])
                out_json = gr.File(label="events.json")
            btn.click(run_demo, inp, [status, out_tab, out_plot, out_vid, out_json])

        with gr.Tab("Results on samples"):
            gr.Markdown(metrics_md())
            sel = gr.Dropdown(RESULTS, value=RESULTS[0] if RESULTS else None, label="Sample video")
            r_note = gr.Markdown()
            r_vid = gr.Video(label="Annotated video (red = jaywalking pedestrian)")
            r_plot = gr.Plot(label="Detected (blue) vs. our labels (orange)")
            r_tab = gr.Dataframe(label="Segments", interactive=False)
            sel.change(show_result, sel, [r_vid, r_plot, r_tab, r_note])
            demo.load(show_result, sel, [r_vid, r_plot, r_tab, r_note])
            gr.Markdown("""### Failure cases
* **Dusk (C3905):** darker video, fewer and shorter person tracks → 2 of 3 events missed.
* **False alarms (C3897, C3902):** pedestrians walking just outside the painted crosswalk; one of them
  (C3897, 238–245 s) may be a real crossing missing from our labels.
* **Boundaries:** at tIoU 0.7 one more event fails because our segment starts ~1.5 s early.""")

        with gr.Tab("Approach"):
            gr.Markdown("### Pipeline")
            gr.HTML(PIPELINE_HTML)
            gr.Markdown(APPROACH_MD)

        with gr.Tab("EDA"):
            gr.Markdown(EDA_MD)
            gr.Dataframe(eda_table(), interactive=False)
            gr.Markdown("""**Findings that shaped the solution**
1. **The camera moves between recordings** (up to ~110 px in 4K) → we align every video to one reference frame.
2. **Two main traffic directions** separated by a median, plus right turns at the bottom-left → zones and the far side of the crossings are defined once on the reference frame.
3. **Jaywalkers mostly cut diagonally** from the top crossing to the islands at the bottom-left → the rule looks for walking movement on the road, not for any pedestrian standing near the curb.
4. **Lighting ranges from bright day to dusk** (brightness 39–70) → dusk has far fewer pedestrian tracks; this is our main failure mode.
5. **4–15 pedestrians and 10–30 cars are in view at any time**; tracker ID switches are common in the crowd at the top crossing → we require ≥1 s and ≥250 px of continuous movement.""")
            for img, cap in [("camera_shift.png", ""), ("counts_over_time.png", "Road users in view over time"),
                             ("heatmap.png", ""), ("directions.png", ""), ("pedestrians.png", "")]:
                if (SITE / "eda" / img).exists():
                    gr.Image(str(SITE / "eda" / img), label=cap or None, show_label=bool(cap),
                             interactive=False)

        with gr.Tab("Report"):
            gr.Markdown(_read(SITE / "report.md", "Report coming soon."))

        with gr.Tab("Team"):
            gr.Markdown(team_md())

if __name__ == "__main__":
    demo.queue(max_size=8).launch(max_file_size=MAX_UPLOAD, css=CSS, theme=gr.themes.Soft())
