"""Drawing helpers: annotated video, zone overlay, event timelines.

Colors (fixed roles, same everywhere):
  pedestrian on the carriageway outside a crossing  -> red
  other pedestrians                                  -> green
  vehicles                                           -> blue
  crosswalks / islands / road outline                -> magenta / green / orange
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .scene import Scene

# BGR
RED, GREEN, BLUE, GRAY = (72, 73, 227), (0, 150, 0), (214, 120, 42), (170, 170, 170)
C_PRED, C_GT = "#2a78d6", "#eb6834"   # timeline: our detections / manual labels


# ---------------------------------------------------------------- zones
def zones_in_video(scene: Scene, H: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Scene polygons (reference px) mapped back into this video's pixels."""
    Hinv = np.linalg.inv(H)
    back = lambda p: cv2.perspectiveTransform(p, Hinv).reshape(-1, 2).astype(np.int32)
    return {"road": [back(scene.road)],
            "crosswalk": [back(p) for p in scene.crosswalks],
            "island": [back(p) for p in scene.islands]}


def draw_zones(img: np.ndarray, zones: dict, scale: float, alpha: float = 0.28) -> np.ndarray:
    over = img.copy()
    for p in zones["crosswalk"]:
        cv2.fillPoly(over, [(p * scale).astype(np.int32)], (255, 0, 255))
    for p in zones["island"]:
        cv2.fillPoly(over, [(p * scale).astype(np.int32)], (0, 200, 0))
    out = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
    for p in zones["road"]:
        cv2.polylines(out, [(p * scale).astype(np.int32)], True, (0, 140, 255), 2)
    return out


# ---------------------------------------------------------------- video
def _writer(path: str, w: int, h: int, fps: float):
    """H.264 via ffmpeg when available (plays in browsers), else OpenCV mp4v."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "26", "-movflags", "+faststart", path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return (lambda f: proc.stdin.write(f.tobytes())), (lambda: (proc.stdin.close(), proc.wait()))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return vw.write, vw.release


def render_video(video_path: str, res, out_path: str, width: int = 1280, stride: int = 3,
                 max_seconds: float | None = None, progress=None) -> str:
    """Annotated copy of the video at fps/stride: zones, boxes, event banner, clock."""
    scene = Scene()
    zones = zones_in_video(scene, res.H)
    W, Hh = res.size
    scale = width / W
    height = int(round(Hh * scale / 2) * 2)
    by_time = {t: g for t, g in res.tracks.groupby("time")}
    on_road = set(res.persons.index[res.persons["on_road"]]) if len(res.persons) else set()

    write, close = _writer(out_path, width, height, res.fps / stride)
    cap = cv2.VideoCapture(video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_seconds is not None:
        n_total = min(n_total, int(max_seconds * res.fps))
    i = 0
    while i < n_total and cap.grab():
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = round(i / res.fps, 2)
            img = draw_zones(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), zones, scale)
            g = by_time.get(t)
            if g is not None:
                for idx, r in g.iterrows():
                    x0, y0 = (r.x - r.w / 2) * scale, (r.y - r.h) * scale
                    x1, y1 = (r.x + r.w / 2) * scale, r.y * scale
                    if r["class"] == "person":
                        col, th = (RED, 3) if idx in on_road else (GREEN, 2)
                    else:
                        col, th = BLUE, 1
                    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, th)
            active = [e for e in res.events if e[0] <= t <= e[1]]
            cv2.rectangle(img, (0, 0), (width, 40), (20, 20, 20), -1)
            cv2.putText(img, f"{t:7.1f} s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if active:
                cv2.rectangle(img, (170, 4), (560, 36), RED, -1)
                cv2.putText(img, "JAYWALKING", (185, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            write(img)
            if progress is not None and i % (stride * 30) == 0:
                progress(i / max(n_total, 1))
        i += 1
    cap.release()
    close()
    return out_path


# ---------------------------------------------------------------- timelines
def timeline_png(pred: list, duration: float, out_path: str, gt: list | None = None, title: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("Detected", pred, C_PRED)] + ([("Our labels", gt, C_GT)] if gt is not None else [])
    fig, ax = plt.subplots(figsize=(10, 0.7 + 0.55 * len(rows)), dpi=120)
    for k, (name, evs, col) in enumerate(reversed(rows)):
        ax.broken_barh([(a, b - a) for a, b, *_ in evs], (k - 0.3, 0.6), facecolors=col, edgecolor="white", linewidth=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], color="#52514e")
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("time, s", color="#52514e")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b5b4ad")
    ax.tick_params(colors="#52514e", length=0)
    ax.grid(axis="x", color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def timeline_plotly(pred: list, duration: float, gt: list | None = None):
    """Interactive timeline (hover shows start/end) for the web page."""
    import plotly.graph_objects as go

    fig = go.Figure()
    rows = [("Detected", pred, C_PRED)] + ([("Our labels", gt, C_GT)] if gt is not None else [])
    for name, evs, col in rows:
        # invisible anchor so the row is shown even with no events
        fig.add_trace(go.Bar(x=[0], base=[0], y=[name], orientation="h", showlegend=False,
                             marker_color=col, hoverinfo="skip"))
        for a, b, *lab in evs:
            fig.add_trace(go.Bar(
                x=[b - a], base=[a], y=[name], orientation="h", marker_color=col, width=0.55,
                showlegend=False, marker_line_width=0,
                hovertemplate=f"{lab[0] if lab else 'event'}<br>{a:.1f}–{b:.1f} s<extra>{name}</extra>"))
    fig.update_layout(height=110 + 50 * len(rows), margin=dict(l=10, r=10, t=10, b=40),
                      xaxis=dict(range=[0, duration], title="time, s", gridcolor="#e8e7e2"),
                      yaxis=dict(categoryorder="array", categoryarray=[r[0] for r in reversed(rows)]),
                      plot_bgcolor="white", paper_bgcolor="white", barmode="overlay")
    return fig
