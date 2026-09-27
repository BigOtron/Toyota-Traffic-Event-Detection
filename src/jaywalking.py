"""Jaywalking: a pedestrian on the carriageway outside a crossing.

Rule (all distances in reference-frame pixels, 3840x2160):
  1. a person point is "on the road" if it is inside the road polygon by more than
     M_ROAD px, outside every crosswalk by more than M_CW px, not on an island, not
     on the far side of the crossings (bus stop / waiting area), and not clipped
     by the bottom of the frame;
  2. people riding a bicycle or motorcycle are not pedestrians;
  3. a track must stay on the road for >= MIN_DUR s and move >= MIN_DISP px at
     walking speed (<= SPD_MAX px/s): people waiting at a curb do not count,
     tracker ID jumps do not count;
  4. runs from all tracks are padded, merged when closer than GAP s, and kept if
     longer than MIN_LEN s. Several people crossing together = one event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scene import Scene, to_reference

PARAMS = dict(
    M_ROAD=80, M_CW=100, M_ISL=10, Y_MAX=2050,
    RIDER_FRAC=0.3, MIN_DUR=1.0, MIN_DISP=250, SPD_MAX=350, MAX_HOLE=1.0,
    PAD=1.5, GAP=4.0, MIN_LEN=2.0,
)


def _rider_flags(tracks: pd.DataFrame, persons: pd.DataFrame) -> np.ndarray:
    """True for person detections standing on a bicycle / motorcycle box."""
    bikes = tracks[tracks["class"].isin(["bicycle", "motorcycle"])]
    by_time = {t: g[["x", "y", "w", "h"]].to_numpy() for t, g in bikes.groupby("time")}
    out = np.zeros(len(persons), bool)
    for i, (t, x, y) in enumerate(zip(persons.time, persons.x, persons.y)):
        b = by_time.get(t)
        if b is not None:
            out[i] = np.any((np.abs(b[:, 0] - x) < b[:, 2] * 0.7)
                            & (y > b[:, 1] - b[:, 3] * 1.2) & (y < b[:, 1] + b[:, 3] * 0.5))
    return out


def person_points(tracks: pd.DataFrame, H: np.ndarray, scene: Scene) -> pd.DataFrame:
    p = tracks[tracks["class"] == "person"].copy()
    ref = to_reference(H, p[["x", "y"]].to_numpy(np.float32))
    p["rx"], p["ry"] = ref[:, 0], ref[:, 1]
    feats = [scene.zone_features(x, y) for x, y in ref]
    p["d_road"], p["d_cw"], p["d_isl"], p["far"] = (list(c) for c in zip(*feats)) if feats else ([],) * 4
    p["rider"] = _rider_flags(tracks, p)
    p["rider_trk"] = p.groupby("track_id")["rider"].transform("mean")
    return p.sort_values(["track_id", "time"])


def on_road_mask(p: pd.DataFrame, P: dict = PARAMS) -> pd.Series:
    """Person points that count as 'pedestrian on the carriageway outside a crossing'."""
    return ((p.d_road > P["M_ROAD"]) & (p.d_cw < -P["M_CW"]) & (p.d_isl < -P["M_ISL"])
            & ~p.far.astype(bool) & (p.ry < P["Y_MAX"]) & (p.rider_trk < P["RIDER_FRAC"]))


def detect_jaywalking(p: pd.DataFrame, duration: float, P: dict = PARAMS) -> list[list]:
    if p.empty:
        return []
    on_road = on_road_mask(p, P)
    runs = []
    for _, g in p[on_road].groupby("track_id"):
        t = g.time.to_numpy()
        xy = g[["rx", "ry"]].to_numpy()
        for idx in np.split(np.arange(len(t)), np.where(np.diff(t) > P["MAX_HOLE"])[0] + 1):
            if len(idx) < 3:
                continue
            dur = t[idx[-1]] - t[idx[0]]
            disp = float(np.linalg.norm(xy[idx].max(0) - xy[idx].min(0)))
            if dur >= P["MIN_DUR"] and disp >= P["MIN_DISP"] and disp / dur <= P["SPD_MAX"]:
                runs.append([t[idx[0]] - P["PAD"], t[idx[-1]] + P["PAD"]])
    events = []
    for a, b in merge(runs, P["GAP"]):
        a, b = round(float(max(0.0, a)), 2), round(float(min(duration, b)), 2)
        if b - a >= P["MIN_LEN"]:
            events.append([a, b, "jaywalking"])
    return events


def merge(runs: list[list[float]], gap: float) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(runs):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out
