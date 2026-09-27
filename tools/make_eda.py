"""EDA figures for the website, from saved tracks + one clean frame per video (no YOLO).

    python tools/make_eda.py --tracks tracks/ --frames frames/ --out site/eda

Writes:
    overview.json          resolution, fps, duration, brightness, camera shift, tracks per class
    counts_over_time.png   road users visible per second, per video (small multiples)
    heatmap.png            where road users are (all videos, reference frame)
    directions.png         vehicle trajectories coloured by direction of travel
    pedestrians.png        pedestrian points: on the road outside a crossing (red) vs. elsewhere
    camera_shift.png       how far each video's camera is from the reference view
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.jaywalking import on_road_mask, person_points  # noqa: E402
from src.scene import REF_SIZE, Scene, to_reference  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e7e2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots, fixed order
VEH = ["car", "bus", "truck", "motorcycle"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ad")
    ax.tick_params(colors=INK2, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ref_image(scene_ref_path):
    img = cv2.imread(str(scene_ref_path))
    return cv2.cvtColor(cv2.resize(img, (1920, 1080)), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="site/eda")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    scene = Scene()
    base = ref_image(ROOT / "configs" / "reference.jpg")
    names = sorted(p.stem for p in Path(a.tracks).glob("*.csv"))
    data, overview = {}, []
    probe = np.float32([[500, 1500], [2000, 1100], [3500, 1000], [1500, 2000]]).reshape(-1, 1, 2)
    for n in names:
        tr = pd.read_csv(Path(a.tracks) / f"{n}.csv")
        frame = cv2.imread(str(Path(a.frames) / f"{n}.png"))
        H, inl = scene.homography_with_score(frame)
        ref = to_reference(H, tr[["x", "y"]].to_numpy(np.float32))
        tr["rx"], tr["ry"] = ref[:, 0], ref[:, 1]
        pp = person_points(tr, H, scene)
        pp["on_road"] = on_road_mask(pp)
        data[n] = (tr, pp)
        s = frame.shape[1] / REF_SIZE[0]
        shift = cv2.perspectiveTransform(probe * s, H).reshape(-1, 2) - probe.reshape(-1, 2)
        overview.append({
            "video": n, "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
            "duration_s": round(float(tr.time.max()), 1),
            "brightness": round(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()), 1),
            "camera_shift_px": round(float(np.linalg.norm(shift, axis=1).mean()), 1),
            "alignment_inliers": inl,
            "tracks": {c: int(tr[tr["class"] == c].track_id.nunique()) for c in ["person", *VEH, "bicycle"]},
        })
    (out / "overview.json").write_text(json.dumps(overview, indent=1))

    # 1. counts over time (small multiples, shared y)
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.1 * len(names)), dpi=120, sharey=True)
    groups = [("Cars", ["car"]), ("Pedestrians", ["person"]), ("Buses & trucks", ["bus", "truck"])]
    for ax, n in zip(np.atleast_1d(axes), names):
        tr = data[n][0]
        sec = tr.assign(s=tr.time.astype(int))
        for (label, cls), col in zip(groups, SERIES):
            c = sec[sec["class"].isin(cls)].groupby(["s", "time"]).size().groupby("s").mean()
            c = c.reindex(range(int(sec.s.max()) + 1), fill_value=0).rolling(5, center=True, min_periods=1).mean()
            ax.plot(c.index, c.values, color=col, lw=2, label=label)
        ax.set_title(n, loc="left", fontsize=10, color=INK)
        style(ax)
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=3, loc="upper right", fontsize=9, labelcolor=INK2)
    np.atleast_1d(axes)[-1].set_xlabel("time, s", color=INK2)
    fig.supylabel("objects in view (5 s mean)", color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "counts_over_time.png")
    plt.close(fig)

    # 2. heatmap of all road users
    allp = pd.concat([d[0] for d in data.values()])
    hm, _, _ = np.histogram2d(allp.ry / 2, allp.rx / 2, bins=[108, 192], range=[[0, 1080], [0, 1920]])
    hm = cv2.GaussianBlur(np.log1p(hm), (0, 0), 1.2)
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=120)
    ax.imshow(base, alpha=0.55)
    im = ax.imshow(np.ma.masked_less(hm, 0.3), extent=(0, 1920, 1080, 0), cmap="Blues", alpha=0.85)
    ax.set_axis_off()
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("detections (log)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Where road users are (all sample videos)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "heatmap.png")
    plt.close(fig)

    # 3. vehicle directions
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=120)
    ax.imshow(base, alpha=0.6)
    dirs = [("towards bottom-right", SERIES[0]), ("towards top-left", SERIES[1]),
            ("turning / other", SERIES[2])]
    for n in names:
        tr = data[n][0]
        for _, g in tr[tr["class"].isin(VEH)].groupby("track_id"):
            if len(g) < 15:
                continue
            xy = g[["rx", "ry"]].to_numpy() / 2
            d = xy[-1] - xy[0]
            if np.linalg.norm(d) < 150:
                continue
            ang = np.degrees(np.arctan2(d[1], d[0]))
            k = 0 if 0 <= ang <= 60 else 1 if -180 <= ang <= -120 else 2
            ax.plot(xy[:, 0], xy[:, 1], color=dirs[k][1], lw=0.7, alpha=0.5)
    for label, col in dirs:
        ax.plot([], [], color=col, lw=2, label=label)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK)
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_axis_off()
    ax.set_title("Vehicle trajectories by direction of travel", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "directions.png")
    plt.close(fig)

    # 4. pedestrians on the road
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=120)
    ax.imshow(base, alpha=0.6)
    for poly, col in [(p, "#c040c0") for p in scene.crosswalks] + [(p, "#008300") for p in scene.islands]:
        q = poly.reshape(-1, 2) / 2
        ax.fill(q[:, 0], q[:, 1], color=col, alpha=0.18, lw=0)
    allpp = pd.concat([d[1] for d in data.values()])
    off = allpp[~allpp.on_road]
    on = allpp[allpp.on_road]
    ax.scatter(off.rx / 2, off.ry / 2, s=1, color="#8a8983", alpha=0.25, label="elsewhere", rasterized=True)
    ax.scatter(on.rx / 2, on.ry / 2, s=2, color="#e34948", alpha=0.6, label="on the road, outside a crossing",
               rasterized=True)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK, markerscale=6)
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_axis_off()
    ax.set_title("Pedestrian positions: most jaywalking cuts diagonally past the islands", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "pedestrians.png")
    plt.close(fig)

    # 5. camera shift
    fig, ax = plt.subplots(figsize=(6, 2.6), dpi=120)
    ov = pd.DataFrame(overview)
    ax.barh(ov.video, ov.camera_shift_px, color=SERIES[0], height=0.55)
    for y, v in enumerate(ov.camera_shift_px):
        ax.text(v + 2, y, f"{v:.0f} px", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("mean shift vs. reference view, px (4K)", color=INK2)
    ax.invert_yaxis()
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_title("The camera moves between recordings", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "camera_shift.png")
    plt.close(fig)
    print(json.dumps(overview, indent=1))


if __name__ == "__main__":
    main()
