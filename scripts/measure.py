"""Compare the model's outputs with the correct answers and measure the difference in millimeters.

Run:  python scripts/measure.py --data PREPARED_FOLDER --pred predictions/test --out results/test

--data  folder made by prepare_tf3.py (holds the correct answer for every scan)
--pred  folder made by predict.py (holds the model's output for some scans)
--out   folder for the numbers (results.csv, summary.json) and the pictures

Everything is measured in the block. In the block one row is 0.4 mm of height, one column is 0.4 mm
along the jaw curve, and one depth step is 0.5 mm from the tongue side toward the cheek side.

What is measured for each scan:

1. Nerve canal (the tube in the lower jaw bone that holds the nerve).
   For each column that holds canal, the middle point of the canal in that column is found. All
   middle points together are the "middle line" of the canal. Then:
     "canal mm"          average distance between the model's middle line and the true middle line.
                         It is the average of two numbers: from each true point to the nearest model
                         point, and from each model point to the nearest true point.
     "canal found share" share of the true middle line that has a model point within 2 mm
                         (1 = the model found the whole canal, 0 = it found nothing).

2. Sinus (the air space above the upper back teeth). Only its bottom edge is in the training scans.
   For each column that holds sinus, the lowest row with sinus is the bottom edge.
     "sinus edge mm"        average height difference between the model's edge and the true edge.
     "sinus edge signed mm" the same, but with a sign: positive = the model put the edge too low.
     "sinus columns found share" share of the true sinus columns in which the model output any sinus.

3. Lower jaw thickness, measured from the tongue side to the cheek side, at two heights
   (12 mm and 20 mm below the line where the teeth meet), at every column within 45 mm of the
   front of the jaw (the part of the jaw that holds teeth). Tooth roots and the nerve canal count
   as part of the jaw.
     "jaw thickness true mm 12", "jaw thickness model mm 12"  average thickness
     "jaw thickness diff mm 12"   average difference per column (always counted as positive)

4. Bone density. Brightness in a 3D scan is given in Hounsfield units (air is -1000, water is 0,
   dense bone is above 1000). Brighter means denser.
     "lower jaw HU true", "lower jaw HU model"  average brightness inside the true lower jaw bone
     "upper jaw HU true", "upper jaw HU model"  the same for the upper jaw bone
     "bone err HU"  average brightness difference, point by point, inside the true bone

5. The same numbers the training script printed: "err_hu" (average brightness difference over the
   whole block), "psnr" (a standard score for how close two pictures are, higher is better), and
   "dice PART" (overlap score from 0 to 1: 1 means the model put the part in exactly the right place).

Before the distances for the canal and the sinus are measured, pieces of the model's canal or sinus
that are smaller than 50 points (4 cubic millimeters) are removed. The overlap scores use the
output unchanged.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.curved_volume import DEPTH_IN_MM, DEPTH_STEP_MM, FRAME_BITE_ROW, FRAME_MIDDLE_COL, OUTSIDE, VOXEL_MM

CLASS_NAMES = ["background", "lower jaw", "upper jaw", "nerve canal", "sinus", "teeth"]
LABEL_COLORS = np.array([[0, 0, 0], [70, 110, 200], [60, 170, 170], [255, 40, 40], [255, 220, 0], [240, 240, 240]], np.uint8)
CANAL, SINUS = 3, 4
SPECK_POINTS = 50              # model pieces smaller than this are removed before measuring distances
NEAR_MM = 2.0                  # a true canal point counts as found if a model point is this close
TEETH_REGION_MM = 45.0         # thickness is measured this far to each side of the front of the jaw
THICKNESS_BELOW_BITE_MM = (12, 20)


def remove_specks(mask):
    """Drop connected pieces that are smaller than SPECK_POINTS."""
    pieces, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(pieces.ravel())
    keep = sizes >= SPECK_POINTS
    keep[0] = False
    return keep[pieces]


def middle_line(mask):
    """One point per column that holds the part: the average place of the part in that column.
    Returns [n, 3] in millimeters: (height, along the jaw curve, tongue-to-cheek)."""
    rows, cols, deps = np.nonzero(mask)
    if len(cols) == 0:
        return np.zeros((0, 3))
    n = np.bincount(cols, minlength=mask.shape[1])
    has = n > 0
    r = np.bincount(cols, weights=rows, minlength=mask.shape[1])[has] / n[has]
    d = np.bincount(cols, weights=deps, minlength=mask.shape[1])[has] / n[has]
    return np.stack([r * VOXEL_MM, np.flatnonzero(has) * VOXEL_MM, d * DEPTH_STEP_MM], axis=1)


def bottom_rows(mask):
    """For each column: the lowest row (largest row number) that holds the part, or -1 if none."""
    rows = np.arange(mask.shape[0])[:, None]
    return np.where(mask.any(axis=2), rows, -1).max(axis=0)


def measure_one(true_hu, true_lab, pred_hu, pred_lab):
    """All arrays are [rows, columns, depth]. Returns a dict of numbers for one scan."""
    nan = float("nan")
    res = {}
    inside = true_lab != OUTSIDE  # places outside the scan have no correct answer
    true_hu, pred_hu = true_hu.astype(np.float32), pred_hu.astype(np.float32)

    # 5. the numbers from the training script
    diff = (pred_hu - true_hu)[inside] / 4000.0
    res["err_hu"] = float(np.abs(diff).mean() * 4000)
    mean_square = float((diff ** 2).mean())
    res["psnr"] = float(-10 * math.log10(mean_square)) if mean_square > 0 else float("inf")
    for c in range(1, 6):
        t, p = true_lab == c, (pred_lab == c) & inside
        res[f"dice {CLASS_NAMES[c]}"] = float(2 * (t & p).sum() / (t.sum() + p.sum())) if t.any() else nan

    # 1. nerve canal
    true_line = middle_line(true_lab == CANAL)
    pred_line = middle_line(remove_specks((pred_lab == CANAL) & inside))
    res["canal mm"] = res["canal true to model mm"] = res["canal model to true mm"] = res["canal found share"] = nan
    if len(true_line):
        res["canal found share"] = 0.0
        if len(pred_line):
            d_true = cKDTree(pred_line).query(true_line)[0]
            d_pred = cKDTree(true_line).query(pred_line)[0]
            res["canal true to model mm"] = float(d_true.mean())
            res["canal model to true mm"] = float(d_pred.mean())
            res["canal mm"] = float((d_true.mean() + d_pred.mean()) / 2)
            res["canal found share"] = float((d_true <= NEAR_MM).mean())

    # 2. bottom edge of the sinus
    true_edge = bottom_rows(true_lab == SINUS)
    pred_edge = bottom_rows(remove_specks((pred_lab == SINUS) & inside))
    res["sinus edge mm"] = res["sinus edge signed mm"] = res["sinus columns found share"] = nan
    if (true_edge >= 0).any():
        both = (true_edge >= 0) & (pred_edge >= 0)
        res["sinus columns found share"] = float(both.sum() / (true_edge >= 0).sum())
        if both.any():
            d = (pred_edge[both] - true_edge[both]) * VOXEL_MM  # rows count downward, so positive = lower
            res["sinus edge mm"] = float(np.abs(d).mean())
            res["sinus edge signed mm"] = float(d.mean())

    # 3. thickness of the lower jaw, tongue side to cheek side
    jaw_true = np.isin(true_lab, (1, 3, 5))
    jaw_pred = np.isin(pred_lab, (1, 3, 5)) & inside
    half = int(round(TEETH_REGION_MM / VOXEL_MM))
    cols = slice(FRAME_MIDDLE_COL - half, FRAME_MIDDLE_COL + half + 1)
    for mm in THICKNESS_BELOW_BITE_MM:
        row = FRAME_BITE_ROW + int(round(mm / VOXEL_MM))
        t = jaw_true[row, cols].sum(axis=1) * DEPTH_STEP_MM
        p = jaw_pred[row, cols].sum(axis=1) * DEPTH_STEP_MM
        ok = t > 0
        res[f"jaw thickness true mm {mm}"] = float(t[ok].mean()) if ok.any() else nan
        res[f"jaw thickness model mm {mm}"] = float(p[ok].mean()) if ok.any() else nan
        res[f"jaw thickness diff mm {mm}"] = float(np.abs(p[ok] - t[ok]).mean()) if ok.any() else nan

    # 4. bone density
    for c in (1, 2):
        bone = true_lab == c
        res[f"{CLASS_NAMES[c]} HU true"] = float(true_hu[bone].mean()) if bone.any() else nan
        res[f"{CLASS_NAMES[c]} HU model"] = float(pred_hu[bone].mean()) if bone.any() else nan
    bone = (true_lab == 1) | (true_lab == 2)
    res["bone err HU"] = float(np.abs(pred_hu[bone] - true_hu[bone]).mean()) if bone.any() else nan
    return res


def colors(lab):
    return LABEL_COLORS[np.minimum(lab, 5)] * (lab[..., None] != OUTSIDE)


def save_overview(path, title, xray, pred_hu, pred_lab, true_hu, true_lab):
    """Model output next to the correct answer: one cut along the jaw curve and one cut seen from above."""
    outside = true_lab == OUTSIDE  # blank the model output where there is no correct answer
    pred_hu, pred_lab = np.where(outside, -1000, pred_hu), np.where(outside, OUTSIDE, pred_lab)
    at_curve = int(round(DEPTH_IN_MM / DEPTH_STEP_MM))  # the depth step that is on the jaw curve
    row = FRAME_BITE_ROW + int(12 / VOXEL_MM)            # 12 mm below where the teeth meet
    fig, ax = plt.subplots(5, 2, figsize=(18, 17), gridspec_kw=dict(height_ratios=[3, 3, 3, 1.2, 1.2]))
    ax[0, 0].imshow(xray, cmap="gray", vmin=0, vmax=1); ax[0, 0].set_title("input: X-ray")
    legend = "colors: blue lower jaw, teal upper jaw, red nerve canal, yellow sinus, white teeth"
    ax[0, 1].text(0.0, 0.5, f"{title}\n\n{legend}", fontsize=10, va="center", family="monospace")
    for col, (hu, lab, name) in enumerate([(pred_hu, pred_lab, "MODEL OUTPUT"), (true_hu, true_lab, "CORRECT ANSWER")]):
        ax[1, col].imshow(hu[:, :, at_curve], cmap="gray", vmin=-1000, vmax=2500); ax[1, col].set_title(f"{name}: cut along the jaw curve")
        ax[2, col].imshow(colors(lab[:, :, at_curve])); ax[2, col].set_title(f"{name}: parts on the same cut")
        ax[3, col].imshow(hu[row].T, cmap="gray", vmin=-1000, vmax=2500, aspect=DEPTH_STEP_MM / VOXEL_MM); ax[3, col].set_title(f"{name}: seen from above, 12 mm below the bite (top edge = tongue side)")
        ax[4, col].imshow(colors(lab[row]).transpose(1, 0, 2), aspect=DEPTH_STEP_MM / VOXEL_MM)
    for a in ax.ravel():
        a.axis("off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.02, hspace=0.12)
    plt.savefig(path, dpi=70); plt.close(fig)


def save_cross_sections(path, title, pred_hu, pred_lab, true_hu, true_lab, places_mm=(-35, -25, -15, 15, 25, 35)):
    """Cuts across the jaw (the view a dentist uses to plan an implant) at fixed distances from the
    front of the jaw. Top row: model output. Bottom row: correct answer.
    Red line = the model's nerve canal. Green line = the true nerve canal. Yellow = sinus."""
    outside = true_lab == OUTSIDE
    pred_hu, pred_lab = np.where(outside, -1000, pred_hu), np.where(outside, 0, pred_lab)
    true_lab = np.where(outside, 0, true_lab)
    top, bottom = FRAME_BITE_ROW - int(24 / VOXEL_MM), FRAME_BITE_ROW + int(56 / VOXEL_MM)
    fig, ax = plt.subplots(2, len(places_mm), figsize=(2.6 * len(places_mm), 11))
    for j, mm in enumerate(places_mm):
        col = FRAME_MIDDLE_COL + int(round(mm / VOXEL_MM))
        side = "patient's right" if mm < 0 else "patient's left"
        for i, (hu, name) in enumerate([(pred_hu, "MODEL"), (true_hu, "CORRECT")]):
            a = ax[i, j]
            a.imshow(hu[top:bottom, col, :], cmap="gray", vmin=-1000, vmax=2500, aspect=VOXEL_MM / DEPTH_STEP_MM)
            own = pred_lab if i == 0 else true_lab
            if (own[top:bottom, col, :] == SINUS).any():
                a.contour(own[top:bottom, col, :] == SINUS, levels=[0.5], colors="yellow", linewidths=1.2)
            if (true_lab[top:bottom, col, :] == CANAL).any():
                a.contour(true_lab[top:bottom, col, :] == CANAL, levels=[0.5], colors="lime", linewidths=1.5)
            if i == 0 and (pred_lab[top:bottom, col, :] == CANAL).any():
                a.contour(pred_lab[top:bottom, col, :] == CANAL, levels=[0.5], colors="red", linewidths=1.5)
            a.set_title(f"{name}\n{abs(mm)} mm from the front,\n{side}", fontsize=9)
            a.axis("off")
    fig.suptitle(f"{title}\ncuts across the jaw (left edge = tongue side, right edge = cheek side). "
                 "Red = model's nerve canal, green = true nerve canal, yellow = sinus", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close(fig)


def describe(res):
    """A few lines of text for the picture and the printed log."""
    f = lambda key, digits=1: "none" if math.isnan(res[key]) else f"{res[key]:.{digits}f}"
    return "\n".join([
        f"nerve canal: middle line off by {f('canal mm')} mm, share found {f('canal found share', 2)}",
        f"sinus bottom edge: off by {f('sinus edge mm')} mm",
        f"lower jaw thickness 12 mm below bite: model {f('jaw thickness model mm 12')} mm, true {f('jaw thickness true mm 12')} mm",
        f"lower jaw thickness 20 mm below bite: model {f('jaw thickness model mm 20')} mm, true {f('jaw thickness true mm 20')} mm",
        f"lower jaw bone brightness: model {f('lower jaw HU model', 0)}, true {f('lower jaw HU true', 0)} Hounsfield units",
        f"upper jaw bone brightness: model {f('upper jaw HU model', 0)}, true {f('upper jaw HU true', 0)} Hounsfield units",
        "overlap (0 to 1): " + ", ".join(f"{CLASS_NAMES[c]} {f('dice ' + CLASS_NAMES[c], 2)}" for c in range(1, 6)),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder made by prepare_tf3.py")
    ap.add_argument("--pred", required=True, help="folder made by predict.py")
    ap.add_argument("--out", required=True, help="folder for the numbers and pictures")
    ap.add_argument("--no-pictures", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    numbers = sorted(p.stem for p in Path(args.pred).glob("*.npz"))
    if not numbers:
        raise SystemExit(f"ERROR: no model outputs found in {args.pred}")

    rows = []
    for num in numbers:
        with np.load(Path(args.data) / f"{num}.npz") as f:
            xray, true_hu, true_lab = f["xray"].astype(np.float32), f["hu"], f["labels"]
        with np.load(Path(args.pred) / f"{num}.npz") as f:
            pred_hu, pred_lab = f["hu"], f["labels"]
        if pred_hu.shape != true_hu.shape:
            raise SystemExit(f"ERROR: scan {num}: model output {pred_hu.shape} and correct answer {true_hu.shape} have different sizes")
        res = measure_one(true_hu, true_lab, pred_hu, pred_lab)
        rows.append(dict(scan=num, **res))
        text = describe(res)
        print(f"scan {num}\n  " + text.replace("\n", "\n  "), flush=True)
        if not args.no_pictures:
            save_overview(out / f"overview_{num}.png", f"scan {num}\n\n{text}", xray, pred_hu, pred_lab, true_hu, true_lab)
            save_cross_sections(out / f"cross_{num}.png", f"scan {num}", pred_hu, pred_lab, true_hu, true_lab)

    keys = [k for k in rows[0] if k != "scan"]
    table = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    counted = (~np.isnan(table)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.where(counted > 0, np.nansum(table, axis=0) / np.maximum(counted, 1), np.nan)
        spread = np.sqrt(np.where(counted > 0, np.nansum((table - mean) ** 2, axis=0) / np.maximum(counted, 1), np.nan))
    summary = {k: dict(average=float(mean[i]), spread=float(spread[i]), scans_counted=int(counted[i])) for i, k in enumerate(keys)}
    # Does the model tell a scan with dense bone from a scan with less dense bone?
    # 1 = the model's order of the scans is the same as the true order, 0 = no link.
    for name in ("lower jaw", "upper jaw"):
        t, p = table[:, keys.index(f"{name} HU true")], table[:, keys.index(f"{name} HU model")]
        ok = ~np.isnan(t) & ~np.isnan(p)
        summary[f"{name} HU link between model and true"] = float(np.corrcoef(t[ok], p[ok])[0, 1]) if ok.sum() >= 3 else float("nan")

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scan"] + keys)
        w.writeheader()
        w.writerows(rows)
        w.writerow(dict(scan="average", **{k: float(mean[i]) for i, k in enumerate(keys)}))
    (out / "summary.json").write_text(json.dumps(dict(scans=numbers, numbers=summary), indent=1))

    print(f"\nAVERAGE OVER {len(numbers)} SCANS (spread = how much the scans differ from the average)")
    for i, k in enumerate(keys):
        print(f"  {k:32s} {mean[i]:9.2f}   spread {spread[i]:7.2f}   scans counted {counted[i]}")
    for name in ("lower jaw", "upper jaw"):
        print(f"  {name} HU link between model and true: {summary[f'{name} HU link between model and true']:.2f}")
    print(f"numbers and pictures are in {out}")


if __name__ == "__main__":
    main()
