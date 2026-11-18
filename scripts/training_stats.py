"""Facts about the 46 training scans that are needed to use the model on a real X-ray.

Run:  .venv/bin/python scripts/training_stats.py --data PREPARED_FOLDER

It saves data/training_stats.npz with:
  px, py, nx, ny, middle   the average jaw curve of the training scans (same layout as the curve in a
                           prepared scan file: one point per column, 1 voxel = 0.4 mm apart, plus the
                           direction across the curve). A real X-ray does not show the patient's own
                           jaw curve, so this average curve is used to bend the model's output into a
                           normal 3D scan.
  first_col                the frame column of the first point of the average curve
  xray_quantiles           how bright the fake X-rays are (1001 evenly spaced steps from the darkest
                           to the brightest pixel). A real X-ray is changed to have the same spread
                           of brightness before the model gets it.
  picture_box              the usual first row, last row, first column, last column of the part of
                           the frame that holds picture (the rest of the frame is black)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.curved_volume import FRAME_MIDDLE_COL, OUTSIDE
from src.train_data import make_split


def average_curve(curves, need=0.5):
    """curves: list of (px, py, middle). Returns (px, py, nx, ny, middle) of the average curve.

    For every step along the curve, counted from the front of the jaw, the direction in which the
    curve runs is averaged over the scans. Left and right are made equal. Then the average directions
    are put one after another, 1 voxel per step, which gives the average curve. Steps that fewer than
    `need` of the scans reach are left out."""
    half = FRAME_MIDDLE_COL
    angle_sum, count = np.zeros(2 * half), np.zeros(2 * half)
    for px, py, middle in curves:
        angle = np.unwrap(np.arctan2(np.gradient(py), np.gradient(px)))
        angle = angle - angle[middle]  # turn every scan so its curve runs the same way at the front
        k = np.arange(len(px)) - middle + half
        ok = (k >= 0) & (k < 2 * half)
        angle_sum[k[ok]] += angle[ok]; count[k[ok]] += 1
    enough = count >= need * len(curves)
    reach = min(np.flatnonzero(enough).max() - half, half - np.flatnonzero(enough).min())  # same on both sides
    k = np.arange(-reach, reach + 1)
    angle = angle_sum[k + half] / count[k + half]
    angle = (angle - angle[::-1]) / 2  # left and right equal
    dx, dy = np.cos(angle), np.sin(angle)
    px = np.concatenate([-np.cumsum(dx[:reach][::-1])[::-1], [0.0], np.cumsum(dx[reach + 1:])])
    py = np.concatenate([-np.cumsum(dy[:reach][::-1])[::-1], [0.0], np.cumsum(dy[reach + 1:])])
    # In a scan, x grows toward the patient's left and y toward the back of the head. The curve
    # starts at the patient's right end, so at the front it runs in the +x direction, and the
    # two ends must be further back (larger y) than the front.
    if py[0] < 0:
        py = -py
    nx, ny = np.gradient(py), -np.gradient(px)
    norm = np.hypot(nx, ny); nx, ny = nx / norm, ny / norm
    if ny[reach] > 0:  # the direction across the curve must point outward: toward the front at the front
        nx, ny = -nx, -ny
    px, py = px - px.min() + 40, py - py.min() + 40  # leave room around the curve
    return px, py, nx, ny, reach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder made by prepare_tf3.py")
    args = ap.parse_args()
    numbers = make_split(args.data)["train"]
    curves, values, boxes = [], [], []
    for num in numbers:
        with np.load(Path(args.data) / f"{num}.npz") as f:
            info = json.loads(str(f["info"]))
            curves.append((f["px"], f["py"], info["middle"]))
            has_picture = (f["labels"] != OUTSIDE).any(axis=2)
            xray = f["xray"].astype(np.float32)
        values.append(xray[has_picture])
        rows, cols = np.flatnonzero(has_picture.any(axis=1)), np.flatnonzero(has_picture.any(axis=0))
        boxes.append([rows[0], rows[-1], cols[0], cols[-1]])
    px, py, nx, ny, middle = average_curve(curves)
    quantiles = np.quantile(np.concatenate(values), np.linspace(0, 1, 1001))
    box = np.median(np.array(boxes), axis=0).astype(int)
    out = ROOT / "data" / "training_stats.npz"
    np.savez(out, px=px, py=py, nx=nx, ny=ny, middle=middle, first_col=FRAME_MIDDLE_COL - middle,
             xray_quantiles=quantiles, picture_box=box)
    print(f"{len(numbers)} training scans")
    print(f"average jaw curve: {len(px)} points, {len(px) * 0.4:.0f} mm long, "
          f"{(px.max() - px.min()) * 0.4:.0f} mm wide, {(py.max() - py.min()) * 0.4:.0f} mm front to back")
    print("usual picture box (first row, last row, first column, last column):", box.tolist())
    print("fake X-ray brightness at 5%, 25%, 50%, 75%, 95%:", np.round(quantiles[[50, 250, 500, 750, 950]], 2).tolist())
    print("saved", out)


if __name__ == "__main__":
    main()
