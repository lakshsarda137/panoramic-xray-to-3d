"""Turn every ToothFairy3 scan into one training example.

For each scan this makes:
  - the fake panoramic X-ray (what the model gets as input)
  - the straightened 3D block of the scan and of the labels (what the model must output)
and saves them in one file per scan, plus one small picture per scan to look at.

Run:  python scripts/prepare_tf3.py --data data/tf3F_0.4mm --out data/prepared
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import curved_volume as cv
from src.fake_panoramic import hu_to_mu, look_like_real, make_panoramic
from src.tf3_io import CLASS_NAMES, list_scans, load_labels, load_scan, merge_labels

MIN_SLICES = 150       # scans shorter than this (60 mm) do not hold the whole jaw and are skipped
MIN_ROWS_ABOVE = 30    # scans with less than this (12 mm) above the bite do not hold the upper teeth and are skipped
LABEL_COLORS = np.array([[0, 0, 0], [70, 110, 200], [60, 170, 170], [255, 40, 40], [255, 220, 0], [240, 240, 240]], np.uint8)


def prepare_one(scan_path, label_path):
    vol = load_scan(scan_path)
    lab_original = load_labels(label_path)
    lab = merge_labels(lab_original)
    if vol.shape != lab.shape:
        raise ValueError(f"scan {vol.shape} and labels {lab.shape} have different sizes")
    if vol.shape[0] < MIN_SLICES:
        raise ValueError(f"only {vol.shape[0]} slices")
    bite, curve = cv.jaw_curve_from_labels(lab_original)
    if vol.shape[0] - 1 - bite < MIN_ROWS_ABOVE:
        raise ValueError(f"only {vol.shape[0] - 1 - bite} slices above the bite")
    xray = look_like_real(make_panoramic(hu_to_mu(vol), curve["cx"], curve["cy"], curve["center"], cv.VOXEL_MM))
    hu = cv.straighten(vol, curve, is_label=False)
    labels = cv.straighten(lab, curve, is_label=True)
    bite_row = vol.shape[0] - 1 - bite  # row 0 is the top of the head
    xray_f, dr, dc = cv.to_frame(xray.astype(np.float32), bite_row, curve["middle"], 0.0)
    hu_f, _, _ = cv.to_frame(np.clip(hu, -1000, 3000).round().astype(np.int16), bite_row, curve["middle"], -1000)
    lab_f, _, _ = cv.to_frame(labels, bite_row, curve["middle"], cv.OUTSIDE)
    info = dict(shape_zyx=list(vol.shape), bite=int(bite), row_shift=int(dr), col_shift=int(dc),
                middle=int(curve["middle"]), n_columns=int(len(curve["px"])))
    return dict(xray=xray_f.astype(np.float16), hu=hu_f, labels=lab_f,
                px=curve["px"], py=curve["py"], nx=curve["nx"], ny=curve["ny"], info=json.dumps(info)), info


def save_preview(ex, path, title):
    """One picture per scan: the X-ray, then two cuts through the block with the labels in color."""
    hu, lab = ex["hu"].astype(np.float32), ex["labels"]
    at_curve = int(round(cv.DEPTH_IN_MM / cv.DEPTH_STEP_MM))  # the depth sample that is on the jaw curve
    row = cv.FRAME_BITE_ROW + int(12 / cv.VOXEL_MM)            # 12 mm below where the teeth meet
    fig, ax = plt.subplots(5, 1, figsize=(10, 14), gridspec_kw=dict(height_ratios=[3, 3, 3, 1, 1]))
    ax[0].imshow(ex["xray"].astype(np.float32), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title(f"{title}: fake X-ray (model input)")
    ax[1].imshow(hu[:, :, at_curve], cmap="gray", vmin=-1000, vmax=2500)
    ax[1].set_title("block, cut along the jaw curve")
    ax[2].imshow(LABEL_COLORS[np.minimum(lab[:, :, at_curve], 5)] * (lab[:, :, at_curve, None] != cv.OUTSIDE))
    ax[2].set_title("labels on the same cut: blue lower jaw, teal upper jaw, red nerve canal, yellow sinus, white teeth")
    ax[3].imshow(hu[row].T, cmap="gray", vmin=-1000, vmax=2500, aspect=cv.DEPTH_STEP_MM / cv.VOXEL_MM)
    ax[3].set_title("block seen from above, 12 mm below the bite (top edge = tongue side, bottom edge = cheek side)")
    ax[4].imshow((LABEL_COLORS[np.minimum(lab[row], 5)] * (lab[row][..., None] != cv.OUTSIDE)).transpose(1, 0, 2),
                 aspect=cv.DEPTH_STEP_MM / cv.VOXEL_MM)
    for a in ax:
        a.axis("off")
    ax[1].axhline(cv.FRAME_BITE_ROW, color="lime", lw=0.5)
    ax[1].axvline(cv.FRAME_MIDDLE_COL, color="lime", lw=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=60)
    plt.close(fig)


def make_sheets(out, numbers, per_sheet=8):
    """Pictures that show many scans at once: left the X-ray, right the block cut along the jaw curve
    with the labels drawn over it in color. The green lines mark the bite row and the front of the jaw."""
    at_curve = int(round(cv.DEPTH_IN_MM / cv.DEPTH_STEP_MM))
    for start in range(0, len(numbers), per_sheet):
        tiles = []
        for num in numbers[start:start + per_sheet]:
            with np.load(out / f"{num}.npz") as f:
                xray, hu, lab = f["xray"].astype(np.float32), f["hu"][:, :, at_curve].astype(np.float32), f["labels"][:, :, at_curve]
            gray = np.clip((hu + 1000) / 3500, 0, 1)[..., None] * np.ones(3)
            color = LABEL_COLORS[np.minimum(lab, 5)] / 255.0
            marked = (lab > 0) & (lab != cv.OUTSIDE)
            cut = np.where(marked[..., None], 0.55 * gray + 0.45 * color, gray)
            left = xray[..., None] * np.ones(3)
            for img in (left, cut):
                img[cv.FRAME_BITE_ROW, ::4] = (0, 1, 0)
                img[::4, cv.FRAME_MIDDLE_COL] = (0, 1, 0)
            tiles.append(np.concatenate([left, cut], axis=1))
        fig, ax = plt.subplots(len(tiles), 1, figsize=(13.6, 3.2 * len(tiles)), squeeze=False)
        for a, tile, num in zip(ax[:, 0], tiles, numbers[start:start + per_sheet]):
            a.imshow(tile); a.axis("off"); a.set_title(f"scan {num}", fontsize=9, loc="left", pad=2)
        plt.subplots_adjust(left=0, right=1, top=0.985, bottom=0, hspace=0.08)
        plt.savefig(out / "previews" / f"sheet_{start // per_sheet + 1:02d}.png", dpi=80)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder that holds imagesTr and labelsTr")
    ap.add_argument("--out", required=True, help="folder for the prepared examples")
    ap.add_argument("--only", nargs="*", help="scan numbers to do, for a quick test (default: all)")
    ap.add_argument("--redo", action="store_true", help="make files again even if they exist")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "previews").mkdir(parents=True, exist_ok=True)
    scans = [s for s in list_scans(args.data) if not args.only or s[0] in args.only]
    print(f"{len(scans)} scans found", flush=True)
    skipped = {}
    for i, (num, scan_path, label_path) in enumerate(scans):
        target = out / f"{num}.npz"
        if target.exists() and not args.redo:
            continue
        t0 = time.time()
        try:
            ex, info = prepare_one(scan_path, label_path)
        except Exception as e:  # one bad scan must not stop the others
            skipped[num] = repr(e)
            target.unlink(missing_ok=True)  # remove a file left by an older run
            print(f"[{i + 1}/{len(scans)}] {num} SKIPPED: {e!r}", flush=True)
            continue
        np.savez_compressed(target, **ex)
        save_preview(ex, out / "previews" / f"{num}.png", f"scan {num}")
        present = [CLASS_NAMES[c] for c in range(1, 6) if (ex["labels"] == c).any()]
        print(f"[{i + 1}/{len(scans)}] {num} done in {time.time() - t0:.0f}s, columns {info['n_columns']}, "
              f"has: {', '.join(present)}", flush=True)
    done = sorted(p.stem for p in out.glob("*.npz"))
    make_sheets(out, done)
    (out / "prepare_report.json").write_text(json.dumps(dict(done=done, skipped=skipped), indent=1))
    print(f"{len(done)} examples are in {out}. Skipped: {skipped if skipped else 'none'}")


if __name__ == "__main__":
    main()
