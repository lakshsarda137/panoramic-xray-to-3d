"""Patient 1: is the error caused by the real X-ray looking different from the fake X-rays used in training?

Run:  .venv/bin/python scripts/compare_fake_patient1.py

The model was run on two X-rays of Patient 1 (Kaggle notebook xray-to-3d-patient1):
  p1_matched  the real panoramic X-ray (scripts/place_xray.py)
  p1_fake     an X-ray computed from Patient 1's real 3D scan with the same program that made the training
              X-rays (the "xray" array in data/patient1/real_block.npz, scripts/prepare_patient1.py).
              This X-ray uses Patient 1's own jaw curve, so the model's output lines up with the real block
              column by column. This is a check only: in real use no 3D scan exists to make such an X-ray.

If the result from p1_fake is much better than from p1_matched, the main problem is that real X-rays look
different from the fake ones. If both are equally bad, the main problem is something else (new scanner,
much metal, the shape of this patient's jaw).

Measured the same way as scripts/compare_patient1.py: jaw thickness and brightness between the two bone
surfaces, 8 mm and 16 mm above the lower edge of the jaw. For p1_fake, column c of the output is compared
with column c of the real block. For p1_matched, columns are matched by teeth (MATCH_COLUMNS).
Also: brightness difference over the whole block (same score as on the 10 test scans).

Saves outputs/patient1/numbers_fake_vs_real.json and outputs/patient1/4_fake_vs_real_xray.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import compare_patient1 as cp  # noqa: E402
from src import curved_volume as cv  # noqa: E402

P1, OUT = cp.P1, cp.OUT


def load(version):
    """The model's output, with the part outside the X-ray picture thrown away."""
    xray = np.load(P1 / "xray" / f"{version}.npz")["xray"].astype(np.float32)
    with np.load(P1 / "kaggle_out" / "predictions" / "patient1" / f"{version}.npz") as f:
        hu, lab = f["hu"].astype(np.float32), f["labels"].copy()
    rows, cols = np.flatnonzero((xray > 0).any(axis=1)), np.flatnonzero((xray > 0).any(axis=0))
    keep = np.zeros(xray.shape, bool)
    keep[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = True
    hu[~keep] = -1000
    lab[~keep] = cv.OUTSIDE
    return xray, hu, lab


def measure(m_hu, m_lab, r_hu, r_inside, column_map):
    both = r_inside & (m_lab != cv.OUTSIDE)
    diff = (m_hu - r_hu)[both] / 4000.0
    res = {"brightness difference HU, whole block": float(np.abs(diff).mean() * 4000),
           "PSNR, whole block": float(-10 * np.log10((diff ** 2).mean()))}
    lower = np.zeros(m_lab.shape, bool); lower[cp.BELOW_BITE:] = True
    m_jaw = cp.biggest_piece(np.isin(m_lab, (1, 3, 5)) & lower)
    r_jaw = cp.biggest_piece((r_hu > cp.REAL_BONE_HU) & r_inside & lower)
    for mm in cp.HEIGHTS_MM:
        mt, rt, mb, rb = [], [], [], []
        for c in range(100, 450, 10):
            m, r = cp.jaw_lines(m_jaw, m_hu, c), cp.jaw_lines(r_jaw, r_hu, column_map(c))
            if m and r and m[mm] and r[mm]:
                mt.append(m[mm]["thickness"]); rt.append(r[mm]["thickness"])
                mb.append(m[mm]["brightness"]); rb.append(r[mm]["brightness"])
        mt, rt, mb, rb = map(np.array, (mt, rt, mb, rb))
        res[f"{mm} mm above the lower edge"] = {
            "places": len(mt), "thickness mm, model": float(mt.mean()), "thickness mm, real": float(rt.mean()),
            "thickness difference mm per place": float(np.abs(mt - rt).mean()),
            "brightness HU, model": float(mb.mean()), "brightness HU, real": float(rb.mean())}
    return res


def main():
    real = np.load(P1 / "real_block.npz")
    r_hu, r_inside = real["hu"].astype(np.float32), real["labels"] != cv.OUTSIDE
    x_f, hu_f, lab_f = load("p1_fake")
    x_m, hu_m, lab_m = load("p1_matched")
    numbers = {"p1_fake (X-ray computed from the real 3D scan)": measure(hu_f, lab_f, r_hu, r_inside, lambda c: c),
               "p1_matched (real X-ray)": measure(hu_m, lab_m, r_hu, r_inside, lambda c: int(round(cp.to_real_column(c))))}
    (OUT / "numbers_fake_vs_real.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))

    gray = dict(cmap="gray", vmin=-1000, vmax=2500)
    on = cp.ON_CURVE
    fig, ax = plt.subplots(3, 2, figsize=(17, 11))
    ax[0, 0].imshow(x_f, cmap="gray", vmin=0, vmax=1); ax[0, 0].set_title("Input A: X-ray computed from Patient 1's real 3D scan")
    ax[0, 1].imshow(x_m, cmap="gray", vmin=0, vmax=1); ax[0, 1].set_title("Input B: Patient 1's real X-ray")
    ax[1, 0].imshow(np.where(lab_f[:, :, on] == cv.OUTSIDE, -1000, hu_f[:, :, on]), **gray); ax[1, 0].set_title("3D scan built from input A, cut along the jaw curve")
    ax[1, 1].imshow(np.where(lab_m[:, :, on] == cv.OUTSIDE, -1000, hu_m[:, :, on]), **gray); ax[1, 1].set_title("3D scan built from input B, cut along the jaw curve")
    ax[2, 0].imshow(r_hu[:, :, on], **gray); ax[2, 0].set_title("Real 3D scan, cut along Patient 1's own jaw curve")
    ax[2, 1].axis("off")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "4_fake_vs_real_xray.png", dpi=80); plt.close(fig)

    cols = [140, 200, 272, 350, 400]
    top, bottom = 40, 200
    fig, ax = plt.subplots(3, len(cols), figsize=(3 * len(cols), 13))
    for j, c in enumerate(cols):
        rc = int(round(cp.to_real_column(c)))
        for i, (img, title) in enumerate(((hu_f[top:bottom, c], f"from input A, column {c}"),
                                          (hu_m[top:bottom, c], f"from input B, column {c}"),
                                          (r_hu[top:bottom, c], f"real, column {c}"))):
            ax[i, j].imshow(img, aspect=cv.VOXEL_MM / cv.DEPTH_STEP_MM, **gray); ax[i, j].set_title(title, fontsize=9); ax[i, j].axis("off")
    fig.suptitle("Cuts across the jaw (left = tongue side, right = cheek side). Column 272 = front of the jaw.\n"
                 "Input B is not lined up with the real scan column by column (see compare_patient1.py).", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT / "5_fake_vs_real_across.png", dpi=80); plt.close(fig)
    print("pictures are in", OUT)


if __name__ == "__main__":
    main()
