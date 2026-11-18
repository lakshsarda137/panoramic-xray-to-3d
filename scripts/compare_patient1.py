"""Patient 1: turn the model's output into a "virtual" 3D scan and compare it with the real 3D scan.

Run:  .venv/bin/python scripts/compare_patient1.py [VERSION]     (VERSION is p1_matched if left out)

Needs (all in data/, ignored by git):
  data/patient1/xray/VERSION.npz                              the X-ray the model got (scripts/place_xray.py)
  data/patient1/kaggle_out/predictions/patient1/VERSION.npz   the model's output (Kaggle notebook xray-to-3d-patient1)
  data/training_stats.npz                                     the average jaw curve (scripts/training_stats.py)
  data/patient1/real_block.npz, cbct_0.4mm.npy                the real 3D scan (scripts/prepare_patient1.py)

PART 1 builds the virtual 3D scan. It uses only the model's output and the average jaw curve of the
training scans. It does not use Patient 1's real 3D scan.
  - Where the X-ray picture is black (outside the picture), the model's output is thrown away.
  - The block is bent along the average jaw curve into a normal 3D scan.
  Saved as data/patient1/virtual_scan_VERSION.npz (hu and labels, [z, y, x], 0.4 mm per voxel).

PART 2 compares with the real 3D scan. Only this part uses the real scan.
  The X-ray and the real 3D scan do not line up by themselves, for three reasons:
    a. A panoramic X-ray is stretched sideways by a different amount in different places.
    b. The patient's lower jaw was more open at the front in the 3D scan than in the X-ray.
    c. The real jaw curve of the patient is not the average jaw curve.
  So places are matched by teeth: MATCH_COLUMNS below lists the column of the same tooth in the
  X-ray and in the real block. The pairs were placed by eye. They are used only for comparing.
  Heights are counted from the lower edge of the jaw bone, which does not depend on how open the mouth is.

  Measured at every 10th column that holds lower jaw in both, at 8 mm and 16 mm above the lower edge:
    jaw thickness      from the tongue-side surface to the cheek-side surface of the bone, in mm
    jaw brightness     average brightness between those two surfaces, in Hounsfield units (brighter = denser)
  For the real scan "bone" means brighter than 700 Hounsfield units (the hard outer layer of the jaw).
  For the model "bone" means the points the model marked as lower jaw, nerve canal or teeth.

  Patient 1 has no label file, so the nerve canal and the sinus of the real scan are not marked.
  Their position cannot be measured here. The pictures show the model's canal drawn on the real scan.

Pictures and numbers are saved in outputs/patient1/ (ignored by git).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import curved_volume as cv

P1 = ROOT / "data" / "patient1"
OUT = ROOT / "outputs" / "patient1"
ON_CURVE = int(round(cv.DEPTH_IN_MM / cv.DEPTH_STEP_MM))  # the depth step that is on the jaw curve
BELOW_BITE = cv.FRAME_BITE_ROW + 5                        # rows below this hold only the lower jaw
REAL_BONE_HU = 700

# The same tooth in the X-ray (first number) and in the real block (second number), as frame columns.
# Placed by eye on enlarged pictures. From the patient's right to the patient's left:
# back end of the last right molar, right implant, gap between the two lower front teeth,
# root-filled left molar, back end of the last left molar.
MATCH_COLUMNS = [(130, 157), (179, 206), (273, 277), (366, 374), (400, 405)]
HEIGHTS_MM = (8, 16)


def to_real_column(c):
    """The real-block column that shows the same place as X-ray column c."""
    xs, rs = zip(*MATCH_COLUMNS)
    if c <= xs[0]:
        return c + rs[0] - xs[0]
    if c >= xs[-1]:
        return c + rs[-1] - xs[-1]
    return float(np.interp(c, xs, rs))


def biggest_piece(mask):
    pieces, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(pieces.ravel()); sizes[0] = 0
    return pieces == sizes.argmax()


def jaw_lines(jaw, hu, col):
    """For one column of a block: the lower edge of the jaw, and for each height above it the
    thickness in mm and the average brightness between the two surfaces. `jaw` is a yes/no block."""
    cut = jaw[:, col, :]
    rows = np.flatnonzero(cut.any(axis=1))
    if len(rows) == 0:
        return None
    edge = rows.max()
    res = dict(edge=int(edge))
    for mm in HEIGHTS_MM:
        r = edge - int(round(mm / cv.VOXEL_MM))
        deps = np.flatnonzero(cut[r - 1:r + 2].any(axis=0))  # 3 rows (1.2 mm) together, to be less jumpy
        if r < BELOW_BITE or len(deps) == 0:
            res[mm] = None
            continue
        res[mm] = dict(row=int(r), first=int(deps.min()), last=int(deps.max()),
                       thickness=float((deps.max() - deps.min() + 1) * cv.DEPTH_STEP_MM),
                       brightness=float(hu[r - 1:r + 2, col, deps.min():deps.max() + 1].mean()))
    return res


def build_virtual_scan(version):
    """PART 1. Uses only the X-ray, the model's output and the average jaw curve."""
    xray = np.load(P1 / "xray" / f"{version}.npz")["xray"].astype(np.float32)
    with np.load(P1 / "kaggle_out" / "predictions" / "patient1" / f"{version}.npz") as f:
        hu, lab = f["hu"].astype(np.float32), f["labels"].copy()
    has_picture = np.zeros(xray.shape, bool)
    rows, cols = np.flatnonzero((xray > 0).any(axis=1)), np.flatnonzero((xray > 0).any(axis=0))
    has_picture[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = True
    hu[~has_picture] = -1000
    lab[~has_picture] = cv.OUTSIDE

    stats = np.load(ROOT / "data" / "training_stats.npz")
    curve = {k: stats[k] for k in ("px", "py", "nx", "ny")}
    c0 = int(stats["first_col"]); c1 = c0 + len(curve["px"])
    shape = (hu.shape[0], int(curve["py"].max()) + 40, int(curve["px"].max()) + 40)
    scan_hu = cv.bend_back(hu[:, c0:c1], curve, shape, fill=-1000.0)
    scan_lab = cv.bend_back(np.where(lab == cv.OUTSIDE, 0, lab)[:, c0:c1], curve, shape, fill=0, is_label=True)
    np.savez_compressed(P1 / f"virtual_scan_{version}.npz", hu=scan_hu.round().astype(np.int16), labels=scan_lab.astype(np.uint8))
    return xray, hu, lab, scan_hu, scan_lab, curve


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "p1_matched"
    OUT.mkdir(parents=True, exist_ok=True)
    xray, m_hu, m_lab, v_hu, v_lab, avg_curve = build_virtual_scan(version)
    print(f"virtual 3D scan: {v_hu.shape} voxels of {cv.VOXEL_MM} mm, saved in {P1 / f'virtual_scan_{version}.npz'}")

    # ---------------- PART 2: compare with the real scan
    real = np.load(P1 / "real_block.npz")
    r_hu, r_inside = real["hu"].astype(np.float32), real["labels"] != cv.OUTSIDE
    real_info = json.loads(str(real["info"]))
    real_scan = np.load(P1 / "cbct_0.4mm.npy")

    both = r_inside & (m_lab != cv.OUTSIDE)
    diff = (m_hu - r_hu)[both] / 4000.0
    numbers = {"brightness error HU, blocks as they are (not lined up)": float(np.abs(diff).mean() * 4000),
               "PSNR, blocks as they are (not lined up)": float(-10 * np.log10((diff ** 2).mean()))}

    lower = np.zeros(m_lab.shape, bool); lower[BELOW_BITE:] = True
    m_jaw = biggest_piece(np.isin(m_lab, (1, 3, 5)) & lower)
    r_jaw = biggest_piece((r_hu > REAL_BONE_HU) & r_inside & lower)
    places = []
    for c in range(100, 450, 10):
        rc = int(round(to_real_column(c)))
        m, r = jaw_lines(m_jaw, m_hu, c), jaw_lines(r_jaw, r_hu, rc)
        if m is None or r is None:
            continue
        for mm in HEIGHTS_MM:
            if m[mm] and r[mm]:
                places.append(dict(xray_column=c, real_column=rc, height_mm=mm, model=m[mm], real=r[mm],
                                   model_edge=m["edge"], real_edge=r["edge"]))
    for mm in HEIGHTS_MM:
        here = [p for p in places if p["height_mm"] == mm]
        mt, rt = np.array([p["model"]["thickness"] for p in here]), np.array([p["real"]["thickness"] for p in here])
        mb, rb = np.array([p["model"]["brightness"] for p in here]), np.array([p["real"]["brightness"] for p in here])
        numbers[f"{mm} mm above the lower edge of the jaw"] = {
            "places": len(here),
            "thickness mm, model": float(mt.mean()), "thickness mm, real": float(rt.mean()),
            "thickness difference mm per place": float(np.abs(mt - rt).mean()),
            "brightness HU, model": float(mb.mean()), "brightness HU, real": float(rb.mean()),
        }
    (OUT / f"numbers_{version}.json").write_text(json.dumps(dict(numbers=numbers, places=places), indent=1))
    print(json.dumps(numbers, indent=1))

    # ---------------- pictures
    colors = np.array([[0, 0, 0], [70, 110, 200], [60, 170, 170], [255, 40, 40], [255, 220, 0], [240, 240, 240]], np.uint8)
    paint = lambda lab: colors[np.minimum(lab, 5)] * (lab[..., None] != cv.OUTSIDE)
    gray = dict(cmap="gray", vmin=-1000, vmax=2500)

    # picture 1: the X-ray, and cuts along the jaw curve
    fig, ax = plt.subplots(2, 2, figsize=(17, 8.5))
    ax[0, 0].imshow(xray, cmap="gray", vmin=0, vmax=1); ax[0, 0].set_title("Patient 1: real panoramic X-ray, as given to the model")
    ax[0, 1].imshow(paint(m_lab[:, :, ON_CURVE])); ax[0, 1].set_title("model output: parts (blue lower jaw, teal upper jaw, red nerve canal, yellow sinus, white teeth)")
    ax[1, 0].imshow(np.where(m_lab[:, :, ON_CURVE] == cv.OUTSIDE, -1000, m_hu[:, :, ON_CURVE]), **gray); ax[1, 0].set_title("VIRTUAL 3D scan (model output), cut along the jaw curve")
    ax[1, 1].imshow(r_hu[:, :, ON_CURVE], **gray); ax[1, 1].set_title("REAL 3D scan, cut along the patient's own jaw curve")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / f"1_along_the_jaw_{version}.png", dpi=85); plt.close(fig)

    # picture 2: slices seen from above, at the same distance below the bite
    fig, ax = plt.subplots(2, 4, figsize=(18, 9.5))
    v_bite = v_hu.shape[0] - 1 - cv.FRAME_BITE_ROW
    half = 150
    for j, mm in enumerate((-6, 8, 16, 24)):
        dz = int(round(mm / cv.VOXEL_MM))
        vz, rz = v_bite - dz, real_info["bite"] - dz
        v_mid = (avg_curve["px"][len(avg_curve["px"]) // 2], avg_curve["py"][len(avg_curve["py"]) // 2])
        r_mid = (real["px"][real_info["middle"]], real["py"][real_info["middle"]])
        ax[0, j].imshow(v_hu[vz], **gray); ax[1, j].imshow(real_scan[rz], **gray)
        for a, mid in ((ax[0, j], v_mid), (ax[1, j], r_mid)):
            a.set_xlim(mid[0] - half, mid[0] + half); a.set_ylim(mid[1] + 2 * half - 60, mid[1] - 60); a.axis("off")
        where = f"{abs(mm)} mm {'above' if mm < 0 else 'below'} the bite"
        ax[0, j].set_title(f"VIRTUAL, {where}"); ax[1, j].set_title(f"REAL, {where}")
    fig.suptitle("Patient 1, seen from above (front of the jaw at the top). The virtual scan only holds a band around the jaw curve.")
    plt.tight_layout(); plt.savefig(OUT / f"2_from_above_{version}.png", dpi=85); plt.close(fig)

    # picture 3: cuts across the jaw at matched places, with the model's canal drawn on the real cut
    xcols = [115, 140, 180, 366, 400, 425]
    top, bottom = 60, 200
    fig, ax = plt.subplots(2, len(xcols), figsize=(2.9 * len(xcols), 12))
    for j, c in enumerate(xcols):
        rc = int(round(to_real_column(c)))
        m, r = jaw_lines(m_jaw, m_hu, c), jaw_lines(r_jaw, r_hu, rc)
        side = "patient's right" if c < cv.FRAME_MIDDLE_COL else "patient's left"
        m_cut = np.where(m_lab[top:bottom, c] == cv.OUTSIDE, -1000, m_hu[top:bottom, c])
        ax[0, j].imshow(m_cut, aspect=cv.VOXEL_MM / cv.DEPTH_STEP_MM, **gray)
        ax[1, j].imshow(r_hu[top:bottom, rc], aspect=cv.VOXEL_MM / cv.DEPTH_STEP_MM, **gray)
        canal = m_lab[top:bottom, c] == 3
        if canal.any():
            ax[0, j].contour(canal, levels=[0.5], colors="red", linewidths=1.5)
            if m and r and m[8] and r[8]:
                # move the model's canal so that the lower edge and the middle of the jaw match the real cut
                d_row = r["edge"] - m["edge"]
                d_dep = int(round((r[8]["first"] + r[8]["last"]) / 2 - (m[8]["first"] + m[8]["last"]) / 2))
                rr, dd = np.nonzero(canal)
                moved = np.zeros_like(canal)
                ok = (rr + d_row >= 0) & (rr + d_row < canal.shape[0]) & (dd + d_dep >= 0) & (dd + d_dep < canal.shape[1])
                moved[(rr + d_row)[ok], (dd + d_dep)[ok]] = True
                ax[1, j].contour(moved, levels=[0.5], colors="red", linewidths=1.5)
        ax[0, j].set_title(f"VIRTUAL\n{abs(c - cv.FRAME_MIDDLE_COL) * cv.VOXEL_MM:.0f} mm from the front,\n{side}", fontsize=9)
        ax[1, j].set_title("REAL, same tooth position", fontsize=9)
        ax[0, j].axis("off"); ax[1, j].axis("off")
    fig.suptitle("Patient 1, cuts across the jaw (left edge = tongue side, right edge = cheek side).\n"
                 "Red = nerve canal from the model. On the real cuts it is drawn after matching the lower edge and the middle of the jaw.", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT / f"3_across_the_jaw_{version}.png", dpi=85); plt.close(fig)
    print("pictures are in", OUT)


if __name__ == "__main__":
    main()
