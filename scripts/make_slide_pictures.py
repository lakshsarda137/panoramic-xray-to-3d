"""Make the pictures that go into the slides. They are saved in outputs/slides/img/ (ignored by git).

Run:  .venv/bin/python scripts/make_slide_pictures.py PREPARED_SCAN.npz MODEL_OUTPUT.npz
      (without the two files, only the Patient 1 pictures are made again)

PREPARED_SCAN.npz  one prepared test scan (for example 032.npz made by scripts/prepare_tf3.py)
MODEL_OUTPUT.npz   the model's output for the same scan (from the Kaggle notebook xray-to-3d-scoring,
                   predictions/test/032.npz)
The Patient 1 files are read from data/patient1/ (see scripts/compare_patient1.py).
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_patient1 import BELOW_BITE, MATCH_COLUMNS, REAL_BONE_HU, biggest_piece, jaw_lines, to_real_column
from src import curved_volume as cv

P1 = ROOT / "data" / "patient1"
OUT = ROOT / "outputs" / "slides" / "img"
TEAL, AMBER, RED, GREEN = "#2DD4BF", "#FBBF24", "#FF4D4D", "#4ADE80"


def save_gray(name, img, lo, hi, scale=1.0, aspect=1.0):
    """Save a picture with nothing around it. aspect = height of one pixel / width of one pixel."""
    x = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
    h, w = x.shape
    size = (max(1, round(w * scale)), max(1, round(h * scale * aspect)))
    x = cv2.resize(x, size, interpolation=cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA)
    cv2.imwrite(str(OUT / name), (np.clip(x, 0, 1) * 255).astype(np.uint8))


def surface_view(vol, hard_hu, turn_deg):
    """A picture of the hard parts (bone, teeth) of a scan [z, y, x] seen from the front, lit from the
    upper left, the way a 3D viewer shows a skull. turn_deg turns the head around the up-down line."""
    v = ndimage.rotate(vol, turn_deg, axes=(1, 2), reshape=True, order=1, cval=-1000) if turn_deg else vol
    v = ndimage.gaussian_filter(v, 0.7)
    hard = v > hard_hu
    hit = hard.any(axis=1)
    depth = hard.argmax(axis=1).astype(np.float32)   # how far behind the front of the scan the surface is
    depth = np.where(hit, depth, depth[hit].max() if hit.any() else 0)
    depth = ndimage.gaussian_filter(depth, 0.8)
    dz, dx = np.gradient(depth)
    normal = np.stack([-dx, dz, np.ones_like(dx)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.array([-0.45, 0.55, 0.7]); light /= np.linalg.norm(light)
    shade = np.clip(normal @ light, 0, 1)
    near = 1 - 0.45 * (depth - depth[hit].min()) / max(1.0, depth[hit].max() - depth[hit].min())
    return (np.where(hit, (0.18 + 0.82 * shade) * near, 0))[::-1]


def bone_colored(img):
    """Gray picture (0 to 1) -> a warm bone color on black, as 8-bit BGR for saving."""
    tint = np.array([200, 228, 245], np.float32)  # blue, green, red
    return (np.clip(img, 0, 1)[..., None] * tint).astype(np.uint8)


def real_band(real):
    """Patient 1's real scan, cut down to the same band around the jaw curve and the same height range
    as the virtual scan, so both show the same part of the head."""
    info = json.loads(str(real["info"]))
    curve = {k: real[k] for k in ("px", "py", "nx", "ny")}
    c0 = info["col_shift"]; c1 = c0 + info["n_columns"]
    hu = np.where(real["labels"] == cv.OUTSIDE, -1000, real["hu"]).astype(np.float32)
    shape = (hu.shape[0], info["shape_zyx"][1], info["shape_zyx"][2])
    return cv.bend_back(hu[:, c0:c1], curve, shape, fill=-1000.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Patient 1: the real X-ray (pixels only; the identity fields of the file are not read)
    file = pydicom.dcmread(ROOT / "data/raw/Patient 1/PX_Panoramic_20260929_144042.dcm")
    x = file.pixel_array.astype(np.float32) / 4095
    h, w = x.shape
    save_gray("p1_xray.png", x[int(0.04 * h):int(0.93 * h), int(0.03 * w):int(0.965 * w)], 0, 1, scale=0.5)

    # ---- Patient 1: virtual scan and real scan, shown the same way
    virtual = np.load(P1 / "virtual_scan_p1_matched.npz")
    v_hu, v_lab = virtual["hu"].astype(np.float32), virtual["labels"]
    real = np.load(P1 / "real_block.npz")
    r_band = real_band(real)
    for name, vol in (("virtual", v_hu), ("real", r_band)):
        for turn in (0, 35):
            img = surface_view(vol, 650, turn)
            rows, cols = np.flatnonzero(img.any(axis=1)), np.flatnonzero(img.any(axis=0))
            img = img[max(rows[0] - 6, 0):rows[-1] + 7, max(cols[0] - 6, 0):cols[-1] + 7]
            big = cv2.resize(bone_colored(img), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(OUT / f"p1_3d_{name}_{turn}.png"), big)

    # seen from above, 12 mm below the bite, both cut to the same window around the front of the jaw
    stats = np.load(ROOT / "data" / "training_stats.npz")
    r_info = json.loads(str(real["info"]))
    dz = int(round(12 / cv.VOXEL_MM))
    v_z = v_hu.shape[0] - 1 - cv.FRAME_BITE_ROW - dz
    v_mid = (stats["px"][int(stats["middle"])], stats["py"][int(stats["middle"])])
    r_mid = (real["px"][r_info["middle"]], real["py"][r_info["middle"]])
    for name, vol, mid in (("virtual", v_hu, v_mid), ("real", r_band, r_mid)):
        pad = np.pad(vol[v_z], 200, constant_values=-1000)
        cx, cy = int(round(mid[0])) + 200, int(round(mid[1])) + 200
        save_gray(f"p1_above_{name}.png", pad[cy - 35:cy + 215, cx - 150:cx + 150], -1000, 2500, scale=4)

    # cuts across the jaw at the same tooth position, with the model's nerve canal in red
    with np.load(P1 / "kaggle_out/predictions/patient1/p1_matched.npz") as f:
        m_hu, m_lab = f["hu"].astype(np.float32), f["labels"]
    r_hu, r_inside = real["hu"].astype(np.float32), real["labels"] != cv.OUTSIDE
    lower = np.zeros(m_lab.shape, bool); lower[BELOW_BITE:] = True
    m_jaw = biggest_piece(np.isin(m_lab, (1, 3, 5)) & lower)
    r_jaw = biggest_piece((r_hu > REAL_BONE_HU) & r_inside & lower)
    for c in (115, 425):
        rc = int(round(to_real_column(c)))
        m, r = jaw_lines(m_jaw, m_hu, c), jaw_lines(r_jaw, r_hu, rc)
        d_row = r["edge"] - m["edge"]
        d_dep = int(round((r[8]["first"] + r[8]["last"]) / 2 - (m[8]["first"] + m[8]["last"]) / 2))
        canal = m_lab[:, c] == 3
        moved = np.roll(np.roll(canal, d_row, axis=0), d_dep, axis=1)
        for name, cut, mask, top in (("virtual", m_hu[:, c], canal, m["edge"] - 95), ("real", r_hu[:, rc], moved, r["edge"] - 95)):
            fig = plt.figure(figsize=(64 * 0.5 / 10, 110 * 0.4 / 10), dpi=200)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.imshow(cut[top:top + 110], cmap="gray", vmin=-1000, vmax=2500, aspect="auto", interpolation="bicubic")
            ax.contour(mask[top:top + 110], levels=[0.5], colors=RED, linewidths=2.2)
            fig.savefig(OUT / f"p1_across_{name}_{c}.png", facecolor="black"); plt.close(fig)

    # the sideways stretch of the X-ray: the same teeth in the X-ray (top) and in the real scan (bottom)
    xr = np.load(P1 / "xray/p1_boxed.npz")["xray"].astype(np.float32)
    mip = np.clip((r_hu.max(axis=2) + 300) / 3300, 0, 1) ** 1.7
    c0, c1, r0, r1 = 95, 450, 35, 140
    fig = plt.figure(figsize=(10, 7.4), dpi=160, facecolor="black")
    top = fig.add_axes([0, 0.56, 1, 0.44]); bottom = fig.add_axes([0, 0, 1, 0.44]); mid = fig.add_axes([0, 0.44, 1, 0.12])
    top.imshow(xr[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, aspect="auto", extent=(c0, c1, r1, r0))
    bottom.imshow(mip[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, aspect="auto", extent=(c0, c1, r1, r0))
    mid.set_xlim(c0, c1); mid.set_ylim(0, 1); mid.set_facecolor("black")
    for xc, rc in MATCH_COLUMNS:
        off = abs(rc - xc) > 12
        color = RED if off else "#FFFFFF"
        mid.plot([xc, rc], [1, 0], color=color, lw=3.5 if off else 2.2, solid_capstyle="round")
        top.plot([xc, xc], [r1 - 12, r1], color=color, lw=3.5 if off else 2.2)
        bottom.plot([rc, rc], [r0, r0 + 12], color=color, lw=3.5 if off else 2.2)
    for a in (top, bottom, mid):
        a.axis("off")
    fig.savefig(OUT / "p1_stretch.png", facecolor="black"); plt.close(fig)

    if len(sys.argv) > 2:
        prepared, predicted = np.load(sys.argv[1]), np.load(sys.argv[2])
        # ---- one public test scan: the three steps of the method, and the nerve canal check
        t_xray, t_hu, t_lab = prepared["xray"].astype(np.float32), prepared["hu"].astype(np.float32), prepared["labels"]
        p_hu, p_lab = predicted["hu"].astype(np.float32), predicted["labels"]
        inside = t_lab != cv.OUTSIDE
        rows, cols = np.flatnonzero(inside.any(axis=(1, 2))), np.flatnonzero(inside.any(axis=(0, 2)))
        box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        on_curve = int(round(cv.DEPTH_IN_MM / cv.DEPTH_STEP_MM))
        save_gray("test_xray.png", t_xray[box], 0, 1, scale=3)
        save_gray("test_block_true.png", t_hu[:, :, on_curve][box], -1000, 2500, scale=3)
        save_gray("test_block_model.png", np.where(inside, p_hu, -1000)[:, :, on_curve][box], -1000, 2500, scale=3)
        info = json.loads(str(prepared["info"]))
        curve = {k: prepared[k] for k in ("px", "py", "nx", "ny")}
        c0 = info["col_shift"]
        shape = (t_hu.shape[0], info["shape_zyx"][1], info["shape_zyx"][2])
        z = t_hu.shape[0] - 1 - cv.FRAME_BITE_ROW - int(round(12 / cv.VOXEL_MM))
        for name, block in (("true", t_hu), ("model", np.where(inside, p_hu, -1000))):
            bent = cv.bend_back(block[:, c0:c0 + info["n_columns"]], curve, shape, fill=-1000.0)
            rr, cc = np.flatnonzero((bent[z] > -999).any(axis=1)), np.flatnonzero((bent[z] > -999).any(axis=0))
            save_gray(f"test_above_{name}.png", bent[z][rr[0]:rr[-1] + 1, cc[0]:cc[-1] + 1], -1000, 2500, scale=4)
        # cut across the jaw with the model's canal (red) and the true canal (green)
        canal_cols = np.flatnonzero(((t_lab == 3).any(axis=(0, 2))) & ((p_lab == 3).any(axis=(0, 2))))
        left = canal_cols[canal_cols > cv.FRAME_MIDDLE_COL]
        c = int(left[len(left) // 3]) if len(left) else int(canal_cols[len(canal_cols) // 2])
        rows = np.flatnonzero(np.isin(t_lab[:, c], (1, 3, 5)).any(axis=1))
        top = max(rows.max() - 95, 0)
        for name, cut in (("model", np.where(inside, p_hu, -1000)[:, c]), ("true", t_hu[:, c])):
            fig = plt.figure(figsize=(64 * 0.5 / 10, 110 * 0.4 / 10), dpi=200)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.imshow(cut[top:top + 110], cmap="gray", vmin=-1000, vmax=2500, aspect="auto", interpolation="bicubic")
            ax.contour((t_lab[:, c] == 3)[top:top + 110], levels=[0.5], colors=GREEN, linewidths=2.2)
            if name == "model":
                ax.contour((p_lab[:, c] == 3)[top:top + 110], levels=[0.5], colors=RED, linewidths=2.2)
            fig.savefig(OUT / f"test_across_{name}.png", facecolor="black"); plt.close(fig)
    # ---- three small signs for the score list: good, partly, weak
    for name, color in (("good", GREEN), ("partly", AMBER), ("weak", RED)):
        fig = plt.figure(figsize=(1, 1), dpi=256)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.axis("off")
        ax.add_patch(plt.Circle((0, 0), 0.94, color=color))
        style = dict(color="#0B1220", lw=7, solid_capstyle="round", solid_joinstyle="round")
        if name == "good":
            ax.plot([-0.42, -0.1, 0.46], [0.0, -0.34, 0.38], **style)
        elif name == "partly":
            ax.plot([-0.42, 0.42], [0, 0], **style)
        else:
            ax.plot([-0.34, 0.34], [-0.34, 0.34], **style); ax.plot([-0.34, 0.34], [0.34, -0.34], **style)
        fig.savefig(OUT / f"sign_{name}.png", transparent=True); plt.close(fig)
    print("pictures are in", OUT)


if __name__ == "__main__":
    main()
