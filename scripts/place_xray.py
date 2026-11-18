"""Put Patient 1's real panoramic X-ray on the model's fixed frame, so the model can use it.

Run:  .venv/bin/python scripts/place_xray.py

This script uses ONLY the X-ray file and data/training_stats.npz (facts about the training scans).
It does not use Patient 1's real 3D scan.

The model was trained on pictures of 240 rows by 544 columns, 0.4 mm per pixel, with the line where
the teeth meet on row 80 and the front of the jaw on column 272. The steps here do the same for
the real X-ray:
  1. Read the pixels. The identity fields of the file (name, ID, birth date) are not read.
  2. Shrink the picture to 0.4 mm per pixel, using the pixel size written in the file (0.0693 mm).
  3. Move it so that the two marks below are on row 80 and column 272.
     The marks were placed by eye on the X-ray:
       BITE_Y   = the dark line between the upper and the lower teeth
       MIDDLE_X = the gap between the two front teeth
  4. Make black: the white bar and the maker's logo at the bottom, and everything outside the part
     of the frame that held picture in the training X-rays (the training X-rays stop behind the last
     teeth; the real one goes on to the ears).
  5. Change the brightness so that the picture has the same spread of dark and bright pixels as
     the training X-rays.

It saves three versions in data/patient1/xray/ (ignored by git), each as NAME.npz with one array "xray":
  p1_plain    steps 1 to 3, and only the white bar and logo made black
  p1_boxed    steps 1 to 4
  p1_matched  steps 1 to 5. This is the one the results are reported for.
and a picture of the three, outputs/patient1_xray_on_frame.png.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.curved_volume import FRAME_BITE_ROW, FRAME_COLS, FRAME_MIDDLE_COL, FRAME_ROWS, VOXEL_MM, to_frame

XRAY_FILE = ROOT / "data/raw/Patient 1/PX_Panoramic_20260929_144042.dcm"
OUT = ROOT / "data" / "patient1" / "xray"
# Marks placed by eye, counted in pixels of the shrunk picture (0.4 mm per pixel).
BITE_Y = 148     # the dark line between the upper and the lower teeth
MIDDLE_X = 275   # the gap between the two front teeth
BAR_Y = 253      # the white bar of the chin rest and the logo start below this line


def match_brightness(values, target_quantiles):
    """Give `values` the same spread of dark and bright as the training X-rays: the darkest 1% of
    the pixels get the brightness of the darkest 1% of training pixels, and so on."""
    order = np.argsort(values)
    rank = np.empty(len(values)); rank[order] = np.linspace(0, 1, len(values))
    return np.interp(rank, np.linspace(0, 1, len(target_quantiles)), target_quantiles)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = np.load(ROOT / "data" / "training_stats.npz")
    file = pydicom.dcmread(XRAY_FILE, specific_tags=["PixelSpacing", "Rows", "Columns", "BitsStored", "PixelData",
                                                      "BitsAllocated", "PixelRepresentation", "SamplesPerPixel",
                                                      "PhotometricInterpretation", "HighBit"])
    pixel_mm = float(file.PixelSpacing[0])
    big = file.pixel_array.astype(np.float32) / (2 ** int(file.BitsStored) - 1)  # 0 = black, 1 = white; bone is bright
    k = pixel_mm / VOXEL_MM
    small = cv2.resize(big, (round(big.shape[1] * k), round(big.shape[0] * k)), interpolation=cv2.INTER_AREA)
    print(f"X-ray: {big.shape[0]} x {big.shape[1]} pixels of {pixel_mm} mm -> {small.shape[0]} x {small.shape[1]} pixels of {VOXEL_MM} mm")

    small[BAR_Y:] = 0.0
    plain, row_shift, col_shift = to_frame(small, BITE_Y, MIDDLE_X, 0.0)

    r0, r1, c0, c1 = (int(v) for v in stats["picture_box"])
    r1 = min(r1, BAR_Y + row_shift - 1)
    box = np.zeros((FRAME_ROWS, FRAME_COLS), bool)
    box[r0:r1 + 1, c0:c1 + 1] = True
    boxed = np.where(box, plain, 0.0).astype(np.float32)
    matched = boxed.copy()
    matched[box] = match_brightness(boxed[box], stats["xray_quantiles"])

    versions = dict(p1_plain=plain, p1_boxed=boxed, p1_matched=matched)
    for name, x in versions.items():
        np.savez_compressed(OUT / f"{name}.npz", xray=x.astype(np.float16))
    fig, ax = plt.subplots(3, 1, figsize=(12, 16))
    for a, (name, x) in zip(ax, versions.items()):
        a.imshow(x, cmap="gray", vmin=0, vmax=1); a.set_title(name)
        a.axhline(FRAME_BITE_ROW, color="lime", lw=0.5); a.axvline(FRAME_MIDDLE_COL, color="lime", lw=0.5); a.axis("off")
    plt.tight_layout(); plt.savefig(ROOT / "outputs" / "patient1_xray_on_frame.png", dpi=80); plt.close(fig)
    print(f"picture box: rows {r0} to {r1}, columns {c0} to {c1}")
    print("saved 3 versions in", OUT)


if __name__ == "__main__":
    main()
