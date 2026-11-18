"""Prepare Patient 1's real 3D scan for comparing with the model's output.

Run:  .venv/bin/python scripts/prepare_patient1.py

This script uses Patient 1's real 3D scan. Nothing it makes is given to the model when the
"virtual" 3D scan is built. It only makes the correct answer that the model's output is compared with.

It makes, in data/patient1/ (ignored by git):
  cbct_0.4mm.npy   the real 3D scan shrunk to 0.4 mm per voxel, [z, y, x], Hounsfield units, capped at 3000
  real_block.npz   the same layout as a prepared training scan:
                   xray   = a fake X-ray made from the real 3D scan (only for development checks)
                   hu     = the real 3D scan as a straightened block, on the fixed frame
                   labels = 0 inside the scan, 255 outside (Patient 1 has no label file)
                   px, py, nx, ny, info = Patient 1's own jaw curve, found from brightness
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import curved_volume as cv
from src.fake_panoramic import find_bite_level, hu_to_mu, look_like_real, make_panoramic, resample
from src.patient_io import load_cbct

OUT = ROOT / "data" / "patient1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    small_path = OUT / "cbct_0.4mm.npy"
    if small_path.exists():
        vol = np.load(small_path).astype(np.float32)
    else:
        big, spacing = load_cbct(next((ROOT / "data/raw/Patient 1").glob("CT_*")))
        print("real 3D scan:", big.shape, "voxel size in mm:", spacing, flush=True)
        vol = np.clip(resample(big, spacing, cv.VOXEL_MM), -1000, 3000)  # metal gives extreme values, so cap them
        del big
        np.save(small_path, vol.round().astype(np.int16))
    print("shrunk scan:", vol.shape, flush=True)

    bite = find_bite_level(vol)
    curve = cv.jaw_curve_from_brightness(vol, bite)
    xray = look_like_real(make_panoramic(hu_to_mu(vol), curve["cx"], curve["cy"], curve["center"], cv.VOXEL_MM))
    hu = cv.straighten(vol, curve, is_label=False)
    inside = cv.straighten(np.ones(vol.shape, np.uint8), curve, is_label=True)  # 1 inside the scan, 255 outside
    labels = np.where(inside == cv.OUTSIDE, cv.OUTSIDE, 0).astype(np.uint8)
    bite_row = vol.shape[0] - 1 - bite  # row 0 is the top of the head
    xray_f, dr, dc = cv.to_frame(xray.astype(np.float32), bite_row, curve["middle"], 0.0)
    hu_f, _, _ = cv.to_frame(hu.round().astype(np.int16), bite_row, curve["middle"], -1000)
    lab_f, _, _ = cv.to_frame(labels, bite_row, curve["middle"], cv.OUTSIDE)
    info = dict(shape_zyx=list(vol.shape), bite=int(bite), row_shift=int(dr), col_shift=int(dc),
                middle=int(curve["middle"]), n_columns=int(len(curve["px"])))
    np.savez_compressed(OUT / "real_block.npz", xray=xray_f.astype(np.float16), hu=hu_f, labels=lab_f,
                        px=curve["px"], py=curve["py"], nx=curve["nx"], ny=curve["ny"], info=json.dumps(info))
    print("saved", OUT / "real_block.npz", info)


if __name__ == "__main__":
    main()
