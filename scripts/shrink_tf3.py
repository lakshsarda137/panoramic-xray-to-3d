"""Watch ~/Downloads for ToothFairy3 files saved by the browser, shrink each one, then delete the original.

Scans are resampled to SPACING mm per voxel (straight-line blending for scans, nearest value for label masks).
Stops when the output folder holds TOTAL files. Safe to stop and start again: finished files are kept.
Usage: python scripts/shrink_tf3.py <output folder>
"""
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

SPACING = 0.4
TOTAL = 126  # 63 scans + 63 label masks
PREFIX = "tf3dl__"
downloads = Path.home() / "Downloads"
out_root = Path(sys.argv[1])


def finished() -> int:
    return len([p for p in out_root.glob("*/*.nii.gz") if not p.name.startswith("tmp_")])


def shrink(path: Path) -> Path:
    # file name looks like tf3dl__imagesTr__ToothFairy3F_001_0000.nii.gz
    _, folder, name = path.name.split("__")
    is_label = folder == "labelsTr"
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    factors = np.array(img.header.get_zooms()[:3], dtype=float) / SPACING
    small = zoom(data, factors, order=0 if is_label else 1).astype(np.uint8 if is_label else np.int16)
    affine = img.affine.copy()
    affine[:3, :3] = affine[:3, :3] @ np.diag(1 / factors)
    out = nib.Nifti1Image(small, affine)
    out.header.set_zooms((SPACING,) * 3)
    dest = out_root / folder / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name("tmp_" + dest.name)
    nib.save(out, str(tmp))
    tmp.rename(dest)  # only a complete file ever gets the real name
    return dest


print(f"{finished()}/{TOTAL} already done", flush=True)
while finished() < TOTAL:
    files = sorted(downloads.glob(PREFIX + "*.nii.gz"))
    if not files:
        time.sleep(2)
        continue
    for f in files:
        size = f.stat().st_size
        time.sleep(1)
        if f.stat().st_size != size:  # still being written
            continue
        try:
            dest = shrink(f)
        except Exception as e:  # half-written or broken file: leave it and retry later
            print(f"retry later {f.name}: {e}", flush=True)
            continue
        f.unlink()
        print(f"{finished()}/{TOTAL} {dest.name} {size/1e6:.0f}MB -> {dest.stat().st_size/1e6:.1f}MB", flush=True)
print("all done", flush=True)
