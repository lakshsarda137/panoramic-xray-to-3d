"""Save pictures of Patient 1's panoramic X-ray and CBCT so we can see what the data looks like."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.patient_io import load_cbct, load_panoramic

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/raw/Patient 1"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

pano = load_panoramic(P / "PX_Panoramic_20260929_144042.dcm")
plt.imsave(OUT / "pano_real.png", pano, cmap="gray")

vol, sp = load_cbct(next(P.glob("CT_*")))
np.save(ROOT / "data/patient1_cbct.npy", vol)
print("cbct", vol.shape, sp, "HU min/max", vol.min(), vol.max(), "percentiles", np.percentile(vol, [1, 50, 99, 99.9]))

z, y, x = (s // 2 for s in vol.shape)
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
kw = dict(cmap="gray", vmin=-500, vmax=2500)
ax[0, 0].imshow(vol[z], **kw); ax[0, 0].set_title(f"top-down slice z={z}")
ax[0, 1].imshow(vol[:, y], origin="lower", aspect=sp[0] / sp[2], **kw); ax[0, 1].set_title("front view slice")
ax[0, 2].imshow(vol[:, :, x], origin="lower", aspect=sp[0] / sp[1], **kw); ax[0, 2].set_title("side view slice")
ax[1, 0].imshow(vol.max(0), **kw); ax[1, 0].set_title("brightest point, looking down")
ax[1, 1].imshow(vol.max(1), origin="lower", aspect=sp[0] / sp[2], **kw); ax[1, 1].set_title("brightest point, from front")
ax[1, 2].imshow(vol.max(2), origin="lower", aspect=sp[0] / sp[1], **kw); ax[1, 2].set_title("brightest point, from side")
plt.tight_layout(); plt.savefig(OUT / "cbct_views.png", dpi=80)
