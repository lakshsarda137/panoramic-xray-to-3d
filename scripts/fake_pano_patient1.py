"""Make a fake panoramic X-ray from Patient 1's CBCT and put it next to the real one.

This only checks that the fake-X-ray code looks realistic. Patient 1's CBCT is not used to train anything.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fake_panoramic import find_arch_curve, find_bite_level, hu_to_mu, look_like_real, make_panoramic, resample
from src.patient_io import load_panoramic

VOX = 0.4
TAG = sys.argv[1] if len(sys.argv) > 1 else "v4"
vol = resample(np.load(ROOT / "data/patient1_cbct.npy"), (0.2, 0.2, 0.2), VOX)
bite = find_bite_level(vol)
z_lo, z_hi = max(bite - int(20 / VOX), 0), bite + int(5 / VOX)
cx, cy, center, _ = find_arch_curve(vol, z_lo, z_hi)
raw = make_panoramic(hu_to_mu(vol), cx, cy, center, VOX)
fake = look_like_real(raw)
real = load_panoramic(ROOT / "data/raw/Patient 1/PX_Panoramic_20260929_144042.dcm")
real = (real - real.min()) / (real.max() - real.min())
real_small = cv2.resize(real, (fake.shape[1], fake.shape[0]), interpolation=cv2.INTER_AREA)

fig, ax = plt.subplots(2, 1, figsize=(14, 11))
ax[0].imshow(fake, cmap="gray"); ax[0].set_title("Fake panoramic (made from Patient 1's CBCT)")
ax[1].imshow(real_small, cmap="gray"); ax[1].set_title("Real panoramic (resized to the same size)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(ROOT / f"outputs/fake_vs_real_pano_{TAG}.png", dpi=70)
np.save(ROOT / f"outputs/fake_pano_{TAG}.npy", fake)
