"""Load Patient 1's panoramic X-ray and CBCT from DICOM files."""
from pathlib import Path

import numpy as np
import pydicom


def load_panoramic(path: Path) -> np.ndarray:
    """Return the panoramic X-ray as a 2D float array (bigger number = brighter)."""
    return pydicom.dcmread(path).pixel_array.astype(np.float32)


def load_cbct(folder: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return the CBCT as a 3D array indexed [z, y, x] and the spacing in mm (z, y, x).

    z goes along the slice order (top to bottom of the head or the reverse, see ImagePositionPatient).
    Values are in Hounsfield units: air is about -1000, water 0, bone several hundred and up.
    """
    slices = [pydicom.dcmread(p) for p in sorted(folder.glob("*.dcm"))]
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    vol = np.stack([d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept)
                    for d in slices])
    dz = float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2])
    dy, dx = (float(v) for v in slices[0].PixelSpacing)
    return vol, (dz, dy, dx)
