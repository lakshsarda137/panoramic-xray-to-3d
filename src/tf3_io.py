"""Load ToothFairy3 scans and label files.

Every array returned here is indexed [z, y, x], the same as the rest of the code:
z grows toward the top of the head, y grows toward the back of the head, x grows toward the patient's left.
"""
from pathlib import Path

import nibabel as nib
import numpy as np

# The numbers the model predicts. Left and right parts are joined into one number.
CLASS_NAMES = ["background", "lower jaw", "upper jaw", "nerve canal", "sinus", "teeth"]
N_CLASSES = len(CLASS_NAMES)


def _to_zyx(img) -> np.ndarray:
    # as_closest_canonical gives [x toward right, y toward front, z toward top]; turn that into our order
    arr = np.asarray(nib.as_closest_canonical(img).dataobj)
    return np.ascontiguousarray(arr.transpose(2, 1, 0)[:, ::-1, ::-1])


def load_scan(path) -> np.ndarray:
    """The 3D scan in Hounsfield units, float32."""
    return _to_zyx(nib.load(str(path))).astype(np.float32)


def load_labels(path) -> np.ndarray:
    """The label file with the original ToothFairy3 numbers."""
    return _to_zyx(nib.load(str(path))).astype(np.int16)


def merge_labels(lab: np.ndarray) -> np.ndarray:
    """Turn the ToothFairy3 numbers into the 6 numbers in CLASS_NAMES.

    ToothFairy3 numbers: 1 lower jaw, 2 upper jaw, 3 and 4 nerve canal, 5 and 6 sinus, 7 throat,
    8 bridge, 9 crown, 10 implant, 11 to 48 single teeth, 103 to 105 small canals at the front of the
    lower jaw, 111 to 148 the soft inside of each tooth.
    """
    out = np.zeros(lab.shape, np.uint8)
    out[lab == 1] = 1
    out[(lab >= 103) & (lab <= 105)] = 1  # small front canals are inside the lower jaw bone
    out[lab == 2] = 2
    out[(lab == 3) | (lab == 4)] = 3
    out[(lab == 5) | (lab == 6)] = 4
    out[((lab >= 8) & (lab <= 48)) | ((lab >= 111) & (lab <= 148))] = 5
    return out


def list_scans(data_dir):
    """Pairs of (scan number as text, scan file, label file) for every scan in the folder."""
    data_dir = Path(data_dir)
    pairs = []
    for scan in sorted((data_dir / "imagesTr").glob("ToothFairy3F_*_0000.nii.gz")):
        num = scan.name.split("_")[1]
        pairs.append((num, scan, data_dir / "labelsTr" / f"ToothFairy3F_{num}.nii.gz"))
    return pairs
