"""Turn a 3D scan into a "straightened" 3D block that lines up with the fake panoramic X-ray.

The fake X-ray has one column for each point along the U-shaped jaw curve and one row for each height.
The straightened block has the same rows and columns, plus a third direction: how far a point is from
the jaw curve, measured straight across the curve (toward the tongue or toward the cheek).

So pixel (row, column) of the X-ray and the line block[row, column, :] show the same place in the head.
The model gets the X-ray as input and must output the block.

Arrays are indexed [z, y, x] for scans and [row, column, depth] for blocks. Row 0 is the top of the head.
"""
import numpy as np
from scipy.ndimage import map_coordinates

from .fake_panoramic import arch_curve_from_shape, curve_points, find_arch_curve, find_bite_level

VOXEL_MM = 0.4      # size of one voxel in the scans, and of one row or column in the X-ray and block
DEPTH_IN_MM = 22.0  # how far the block reaches from the jaw curve toward the tongue
DEPTH_OUT_MM = 10.0 # how far the block reaches from the jaw curve toward the cheek
DEPTH_STEP_MM = 0.5 # distance between two neighbouring depth samples
OUTSIDE = 255       # label value for places that are outside the scan


def depth_positions_mm() -> np.ndarray:
    """Distance from the jaw curve of each depth sample. Negative = toward the tongue."""
    return np.arange(-DEPTH_IN_MM, DEPTH_OUT_MM, DEPTH_STEP_MM)


def jaw_curve(vol_hu: np.ndarray):
    """Find the jaw curve of a scan. Returns (bite slice, a dict with the curve)."""
    bite = find_bite_level(vol_hu)
    z_lo, z_hi = max(bite - int(20 / VOXEL_MM), 0), bite + int(5 / VOXEL_MM)
    cx, cy, center, _ = find_arch_curve(vol_hu, z_lo, z_hi)
    px, py, nx, ny = curve_points(cx, cy, center, step=1.0)
    middle = int(np.argmin(np.abs(px - center[0]) + 1e3 * (py > center[1])))  # the front-most point of the U
    return bite, dict(cx=cx, cy=cy, center=np.array(center), px=px, py=py, nx=nx, ny=ny, middle=middle)


def straighten(vol: np.ndarray, curve: dict, is_label: bool):
    """Sample the scan (or label file) along the jaw curve. Returns block[row, column, depth].

    For a scan, places outside the scan get -1000 (air). For labels they get OUTSIDE.
    """
    t = depth_positions_mm() / VOXEL_MM
    sx = curve["px"][:, None] + t[None, :] * curve["nx"][:, None]
    sy = curve["py"][:, None] + t[None, :] * curve["ny"][:, None]
    coords = np.stack([sy, sx])
    out = np.empty((vol.shape[0], sx.shape[0], sx.shape[1]), np.uint8 if is_label else np.float32)
    for z in range(vol.shape[0]):
        if is_label:
            out[z] = map_coordinates(vol[z], coords, order=0, cval=OUTSIDE)
        else:
            out[z] = map_coordinates(vol[z], coords, order=1, cval=-1000.0)
    return out[::-1]


def bend_back(block: np.ndarray, curve: dict, shape_zyx, fill: float, is_label: bool = False) -> np.ndarray:
    """The opposite of straighten: put a block back into the normal [z, y, x] shape of a scan.

    For each (y, x) place it finds the nearest point on the jaw curve and the distance across the curve,
    then reads the block there. Places further from the curve than the block reaches get `fill`.
    """
    from scipy.spatial import cKDTree
    nz, ny_, nx_ = shape_zyx
    yy, xx = np.mgrid[0:ny_, 0:nx_].astype(np.float32)
    pts = np.stack([curve["px"], curve["py"]], axis=1)
    _, idx = cKDTree(pts).query(np.stack([xx.ravel(), yy.ravel()], axis=1))
    dx, dy = xx.ravel() - curve["px"][idx], yy.ravel() - curve["py"][idx]
    # distance across the curve (signed), and distance along the curve from the nearest curve point
    across = dx * curve["nx"][idx] + dy * curve["ny"][idx]
    along = dx * np.gradient(curve["px"])[idx] + dy * np.gradient(curve["py"])[idx]
    col = idx + along  # curve points are 1 voxel apart, so 1 voxel along the curve = 1 column
    dep = (across * VOXEL_MM + DEPTH_IN_MM) / DEPTH_STEP_MM
    n_col, n_dep = block.shape[1], block.shape[2]
    ok = (col >= 0) & (col <= n_col - 1) & (dep >= 0) & (dep <= n_dep - 1) & (np.abs(along) < 3)
    out = np.full((nz, ny_ * nx_), fill, block.dtype)
    coords = np.stack([col[ok], dep[ok]])
    for z in range(nz):
        out[z, ok] = map_coordinates(block[nz - 1 - z], coords, order=0 if is_label else 1, mode="nearest")
    return out.reshape(nz, ny_, nx_)


# Every X-ray and block is put on a frame of one fixed size, so the model always gets the same size.
# The row where the teeth meet and the column of the front of the jaw are always at the same place.
FRAME_ROWS = 240
FRAME_COLS = 544
FRAME_BITE_ROW = 80
FRAME_MIDDLE_COL = FRAME_COLS // 2


def to_frame(arr: np.ndarray, bite_row: int, middle_col: int, fill):
    """Copy arr[row, column, ...] onto the fixed frame. Parts that do not fit are dropped.
    Returns (frame, row shift, column shift), where frame[r + row shift, c + column shift] = arr[r, c]."""
    dr, dc = FRAME_BITE_ROW - bite_row, FRAME_MIDDLE_COL - middle_col
    out = np.full((FRAME_ROWS, FRAME_COLS) + arr.shape[2:], fill, arr.dtype)
    r0, r1 = max(0, -dr), min(arr.shape[0], FRAME_ROWS - dr)
    c0, c1 = max(0, -dc), min(arr.shape[1], FRAME_COLS - dc)
    out[r0 + dr:r1 + dr, c0 + dc:c1 + dc] = arr[r0:r1, c0:c1]
    return out, dr, dc


def jaw_curve_from_labels(lab: np.ndarray):
    """Find the bite slice and the jaw curve of a training scan from its label file
    (the original ToothFairy3 numbers, not the merged ones).

    This is more reliable than jaw_curve, which has to guess from brightness alone.
    Bite slice = halfway between the tips of the lower teeth and the tips of the upper teeth.
    Jaw curve = the U shape of the lower jaw bone and the teeth, seen from above, from 30 mm below
    the bite slice to 5 mm above it.
    """
    lower = ((lab >= 31) & (lab <= 48)) | ((lab >= 131) & (lab <= 148))
    upper = ((lab >= 11) & (lab <= 28)) | ((lab >= 111) & (lab <= 128))
    tips = []
    for teeth, q in ((lower, 97), (upper, 3)):
        per_slice = teeth.sum(axis=(1, 2))
        if per_slice.sum() > 0:
            tips.append(np.searchsorted(np.cumsum(per_slice) / per_slice.sum(), q / 100))
    if not tips:
        raise ValueError("no teeth in the label file")
    bite = int(round(np.mean(tips)))
    z_lo, z_hi = max(bite - int(30 / VOXEL_MM), 0), min(bite + int(5 / VOXEL_MM), lab.shape[0])
    slab = lab[z_lo:z_hi]
    shape = ((slab == 1) | ((slab >= 8) & (slab <= 48)) | ((slab >= 111) & (slab <= 148))).any(axis=0)
    return bite, curve_from_shape(shape)


def curve_from_shape(shape: np.ndarray) -> dict:
    """The jaw curve from a yes/no picture of the lower jaw bone and the teeth, seen from above."""
    cx, cy, center, _ = arch_curve_from_shape(shape)
    # keep only the part of the curve that goes outward from the front; drop ends that turn back
    angle = np.arctan2(cx - center[0], center[1] - cy)
    radius = np.hypot(cx - center[0], cy - center[1])
    front = int(np.argmin(np.abs(angle) + 10 * (radius < 0.5 * np.median(radius))))
    good = radius > 0.35 * radius[front]
    # also drop ends that turn back toward the front, and ends that are past the last bit of jaw
    for side in (slice(front, None), slice(front, None, -1)):
        y = cy[side]
        good[side] &= np.cumsum(y < np.maximum.accumulate(y) - 1.0) == 0
    from scipy.ndimage import binary_dilation
    near_jaw = binary_dilation(shape, iterations=3)
    on_jaw = np.flatnonzero(near_jaw[np.clip(np.round(cy).astype(int), 0, shape.shape[0] - 1),
                                     np.clip(np.round(cx).astype(int), 0, shape.shape[1] - 1)])
    good[:on_jaw[0]] = False
    good[on_jaw[-1] + 1:] = False
    first = front - np.argmax(~good[front::-1]) + 1 if (~good[:front]).any() else 0
    last = front + np.argmax(~good[front:]) if (~good[front:]).any() else len(cx)
    cx, cy = cx[first:last], cy[first:last]
    px, py, nx, ny = curve_points(cx, cy, center, step=1.0)
    middle = int(np.argmin(np.hypot(px - cx[front - first], py - cy[front - first])))
    return dict(cx=cx, cy=cy, center=np.array(center), px=px, py=py, nx=nx, ny=ny, middle=middle)


def jaw_curve_from_brightness(vol_hu: np.ndarray, bite: int, hard_hu: float = 900):
    """The jaw curve of a scan that has no label file, found the same way as for the training scans:
    everything hard (bone, teeth, fillings) from 30 mm below the bite slice to 5 mm above it, seen
    from above, is used in place of the labels. The biggest connected piece is the jaw."""
    from scipy.ndimage import binary_fill_holes, binary_opening, label
    z_lo, z_hi = max(bite - int(30 / VOXEL_MM), 0), min(bite + int(5 / VOXEL_MM), vol_hu.shape[0])
    shape = binary_opening((vol_hu[z_lo:z_hi] > hard_hu).any(axis=0), iterations=2)
    pieces, _ = label(shape)
    sizes = np.bincount(pieces.ravel()); sizes[0] = 0
    return curve_from_shape(binary_fill_holes(pieces == sizes.argmax()))
