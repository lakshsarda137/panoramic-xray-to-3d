"""Make a fake panoramic X-ray from a 3D CBCT.

How it works, in plain steps:
1. Find the curve of the jaw, seen from above (the U shape), at the height of the tooth roots.
2. Walk along that curve. At each point, look straight across the curve, through the jaw.
3. Add up how much X-ray each bit of tissue along that short line would block. Bone blocks a lot, air almost none.
   Tissue near the curve counts fully; tissue further away counts less (real machines blur it out).
4. Each point on the curve becomes one column of the picture; each height in the scan becomes one row.

Volumes are indexed [z, y, x]. z grows toward the top of the head, y grows toward the back of the head.
"""
import numpy as np
from scipy.ndimage import binary_opening, gaussian_filter1d, label, map_coordinates, zoom

MU_WATER = 0.02  # how much 1 mm of water blocks X-rays (per mm), a typical value near 70 keV


def hu_to_mu(vol_hu: np.ndarray, metal_cap: float = 3000.0) -> np.ndarray:
    """Turn CT numbers (Hounsfield units) into how much each mm of tissue blocks X-rays."""
    hu = np.clip(vol_hu, -1000, metal_cap)
    return MU_WATER * (1 + hu / 1000.0)


def resample(vol: np.ndarray, spacing, new_spacing: float) -> np.ndarray:
    return zoom(vol, [s / new_spacing for s in spacing], order=1)


def find_bite_level(vol_hu: np.ndarray, teeth_hu: float = 1800) -> int:
    """Slice index where the most very bright voxels (tooth enamel, fillings) are: roughly where the teeth meet."""
    counts = (vol_hu > teeth_hu).sum(axis=(1, 2))
    return int(np.argmax(gaussian_filter1d(counts.astype(float), 3)))


def find_arch_curve(vol_hu: np.ndarray, z_lo: int, z_hi: int, bone_hu: float = 500, metal_hu: float = 2800,
                    angle_range_deg: float = 100, n_angles: int = 201):
    """Find the jaw's U shape, seen from above, between slices z_lo and z_hi.

    Keeps bone but not metal, keeps the biggest connected piece (the jaw), then stands at a point
    inside the U and looks outward in many directions. Along each direction, the middle of the
    last stretch of bone before the outside of the face is one point of the curve.
    The distances are smoothed so the curve has no jumps.
    Returns (points_x, points_y, center).
    """
    slab = vol_hu[z_lo:z_hi]
    bone = ((slab > bone_hu) & (slab < metal_hu)).any(axis=0)
    bone = binary_opening(bone, iterations=2)
    lab, _ = label(bone)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    bone = lab == sizes.argmax()
    return arch_curve_from_shape(bone, angle_range_deg, n_angles)


def arch_curve_from_shape(bone: np.ndarray, angle_range_deg: float = 100, n_angles: int = 201):
    """The second half of find_arch_curve. `bone` is a yes/no picture of the jaw seen from above."""
    ys, xs = np.nonzero(bone)
    cx = (xs.min() + xs.max()) / 2
    cy = ys.max() - 0.15 * (ys.max() - ys.min())  # inside the U, near the back ends of the jaw
    angles = np.deg2rad(np.linspace(-angle_range_deg, angle_range_deg, n_angles))  # 0 = straight to the front
    r_max = np.hypot(*bone.shape)
    radii = np.arange(0, r_max, 0.5)
    r_mid = np.full(len(angles), np.nan)
    for i, a in enumerate(angles):
        dx, dy = np.sin(a), -np.cos(a)
        px, py = cx + radii * dx, cy + radii * dy
        inside = (px >= 0) & (px < bone.shape[1] - 1) & (py >= 0) & (py < bone.shape[0] - 1)
        hit = np.zeros(len(radii), bool)
        hit[inside] = bone[np.round(py[inside]).astype(int), np.round(px[inside]).astype(int)]
        idx = np.flatnonzero(hit)
        if idx.size == 0:
            continue
        end = idx[-1]
        start = end
        while start - 1 >= 0 and hit[start - 1]:
            start -= 1
        r_mid[i] = radii[(start + end) // 2]
    ok = ~np.isnan(r_mid)
    # smooth the distance-vs-direction curve with a low-order polynomial, dropping outliers
    keep = ok.copy()
    for _ in range(3):
        coef = np.polyfit(angles[keep], r_mid[keep], 6)
        res = np.abs(r_mid - np.polyval(coef, angles))
        keep = ok & (res < max(3 * np.nanmedian(res[keep]), 2.0))
    a_used = angles[ok]
    a_fine = np.linspace(a_used.min(), a_used.max(), 2000)
    r_fine = np.polyval(coef, a_fine)
    return cx + r_fine * np.sin(a_fine), cy - r_fine * np.cos(a_fine), (cx, cy), (angles[keep], r_mid[keep])


def curve_points(cx_pts, cy_pts, center, step: float):
    """Points spaced `step` voxels apart along the curve, plus the direction across the curve (pointing outward)."""
    seg = np.hypot(np.diff(cx_pts), np.diff(cy_pts))
    arc = np.concatenate([[0], np.cumsum(seg)])
    s = np.arange(0, arc[-1], step)
    px, py = np.interp(s, arc, cx_pts), np.interp(s, arc, cy_pts)
    tx, ty = np.gradient(px), np.gradient(py)
    nx, ny = ty, -tx
    norm = np.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    flip = (px - center[0]) * nx + (py - center[1]) * ny < 0
    nx[flip] *= -1; ny[flip] *= -1
    return px, py, nx, ny


def make_panoramic(mu: np.ndarray, curve_x, curve_y, center, voxel_mm: float,
                   depth_in_mm: float = 150, depth_out_mm: float = 30, sharp_mm: float = 4,
                   tilt_deg: float = 7, near_mm: float = 25, near_blur_px: float = 6,
                   far_blur_px: float = 25) -> np.ndarray:
    """Return the fake panoramic picture: rows = height (top of head first), columns = along the jaw
    (patient's right side on the left of the picture, like a real panoramic X-ray).

    Each column's light line starts outside the face and goes depth_in_mm into the head, so it
    passes through the whole head (tongue, throat, spine), as in a real machine.
    The line tilts upward by tilt_deg toward the front, like a real machine's beam.
    Everything along the line blocks light, but only tissue within about sharp_mm of the jaw curve is
    sharp. Tissue up to near_mm away is smeared sideways by near_blur_px columns, and tissue further
    away (for example the spine) by far_blur_px, because the machine moves while taking the picture.
    """
    px, py, nx, ny = curve_points(curve_x, curve_y, center, step=1.0)
    t = np.arange(-depth_in_mm, depth_out_mm, voxel_mm) / voxel_mm  # negative = into the head
    d = np.abs(t * voxel_mm)
    w_sharp = np.exp(-0.5 * (d / sharp_mm) ** 2)
    w_near = (1 - w_sharp) * (d <= near_mm)
    w_far = (1 - w_sharp) * (d > near_mm)
    dz = t * np.tan(np.deg2rad(tilt_deg))  # points deeper in the head sit lower on the line
    nz = mu.shape[0]
    sharp, near, far = (np.zeros((nz, len(px)), np.float32) for _ in range(3))
    zz = np.arange(nz)[:, None] + dz[None, :]
    for i in range(len(px)):
        sx = np.broadcast_to(px[i] + t * nx[i], zz.shape)
        sy = np.broadcast_to(py[i] + t * ny[i], zz.shape)
        vals = map_coordinates(mu, np.stack([zz.ravel(), sy.ravel(), sx.ravel()]), order=1, cval=0.0)
        vals = vals.reshape(zz.shape) * voxel_mm
        sharp[:, i] = vals @ w_sharp
        near[:, i] = vals @ w_near
        far[:, i] = vals @ w_far
    img = sharp + gaussian_filter1d(near, near_blur_px, axis=1) + gaussian_filter1d(far, far_blur_px, axis=1)
    return img[::-1]


def look_like_real(img: np.ndarray, clip_limit: float = 1.2, sharpen: float = 0.6) -> np.ndarray:
    """Imitate the picture clean-up that panoramic machines do: gentle local brightness balancing
    and edge sharpening. Returns values between 0 and 1."""
    import cv2
    lo, hi = np.percentile(img, [0.5, 99.8])
    x = np.clip((img - lo) / (hi - lo), 0, 1)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 8))
    x = clahe.apply((x * 65535).astype(np.uint16)).astype(np.float32) / 65535
    blurred = cv2.GaussianBlur(x, (0, 0), 3)
    return np.clip(x + sharpen * (x - blurred), 0, 1)
