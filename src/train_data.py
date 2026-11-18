"""Load the prepared examples and cut random training pieces from them."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUTSIDE = 255
HU_LOW, HU_HIGH = -1000.0, 3000.0


def hu_to_unit(hu):
    """Hounsfield units -> numbers between 0 and 1 (what the model outputs)."""
    return (hu - HU_LOW) / (HU_HIGH - HU_LOW)


def unit_to_hu(x):
    return x * (HU_HIGH - HU_LOW) + HU_LOW


def make_split(folder, n_check=5, n_test=10, seed=0):
    """Split the scans into train / check / test. Saved in split.json so it never changes.
    "check" scans are used during training to see if the model is getting better.
    "test" scans are never used during training."""
    folder = Path(folder)
    path = folder / "split.json"
    if path.exists():
        return json.loads(path.read_text())
    nums = sorted(p.stem for p in folder.glob("*.npz"))
    random.Random(seed).shuffle(nums)
    split = dict(test=sorted(nums[:n_test]), check=sorted(nums[n_test:n_test + n_check]),
                 train=sorted(nums[n_test + n_check:]))
    path.write_text(json.dumps(split, indent=1))
    return split


class Examples:
    """Holds a list of prepared scans in memory.
    xray: [n, rows, columns] 0..1,  hu: [n, depth, rows, columns] Hounsfield units,
    labels: [n, depth, rows, columns] part numbers, 255 = outside the scan."""

    def __init__(self, folder, numbers):
        xs, hs, ls = [], [], []
        for num in numbers:
            with np.load(Path(folder) / f"{num}.npz") as f:
                xs.append(torch.from_numpy(f["xray"].astype(np.float32)))
                hs.append(torch.from_numpy(f["hu"]).permute(2, 0, 1).contiguous())
                ls.append(torch.from_numpy(f["labels"]).permute(2, 0, 1).contiguous())
        self.numbers = list(numbers)
        self.xray, self.hu, self.labels = torch.stack(xs), torch.stack(hs), torch.stack(ls)

    def __len__(self):
        return len(self.numbers)

    def whole(self, i):
        """One whole example, as a batch of 1."""
        return self.xray[i:i + 1, None], self.hu[i:i + 1], self.labels[i:i + 1]

    def random_batch(self, batch_size, piece_cols, max_shift_rows=8):
        """Random pieces for training. Each piece is piece_cols columns wide and is randomly
        mirrored left-right and moved up or down a little, so the model sees more variety."""
        rows, cols = self.xray.shape[1:]
        xs, hs, ls = [], [], []
        for _ in range(batch_size):
            i = random.randrange(len(self))
            c = random.randrange(0, cols - piece_cols + 1)
            x, h, l = self.xray[i, :, c:c + piece_cols], self.hu[i, :, :, c:c + piece_cols], self.labels[i, :, :, c:c + piece_cols]
            if random.random() < 0.5:
                x, h, l = x.flip(-1), h.flip(-1), l.flip(-1)
            s = random.randint(-max_shift_rows, max_shift_rows)
            if s != 0:
                x, h, l = _shift_rows(x, s, 0.0), _shift_rows(h, s, int(HU_LOW)), _shift_rows(l, s, OUTSIDE)
            xs.append(x); hs.append(h); ls.append(l)
        return torch.stack(xs)[:, None], torch.stack(hs), torch.stack(ls)


def _shift_rows(t, s, fill):
    """Move the picture s rows down (or up if s is negative). Rows that become empty get `fill`."""
    out = torch.full_like(t, fill)
    if s > 0:
        out[..., s:, :] = t[..., :-s, :]
    else:
        out[..., :s, :] = t[..., -s:, :]
    return out


def change_xray_look(xray):
    """Randomly change brightness curve, contrast, blur and noise of a batch of X-rays [b, 1, rows, columns].
    Real X-rays do not look exactly like our fake ones, so the model must not depend on the exact look."""
    b = xray.shape[0]
    dev = xray.device
    gamma = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.5)
    x = xray.clamp(0, 1) ** gamma
    contrast = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    bright = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.08, 0.08)
    x = (x - 0.5) * contrast + 0.5 + bright
    if random.random() < 0.5:
        sigma = random.uniform(0.5, 2.0)
        k = torch.arange(-6, 7, device=dev, dtype=x.dtype)
        k = torch.exp(-0.5 * (k / sigma) ** 2)
        k = k / k.sum()
        x = F.conv2d(F.pad(x, (6, 6, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        x = F.conv2d(F.pad(x, (0, 0, 6, 6), mode="replicate"), k.view(1, 1, -1, 1))
    x = x + torch.randn_like(x) * random.uniform(0.0, 0.04)
    return x.clamp(0, 1)
