"""The model: gets a panoramic X-ray, outputs the straightened 3D block.

Plain description:
1. A "U-Net" looks at the X-ray. A U-Net is a common program for pictures. It shrinks the picture
   step by step to find large shapes (a whole jaw), then grows it back step by step to find small
   shapes (one tooth root), and at each size it reuses what it found on the way down.
2. For each X-ray pixel the U-Net outputs a list of numbers. The list is turned into a short line
   of points going from the tongue side to the cheek side. All lines together are the 3D block.
3. A small 3D step looks at each point together with its neighbours in all three directions and
   outputs, for each point: how bright it is in a 3D scan, and which part it is (bone, tooth, ...).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(n_in, n_out):
    return nn.Sequential(
        nn.Conv2d(n_in, n_out, 3, padding=1), nn.GroupNorm(8, n_out), nn.ReLU(inplace=True),
        nn.Conv2d(n_out, n_out, 3, padding=1), nn.GroupNorm(8, n_out), nn.ReLU(inplace=True))


class XrayTo3D(nn.Module):
    def __init__(self, in_channels=1, depth=64, n_classes=6, widths=(48, 96, 192, 384, 512), lift=8):
        """in_channels: 1 for the X-ray alone. More when extra inputs (from the mouth video) are added.
        depth: number of points on each tongue-to-cheek line."""
        super().__init__()
        self.depth, self.n_classes, self.lift = depth, n_classes, lift
        self.down = nn.ModuleList()
        n = in_channels
        for w in widths:
            self.down.append(conv_block(n, w))
            n = w
        self.up = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.up.append(conv_block(n + w, w))
            n = w
        self.to_lines = nn.Conv2d(n, lift * depth, 1)
        self.head3d = nn.Sequential(
            nn.Conv3d(lift, 24, 3, padding=1), nn.GroupNorm(8, 24), nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, 3, padding=1), nn.GroupNorm(8, 24), nn.ReLU(inplace=True),
            nn.Conv3d(24, 1 + n_classes, 1))

    def forward(self, xray):
        """xray: [batch, in_channels, rows, columns], rows and columns must be multiples of 16.
        Returns (brightness [batch, depth, rows, columns] between 0 and 1,
                 part scores [batch, n_classes, depth, rows, columns])."""
        kept = []
        x = xray
        for i, block in enumerate(self.down):
            x = block(x)
            if i < len(self.down) - 1:
                kept.append(x)
                x = F.max_pool2d(x, 2)
        for block in self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(torch.cat([x, kept.pop()], dim=1))
        b, _, h, w = x.shape
        x = self.to_lines(x).view(b, self.lift, self.depth, h, w)
        x = self.head3d(x)
        return torch.sigmoid(x[:, 0]), x[:, 1:]
