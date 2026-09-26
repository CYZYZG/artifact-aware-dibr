"""Geometric cost of each widening mode: how much does the foreground layer grow?

The foreground mask is warped with the same disparity field, so its area shows how many
background pixels the (widened) foreground layer swallows.
"""
import os
import sys

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import io_utils  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.pipeline import warp_and_fill, warp_view  # noqa: E402

rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
ref_fg = (inv > 0.62).astype(np.float32) * 255.0
print(f"reference foreground: {int((ref_fg > 127).sum())} px")
print(f"{'mode':>6s} | {'warped fg px':>12s} {'vs ref':>8s} | {'filled px':>9s}")
for mode in (0, 1, 2, 3, "auto", "auto5", "auto7"):
    cfg = FillConfig(scale=-44.8, depth_dilate=mode)
    I_w, _, _, _, _, _ = warp_view(ref_fg, inv * 255.0, cfg)
    fg_px = int((np.asarray(I_w) > 127).sum())
    r = warp_and_fill(rgb, inv, cfg)
    print(f"{str(mode):>6s} | {fg_px:>12d} {fg_px / (ref_fg > 127).sum() - 1:>+7.2%} | "
          f"{int(r['hole_mask'].sum()):>9d}")

# visual: 3 vs auto7
outs = {}
for mode in (3, "auto7"):
    outs[mode] = warp_and_fill(rgb, inv, FillConfig(scale=-44.8, depth_dilate=mode))["filled"]
x0, x1, y0, y1 = 640, 900, 60, 700
sepr = np.full((y1 - y0, 3, 3), 255, np.uint8)
row = np.hstack([outs[3][y0:y1, x0:x1], sepr, outs["auto7"][y0:y1, x0:x1]])
io_utils.imwrite(os.path.join(HERE, "_work", "seam", "dil3_vs_auto7.png"),
                 cv2.resize(row, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_NEAREST))
print("dil3_vs_auto7.png (x2.4) = depth_dilate=3 | auto7")
