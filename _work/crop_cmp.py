"""Side-by-side zoom: reference | warped | filled, around each dancer and the OOFA edge."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补")
from dibr import io_utils  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.pipeline import warp_and_fill  # noqa: E402

OUT = r"D:\项目\空洞填补\_work\warpcheck"
os.makedirs(OUT, exist_ok=True)
rgb, inv = vio.load_pair(r"D:\项目\空洞填补\input\color-cam6-f000.jpg",
                         r"D:\项目\空洞填补\input\depth-cam6-f000.png")
cfg = FillConfig(scale=-44.8)
res = warp_and_fill(rgb, inv, cfg)
I_w = np.clip(res["I_w"], 0, 255).astype(np.uint8)
I_f = res["filled"]
hole = res["hole_mask"]
print(f"holes {int(hole.sum())} px -> residual {int(res['remaining'].sum())} px")


def strip(x0, x1, y0, y1, scale=1.6):
    ref = rgb[y0:y1, x0:x1]
    wrp = I_w[y0:y1, x0:x1].copy()
    fil = I_f[y0:y1, x0:x1]
    ovl = wrp.copy()
    ovl[hole[y0:y1, x0:x1]] = (255, 0, 0)
    sepr = np.full((y1 - y0, 3, 3), 255, np.uint8)
    row = np.hstack([ref, sepr, wrp, sepr, ovl, sepr, fil])
    return cv2.resize(row, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


io_utils.imwrite(os.path.join(OUT, "cmp_man.png"), strip(600, 960, 20, 720))
io_utils.imwrite(os.path.join(OUT, "cmp_woman.png"), strip(120, 480, 120, 720))
io_utils.imwrite(os.path.join(OUT, "cmp_right_edge.png"), strip(880, 1024, 100, 700))
print("cmp_man.png / cmp_woman.png / cmp_right_edge.png = "
      "reference | warped | warped+holes(red) | filled")

# is the filled content plausible at the disocclusion band?  compare its statistics
band = res["hole_mask"] & (np.arange(hole.shape[1])[None, :] > 200)
print(f"  disocclusion band px {int(band.sum())}, "
      f"filled mean {I_f[band].mean():.1f}, "
      f"neighbourhood (reference) mean {rgb[band].mean():.1f}")
