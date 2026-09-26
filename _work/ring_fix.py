"""Which combination removes the dark rim, and does it cost image quality?

Metric "rim" = mean luma of the filled pixels 1-3 px from the hole boundary minus the
mean luma of the filled pixels 3-10 px inside (negative = darker rim).
Evaluated on the user's 1D flow, against the real cam7 for the quality numbers.
"""
import os
import sys

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import io_utils, viz  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.pipeline import warp_and_fill  # noqa: E402

rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, "f000")["color"]
gray = lambda a: 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

print(f"{'bg_template':>11s} {'full_bg':>8s} | {'rim':>7s} {'filled 1-3':>10s} "
      f"{'filled 3-10':>11s} | {'holes':>6s} {'iters':>6s} {'sizes (9/7/5/3)':>20s} | "
      f"{'GT PSNR':>8s} {'GT SSIM':>8s}")
for bg_t in (False, True):
    for full_bg in (False, True):
        cfg = FillConfig(scale=-44.8, bg_template=bg_t, require_full_bg=full_bg)
        r = warp_and_fill(rgb, inv, cfg)
        I_f, hole = r["I_filled"], r["hole_mask"]
        di = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
        near = hole & (di <= 3)
        far = hole & (di > 3) & (di <= 10)
        g = gray(I_f)
        rim = float(g[near].mean() - g[far].mean())
        m = ~r["remaining"]
        sz = r["stats"]["patch_sizes"]
        print(f"{str(bg_t):>11s} {str(full_bg):>8s} | {rim:>+7.2f} {g[near].mean():>10.2f} "
              f"{g[far].mean():>11.2f} | {int(hole.sum()):>6d} "
              f"{r['stats']['iterations']:>6d} "
              f"{str([sz.get(k,0) for k in (9,7,5,3)]):>20s} | "
              f"{viz.psnr(gt, I_f, mask=m):>8.2f} {viz.ssim(gt, I_f, mask=m):>8.4f}")
