"""Dark-rim fix: cracks on a depth step are filled from the background side.

rim(all)   = mean luma of the filled pixels 1-3 px from the hole boundary minus 3-10 px
crack mean = mean luma of the crack (thin-hole) pixels
disocc rim = same rim computed only over the disocclusion pixels
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

hdr = ["crack_fill", "rim(all)", "crack mean", "disocc rim", "GT PSNR", "GT SSIM"]
print(" | ".join(f"{h:>11s}" for h in hdr))
for mode in ("hhf", "auto"):
    cfg = FillConfig(scale=-44.8, crack_fill=mode)
    r = warp_and_fill(rgb, inv, cfg)
    I_f, hole, crack = r["I_filled"], r["hole_mask"], r["crack"]
    g = gray(I_f)
    di = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
    near, far = di <= 3, (di > 3) & (di <= 10)
    m = ~r["remaining"]
    vals = [mode,
            f"{g[hole & near].mean() - g[hole & far].mean():+.2f}",
            f"{g[crack].mean():.2f}",
            f"{g[hole & ~crack & near].mean() - g[hole & ~crack & far].mean():+.2f}",
            f"{viz.psnr(gt, I_f, mask=m):.2f}",
            f"{viz.ssim(gt, I_f, mask=m):.4f}"]
    print(" | ".join(f"{v:>11s}" for v in vals))
    print(f"            bg-side filled px: {r['stats'].get('crack_bg_side_px')}")
