"""Seam metric across the filled/background boundary for each crack-fill mode."""
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
K = 9


def local(img, mask):
    m = mask.astype(np.float32)
    num = cv2.boxFilter(img * m, -1, (K, K), normalize=False,
                        borderType=cv2.BORDER_REPLICATE)
    den = cv2.boxFilter(m, -1, (K, K), normalize=False, borderType=cv2.BORDER_REPLICATE)
    return num / np.maximum(den, 1e-6)


print(f"{'crack_fill':>10s} | {'seam mean':>9s} {'seam p90':>8s} {'seam max':>8s} | "
      f"{'GT PSNR':>8s} {'GT SSIM':>8s} {'crack luma':>10s}")
for mode in ("hhf", "bg", "linear", "auto"):
    r = warp_and_fill(rgb, inv, FillConfig(scale=-44.8, crack_fill=mode))
    I_f, I_w, hole = r["I_filled"], r["I_w"], r["hole_mask"]
    gf, gw = gray(I_f), gray(I_w)
    valid = ~hole
    inner, outer = local(gf, hole), local(gw, valid)
    bnd = hole & cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    j = np.abs(inner[bnd] - outer[bnd])
    m = ~r["remaining"]
    print(f"{mode:>10s} | {j.mean():>9.2f} {np.percentile(j, 90):>8.2f} {j.max():>8.1f} | "
          f"{viz.psnr(gt, I_f, mask=m):>8.2f} {viz.ssim(gt, I_f, mask=m):>8.4f} "
          f"{gf[r['crack']].mean():>10.2f}")
