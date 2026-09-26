"""Uniform vs demand-driven splat widening."""
import os
import sys
import time

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
K = 9


def report(I_f, I_w, hole):
    g_f = 0.299 * I_f[..., 0] + 0.587 * I_f[..., 1] + 0.114 * I_f[..., 2]
    g_w = 0.299 * I_w[..., 0] + 0.587 * I_w[..., 1] + 0.114 * I_w[..., 2]
    valid = ~hole

    def loc(img, m):
        mm = m.astype(np.float32)
        num = cv2.boxFilter(img * mm, -1, (K, K), normalize=False,
                            borderType=cv2.BORDER_REPLICATE)
        den = cv2.boxFilter(mm, -1, (K, K), normalize=False,
                            borderType=cv2.BORDER_REPLICATE)
        return num / np.maximum(den, 1e-6)

    bnd = hole & cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    j = np.abs(loc(g_f, hole)[bnd] - loc(g_w, valid)[bnd])
    fg = inv > 0.62
    return j.mean(), np.percentile(j, 90), int(((g_f > 0) & fg).sum()), int(hole.sum())


print(f"{'depth_dilate':>13s} | {'holes':>6s} {'cracks':>7s} | {'seam':>6s} {'p90':>6s} | "
      f"{'GT PSNR':>8s} {'GT SSIM':>8s} | {'fg px':>7s} | {'t s':>6s}")
for mode in (0, 1, 2, 3, "auto"):
    t = time.time()
    r = warp_and_fill(rgb, inv, FillConfig(scale=-44.8, depth_dilate=mode))
    dt = time.time() - t
    I_f, I_w, hole = r["I_filled"], r["I_w"], r["hole_mask"]
    s, p90, fgpx, nh = report(I_f, I_w, hole)
    m = ~r["remaining"]
    print(f"{str(mode):>13s} | {nh:>6d} {int(r['crack'].sum()):>7d} | {s:>6.2f} {p90:>6.2f} | "
          f"{viz.psnr(gt, I_f, mask=m):>8.2f} {viz.ssim(gt, I_f, mask=m):>8.4f} | "
          f"{fgpx:>7d} | {dt:>6.1f}")
    if mode in ("auto", 2):
        io_utils.imwrite(os.path.join(HERE, "_work", "seam", f"dil_{mode}.png"),
                         I_f.astype(np.uint8)[:, :, ::-1])
