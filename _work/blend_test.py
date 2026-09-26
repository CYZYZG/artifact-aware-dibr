"""Does edge-domain blending remove the seam, and what does it cost?

seam   = mean |filled side - untouched side| at the boundary (9x9 local means)
p90    = 90th percentile of the same
GT     = against the real cam7, over the valid (non-hole) pixels
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import io_utils, viz  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.blend import offset_blend, poisson_blend  # noqa: E402
from viewfill.pipeline import warp_and_fill  # noqa: E402

rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, "f000")["color"]
K = 9


def metrics(I_f, I_w, hole):
    g_f = 0.299 * I_f[..., 0] + 0.587 * I_f[..., 1] + 0.114 * I_f[..., 2]
    g_w = 0.299 * I_w[..., 0] + 0.587 * I_w[..., 1] + 0.114 * I_w[..., 2]
    valid = ~hole

    def loc(img, mask):
        m = mask.astype(np.float32)
        num = cv2.boxFilter(img * m, -1, (K, K), normalize=False,
                            borderType=cv2.BORDER_REPLICATE)
        den = cv2.boxFilter(m, -1, (K, K), normalize=False,
                            borderType=cv2.BORDER_REPLICATE)
        return num / np.maximum(den, 1e-6)

    bnd = hole & cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    j = np.abs(loc(g_f, hole)[bnd] - loc(g_w, valid)[bnd])
    return float(j.mean()), float(np.percentile(j, 90))


res = warp_and_fill(rgb, inv, FillConfig(scale=-44.8))
I_f, I_w, hole = res["I_filled"], res["I_w"], res["hole_mask"]
m = ~res["remaining"]

print(f"{'blend':>12s} | {'seam':>7s} {'p90':>7s} | {'GT PSNR':>8s} {'GT SSIM':>8s} | "
      f"{'time s':>7s}")
seam0, p900 = metrics(I_f, I_w, hole)
print(f"{'none':>12s} | {seam0:>7.2f} {p900:>7.2f} | "
      f"{viz.psnr(gt, I_f, mask=m):>8.2f} {viz.ssim(gt, I_f, mask=m):>8.4f} | {'-':>7s}")

for name, fn in (("offset", lambda a: offset_blend(a, hole, iters=300)),
                 ("poisson", lambda a: poisson_blend(a, hole))):
    t = time.time()
    out = fn(I_f)
    dt = time.time() - t
    out = np.clip(out, 0, 255)
    s, p = metrics(out, I_w, hole)
    print(f"{name:>12s} | {s:>7.2f} {p:>7.2f} | {viz.psnr(gt, out, mask=m):>8.2f} "
          f"{viz.ssim(gt, out, mask=m):>8.4f} | {dt:>7.2f}")
    if name == "poisson":
        # visual
        x0, x1, y0, y1 = 600, 900, 20, 720
        sepr = np.full((y1 - y0, 3, 3), 255, np.uint8)
        row = np.hstack([np.clip(I_f, 0, 255).astype(np.uint8)[y0:y1, x0:x1], sepr,
                         out.astype(np.uint8)[y0:y1, x0:x1]])
        io_utils.imwrite(os.path.join(HERE, "_work", "seam", "poisson_compare.png"),
                         cv2.resize(row, None, fx=1.6, fy=1.6,
                                    interpolation=cv2.INTER_NEAREST))
        print("    poisson_compare.png = before | after  (man, x 600-900, y 20-720)")
