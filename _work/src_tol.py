"""Does rejecting source patches that straddle a depth edge remove the ghost contour?

seam(all)  = mean |filled side - untouched side| at the boundary (9x9 local means)
seam(BG)   = same, restricted to boundary pixels whose outside neighbour is background
ghost      = std of the filled band's high-frequency content (lower = fewer duplicated
             edges);  edges = number of |Laplacian| > 30 pixels inside the filled band
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
    lap = np.abs(cv2.Laplacian(g_f, cv2.CV_32F))
    return j.mean(), np.percentile(j, 90), (lap[hole] > 30).sum(), g_f[hole].std()


print(f"{'src_depth_tol':>13s} | {'seam':>6s} {'p90':>6s} {'band edges':>10s} | "
      f"{'GT PSNR':>8s} {'GT SSIM':>8s} | {'iters':>6s} {'sizes 9/7/5/3':>16s} | {'t s':>6s}")
best = None
for tol in (0.0, 3.0, 6.0, 12.0, 24.0):
    t = time.time()
    r = warp_and_fill(rgb, inv, FillConfig(scale=-44.8, src_depth_tol=tol))
    dt = time.time() - t
    I_f, I_w, hole = r["I_filled"], r["I_w"], r["hole_mask"]
    s, p90, edges, std = report(I_f, I_w, hole)
    m = ~r["remaining"]
    sz = r["stats"]["patch_sizes"]
    print(f"{tol:>13.1f} | {s:>6.2f} {p90:>6.2f} {edges:>10d} | "
          f"{viz.psnr(gt, I_f, mask=m):>8.2f} {viz.ssim(gt, I_f, mask=m):>8.4f} | "
          f"{r['stats']['iterations']:>6d} "
          f"{str([sz.get(k,0) for k in (9,7,5,3)]):>16s} | {dt:>6.1f}")
    if best is None or (s < best[1] and viz.psnr(gt, I_f, mask=m) >= viz.psnr(gt, best[0], mask=m) - 0.05):
        best = (I_f, s, tol)
    if tol in (0.0, 6.0):
        io_utils.imwrite(os.path.join(HERE, "_work", "seam", f"tol{int(tol)}.png"),
                         I_f.astype(np.uint8)[:, :, ::-1])
