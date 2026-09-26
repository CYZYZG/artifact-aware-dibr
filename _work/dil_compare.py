"""depth_dilate = 2 (uniform) vs "auto" (demand-driven): where exactly do they differ?

Metrics
  seam      mean / p90 / p99 / max of |filled side - untouched side| at the boundary
  cracks    thin-hole pixels detected
  GT        against the real cam7, over valid pixels
  fg px     area of the warped foreground mask (geometric cost of uniform dilation)
  bg px     area of the filled region (how much background got eaten)
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
from viewfill.pipeline import warp_and_fill, warp_view  # noqa: E402

rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, "f000")["color"]
K = 9


def seam_stats(I_f, I_w, hole):
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
    return (j.mean(), np.percentile(j, 90), np.percentile(j, 99), j.max())


print(f"{'mode':>6s} | {'seam':>6s} {'p90':>6s} {'p99':>6s} {'max':>6s} | {'cracks':>7s} | "
      f"{'GT PSNR':>7s} {'GT SSIM':>7s} | {'war fg px':>9s} {'hole px':>8s} | {'t s':>5s}")
outs = {}
for mode in (0, 2, 3, "auto", "auto5", "auto7"):
    t = time.time()
    r = warp_and_fill(rgb, inv, FillConfig(scale=-44.8, depth_dilate=mode))
    dt = time.time() - t
    I_f, I_w, hole = r["I_filled"], r["I_w"], r["hole_mask"]
    # geometric cost: where does the foreground silhouette land after the warp?
    Iw_fg, _, hole_fg, _, _, _ = warp_view((inv > 0.62).astype(np.float32) * 255, inv * 255,
                                           FillConfig(scale=-44.8, depth_dilate=mode))
    fg_px = int((Iw_fg[..., 0] > 127).sum())
    s = seam_stats(I_f, I_w, hole)
    m = ~r["remaining"]
    print(f"{str(mode):>6s} | {s[0]:>6.2f} {s[1]:>6.2f} {s[2]:>6.1f} {s[3]:>6.1f} | "
          f"{int(r['crack'].sum()):>7d} | {viz.psnr(gt, I_f, mask=m):>7.2f} "
          f"{viz.ssim(gt, I_f, mask=m):>7.4f} | {fg_px:>9d} {int(hole.sum()):>8d} | "
          f"{dt:>5.1f}")
    outs[mode] = (I_f, r)

# where are the remaining strong seams?  map them for both modes
x0, x1, y0, y1 = 640, 900, 60, 700
sepr = np.full((y1 - y0, 3, 3), 255, np.uint8)
row = None
for mode in (2, "auto7"):
    I_f = outs[mode][0]
    tile = np.clip(I_f, 0, 255).astype(np.uint8)[y0:y1, x0:x1]
    row = tile if row is None else np.hstack([row, sepr, tile])
io_utils.imwrite(os.path.join(HERE, "_work", "seam", "dil2_vs_auto_zoom.png"),
                 cv2.resize(row, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_NEAREST))
print("dil2_vs_auto_zoom.png (x2.4) = depth_dilate=2 | auto   (man, x 640-900, y 60-700)")
