"""Background-anchored Poisson blending vs none.

The hole's boundary has two sides: the untouched BACKGROUND (which the fill should match)
and the foreground silhouette (which it must NOT be dragged towards).  Anchor only the
background side, and measure the seam separately on each side.
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
from viewfill.blend import poisson_blend  # noqa: E402
from viewfill.pipeline import warp_and_fill  # noqa: E402

rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, "f000")["color"]
K = 9


def seam(I_f, I_w, hole, mask_sel):
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

    bnd = mask_sel & hole & cv2.dilate(valid.astype(np.uint8),
                                       np.ones((3, 3), np.uint8)).astype(bool)
    if not bnd.any():
        return float("nan"), float("nan")
    j = np.abs(loc(g_f, hole)[bnd] - loc(g_w, valid)[bnd])
    return float(j.mean()), float(np.percentile(j, 90))


res = warp_and_fill(rgb, inv, FillConfig(scale=-44.8))
I_f, I_w, hole = res["I_filled"], res["I_w"], res["hole_mask"]
m = ~res["remaining"]
D_f = np.asarray(res["D_filled"], np.float32)

# background side = an untouched neighbour that is FARTHER than the hole pixel
near = cv2.dilate(D_f, np.ones((3, 3), np.uint8))
far = cv2.erode(np.where(~hole, D_f, 1e9).astype(np.float32), np.ones((3, 3), np.uint8))
near_n = cv2.dilate(D_f, np.ones((3, 3), np.uint8))
# anchor: untouched pixels that are not the nearest (foreground) layer around the hole
anchor = (~hole) & (D_f <= (cv2.erode(np.where(hole, 0.0, D_f), np.ones((5, 5), np.uint8))
                            + 0.0) + 1e-3)
# simpler and more robust: untouched pixels with a LOWER inverse depth than the local hole
anchor = (~hole) & (D_f < cv2.dilate(np.where(hole, 0.0, -1e9).astype(np.float32),
                                     np.ones((3, 3), np.uint8)))
anchor = (~hole) & (D_f < cv2.dilate(np.where(hole, -1e9, D_f).astype(np.float32),
                                     np.ones((3, 3), np.uint8)) - 1e-3)
print(f"anchor pixels: {int(anchor.sum())} of {int((~hole).sum())} untouched")

rows = []
for name, anc in (("none", None), ("poisson(bg-anchor)", anchor),
                  ("poisson(all)", np.ones_like(hole))):
    t = time.time()
    out = I_f if anc is None else np.clip(poisson_blend(I_f, hole, anchor=anc), 0, 255)
    dt = time.time() - t
    sa, pa = seam(out, I_w, hole, np.ones_like(hole))
    sb, pb = seam(out, I_w, hole, ~anchor if anc is not None else np.ones_like(hole))
    print(f"{name:>18s} | seam(all) {sa:6.2f} p90 {pa:6.2f} | seam(BG side) {sb:6.2f} "
          f"p90 {pb:6.2f} | GT {viz.psnr(gt, out, mask=m):6.2f} "
          f"{viz.ssim(gt, out, mask=m):7.4f} | {dt:5.2f}s")
    rows.append((name, out))

# visual of the two best
x0, x1, y0, y1 = 600, 900, 20, 720
sepr = np.full((y1 - y0, 3, 3), 255, np.uint8)
tiles = [np.clip(r, 0, 255).astype(np.uint8)[y0:y1, x0:x1] for _, r in rows]
row = tiles[0]
for t in tiles[1:]:
    row = np.hstack([row, sepr, t])
io_utils.imwrite(os.path.join(HERE, "_work", "seam", "blend_compare.png"),
                 cv2.resize(row, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))
print("blend_compare.png = none | poisson(bg-anchor) | poisson(all)")
