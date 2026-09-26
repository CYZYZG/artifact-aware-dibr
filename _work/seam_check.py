"""Is there a visible seam at the boundary between FILLED pixels and UNTOUCHED background?

For every filled pixel that touches untouched content, compare the local mean of the
filled side (<=4 px inside) with the local mean of the untouched side (<=4 px outside).
That jump is then compared with the typical local variation of the background, measured
the same way at random untouched locations (patch vs patch offset by 8 px).
"""
import os
import sys

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import io_utils  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.pipeline import warp_and_fill  # noqa: E402

OUT = os.path.join(HERE, "_work", "seam")
os.makedirs(OUT, exist_ok=True)
rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
res = warp_and_fill(rgb, inv, FillConfig(scale=-44.8))
I_f = res["I_filled"]
I_w = res["I_w"]
hole = res["hole_mask"]
gray = lambda a: 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
gf, gw = gray(I_f), gray(I_w)

valid = ~hole
K = 2 * 4 + 1


def local_mean(img, mask, k=K):
    m = mask.astype(np.float32)
    num = cv2.boxFilter(img * m, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE)
    den = cv2.boxFilter(m, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE)
    return num / np.maximum(den, 1e-6), den


inner, n_in = local_mean(gf, hole)       # mean of the filled content around a pixel
outer, n_out = local_mean(gw, valid)     # mean of the untouched content around a pixel
bnd = hole & cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
jump = np.zeros_like(gf)
jump[bnd] = inner[bnd] - outer[bnd]

# natural local variation of the background (reference patch vs patch 8 px away)
gref = gray(np.asarray(rgb, np.float32))
k5 = 5
loc = cv2.boxFilter(gref, -1, (k5, k5), normalize=True, borderType=cv2.BORDER_REPLICATE)
nat = np.zeros_like(gref)
nat[:, :-8] = np.abs(loc[:, :-8] - loc[:, 8:])
nat_valid = nat[valid & (np.roll(valid, 8, axis=1))]
print(f"boundary pixels: {int(bnd.sum())}")
print(f"jump at the filled/background boundary: "
      f"median {np.median(np.abs(jump[bnd])):.2f}, mean {np.abs(jump[bnd]).mean():.2f}, "
      f"p90 {np.percentile(np.abs(jump[bnd]), 90):.2f}, "
      f"p99 {np.percentile(np.abs(jump[bnd]), 99):.2f} gray levels")
print(f"signed jump: median {np.median(jump[bnd]):+.2f}  (>0 = filled side brighter)")
print(f"natural background variation (5x5 patch vs 8 px away): "
      f"median {np.median(nat_valid):.2f}, mean {nat_valid.mean():.2f}, "
      f"p90 {np.percentile(nat_valid, 90):.2f}")
print(f"-> the seam is {np.abs(jump[bnd]).mean() / max(nat_valid.mean(), 1e-6):.1f}x "
      f"the natural local variation")

# where is the worst (excluding the silhouette neighbourhood)?
fg = inv > 0.62
far_from_fg = cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, 5) > 25
cand = bnd & far_from_fg & (np.abs(jump) > 12)
print(f"\nfilled pixels on a <-12|+12> gray-level jump and >25 px away from the "
      f"subject: {int(cand.sum())}")
if cand.any():
    ys, xs = np.nonzero(cand)
    order = np.argsort(-np.abs(jump[ys, xs]))[:200]
    # keep the largest connected clump
    m = np.zeros_like(cand)
    m[ys[order], xs[order]] = True
    n, _, cc, cent = cv2.connectedComponentsWithStats(
        cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)), 8)
    k = 1 + int(np.argmax(cc[1:, cv2.CC_STAT_AREA]))
    cy, cx = int(cent[k][1]), int(cent[k][0])
    print(f"  worst clump around ({cx}, {cy}), |jump| up to "
          f"{np.abs(jump[ys[order[:50]], xs[order[:50]]]).max():.1f}")
    half = 90
    x0, x1 = max(0, cx - half), min(I_f.shape[1], cx + half)
    y0, y1 = max(0, cy - half), min(I_f.shape[0], cy + half)
    sepr = np.full((y1 - y0, 3, 3), 255, np.uint8)
    ovl = np.clip(I_f, 0, 255).astype(np.uint8).copy()
    ovl[hole] = (0.35 * ovl[hole] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
    row = np.hstack([np.clip(rgb, 0, 255).astype(np.uint8)[y0:y1, x0:x1], sepr,
                     np.clip(I_w, 0, 255).astype(np.uint8)[y0:y1, x0:x1], sepr,
                     ovl[y0:y1, x0:x1], sepr,
                     np.clip(I_f, 0, 255).astype(np.uint8)[y0:y1, x0:x1]])
    io_utils.imwrite(os.path.join(OUT, "worst_seam.png"),
                     cv2.resize(row, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST))
    print("  worst_seam.png (x3) = reference | warped | filled+hole(red) | filled")
