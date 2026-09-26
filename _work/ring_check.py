"""Diagnose the dark ring at the filled-region / background boundary.

1) radial profile: mean luma of the FILLED pixels vs the distance to the hole boundary,
   compared with the mean luma of the untouched pixels just OUTSIDE the boundary.
2) how much foreground (dark silhouette) content gets copied into the holes?
3) high zoom crops of the boundary.
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

OUT = os.path.join(HERE, "_work", "ring")
os.makedirs(OUT, exist_ok=True)
rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
cfg = FillConfig(scale=-44.8)
res = warp_and_fill(rgb, inv, cfg)
I_f = res["I_filled"]
I_w = res["I_w"]
hole = res["hole_mask"]

gray = lambda a: (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
gf, gw, gr = gray(I_f), gray(I_w), gray(np.asarray(rgb, np.float32))

# ---- radial profile around the hole boundary
dist_in = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
dist_out = cv2.distanceTransform((~hole).astype(np.uint8), cv2.DIST_L2, 5)
print(f"{'distance to boundary':>22s} {'FILLED px mean':>14s} {'n':>8s} "
      f"{'OUTSIDE px mean':>15s} {'n':>8s} {'diff':>7s}")
for d in range(1, 16):
    m_in = hole & (dist_in >= d - 0.5) & (dist_in < d + 0.5)
    m_out = (~hole) & (dist_out >= d - 0.5) & (dist_out < d + 0.5)
    if m_in.sum() and m_out.sum():
        a, b = float(gf[m_in].mean()), float(gf[m_out].mean())
        print(f"{d:>22d} {a:>14.2f} {int(m_in.sum()):>8d} {b:>15.2f} "
              f"{int(m_out.sum()):>8d} {a-b:>+7.2f}")

# a wider view: the first 3 px inside vs the 3-10 px inside
near = hole & (dist_in <= 3)
far = hole & (dist_in > 3) & (dist_in <= 10)
out3 = (~hole) & (dist_out <= 3)
print(f"\nfilled band 1-3 px from the boundary : {gf[near].mean():6.2f}")
print(f"filled band 3-10 px                  : {gf[far].mean():6.2f}")
print(f"untouched pixels 1-3 px outside      : {gf[out3].mean():6.2f}")
print(f"-> darkening at the seam: {gf[near].mean() - gf[far].mean():+.2f} gray levels")

# ---- how much foreground content gets copied in?  (re-run the fill with a counter)
import dibr.inpaint as I  # noqa: E402

orig_copy = None
copied_fg = {"n": 0, "total": 0}
src_depth = inv * 255.0
thresholds = {}


def patched_fill_all(I_w, D_w, hole, ref_color, ref_depth, disp, src_of, **kw):
    """Wrap fill_all to record, per copied pixel, whether the source was foreground."""
    return I.fill_all(I_w, D_w, hole, ref_color, ref_depth, disp, src_of, **kw)


print("\n(foreground-copy audit is reported by the fill log below)")
