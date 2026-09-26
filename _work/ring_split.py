"""Where does the dark rim come from?

(a) split the rim into the crack-filled pixels and the disocclusion-filled pixels
(b) does the REFERENCE image already have a dark halo around the foreground silhouette?
"""
import os
import sys

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.pipeline import warp_and_fill  # noqa: E402

rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
gray = lambda a: 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

res = warp_and_fill(rgb, inv, FillConfig(scale=-44.8, bg_template=True))
I_f, hole, crack = res["I_filled"], res["hole_mask"], res["crack"]
g = gray(I_f)
di = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
near = di <= 3
far = (di > 3) & (di <= 10)

print("rim by region (negative = darker than the core of the filled area)")
for name, m in (("crack pixels        ", hole & crack),
                ("disocclusion pixels ", hole & ~crack),
                ("all hole pixels     ", hole)):
    a = float(g[m & near].mean()) if (m & near).any() else float("nan")
    b = float(g[m & far].mean()) if (m & far).any() else float("nan")
    print(f"  {name} n={int(m.sum()):6d}  1-3px {a:7.2f}  3-10px {b:7.2f}  rim {a-b:+6.2f}")

# ---- does the reference itself have a dark halo around the silhouette?
ref_g = gray(np.asarray(rgb, np.float32))
fg = inv > 0.62
d_fg = cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, 5)
print("\nreference image, background pixels by distance to the foreground silhouette:")
for d in (1, 2, 3, 5, 8, 12, 20, 30, 50):
    m = (~fg) & (d_fg >= d - 0.5) & (d_fg < d + 0.5)
    if m.sum():
        print(f"  distance {d:3d} px: {ref_g[m].mean():7.2f}  (n={int(m.sum())})")
ring = (~fg) & (d_fg <= 3)
out = (~fg) & (d_fg > 3) & (d_fg <= 10)
print(f"  -> reference halo: {ref_g[ring].mean() - ref_g[out].mean():+.2f} gray levels "
      f"(1-3px vs 3-10px)")

# ---- is the warped image's own content near the hole boundary darker?
print("\nwarped image (before filling), valid pixels by distance to the hole boundary:")
gw = gray(res["I_w"])
do = cv2.distanceTransform((~hole).astype(np.uint8), cv2.DIST_L2, 5)
for d in (1, 2, 3, 5, 8, 12, 20):
    m = (~hole) & (do >= d - 0.5) & (do < d + 0.5)
    if m.sum():
        print(f"  distance {d:3d} px: {gw[m].mean():7.2f}  (n={int(m.sum())})")
