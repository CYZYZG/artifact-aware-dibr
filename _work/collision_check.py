"""Is the provided warping.scatter_image eating the foreground?

The provided splat accumulates every sample that lands on a target pixel and, with
inverse_ordering=True, lets the FAR sample win the collision -> the foreground silhouette
is replaced by background.  Compare it with the Z-buffer warp.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补")
from dibr import io_utils, viz  # noqa: E402
from warping import scatter_image  # noqa: E402
from viewfill import io as vio  # noqa: E402
from viewfill.pipeline import disparity_from_depth, depth_to_scale255  # noqa: E402
from dibr import warp as W  # noqa: E402

OUT = r"D:\项目\空洞填补\_work\warpcheck"
os.makedirs(OUT, exist_ok=True)
rgb, inv = vio.load_pair(r"D:\项目\空洞填补\input\color-cam6-f000.jpg",
                         r"D:\项目\空洞填补\input\depth-cam6-f000.png")
P = depth_to_scale255(inv)
h, w = P.shape
dx = disparity_from_depth(P, -44.8)
fg = (P / 255.0) > 0.62
print(f"disparity: dx = inv*(-44.8), range [{dx.min():.2f}, {dx.max():.2f}] px; "
      f"foreground {int(fg.sum())} px")


def fg_damage(img, hole, label):
    """How much of the foreground's colour is replaced by background?"""
    # reference foreground colour vs what sits at its mapped target
    ys, xs = np.nonzero(fg)
    t = np.rint(xs + dx[ys, xs]).astype(int)
    ok = (t >= 0) & (t < w) & ~hole[ys, t]
    err = np.abs(img[ys[ok], t[ok]] - rgb[ys[ok], xs[ok]]).max(axis=1)
    bad = float((err > 40).mean() * 100)
    print(f"  {label:34s} holes {hole.mean()*100:5.2f}%  "
          f"FG px landing on intact target {ok.sum():6d}  "
          f"FG colour wrong by >40 gray: {bad:5.2f}%")
    return bad


# provided implementation, both collision behaviours
for ordering in (True, False):
    img, hm, dep = scatter_image(rgb.astype(np.float32), inv, direction=-1,
                                 scale_factor=44.8, inverse_ordering=ordering,
                                 reproject_depth=True)
    fg_damage(np.asarray(img, np.float32), hm > 0,
              f"provided scatter inverse_ordering={ordering}")

# Z-buffer warp (ours)
for splat, rule in (("sub", "zbuf"), ("sub", "avg"), ("floor", "zbuf")):
    img, _, hole, _ = W.forward_warp(rgb.astype(np.float32), dx, None, z=P,
                                     hole_depth=-1.0, rule=rule, splat=splat)
    fg_damage(img, hole, f"ours splat={splat} rule={rule}")

# ---- visual: crops of the man from each warp
crops = []
prov = {}
for ordering in (True, False):
    img, hm, _ = scatter_image(rgb.astype(np.float32), inv, direction=-1,
                               scale_factor=44.8, inverse_ordering=ordering,
                               reproject_depth=True)
    prov[ordering] = np.asarray(img, np.float32)
ours, _, hole_ours, _ = W.forward_warp(rgb.astype(np.float32), dx, None, z=P,
                                       hole_depth=-1.0, rule="zbuf", splat="sub")
x0, x1, y0, y1 = 600, 900, 20, 720
sepr = np.full((y1 - y0, 3, 3), 255, np.uint8)
row = np.hstack([rgb[y0:y1, x0:x1],
                 sepr, np.clip(prov[True][y0:y1, x0:x1], 0, 255).astype(np.uint8),
                 sepr, np.clip(prov[False][y0:y1, x0:x1], 0, 255).astype(np.uint8),
                 sepr, np.clip(ours[y0:y1, x0:x1], 0, 255).astype(np.uint8)])
io_utils.imwrite(os.path.join(OUT, "collision_compare.png"),
                 cv2.resize(row, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_NEAREST))
print("collision_compare.png = reference | provided(ordering=True) | "
      "provided(ordering=False) | ours(Z-buffer)")
