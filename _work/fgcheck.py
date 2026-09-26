"""Is the foreground shredded by the warp, and if so why?

Metrics
  * FG = reference pixels with a large inverse depth (the dancers).
  * warp the FG mask with the same displacement -> how much of the mapped FG area is
    missing, and how many FG pixels land on a target that is a hole.
  * repeat with a smoothed depth map to show the cause.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补")
from dibr import io_utils, warp as W  # noqa: E402
from viewfill import io as vio  # noqa: E402
from viewfill.pipeline import disparity_from_depth, depth_to_scale255  # noqa: E402

OUT = r"D:\项目\空洞填补\_work\warpcheck"
os.makedirs(OUT, exist_ok=True)
rgb, inv = vio.load_pair(r"D:\项目\空洞填补\input\color-cam6-f000.jpg",
                         r"D:\项目\空洞填补\input\depth-cam6-f000.png")
P = depth_to_scale255(inv)
h, w = P.shape


def analyse(P_used, label, scale=-44.8, splat="sub"):
    dx = disparity_from_depth(P_used, scale)
    fg = (P_used / 255.0) > 0.62                      # the two dancers
    fg_img = np.zeros((h, w, 3), np.float32)
    fg_img[..., 0] = np.where(fg, 255.0, 0.0)         # FG red
    fg_img[..., 1] = np.where(fg, 0.0, 255.0)         # BG green
    _, _, hole, _ = W.forward_warp(rgb.astype(np.float32), dx, None, z=P_used,
                                   hole_depth=-1.0, rule="zbuf", splat=splat)
    fgw, _, hole_fg, _ = W.forward_warp(fg_img, dx, None, z=P_used, hole_depth=-1.0,
                                        rule="zbuf", splat=splat)
    red = (fgw[..., 0] > 127)
    src_fg = int(fg.sum())
    mapped = int(red.sum())
    # how many FG pixels have every one of their targets empty
    ys, xs = np.nonzero(fg)
    t = np.rint(xs + dx[ys, xs]).astype(int)
    ok = (t >= 0) & (t < w)
    lost = int((~ok).sum()) + int(np.sum(hole[ys[ok], t[ok]]))
    print(f"{label:28s} FG px {src_fg:6d} -> mapped {mapped:6d} "
          f"({mapped/src_fg*100:5.1f}%)  FG landing on a hole: {lost:6d} "
          f"({lost/max(1,src_fg)*100:4.1f}%)  holes {hole.mean()*100:5.2f}%")
    return dict(fgw=fgw, hole=hole, fg=fg, dx=dx, lost=lost, mapped=mapped)


base = analyse(P, "raw MSR depth")
# edge preserving smoothing of the depth map (guided filter) - standard DIBR practice
guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
P_s = cv2.ximgproc.guidedFilter((guide * 255).astype(np.uint8), P, 8, 40.0) \
    if hasattr(cv2, "ximgproc") else None
if P_s is None:
    P_s = cv2.bilateralFilter(P, 9, 40, 9)
sm = analyse(P_s, "guided/bilateral smoothed")

# visualise
x0, x1, y0, y1 = 600, 960, 20, 720
tiles = []
for name, m in (("raw", base), ("smoothed", sm)):
    seg = np.clip(m["fgw"][y0:y1, x0:x1], 0, 255).astype(np.uint8)
    ovl = np.clip(rgb[y0:y1, x0:x1].astype(np.float32), 0, 255).astype(np.uint8).copy()
    hm = m["hole"][y0:y1, x0:x1]
    ovl[hm] = (0.4 * ovl[hm] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    tiles += [cv2.resize(seg, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST),
              cv2.resize(ovl, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)]
io_utils.imwrite(os.path.join(OUT, "fg_shredding.png"), np.hstack(tiles))
print(f"  fg_shredding.png = [raw: FG mask | raw: warped+holes] "
      f"[smoothed: FG mask | smoothed: warped+holes]  x {x0}-{x1}, y {y0}-{y1}")

# depth statistics inside the foreground (is it noisy?)
for name, Pm in (("raw", P), ("smoothed", P_s)):
    v = Pm[base["fg"]]
    lap = cv2.Laplacian(Pm, cv2.CV_32F)
    print(f"  {name:9s} depth inside FG: std {v.std():5.1f}, "
          f"|laplacian| median {np.median(np.abs(lap[base['fg']])):5.2f} "
          f"p95 {np.percentile(np.abs(lap[base['fg']]), 95):5.2f}")
