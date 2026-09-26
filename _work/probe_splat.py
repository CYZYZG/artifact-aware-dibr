"""Does the splat rule explain the paper's "cracks can be up to 8.4%"?

Compares sub-pixel 2-tap splatting (our Z-buffer warp) with integer single-tap
splatting (what an integer disparity map + nearest-neighbour splat produces).
"""
import cv2
import numpy as np
import sys

sys.path.insert(0, r"D:\项目\空洞填补")
from dibr import calib, cracks, io_utils, warp  # noqa: E402

ROOT = io_utils.DATASET_ROOT_DEFAULT
cams = calib.load_calib(ROOT)
v6 = io_utils.load_view(ROOT, 6, "f000")
v7 = io_utils.load_view(ROOT, 7, "f000")
color = v6["color"].astype(np.float32)
P = v6["depth"].astype(np.float32)
dx, dy, _, _, _ = calib.displacement_field(cams, 6, 7, P)


def single_tap(color, dx, dy, z, mode="floor"):
    """One target pixel per source pixel (no sub-pixel interpolation)."""
    h, w = dx.shape
    y, x = np.mgrid[0:h, 0:w]
    tx = x + dx
    ty = y + dy
    if mode == "floor":
        x0 = np.floor(tx).astype(np.int64)
        y0 = np.floor(ty).astype(np.int64)
    else:
        x0 = np.round(tx).astype(np.int64)
        y0 = np.round(ty).astype(np.int64)
    sel = (x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h)
    yy, xx, zz = y0[sel], x0[sel], z[sel]
    src = color[sel]
    zb = np.full((h, w), -np.inf, np.float32)
    np.maximum.at(zb, (yy, xx), zz)
    win = zz >= zb[yy, xx]
    zb_img = np.zeros((h, w, 3), np.float32)
    zb_img[yy[win], xx[win]] = src[win]
    hole = ~np.isfinite(zb)
    return zb_img, hole


def report(name, img, hole):
    rows, lab = cracks.component_stats(hole)
    thin = [r for r in rows if r["thickness"] <= 2]
    gt = v7["color"]
    from dibr import viz
    ps = viz.psnr(gt, img, mask=~hole)
    print(f"{name:34s} holes {hole.mean()*100:6.2f}%  comps {len(rows):5d}  "
          f"thin<=2px: {len(thin):4d} comps / {sum(r['area'] for r in thin):7d} px "
          f"({sum(r['area'] for r in thin)/hole.sum()*100 if hole.sum() else 0:5.1f}% of holes)"
          f"  PSNR(valid) {ps:5.2f}")


report("sub-pixel 2-tap + Z-buffer (ours)",
       *(lambda r: (r[0], r[2]))(warp.forward_warp(color, dx, dy, z=P)))
report("sub-pixel 2-tap + avg rule",
       *(lambda r: (r[0], r[2]))(warp.forward_warp(color, dx, dy, z=P, rule="avg")))
report("integer floor, single tap, Z-buffer", *single_tap(color, dx, dy, P, "floor"))
report("integer round, single tap, Z-buffer", *single_tap(color, dx, dy, P, "round"))

# horizontal-only variants
report("dx-only sub-pixel 2-tap + Z-buffer",
       *(lambda r: (r[0], r[2]))(warp.forward_warp(color, dx, z=P)))
report("dx-only integer single tap", *single_tap(color, dx, np.zeros_like(dy), P, "floor"))
