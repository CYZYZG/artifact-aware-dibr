"""Is the warp wrong, or are those black shapes simply the disocclusions?

1) ramp test: warp an image whose value encodes x, so the output directly reads back the
   source coordinate that landed on each target pixel -> exact per-pixel displacement.
2) zoomed crops of the reference / warped / hole mask around the foreground dancer.
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补")
from dibr import io_utils, viz, warp as W  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.pipeline import disparity_from_depth, depth_to_scale255  # noqa: E402

OUT = r"D:\项目\空洞填补\_work\warpcheck"
os.makedirs(OUT, exist_ok=True)
rgb, inv = vio.load_pair(r"D:\项目\空洞填补\input\color-cam6-f000.jpg",
                         r"D:\项目\空洞填补\input\depth-cam6-f000.png")
cfg = FillConfig(scale=-44.8)
P = depth_to_scale255(inv)
dx = disparity_from_depth(P, cfg.scale)
print(f"depth  : P = 255*inv, range [{P.min():.1f}, {P.max():.1f}]")
print(f"disp   : dx = P/255*({cfg.scale}), range [{dx.min():.2f}, {dx.max():.2f}] px")

# ---------------------------------------------------------------- 1) ramp test
h, w = P.shape
ramp = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
for splat in ("sub", "floor"):
    for rule in ("zbuf", "avg"):
        out, zz, hole, wgt = W.forward_warp(ramp, dx, None, z=P, hole_depth=-1.0,
                                            rule=rule, splat=splat)
        valid = ~hole
        src = out[valid]                       # source x that landed on each target
        ry, tgt = np.nonzero(valid)
        tgt = tgt.astype(np.float32)
        dyv = tgt - src                        # measured displacement
        rr = np.clip(ry, 0, h - 1)
        cc = np.clip(np.rint(src).astype(int), 0, w - 1)
        exp = dx[rr, cc]
        err = dyv - exp
        print(f"  splat={splat:5s} rule={rule:4s}: holes {hole.mean()*100:5.2f}%  "
              f"measured dx range [{dyv.min():+7.2f}, {dyv.max():+7.2f}]  "
              f"median err {np.median(np.abs(err)):5.2f} px  p95 {np.percentile(np.abs(err),95):5.2f}")

# ---------------------------------------------------------------- 2) crops
I_w, D_w, hole, wgt = W.forward_warp(rgb.astype(np.float32), dx, None, z=P,
                                     hole_depth=-1.0, rule="zbuf", splat="sub")


def crop(img, x0, x1, y0, y1):
    return img[y0:y1, x0:x1].copy()


def stack(tiles, scale=2):
    import cv2
    out = []
    for t in tiles:
        t = np.clip(t, 0, 255).astype(np.uint8)
        out.append(cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
    return np.hstack(out)


regions = {"man": (620, 940, 40, 700), "woman": (140, 470, 100, 720)}
for name, (x0, x1, y0, y1) in regions.items():
    ref = crop(rgb, x0, x1, y0, y1)
    wrp = crop(I_w, x0, x1, y0, y1)
    msk = crop(np.repeat((hole * 255).astype(np.uint8)[:, :, None], 3, 2), x0, x1, y0, y1)
    ovl = crop(viz.overlay_mask(I_w, hole, (255, 0, 0)), x0, x1, y0, y1)
    io_utils.imwrite(os.path.join(OUT, f"crop_{name}.png"), stack([ref, wrp, ovl, msk]))
    print(f"  crop_{name}.png  (reference | warped | warped+holes red | hole mask, "
          f"nearest 2x, x {x0}-{x1}, y {y0}-{y1})")

# how much of each dancer is a hole vs lost?
for label, sel in (("woman", (slice(100, 720), slice(140, 470))),
                   ("man", (slice(40, 700), slice(620, 940)))):
    m = hole[sel]
    print(f"  {label}: holes {m.mean()*100:5.2f}% of the crop, "
          f"{int(m.sum())} px")
