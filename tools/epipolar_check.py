"""Why does a horizontal rail come out vertically displaced?  Measure it.

For a rectified pair (cam6 -> cam7) warped with the calibrated field:
  * where does the matcher take each patch from?  distribution of the chosen dy (=0 means
    the source patch came from the same row, i.e. the epipolar constraint holds)
  * vertical misalignment of the RESULT: find the global shift d that minimises
        SSD(d) = sum over the filled pixels of (I_filled(y,x) - GT(y+d,x))^2
    d = 0 means the filled content is vertically aligned with the truth.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from dibr import calib, inpaint, io_utils, viz, warp as W  # noqa: E402
from viewfill import FillConfig  # noqa: E402
from viewfill.pipeline import _prep_depth, fill_warped  # noqa: E402

cams = calib.load_calib(io_utils.DATASET_ROOT_DEFAULT)


def run(src, dst, frame, epi):
    view = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, src, frame)
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
    rgb = view["color"]
    P = np.asarray(view["depth"], np.float32).copy()
    cfg = FillConfig(scale=-44.8, epipolar=epi, repair_warp="never")
    Pd = _prep_depth(P, cfg)
    fdx, fdy, *_ = calib.displacement_field(cams, src, dst, Pd, "bottom")
    fdx = np.asarray(fdx, np.float32)
    fdy = None if fdy is None else np.asarray(fdy, np.float32)
    I_w, D_w, hole, _ = W.forward_warp(np.asarray(rgb, np.float32), fdx, fdy, z=Pd,
                                       hole_depth=-1.0, rule="zbuf", splat="sub")
    inpaint.SEARCH_DIAG.pop("dy", None)
    res = fill_warped(I_w, hole, D_w, rgb, P, disp=(fdx, fdy), cfg=cfg)
    dy = np.array(inpaint.SEARCH_DIAG.get("dy", [0]), float)
    I_f = np.asarray(res["I_filled"], np.float32)
    g = np.asarray(gt, np.float32)
    H = g.shape[0]
    ssd = {}
    for d in range(-6, 7):
        ys = np.clip(np.arange(H) + d, 0, H - 1)
        diff = (I_f[hole] - g[ys][hole]) ** 2
        ssd[d] = float(diff.mean())
    best = min(ssd, key=ssd.get)
    m = ~res["remaining"]
    return dict(dy_nonzero=float((dy != 0).mean() * 100),
                dy_abs_p90=float(np.percentile(np.abs(dy), 90)),
                dy_abs_max=float(np.abs(dy).max()),
                best_shift=int(best),
                gt_psnr=viz.psnr(gt, I_f, mask=m), gt_ssim=viz.ssim(gt, I_f, mask=m),
                ssd0=ssd[0])


print(f"{'frame':>6s} {'epipolar':>9s} | {'|dy|>0':>7s} {'dy p90':>7s} {'dy max':>6s} | "
      f"best y-shift | GT PSNR  GT SSIM")
for fr in ("f000", "f004", "f006"):
    for epi in (None, 0):
        r = run(6, 7, fr, epi)
        print(f"{fr:>6s} {str(epi):>9s} | {r['dy_nonzero']:>6.1f}% {r['dy_abs_p90']:>7.1f} "
              f"{r['dy_abs_max']:>6.1f} | {r['best_shift']:>+12d} | "
              f"{r['gt_psnr']:>8.2f} {r['gt_ssim']:>8.4f}", flush=True)
