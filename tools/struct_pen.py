"""Structure-aware cross-row penalty: does it keep the good matches and stop the rail shift?

Sweep struct_pen and report the same decomposition as before
(all / filled-band / barre-row PSNR+SSIM) plus the patch row-offset statistics.
"""
import os
import sys

import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import calib, inpaint, io_utils, viz, warp as W  # noqa: E402
from viewfill import FillConfig  # noqa: E402
from viewfill.pipeline import _prep_depth, fill_warped  # noqa: E402

cams = calib.load_calib(io_utils.DATASET_ROOT_DEFAULT)


def run(fr, pen):
    view = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 6, fr)
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, fr)["color"]
    rgb = view["color"]
    P = np.asarray(view["depth"], np.float32).copy()
    cfg = FillConfig(scale=-44.8, struct_pen=pen, repair_warp="never")
    Pd = _prep_depth(P, cfg)
    fdx, fdy, *_ = calib.displacement_field(cams, 6, 7, Pd, "bottom")
    fdx = np.asarray(fdx, np.float32)
    fdy = None if fdy is None else np.asarray(fdy, np.float32)
    I_w, D_w, hole, _ = W.forward_warp(np.asarray(rgb, np.float32), fdx, fdy, z=Pd,
                                       hole_depth=-1.0, rule="zbuf", splat="sub")
    inpaint.SEARCH_DIAG.pop("dy", None)
    res = fill_warped(I_w, hole, D_w, rgb, P, disp=(fdx, fdy), cfg=cfg)
    dy = np.abs(np.array(inpaint.SEARCH_DIAG.get("dy", [0]), float))
    I_f = np.asarray(res["I_filled"], np.float32)
    g = np.asarray(gt, np.float32)
    gray = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
    rail = int(np.median(np.argsort(-np.abs(np.diff(gray, axis=0)).mean(axis=1))[:60]))
    rows = np.zeros_like(hole)
    rows[max(0, rail - 15):rail + 16] = True
    valid = ~res["remaining"]
    return dict(all_p=viz.psnr(gt, I_f, mask=valid), all_s=viz.ssim(gt, I_f, mask=valid),
                band_p=viz.psnr(gt, I_f, mask=hole), band_s=viz.ssim(gt, I_f, mask=hole),
                rail_p=viz.psnr(gt, I_f, mask=hole & rows),
                rail_s=viz.ssim(gt, I_f, mask=hole & rows),
                offrow=float((dy != 0).mean() * 100),
                p90=float(np.percentile(dy, 90)), mx=float(dy.max()))


PENS = (0.0, 0.5, 2.0, 8.0)
print(f"{'frame':>6s} {'struct_pen':>10s} | {'all PSNR':>8s} {'allSSIM':>8s} | "
      f"{'band PSNR':>9s} {'bandSSIM':>9s} | {'rail PSNR':>9s} {'railSSIM':>9s} | "
      f"{'off-row':>7s} {'p90':>5s} {'max':>5s}")
agg = {}
for fr in ("f000", "f004"):
    for pen in PENS:
        r = run(fr, pen)
        agg.setdefault(pen, []).append(r)
        print(f"{fr:>6s} {pen:>10.1f} | {r['all_p']:>8.2f} {r['all_s']:>8.4f} | "
              f"{r['band_p']:>9.2f} {r['band_s']:>9.4f} | {r['rail_p']:>9.2f} "
              f"{r['rail_s']:>9.4f} | {r['offrow']:>6.1f}% {r['p90']:>5.1f} "
              f"{r['mx']:>5.0f}", flush=True)
print()
for pen, rows in agg.items():
    a = {k: np.mean([x[k] for x in rows]) for k in rows[0]}
    print(f"mean pen={pen:>4.1f}: all {a['all_p']:.2f}/{a['all_s']:.4f}  "
          f"band {a['band_p']:.2f}/{a['band_s']:.4f}  rail {a['rail_p']:.2f}/"
          f"{a['rail_s']:.4f}  off-row {a['offrow']:.1f}%  p90 {a['p90']:.1f}  "
          f"max {a['mx']:.0f}")
