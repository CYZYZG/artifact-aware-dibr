"""Epipolar tolerance trade-off: epipolar = None / 0 / 1 / 2.

off-row  : share of patch choices whose source row differs from the target row
edge_ex  : geometric structure metric = mean |d/dy I_filled| inside the filled band minus
           mean |d/dy GT| over the same pixels.  A horizontal structure (rail/barre) placed
           at the wrong height adds spurious horizontal edges, so lower is better; ~0 means
           the filled content has the same vertical edge energy as the truth.
"""
import os
import sys

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import calib, inpaint, io_utils, viz, warp as W  # noqa: E402
from viewfill import FillConfig  # noqa: E402
from viewfill.pipeline import _prep_depth, fill_warped  # noqa: E402

cams = calib.load_calib(io_utils.DATASET_ROOT_DEFAULT)


def run(fr, epi):
    view = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 6, fr)
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, fr)["color"]
    rgb = view["color"]
    P = np.asarray(view["depth"], np.float32).copy()
    cfg = FillConfig(scale=-44.8, epipolar=epi, repair_warp="never")
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
    gy_f = np.abs(np.diff(I_f, axis=0)).mean(axis=2)
    gy_g = np.abs(np.diff(g, axis=0)).mean(axis=2)
    band = hole[:-1] & hole[1:]
    edge_ex = float(gy_f[band].mean() - gy_g[band].mean())
    m = ~res["remaining"]
    return dict(offrow=float((dy != 0).mean() * 100), p90=float(np.percentile(dy, 90)),
                gt=viz.psnr(gt, I_f, mask=m), ss=viz.ssim(gt, I_f, mask=m),
                edge=edge_ex, holes=int(hole.sum()))


print(f"{'frame':>6s} {'epipolar':>8s} | {'off-row':>8s} {'p90':>6s} | {'edge_ex':>8s} | "
      f"{'GT PSNR':>8s} {'GT SSIM':>8s}")
agg = {}
for fr in ("f000", "f004", "f006"):
    for epi in (None, 0, 1, 2):
        r = run(fr, epi)
        agg.setdefault(epi, []).append(r)
        print(f"{fr:>6s} {str(epi):>8s} | {r['offrow']:>7.1f}% {r['p90']:>6.1f} | "
              f"{r['edge']:>+8.3f} | {r['gt']:>8.2f} {r['ss']:>8.4f}", flush=True)
print()
for epi, rows in agg.items():
    a = {k: np.mean([x[k] for x in rows]) for k in rows[0]}
    print(f"mean epipolar={str(epi):>4s}: off-row {a['offrow']:5.1f}%  p90 {a['p90']:5.1f}  "
          f"edge_ex {a['edge']:+.3f}  GT {a['gt']:.2f}/{a['ss']:.4f}")
