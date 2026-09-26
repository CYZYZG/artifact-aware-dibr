"""Is the vertical misalignment gone?  Measure it on the rail directly.

A) per-row alignment: for each row of the filled band, the shift d that minimises the SSD
   against the truth -> |d| > 0 means the content in that row came from another height.
B) rail displacement: locate the strongest horizontal edge (the barre) and, for every
   column of the filled band, compare the y of that edge in the result with the y in the
   truth -> direct vertical displacement of a horizontal structure.
C) zoomed crops around the rail for visual inspection.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from dibr import calib, io_utils, warp as W  # noqa: E402
from viewfill import FillConfig  # noqa: E402
from viewfill.pipeline import _prep_depth, fill_warped  # noqa: E402

OUT = os.path.join(HERE, "output", "y_misalign")
os.makedirs(OUT, exist_ok=True)
cams = calib.load_calib(io_utils.DATASET_ROOT_DEFAULT)
RANGES = 10


def fill(fr, epi):
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
    res = fill_warped(I_w, hole, D_w, rgb, P, disp=(fdx, fdy), cfg=cfg)
    return rgb, np.asarray(res["I_filled"], np.float32), np.asarray(gt, np.float32), hole


def metrics(I_f, gt, hole):
    H = gt.shape[0]
    gray = lambda a: 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    f, g = gray(I_f), gray(gt)
    # ---- A) per-row best vertical shift
    shifts = []
    for y in range(H):
        xs = np.nonzero(hole[y])[0]
        if xs.size < 20:
            continue
        best, bd = 0, np.inf
        for d in range(-RANGES, RANGES + 1):
            y2 = min(max(y + d, 0), H - 1)
            v = float(((f[y, xs] - g[y2, xs]) ** 2).mean())
            if v < bd:
                bd, best = v, d
        shifts.append(best)
    sh = np.abs(np.array(shifts, float))
    # ---- B) rail edge displacement: strongest horizontal edge row of the truth
    gy = np.abs(np.diff(g, axis=0))
    rail_rows = np.argsort(-gy.mean(axis=1))[:60]
    rail = int(np.median(rail_rows))
    dy_col = []
    for x in range(g.shape[1]):
        ys = np.nonzero(hole[:, x])[0]
        if ys.size < 5 or not (rail - 14 <= ys.min() and ys.max() <= rail + 14):
            continue
        win = slice(max(0, rail - 12), min(H, rail + 13))
        a = np.abs(np.diff(f[:, x]))[win]
        b = np.abs(np.diff(g[:, x]))[win]
        dy_col.append(int(np.argmax(a) - np.argmax(b)))
    dc = np.abs(np.array(dy_col, float)) if dy_col else np.array([0.0])
    return sh, dc, rail


print(f"{'frame':>6s} {'epipolar':>8s} | {'row |d| mean':>12s} {'p90':>5s} {'>2px':>6s} | "
      f"{'rail |dy| mean':>14s} {'p90':>5s} {'>2px':>6s} | rail row")
for fr in ("f000", "f004"):
    for epi in (None, 0, 2):
        rgb, I_f, gt, hole = fill(fr, epi)
        sh, dc, rail = metrics(I_f, gt, hole)
        print(f"{fr:>6s} {str(epi):>8s} | {sh.mean():>12.2f} {np.percentile(sh,90):>5.1f} "
              f"{(sh>2).mean()*100:>5.1f}% | {dc.mean():>14.2f} "
              f"{np.percentile(dc,90):>5.1f} {(dc>2).mean()*100:>5.1f}% | {rail}", flush=True)
        if fr == "f004":
            y0, y1 = max(0, rail - 70), min(gt.shape[0], rail + 70)
            x0, x1 = 640, 900
            sepr = np.full((y1 - y0, 3, 3), 255, np.uint8)
            row = np.hstack([gt[y0:y1, x0:x1].astype(np.uint8), sepr,
                             I_f[y0:y1, x0:x1].astype(np.uint8)])
            io_utils.imwrite(os.path.join(OUT, f"rail_{fr}_epi{epi}.png"),
                             cv2.resize(row, None, fx=3.0, fy=3.0,
                                        interpolation=cv2.INTER_NEAREST))
print(f"crops under {OUT} (left = truth cam7, right = result)")
