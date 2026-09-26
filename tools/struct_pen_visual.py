"""Visual: does struct_pen fix the rail, and what does it cost elsewhere?

Outputs, per frame:
  <fr>_rail.png  = truth | struct_pen=0 | 0.5 | 8   (3x zoom on the rail crossing the band)
  <fr>_full.png  = truth | struct_pen=0 | 0.5 | 8   (whole frame, downscaled)
"""
import os
import sys

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import calib, io_utils, warp as W  # noqa: E402
from viewfill import FillConfig  # noqa: E402
from viewfill.pipeline import _prep_depth, fill_warped  # noqa: E402

OUT = os.path.join(HERE, "output", "struct_pen")
os.makedirs(OUT, exist_ok=True)
cams = calib.load_calib(io_utils.DATASET_ROOT_DEFAULT)
PENS = (0.0, 0.5, 8.0)


def fill(fr, pen):
    view = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 6, fr)
    rgb = view["color"]
    P = np.asarray(view["depth"], np.float32).copy()
    cfg = FillConfig(scale=-44.8, struct_pen=pen, repair_warp="never")
    Pd = _prep_depth(P, cfg)
    fdx, fdy, *_ = calib.displacement_field(cams, 6, 7, Pd, "bottom")
    fdx = np.asarray(fdx, np.float32)
    fdy = None if fdy is None else np.asarray(fdy, np.float32)
    I_w, D_w, hole, _ = W.forward_warp(np.asarray(rgb, np.float32), fdx, fdy, z=Pd,
                                       hole_depth=-1.0, rule="zbuf", splat="sub")
    res = fill_warped(I_w, hole, D_w, rgb, P, disp=(fdx, fdy), cfg=cfg)
    return np.clip(res["I_filled"], 0, 255).astype(np.uint8), hole


for fr in ("f004", "f000"):
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, fr)["color"]
    gray = 0.299 * gt[..., 0] + 0.587 * gt[..., 1] + 0.114 * gt[..., 2]
    rail = int(np.median(np.argsort(-np.abs(np.diff(gray, axis=0)).mean(axis=1))[:60]))
    tiles, full = [], [gt.astype(np.uint8)]
    for pen in PENS:
        I_f, hole = fill(fr, pen)
        full.append(I_f)
        tiles.append(I_f)
    y0, y1, x0, x1 = max(0, rail - 70), min(gt.shape[0], rail + 70), 620, 900
    sepr = np.full((y1 - y0, 3, 3), 255, np.uint8)
    sepf = np.full((gt.shape[0], 3, 3), 255, np.uint8)
    row = gt[y0:y1, x0:x1].astype(np.uint8)
    for t in tiles:
        row = np.hstack([row, sepr, t[y0:y1, x0:x1]])
    io_utils.imwrite(os.path.join(OUT, f"{fr}_rail.png"),
                     cv2.resize(row, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST))
    frow = full[0]
    for t in full[1:]:
        frow = np.hstack([frow, sepf, t])
    io_utils.imwrite(os.path.join(OUT, f"{fr}_full.png"),
                     cv2.resize(frow, None, fx=0.5, fy=0.5))
    print(f"{fr}: rail row {rail}; {fr}_rail.png (truth|pen0|pen0.5|pen8, 3x), "
          f"{fr}_full.png", flush=True)
