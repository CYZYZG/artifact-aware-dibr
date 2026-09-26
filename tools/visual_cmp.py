"""Visual comparison 3 vs auto7 on the calibrated geometry.

Per frame, one row:
  [ reference crop | warped + holes(red) | filled depth_dilate=3 | filled auto7 | |3-auto7|x4 ]
Frames chosen from the paired data: f004 (auto7 much better), f006 (3 much better), f000 (base).
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

OUT = os.path.join(HERE, "output", "cmp_dil")
os.makedirs(OUT, exist_ok=True)
cams = calib.load_calib(io_utils.DATASET_ROOT_DEFAULT)
X0, X1, Y0, Y1 = 600, 900, 40, 720
Z = 1.9


def run(src, dst, frame, mode):
    view = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, src, frame)
    rgb = view["color"]
    P = np.asarray(view["depth"], np.float32).copy()
    cfg = FillConfig(scale=-44.8, depth_dilate=mode, repair_warp="never")
    Pd = _prep_depth(P, cfg)
    fdx, fdy, *_ = calib.displacement_field(cams, src, dst, Pd, "bottom")
    fdx = np.asarray(fdx, np.float32)
    fdy = None if fdy is None else np.asarray(fdy, np.float32)
    I_w, D_w, hole, _ = W.forward_warp(np.asarray(rgb, np.float32), fdx, fdy, z=Pd,
                                       hole_depth=-1.0, rule="zbuf", splat="sub")
    res = fill_warped(I_w, hole, D_w, rgb, P, disp=(fdx, fdy), cfg=cfg)
    return rgb, I_w, hole, res


sepr = np.full((Y1 - Y0, 4, 3), 255, np.uint8)
for fr in ("f004", "f006", "f000"):
    rgb, I_w, hole, r3 = run(6, 7, fr, 3)
    _, _, _, r7 = run(6, 7, fr, "auto7")
    ovl = np.clip(I_w, 0, 255).astype(np.uint8).copy()
    m = hole[Y0:Y1, X0:X1]
    o = ovl[Y0:Y1, X0:X1]
    o[m] = (0.35 * o[m] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
    a = np.clip(r3["I_filled"], 0, 255).astype(np.uint8)[Y0:Y1, X0:X1]
    b = np.clip(r7["I_filled"], 0, 255).astype(np.uint8)[Y0:Y1, X0:X1]
    d = np.clip(np.abs(r3["I_filled"] - r7["I_filled"]).max(axis=2) * 4, 0, 255)
    d = cv2.applyColorMap(d.astype(np.uint8)[Y0:Y1, X0:X1],
                          cv2.COLORMAP_INFERNO)[:, :, ::-1]
    row = np.hstack([np.clip(rgb, 0, 255).astype(np.uint8)[Y0:Y1, X0:X1], sepr, o,
                     sepr, a, sepr, b, sepr, d])
    io_utils.imwrite(os.path.join(OUT, f"{fr}_3_vs_auto7.png"),
                     cv2.resize(row, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST))
    print(f"{fr}: seam 3={r3['stats'].get('seam', float('nan')):.2f} "
          f"auto7={r7['stats'].get('seam', float('nan')):.2f} | "
          f"mean |3-auto7| = {np.abs(r3['I_filled'] - r7['I_filled']).mean():.2f}, "
          f"px differing >8 = {int((np.abs(r3['I_filled'] - r7['I_filled']).max(axis=2) > 8).sum())}",
          flush=True)
print(f"images under {OUT}")
