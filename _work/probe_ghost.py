"""Diagnose the fix_mode='bg' -inf and check where the ghost writes land."""
import numpy as np
import sys

sys.path.insert(0, r"D:\项目\空洞填补")
from dibr import calib, ghosts, io_utils  # noqa: E402

ROOT = io_utils.DATASET_ROOT_DEFAULT
SRC = r"D:\项目\空洞填补\output\step2_cracks_int\cam6_to_cam7_f000\lam5_h_none"
I_w = np.load(SRC + r"\I_filled.npy").astype(np.float32)
D_w = np.load(SRC + r"\D_filled.npy").astype(np.float32)
hole = np.load(SRC + r"\remaining_holes.npy")
cams = calib.load_calib(ROOT)
P = io_utils.load_view(ROOT, 6, "f000")["depth"].astype(np.float32)
dx, dy, _, _, _ = calib.displacement_field(cams, 6, 7, P)
DFG = ghosts.extended_fg(P)
fx, fy, _, _, _ = calib.displacement_field(cams, 6, 7, DFG)

gray = (I_w[..., 0] * .299 + I_w[..., 1] * .587 + I_w[..., 2] * .114).astype(np.float32)
cand, G = ghosts.candidate_band(hole, 2, ghosts.find_oofa(hole, -1))
valid = ~(hole | cand)
med = ghosts.masked_median(gray, valid, 9)
print("med_field finite:", bool(np.isfinite(med).all()),
      " min", np.nanmin(med), " max", np.nanmax(med),
      " inf count", int((~np.isfinite(med)).sum()))
# where would inf come from: windows with no valid pixel at all
import cv2
cnt = cv2.boxFilter(valid.astype(np.float32), -1, (9, 9), normalize=False,
                    borderType=cv2.BORDER_CONSTANT)
print("windows with 0 valid:", int((cnt == 0).sum()), " with 1 valid:",
      int((cnt == 1).sum()), " min cnt", cnt.min())

for mode in ("copy", "move", "bg"):
    r = ghosts.detect_and_fix(I_w, D_w, hole, (dx, dy), (fx, fy), direction=-1,
                              fix_mode=mode)
    I = r["I_fixed"]
    print(f"mode={mode}: I finite {bool(np.isfinite(I).all())}  "
          f"min {np.nanmin(I):.2f} max {np.nanmax(I):.2f}  "
          f"ghost {int(r['ghost'].sum())}  changed "
          f"{int((np.abs(I - I_w).max(axis=2) > 1e-3).sum())}")
    # where do the writes land?
    wys, wxs = r["write_yx"]
    wtx, wty = r["write_xy"]
    if len(wys):
        print(f"   destinations: on holes {int(hole[wty, wtx].sum())}, "
              f"on valid {int((~hole[wty, wtx]).sum())}, "
              f"offset med {np.median(np.hypot(wtx - wxs, wty - wys)):.1f}px, "
              f"dist>8px {int((np.hypot(wtx - wxs, wty - wys) > 8).sum())}")
