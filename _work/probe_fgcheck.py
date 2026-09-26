"""Diagnose the fg-removal check failure."""
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

r = ghosts.detect_and_fix(I_w, D_w, hole, (dx, dy), (fx, fy), direction=-1, fix_mode="copy")
fg, cand_post = r["fg_points"], r["candidates"]
cand_union = cand_post | fg
Tmap = r["T"][np.clip(r["lab_near"], 0, len(r["T"]) - 1)]

print("fg px", int(fg.sum()), "cand_post", int(cand_post.sum()),
      "union", int(cand_union.sum()), "hole px", int(hole.sum()))
print("fg & hole:", int((fg & hole).sum()))
print("cand_union & hole:", int((cand_union & hole).sum()))
print("D_w finite at fg:", bool(np.isfinite(D_w[fg]).all()),
      " Tmap finite at fg:", bool(np.isfinite(Tmap[fg]).all()))
d = D_w[fg] - Tmap[fg]
print("D_w - Tmap at fg: min %.4f max %.4f  #<=0: %d  #nan: %d"
      % (np.nanmin(d), np.nanmax(d), int((d <= 0).sum()), int(np.isnan(d).sum())))
mask_ok = (D_w[fg] > Tmap[fg] - 1e-6)
print("comparison all True:", bool(mask_ok.all()), " #False:", int((~mask_ok).sum()))
if (~mask_ok).any():
    bad = np.nonzero(fg)
    bad = (bad[0][~mask_ok], bad[1][~mask_ok])
    print("  bad examples:", list(zip(bad[0][:5], bad[1][:5], D_w[bad][:5], Tmap[bad][:5])))
print("A:", bool((fg & cand_post).all()), " B:", bool(mask_ok.all()),
      " A and B:", bool((fg & cand_post).all() and mask_ok.all()))
