"""Tabulate all step4_inpaint runs (beta sweep + ablations) from their stats.json."""
import glob
import json
import os

import numpy as np

OUT = r"D:\项目\空洞填补\output\step4_inpaint\cam6_to_cam7_f000"
rows = []
for d in sorted(glob.glob(os.path.join(OUT, "*"))):
    p = os.path.join(d, "stats.json")
    if not os.path.isfile(p):
        continue
    s = json.load(open(p, encoding="utf-8"))
    if s.get("components_filled", 0) < 100:
        continue
    rows.append((os.path.basename(d), s))

hdr = ["run", "holes_after", "iters", "sec", "sz9", "sz3", "inHoles", "framePSNR",
       "frameSSIM"]
print(" | ".join(f"{h:>13s}" for h in hdr))
for name, s in rows:
    sc = lambda k: str(s.get(k, "-"))
    vals = [name, sc("holes_after"), sc("iterations"), sc("seconds_fill"),
            sc("patchsize_9"), sc("patchsize_3"),
            f"{s.get('gt_psnr_in_holes_after', float('nan')):.2f}",
            f"{s.get('gt_psnr_frame_after', float('nan')):.2f}",
            f"{s.get('gt_ssim_frame_after', float('nan')):.4f}"]
    print(" | ".join(f"{v:>13s}" for v in vals))

# progression table: warp -> +cracks -> +ghosts -> +fill, same metric (holes black)
print("\n=== pipeline progression (whole frame vs the real cam7, unfilled holes = black) ===")
from dibr import io_utils, viz  # noqa: E402
import sys
sys.path.insert(0, r"D:\项目\空洞填补")
gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, "f000")["color"]
s1 = r"D:\项目\空洞填补\output\step1_warp_int\cam6_to_cam7_f000"
s2 = r"D:\项目\空洞填补\output\step2_cracks_int\cam6_to_cam7_f000\lam5_h_none"
s3 = r"D:\项目\空洞填补\output\step3_ghosts\cam6_to_cam7_f000\band2_gt_a11_copy"
s4 = os.path.join(OUT, "none")
stages = [("1) integer warp", os.path.join(s1, "warped_color.npy")),
          ("2) + crack filling", os.path.join(s2, "I_filled.npy")),
          ("3) + ghost removal", os.path.join(s3, "I_fixed.npy")),
          ("4-7) + exemplar filling", os.path.join(s4, "final_color.npy"))]
for name, path in stages:
    if not os.path.isfile(path):
        print(f"  {name:26s} (missing {path})")
        continue
    img = np.load(path).astype(np.float32)
    print(f"  {name:26s} PSNR {viz.psnr(gt, img):6.2f} dB   SSIM {viz.ssim(gt, img):.4f}")
