"""depth_dilate = 3 (uniform) vs auto5 / auto7 (demand-driven) over several frames.

Per frame: seam mean/p90, residual cracks, GT PSNR/SSIM against the real cam7.
"""
import glob
import os
import re
import sys

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import io_utils, viz  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.pipeline import warp_and_fill  # noqa: E402

files = sorted(glob.glob(os.path.join(HERE, "input", "color-cam6-f*.jpg")))
frames = [re.search(r"(f\d+)", os.path.basename(f)).group(1) for f in files]
pick = frames[:: max(1, len(frames) // 6)][:6]
print(f"input frames available: {len(frames)}; testing {pick}\n")
K = 9


def seam(I_f, I_w, hole):
    g_f = 0.299 * I_f[..., 0] + 0.587 * I_f[..., 1] + 0.114 * I_f[..., 2]
    g_w = 0.299 * I_w[..., 0] + 0.587 * I_w[..., 1] + 0.114 * I_w[..., 2]
    valid = ~hole

    def loc(img, m):
        mm = m.astype(np.float32)
        num = cv2.boxFilter(img * mm, -1, (K, K), normalize=False,
                            borderType=cv2.BORDER_REPLICATE)
        den = cv2.boxFilter(mm, -1, (K, K), normalize=False,
                            borderType=cv2.BORDER_REPLICATE)
        return num / np.maximum(den, 1e-6)

    bnd = hole & cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    j = np.abs(loc(g_f, hole)[bnd] - loc(g_w, valid)[bnd])
    return j.mean(), np.percentile(j, 90)


modes = (2, 3, "auto5", "auto7")
acc = {m: [] for m in modes}
for fr in pick:
    rgb, inv = vio.load_pair(os.path.join(HERE, "input", f"color-cam6-{fr}.jpg"),
                             os.path.join(HERE, "input", f"depth-cam6-{fr}.png"))
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, fr)["color"]
    line = [f"{fr} "]
    for m in modes:
        r = warp_and_fill(rgb, inv, FillConfig(scale=-44.8, depth_dilate=m))
        I_f, I_w, hole = r["I_filled"], r["I_w"], r["hole_mask"]
        s, p90 = seam(I_f, I_w, hole)
        msk = ~r["remaining"]
        ps, ss = viz.psnr(gt, I_f, mask=msk), viz.ssim(gt, I_f, mask=msk)
        acc[m].append((s, p90, int(r["crack"].sum()), ps, ss))
        line.append(f"| {str(m):>5s} {s:5.2f}/{p90:5.2f} {int(r['crack'].sum()):5d} "
                    f"{ps:5.2f}/{ss:.4f} ")
    print("".join(line))

print(f"\n{'mode':>6s} | {'seam':>6s} {'p90':>6s} {'cracks':>7s} | {'GT PSNR':>8s} "
      f"{'GT SSIM':>8s} | wins(seam/p90/GT)")
for m in modes:
    a = np.array(acc[m], np.float64)
    wins = 0
    for i in range(len(pick)):
        best = min(acc[mm][i][0] for mm in modes)
        if a[i, 0] <= best + 1e-9:
            wins += 1
    print(f"{str(m):>6s} | {a[:,0].mean():>6.2f} {a[:,1].mean():>6.2f} "
          f"{a[:,2].mean():>7.0f} | {a[:,3].mean():>8.2f} {a[:,4].mean():>8.4f} | {wins}")
