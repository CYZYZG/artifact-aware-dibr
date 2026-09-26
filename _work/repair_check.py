"""Reproduce the broken warp (provided scatter_image, inverse_ordering=True) and show
that viewfill detects and repairs it."""
import sys

import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补")
sys.path.insert(0, r"D:\项目\空洞填补\_work")
from warping import scatter_image  # noqa: E402

from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.pipeline import fill_warped  # noqa: E402

rgb, inv = vio.load_pair(r"D:\项目\空洞填补\input\color-cam6-f000.jpg",
                         r"D:\项目\空洞填补\input\depth-cam6-f000.png")

for ordering in (True, False):
    w, hm, dep = scatter_image(rgb.astype(np.float32), inv, direction=-1,
                               scale_factor=44.8, inverse_ordering=ordering,
                               reproject_depth=True)
    hole = hm > 0
    dw = np.where(hole, 0.0, 255.0 / (dep + 1e-6)).astype(np.float32)
    for mode in ("never", "auto"):
        cfg = FillConfig(scale=-44.8, repair_warp=mode)
        r = fill_warped(np.clip(w, 0, 255).astype(np.float32), hole, dw, rgb, inv,
                        cfg=cfg)
        s = r["stats"]
        print(f"provided warp inverse_ordering={ordering!s:5s} repair_warp={mode:5s}: "
              f"deviation {s['warp_deviation_pct']:5.2f}% of valid px "
              f"(fg {s['fg_deviation_pct']:5.2f}%)  repaired={s['warp_repaired']!s:5s}  "
              f"residual {int(r['remaining'].sum())}  "
              f"back-proj {s.get('back_proj_psnr_before', float('nan')):.2f} -> "
              f"{s.get('back_proj_psnr_after', float('nan')):.2f} dB")
