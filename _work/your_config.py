"""Your exact warp configuration: direction=-1, scale_factor=44.8, inverse_ordering=True.

Measures the foreground damage, shows the three fixes and writes a comparison figure.
"""
import os
import sys

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import io_utils  # noqa: E402
from warping import scatter_image  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.compat import scatter_image_safe  # noqa: E402
from viewfill.pipeline import check_warp_quality, disparity_from_depth, fill_warped  # noqa: E402

OUT = os.path.join(HERE, "_work", "warpcheck")
os.makedirs(OUT, exist_ok=True)
rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
cfg = FillConfig(scale=-44.8, splat="sub", rule="zbuf")
disp = (disparity_from_depth(inv * 255.0, -44.8), None)
depth255 = inv * 255.0

rows = []
crops = {}


def evaluate(img, hole, label):
    q, _, _ = check_warp_quality(np.asarray(img, np.float32), hole, rgb, depth255, disp,
                                cfg)
    rows.append((label, int(hole.sum()), q["warp_deviation_pct"], q["fg_deviation_pct"]))
    return q


# 1) your current call
w1, m1, d1 = scatter_image(rgb.astype(np.float32), inv, direction=-1, scale_factor=44.8,
                           inverse_ordering=True, reproject_depth=True)
evaluate(w1, m1 > 0, "yours: inverse_ordering=True  (broken)")

# 2) same call, ordering forced to False
w2, m2, d2 = scatter_image(rgb.astype(np.float32), inv, direction=-1, scale_factor=44.8,
                           inverse_ordering=False, reproject_depth=True)
evaluate(w2, m2 > 0, "yours: inverse_ordering=False (fixed)")

# 3) drop-in replacement (same signature, Z-buffer)
w3, m3, d3 = scatter_image_safe(rgb, inv, direction=-1, scale_factor=44.8,
                                inverse_ordering=True, reproject_depth=True)
evaluate(w3, m3 > 0, "viewfill.compat.scatter_image_safe (Z-buffer)")

print(f"{'variant':44s} {'hole px':>9s} {'deviation %':>12s} {'fg dev %':>9s}")
for lbl, hp, dv, fg in rows:
    print(f"{lbl:44s} {hp:9d} {dv:12.2f} {fg:9.2f}")

# ---- fill your broken warp with auto repair, and compare with the fixed one
def fill(w, m, d, mode):
    hole = m > 0
    dw = np.where(hole, 0.0, 255.0 / (d + 1e-6)).astype(np.float32)
    return fill_warped(np.clip(w, 0, 255).astype(np.float32), hole, dw, rgb, inv,
                       cfg=FillConfig(scale=-44.8, repair_warp=mode))


r_broken = fill(w1, m1, d1, "never")
r_repaired = fill(w1, m1, d1, "auto")
r_fixed = fill(w2, m2, d2, "never")
x0, x1, y0, y1 = 600, 900, 20, 720
sepr = np.full((y1 - y0, 3, 3), 255, np.uint8)
tiles = [rgb[y0:y1, x0:x1],
         np.clip(w1, 0, 255).astype(np.uint8)[y0:y1, x0:x1],
         np.clip(w2, 0, 255).astype(np.uint8)[y0:y1, x0:x1],
         r_broken["filled"][y0:y1, x0:x1],
         r_repaired["filled"][y0:y1, x0:x1],
         r_fixed["filled"][y0:y1, x0:x1]]
row = tiles[0]
for t in tiles[1:]:
    row = np.hstack([row, sepr, t])
io_utils.imwrite(os.path.join(OUT, "your_config_fix.png"),
                 cv2.resize(row, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_NEAREST))
print("\nfill results on your warp:")
for name, r in (("fill(broken, repair=never)", r_broken),
                ("fill(broken, repair=auto) ", r_repaired),
                ("fill(fixed,  repair=never)", r_fixed)):
    s = r["stats"]
    print(f"  {name}: residual {int(r['remaining'].sum()):5d} px  "
          f"repaired={s['warp_repaired']!s:5s}  "
          f"back-proj {s.get('back_proj_psnr_before', float('nan')):.2f} -> "
          f"{s.get('back_proj_psnr_after', float('nan')):.2f} dB")
same = float(np.abs(r_repaired["I_filled"] - r_fixed["I_filled"]).max())
print(f"  repaired(broken) vs fixed: max pixel difference {same:g}")
print(f"\nyour_config_fix.png = reference | yours(True) | yours(False) | "
      f"filled(broken) | filled(repaired) | filled(fixed)")
