"""Which collision rule is actually better?  Measured against the real cam7 image."""
import os
import sys

import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import io_utils, viz  # noqa: E402
from warping import scatter_image  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.compat import scatter_image_safe  # noqa: E402
from viewfill.pipeline import fill_warped  # noqa: E402

rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, "f000")["color"]

variants = []
for ordering in (True, False):
    w, m, d = scatter_image(rgb.astype(np.float32), inv, direction=-1,
                            scale_factor=44.8, inverse_ordering=ordering,
                            reproject_depth=True)
    variants.append((f"scatter inverse_ordering={ordering}", np.asarray(w, np.float32),
                     m > 0, d))
w, m, d = scatter_image_safe(rgb, inv, direction=-1, scale_factor=44.8,
                             inverse_ordering=True, reproject_depth=True)
variants.append(("viewfill Z-buffer (compat)", np.asarray(w, np.float32), m > 0, d))

print(f"{'warp':34s} {'hole px':>8s} | {'warped PSNR':>11s} {'warped SSIM':>11s} | "
      f"{'filled PSNR':>11s} {'filled SSIM':>11s}")
for name, w, hole, d in variants:
    dw = np.where(hole, 0.0, 255.0 / (d + 1e-6)).astype(np.float32)
    r = fill_warped(np.clip(w, 0, 255).astype(np.float32), hole, dw, rgb, inv,
                    cfg=FillConfig(scale=-44.8, repair_warp="never"))
    m = ~hole
    print(f"{name:34s} {int(hole.sum()):8d} | "
          f"{viz.psnr(gt, w, mask=m):11.2f} {viz.ssim(gt, w, mask=m):11.4f} | "
          f"{viz.psnr(gt, r['I_filled'], mask=m):11.2f} "
          f"{viz.ssim(gt, r['I_filled'], mask=m):11.4f}")

print("\n(1D-shift stress test: the Ballet rig is toed-in, so the absolute numbers are "
      "below the calibrated-warp case; the comparison across collision rules is fair "
      "because the geometry is identical.)")
