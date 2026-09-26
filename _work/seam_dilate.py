"""Do the cracks (and the dark ghost line) disappear if the disparity is pre-dilated?

Cracks appear where the disparity changes fast (a gradual depth ramp at a silhouette maps
to a step of >1 px between adjacent source columns -> 1-2 px gaps).  Dilating the disparity
by 1 px widens the splat footprint, closing those gaps at the cost of fattening the
foreground slightly.
"""
import os
import sys

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import io_utils, viz  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.pipeline import warp_and_fill  # noqa: E402

rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, "f000")["color"]
gray = lambda a: 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
K = 9


def local(img, mask):
    m = mask.astype(np.float32)
    num = cv2.boxFilter(img * m, -1, (K, K), normalize=False,
                        borderType=cv2.BORDER_REPLICATE)
    den = cv2.boxFilter(m, -1, (K, K), normalize=False, borderType=cv2.BORDER_REPLICATE)
    return num / np.maximum(den, 1e-6)


print(f"{'depth pre-processing':>26s} | {'holes':>6s} {'cracks':>7s} | {'seam mean':>9s} "
      f"{'seam p90':>8s} | {'GT PSNR':>8s} {'GT SSIM':>8s} {'filled std':>10s}")
for label, d in (("raw", inv),
                 ("dilate 3x3 (max)", cv2.dilate(inv, np.ones((3, 3), np.uint8))),
                 ("dilate 5x5 (max)", cv2.dilate(inv, np.ones((5, 5), np.uint8)))):
    r = warp_and_fill(rgb, d, FillConfig(scale=-44.8))
    I_f, I_w, hole, crack = r["I_filled"], r["I_w"], r["hole_mask"], r["crack"]
    valid = ~hole
    j = np.abs(local(gray(I_f), hole)[hole & cv2.dilate(
        valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)]
        - local(gray(I_w), valid)[hole & cv2.dilate(
            valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)])
    m = ~r["remaining"]
    print(f"{label:>26s} | {int(hole.sum()):>6d} {int(crack.sum()):>7d} | "
          f"{j.mean():>9.2f} {np.percentile(j, 90):>8.2f} | "
          f"{viz.psnr(gt, I_f, mask=m):>8.2f} {viz.ssim(gt, I_f, mask=m):>8.4f} "
          f"{gray(I_f)[hole].std():>10.2f}")
    if label.startswith("dilate 3"):
        io_utils.imwrite(os.path.join(HERE, "_work", "seam", "dilated3_filled.png"),
                         I_f.astype(np.uint8)[:, :, ::-1])
