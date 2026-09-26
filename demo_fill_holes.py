"""One-call interface demo:  fill_holes(image, inv_depth) -> repaired image.

    python demo_fill_holes.py

NOTE on paths: OpenCV's imread/imwrite go through a narrow-char path on Windows and fail
for directories such as D:\\项目\\... .  Use viewfill.io (or np.fromfile + cv2.imdecode),
which is what this demo does.
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from viewfill import fill_holes, io as vio  # noqa: E402

OUT = os.path.join(HERE, "output", "demo_fill_holes")


def main():
    os.makedirs(OUT, exist_ok=True)
    # encoding-safe load; the depth comes back normalised to 0..1 (near = large)
    img, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                             os.path.join(HERE, "input", "depth-cam6-f000.png"))
    print(f"image {img.shape} {img.dtype} (RGB)")
    print(f"inverse depth {inv.shape} range [{inv.min():.3f}, {inv.max():.3f}]")

    for scale in (-44.8, -20.0, -80.0):
        t = time.time()
        res = fill_holes(img, inv, scale=scale, return_info=True)   # one call
        fixed = res["image"]
        s = res["stats"]
        print(f"  scale {scale:+6.1f}: {s['holes_before']:6d} hole px -> "
              f"{s['holes_after']} left  |  filled-region content std "
              f"{fixed[res['hole_mask']].std():5.1f}  |  {time.time()-t:.1f}s")
        vio.save_image(os.path.join(OUT, f"scale{int(scale)}_warped.png"), res["warped"])
        vio.save_image(os.path.join(OUT, f"scale{int(scale)}_filled.png"), fixed)
        grid = np.hstack([img, res["warped"], fixed])
        vio.save_image(os.path.join(OUT, f"scale{int(scale)}_compare.png"),
                       cv2.resize(grid, None, fx=0.62, fy=0.62))

    # 0..255 depth maps and grayscale images are both accepted
    dep8 = (inv * 255.0).astype(np.uint8)
    fixed8 = fill_holes(img, dep8)
    print(f"  0..255 depth map       : ok, output {fixed8.shape} {fixed8.dtype}")
    gray = np.asarray(vio.load_image(os.path.join(HERE, "input", "color-cam6-f000.jpg"))
                      .astype(np.float32) @ np.array([0.299, 0.587, 0.114]))
    fixedg = fill_holes(gray.astype(np.uint8), inv)
    print(f"  grayscale input image  : ok, output {fixedg.shape} {fixedg.dtype}")
    vio.save_image(os.path.join(OUT, "final_filled.png"),
                   fill_holes(img, inv, scale=-44.8))
    print(f"\nartefacts under {OUT}")


if __name__ == "__main__":
    main()

