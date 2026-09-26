"""High zoom of the silhouette/filled boundary: reference | hhf | bg-side(auto)."""
import os
import sys

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import io_utils  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.pipeline import warp_and_fill  # noqa: E402

OUT = os.path.join(HERE, "_work", "ring")
os.makedirs(OUT, exist_ok=True)
rgb, inv = vio.load_pair(os.path.join(HERE, "input", "color-cam6-f000.jpg"),
                         os.path.join(HERE, "input", "depth-cam6-f000.png"))
outs = {}
for mode in ("hhf", "auto"):
    outs[mode] = warp_and_fill(rgb, inv, FillConfig(scale=-44.8, crack_fill=mode))

# head/back of the man: crack band + disocclusion band + untouched background
x0, x1, y0, y1 = 690, 840, 60, 340
sepr = np.full((y1 - y0, 3, 3), 255, np.uint8)
tiles = [np.clip(rgb, 0, 255).astype(np.uint8)[y0:y1, x0:x1],
         np.clip(outs["hhf"]["I_w"], 0, 255).astype(np.uint8)[y0:y1, x0:x1],
         outs["hhf"]["filled"][y0:y1, x0:x1],
         outs["auto"]["filled"][y0:y1, x0:x1]]
row = tiles[0]
for t in tiles[1:]:
    row = np.hstack([row, sepr, t])
row = cv2.resize(row, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
io_utils.imwrite(os.path.join(OUT, "zoom_seam.png"), row)
print(f"zoom_seam.png (x4) = reference | warped | filled(hhf) | filled(bg-side)  "
      f"x {x0}-{x1}, y {y0}-{y1}")

# luma trace across the silhouette, one row, to see the profile
yy = 200
for mode in ("hhf", "auto"):
    g = 0.299 * outs[mode]["filled"][..., 0] + 0.587 * outs[mode]["filled"][..., 1] \
        + 0.114 * outs[mode]["filled"][..., 2]
    print(f"  row {yy} [{mode:4s}]: " + " ".join(f"{int(v):3d}" for v in g[yy, x0:x0 + 30]))
gref = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
print(f"  row {yy} [ref ]: " + " ".join(f"{int(v):3d}" for v in gref[yy, x0:x0 + 30]))
