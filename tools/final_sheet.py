"""Random samples -> ONLY the final filled images, tiled into one sheet for eyeballing.

    python tools/final_sheet.py [n] [scale]
"""
import os
import random
import sys

import cv2
import numpy as np

ROOT = r"D:\项目\空洞填补"
sys.path.insert(0, ROOT)
from dibr import io_utils                                   # noqa: E402
from viewfill import fill_holes, io as vio                   # noqa: E402

OUT = os.path.join(ROOT, "output", "random_demo")
os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else -44.8
SEED = 20260928
COLS = 4
TILE = 0.52          # each filled frame is downscaled by this factor

root = io_utils.DATASET_ROOT_DEFAULT
avail = []
for cam in range(8):
    d = os.path.join(root, "cam%d" % cam)
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.startswith("color-cam%d-" % cam) and fn.endswith(".jpg"):
                avail.append((cam, fn.split("-")[-1].split(".")[0]))
picks = random.Random(SEED).sample(avail, N)
print("picks:", picks, "| scale", SCALE)

tiles = []
for cam, fr in picks:
    view = io_utils.load_view(root, cam, fr)
    rgb = view["color"]
    inv = np.asarray(view["depth"], np.float32) / 255.0
    res = fill_holes(rgb, inv, scale=SCALE, return_info=True)
    t = res["filled"]
    t = cv2.resize(t, None, fx=TILE, fy=TILE, interpolation=cv2.INTER_AREA)
    tiles.append(cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
    print("  cam%d %s: residual %d" % (cam, fr, int(res["remaining"].sum())), flush=True)

sep = 4
h, w = tiles[0].shape[:2]
blank = np.full((h, sep, 3), 255, np.uint8)
rows = []
for i in range(0, len(tiles), COLS):
    chunk = tiles[i:i + COLS]
    while len(chunk) < COLS:
        chunk.append(np.full_like(tiles[0], 255))
    row = chunk[0]
    for t in chunk[1:]:
        row = np.hstack([row, blank, t])
    rows.append(row)
    rows.append(np.full((sep, row.shape[1], 3), 255, np.uint8))
sheet = np.vstack(rows[:-1])
name = "final_scale%d.png" % abs(int(SCALE))
vio.save_image(os.path.join(OUT, name), cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
print("sheet:", os.path.join(OUT, name), sheet.shape)
