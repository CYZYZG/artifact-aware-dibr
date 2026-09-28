"""Random samples from the dataset -> hole filling -> contact sheet + stats table.

Picks N random (camera, frame) pairs (fixed seed, reproducible), runs the public interface
`fill_holes(rgb, inv, scale=...)` with the default configuration, and writes
output/random_demo/sheet.png :  per row  [ original | warp + holes(red) | filled | zoom ]
"""
import os
import random
import sys
import time

import cv2
import numpy as np

ROOT = r"D:\项目\空洞填补"
sys.path.insert(0, ROOT)
from dibr import io_utils, viz                                    # noqa: E402
from viewfill import fill_holes, io as vio                        # noqa: E402

OUT = os.path.join(ROOT, "output", "random_demo")
os.makedirs(OUT, exist_ok=True)
SEED = 20260928
N = 6
SCALE = -44.8
ROOT_DATA = io_utils.DATASET_ROOT_DEFAULT

# ---- which (camera, frame) pairs actually exist
avail = []
for cam in range(8):
    d = os.path.join(ROOT_DATA, "cam%d" % cam)
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        if fn.startswith("color-cam%d-" % cam) and fn.endswith(".jpg"):
            avail.append((cam, fn.split("-")[-1].split(".")[0]))
print("available (cam, frame) pairs:", len(avail))

picks = random.Random(SEED).sample(avail, N)
print("random picks (seed %d): %s" % (SEED, picks))

rows, stats = [], []
for cam, fr in picks:
    view = io_utils.load_view(ROOT_DATA, cam, fr)
    rgb = view["color"]
    inv = np.asarray(view["depth"], np.float32) / 255.0
    t = time.time()
    res = fill_holes(rgb, inv, scale=SCALE, return_info=True)
    dt = time.time() - t
    s = res["stats"]
    filled = res["filled"]
    hole = res["hole_mask"]
    ys, xs = np.nonzero(hole)
    if ys.size:                                    # zoom on the densest hole area
        y0 = int(np.clip(np.median(ys) - 90, 0, rgb.shape[0] - 181))
        x0 = int(np.clip(np.median(xs) - 120, 0, rgb.shape[1] - 241))
    else:
        y0 = x0 = 0
    z = (slice(y0, y0 + 180), slice(x0, x0 + 240))
    ovl = np.clip(res["I_w"], 0, 255).astype(np.uint8).copy()
    o = ovl[z]
    m = hole[z]
    o[m] = (0.3 * o[m] + 0.7 * np.array([255, 0, 0])).astype(np.uint8)
    pan = [np.clip(rgb, 0, 255).astype(np.uint8)[z], ovl[z], filled[z]]
    pan.append(np.clip(res["I_w"], 0, 255).astype(np.uint8)[z])   # 4th: warp without marks
    row = pan[0]
    for t2 in pan[1:]:
        row = np.hstack([row, np.full((row.shape[0], 3, 3), 255, np.uint8), t2])
    rows.append(cv2.resize(row, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_NEAREST))

    stats.append(dict(cam=cam, frame=fr, holes=s["holes_before"],
                      hole_pct=hole.mean() * 100, residual=int(res["remaining"].sum()),
                      iters=s["iterations"], big=int(s.get("crack_big_hole_px", 0)),
                      unsup=int(s.get("hhf_unsupported_px", 0)),
                      bp0=s.get("back_proj_psnr_before", float("nan")),
                      bp1=s.get("back_proj_psnr_after", float("nan")), sec=dt))
    print("  cam%d %s: holes %.2f%% -> residual %d | iters %d | big-hole rej %d | "
          "unsup %d | back-proj %.2f -> %.2f | %.1fs"
          % (cam, fr, hole.mean() * 100, int(res["remaining"].sum()), s["iterations"],
             s.get("crack_big_hole_px", 0), s.get("hhf_unsupported_px", 0),
             s.get("back_proj_psnr_before", float("nan")),
             s.get("back_proj_psnr_after", float("nan")), dt), flush=True)

sheet = rows[0]
for r in rows[1:]:
    w = np.full((6, sheet.shape[1], 3), 255, np.uint8)
    sheet = np.vstack([sheet, w, r])
vio.save_image(os.path.join(OUT, "sheet.png"), sheet)
print("sheet:", os.path.join(OUT, "sheet.png"), sheet.shape)
print("\n%-6s %-7s %8s %8s %8s %10s %8s %9s" %
      ("cam", "frame", "hole%", "resid", "iters", "big-hold", "unsup", "sec"))
for r in stats:
    print("%-6d %-7s %7.2f%% %8d %8d %10d %8d %9.1f" %
          (r["cam"], r["frame"], r["hole_pct"], r["residual"], r["iters"], r["big"],
           r["unsup"], r["sec"]))
