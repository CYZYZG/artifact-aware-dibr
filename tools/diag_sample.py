"""Why is ONE sample much worse?  Per-stage diagnosis of the region in question.

Compares the bad sample (4th pick of the sheet) with a good one, restricted to the left
third of the frame (the left dancer), and dumps the stage masks so the cause is visible.
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
root = io_utils.DATASET_ROOT_DEFAULT
avail = []
for cam in range(8):
    d = os.path.join(root, "cam%d" % cam)
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.startswith("color-cam%d-" % cam) and fn.endswith(".jpg"):
                avail.append((cam, fn.split("-")[-1].split(".")[0]))
picks = random.Random(20260928).sample(avail, 8)
print("sheet order:", picks)
bad, good = picks[3], picks[0]
print("bad  = picks[3] =", bad)
print("good = picks[0] =", good)


def analyse(cam, fr, tag):
    view = io_utils.load_view(root, cam, fr)
    rgb = view["color"]
    dep = np.asarray(view["depth"], np.float32)
    inv = dep / 255.0
    res = fill_holes(rgb, inv, scale=-80.0, return_info=True)
    hole = res["hole_mask"]
    H, W = hole.shape
    left = np.zeros_like(hole)
    left[:, :W // 3] = True
    reg = hole & left
    n, lab, st, cent = cv2.connectedComponentsWithStats(reg.astype(np.uint8), 8)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    comp = lab == k
    y0, x0 = st[k, cv2.CC_STAT_TOP], st[k, cv2.CC_STAT_LEFT]
    h_, w_ = st[k, cv2.CC_STAT_HEIGHT], st[k, cv2.CC_STAT_WIDTH]
    oofa = res["oofa"] & comp
    disocc = res["disocc"] & comp
    ghost = res["stats"]  # ghost is not returned as a mask; report its global effect
    # how much of this component is "crack" vs "big hole"
    print(f"\n[{tag}] cam{cam} {fr}  scale -80")
    print(f"  image holes {hole.mean()*100:.2f}% | left-third holes {reg.mean()*100:.2f}%")
    print(f"  biggest left component: area {int(comp.sum())} px, bbox {w_}x{h_} at ({x0},{y0})")
    print(f"  of that component: OOFA {int(oofa.sum())} px, disocclusion {int(disocc.sum())} px, "
          f"neither {int((comp & ~oofa & ~disocc).sum())} px")
    print(f"  stage stats: crack {res['stats'].get('crack_px')} px, "
          f"big-hole rejected {res['stats'].get('crack_big_hole_px')}, "
          f"iterations {res['stats'].get('iterations')}, sizes "
          f"{res['stats'].get('patch_sizes')}")
    print(f"  ghost: {res['stats'].get('ghost_px')} px moved, "
          f"warp deviation n/a | residual {int(res['remaining'].sum())}")
    # depth consistency inside that component: how many hole pixels carry valid depth?
    print(f"  inside the component, depth sentinel (-1) px: {int((np.asarray(res['warped_depth'])[comp] < 0).sum())}")
    pad = 40
    yy0, yy1 = max(0, y0 - pad), min(H, y0 + h_ + pad)
    xx0, xx1 = max(0, x0 - pad), min(W, x0 + w_ + pad)
    z = (slice(yy0, yy1), slice(xx0, xx1))
    ovl = res["filled"][z].copy()
    m = comp[z]
    ovl[m] = (0.45 * ovl[m] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
    pan = [np.clip(rgb, 0, 255).astype(np.uint8)[z], res["filled"][z], ovl]
    row = pan[0]
    for t in pan[1:]:
        row = np.hstack([row, np.full((row.shape[0], 3, 3), 255, np.uint8), t])
    io_utils.imwrite(os.path.join(OUT, f"diag_{tag}.png"),
                     cv2.resize(row, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_NEAREST))
    return res


rb = analyse(bad[0], bad[1], "bad")
rg = analyse(good[0], good[1], "good")
print("\nzoom sheets: output/random_demo/diag_bad.png , diag_good.png  (原图 | 结果 | 该连通域高亮)")
