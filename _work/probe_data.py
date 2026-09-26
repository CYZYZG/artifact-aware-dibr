"""Exploratory probe: (1) temporal vs multi-view, (2) behaviour of provided warping.py."""
import glob
import os
import sys

import numpy as np
from PIL import Image

BASE = r"D:\项目\空洞填补\input"
sys.path.insert(0, r"D:\项目\空洞填补")
from warping import scatter_image  # noqa: E402


def imread(path, gray=False):
    im = Image.open(path)
    return np.asarray(im.convert("L" if gray else "RGB"))


cs = sorted(glob.glob(os.path.join(BASE, "color-*.jpg")))
ds = sorted(glob.glob(os.path.join(BASE, "depth-*.png")))

# ---------- 1) global best integer x-shift between frame 0 and frame k ----------
def best_shift(a, b, max_shift=40):
    """Find integer dx minimising mean abs diff of b(x+dx) vs a(x), on the given crop."""
    best = None
    for dx in range(-max_shift, max_shift + 1):
        if dx >= 0:
            d = np.abs(a[:, dx:] - b[:, : b.shape[1] - dx]).mean()
        else:
            d = np.abs(a[:, : b.shape[1] + dx] - b[:, -dx:]).mean()
        if best is None or d < best[1]:
            best = (dx, d)
    return best


f0 = imread(cs[0]).astype(np.float32)
print("=== global translation (color) vs f000 ===")
for k in [1, 5, 9]:
    fk = imread(cs[k]).astype(np.float32)
    fr = imread(cs[(k + 5) % 10]).astype(np.float32)
    print(f"f{k:03d}: best dx={best_shift(f0, fk)}   (f{(k+5)%10:03d} vs f{k:03d}: "
          f"{best_shift(fk, fr)})")

print("\n=== per-region best dx f000 -> f005 (far bg vs near fg) ===")
f5 = imread(cs[5]).astype(np.float32)
for name, sl in [("far wall/curtain", (slice(30, 200), slice(300, 700))),
                 ("floor", (slice(600, 760), slice(100, 500))),
                 ("near dancer", (slice(60, 700), slice(650, 900))),
                 ("whole image", (slice(None), slice(None)))]:
    print(f"  {name:16s}", best_shift(f0[sl], f5[sl]))

# ---------- 2) provided scatter_image behaviour ----------
print("\n=== provided warping.scatter_image on f000 ===")
color = imread(cs[0]).astype(np.float32)
inv = imread(ds[0], gray=True).astype(np.float32) / 255.0
print("inv_depth min/max/mean %.4f %.4f %.4f" % (inv.min(), inv.max(), inv.mean()))
for sf in [10.0, 22.4, 44.8]:
    for direction in (-1, 1):
        for ordering in (False, True):
            w, mask, dep = scatter_image(color, inv, direction=direction,
                                        scale_factor=sf, inverse_ordering=ordering,
                                        reproject_depth=True)
            holes = mask > 0
            print(f"  sf={sf:5.1f} dir={direction:+d} inv_order={ordering!s:5s} "
                  f"holes={holes.mean()*100:6.3f}%  maxdisp={sf*inv.max():6.2f}px")
