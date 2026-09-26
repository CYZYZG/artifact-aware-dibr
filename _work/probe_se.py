"""Which SE orientation actually detects the true thin (<=2 px) cracks?"""
import sys

import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补")
from dibr import cracks  # noqa: E402

OUT = r"D:\项目\空洞填补\output\step1_warp\cam6_to_cam7_f000"
D_w = np.load(OUT + r"\warped_P.npy")
hole = D_w < 0
rows, lab = cracks.component_stats(hole)
thin_labels = [r["label"] for r in rows if r["thickness"] <= 2]
thin = np.isin(lab, thin_labels)
big_labels = [r["label"] for r in rows if r["area"] > 200]
big = np.isin(lab, big_labels)
print(f"true thin (<=2px) hole pixels: {int(thin.sum())} in {len(thin_labels)} components")
print(f"big hole pixels: {int(big.sum())}")

print(f"\n{'SE':>3s} {'crack px':>9s} {'of thin':>9s} {'thin rec%':>10s} "
      f"{'empty':>7s} {'transl':>7s} {'in big':>7s}")
for orient in ("v", "h"):
    c, _, _ = cracks.detect_cracks(D_w, lam=5, se_len=4, orientation=orient)
    print(f"{orient:>3s} {c.sum():9d} {int((c & thin).sum()):9d} "
          f"{100*(c & thin).sum()/max(1,thin.sum()):10.1f} "
          f"{int((c&hole).sum()):7d} {int((c&~hole).sum()):7d} {int((c&big).sum()):7d}")

# a crack is a vertical slit -> a horizontal SE must bridge it; check on the thin set
print("\nthin hole components, thickness<=2, sorted by area:")
for r in sorted([r for r in rows if r["thickness"] <= 2], key=lambda r: -r["area"])[:8]:
    print(f"  area {r['area']:4d} bbox {r['bw']}x{r['bh']} thickness {r['thickness']:.1f} "
          f"elong {r['elongation']:.1f} at ({r['cx']:.0f},{r['cy']:.0f})")

# how many detections are within 2 px of a big hole (i.e. not a crack)
import cv2
near_big = cv2.dilate(big.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
print(f"\n{'SE':>3s} {'det':>7s} {'near big hole':>14s} {'away from holes':>16s}")
for orient in ("v", "h"):
    c, _, _ = cracks.detect_cracks(D_w, lam=5, se_len=4, orientation=orient)
    print(f"{orient:>3s} {c.sum():7d} {int((c & near_big).sum()):14d} "
          f"{int((c & ~near_big).sum()):16d}")
