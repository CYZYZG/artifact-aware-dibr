"""Step 2 probe: how crack-like is the warped data, and what does lambda do?"""
import sys

import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补")
from dibr import cracks, io_utils  # noqa: E402

OUT = r"D:\项目\空洞填补\output\step1_warp\cam6_to_cam7_f000"
D_w = np.load(OUT + r"\warped_P.npy")
I_w = np.load(OUT + r"\warped_color.npy")
hole = D_w < 0
print(f"holes {hole.sum()} px ({hole.mean()*100:.2f}%)")

rows, lab = cracks.component_stats(hole)
areas = np.array([r["area"] for r in rows])
thick = np.array([r["thickness"] for r in rows])
elong = np.array([r["elongation"] for r in rows])
print(f"components {len(rows)}   area: median {np.median(areas):.0f} max {areas.max()}")
print("thickness histogram (2*maxEDT): ",
      {f"<={t}": int((thick <= t).sum()) for t in (1, 2, 3, 4, 6, 10)})
print(f"  components with thickness<=2: {int((thick<=2).sum())} "
      f"covering {int(areas[thick<=2].sum())} px")
print(f"  components with thickness<=3: {int((thick<=3).sum())} "
      f"covering {int(areas[thick<=3].sum())} px")
print(f"  elongated (>=3) & thickness<=3: {int(((elong>=3)&(thick<=3)).sum())} "
      f"covering {int(areas[(elong>=3)&(thick<=3)].sum())} px")
big = set(r["label"] for r in rows if r["area"] > 200)
print(f"big components (>200 px): {len(big)} covering {int(areas[areas>200].sum())} px")

print("\n=== luminance range of D_w on valid pixels ===")
print(f"  valid P: min {D_w[~hole].min():.0f} max {D_w[~hole].max():.0f} "
      f"p1 {np.percentile(D_w[~hole],1):.0f} p99 {np.percentile(D_w[~hole],99):.0f}")

print("\n=== detection vs lambda (vertical SE, len 4) ===")
print(f"{'lam':>5s} {'crack px':>9s} {'empty':>8s} {'translucent':>12s} "
      f"{'in big holes':>13s} {'comp':>6s} {'thick<=3 comp':>14s}")
for lam in (1, 2, 3, 5, 8, 12, 20):
    crack, D_hat, diff = cracks.detect_cracks(D_w, lam=lam, se_len=4, orientation="v")
    empty = crack & hole
    trans = crack & ~hole
    inbig = crack & np.isin(lab, list(big))
    crows, _ = cracks.component_stats(crack)
    ct = np.array([r["thickness"] for r in crows]) if crows else np.array([0.0])
    print(f"{lam:5d} {crack.sum():9d} {empty.sum():8d} {trans.sum():12d} "
          f"{inbig.sum():13d} {len(crows):6d} {int((ct<=3).sum()):14d}")

print("\n=== orientation comparison at lam=5 ===")
for orient in ("v", "h"):
    crack, D_hat, diff = cracks.detect_cracks(D_w, lam=5, se_len=4, orientation=orient)
    print(f"  {orient}: crack {crack.sum():7d} px  empty {(crack&hole).sum():7d}  "
          f"translucent {(crack&~hole).sum():7d}  in-big-holes "
          f"{(crack&np.isin(lab, list(big))).sum():7d}")
cb, hat, diff, cv, ch = cracks.detect_cracks_both(D_w, lam=5, se_len=4)
print(f"  both: crack {cb.sum():7d} px (v={cv.sum()}, h={ch.sum()}, "
      f"overlap={(cv&ch).sum()})")
