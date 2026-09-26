"""Paired statistics over output/sweep_all.csv: which depth_dilate mode is best?

For every metric (seam, p90, cracks, gt_psnr, gt_ssim) and every pair of modes, this does a
PAIRED comparison over the same (pair, frame) samples:
  * mean +- std of the per-sample difference
  * win count / ties
  * Wilcoxon signed-rank test (non-parametric, no normality assumption)
and prints the per-mode means plus the best-per-sample counts.

    python _work/sweep_stats.py            # all pairs
    python _work/sweep_stats.py --pair 6-7
"""
import argparse
import os
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "output", "sweep_all.csv")
LOWER_IS_BETTER = {"seam": True, "p90": True, "cracks": True, "holes": True,
                   "gt_psnr": False, "gt_ssim": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="")
    ap.add_argument("--metric", default="seam")
    a = ap.parse_args()
    df = pd.read_csv(CSV)
    df["mode"] = df["mode"].astype(str)
    df = df.drop_duplicates(subset=["pair", "frame", "mode"], keep="last")
    if a.pair:
        df = df[df["pair"] == a.pair]
    n = df.groupby(["pair", "frame"]).size().min()
    full = df.groupby(["pair", "frame"]).size()
    complete = full[full == full.max()].index
    df = df.set_index(["pair", "frame"]).loc[complete].reset_index()
    print(f"{len(df)} runs, {df.groupby(['pair','frame']).ngroups} (pair, frame) samples, "
          f"{df['mode'].nunique()} modes, {df['pair'].nunique()} pairs "
          f"(each sample has {n} modes)\n")

    modes = sorted(df["mode"].unique())
    for metric in ("seam", "p90", "gt_psnr", "gt_ssim"):
        low = LOWER_IS_BETTER[metric]
        print(f"=== {metric} ({'lower' if low else 'higher'} is better) ===")
        piv = df.pivot_table(index=["pair", "frame"], columns="mode", values=metric)
        for m in modes:
            arr = piv[m].to_numpy()
            print(f"  {m:>6s}: mean {arr.mean():8.3f}  sd {arr.std(ddof=1):7.3f}  "
                  f"median {np.median(arr):8.3f}")
        # best per sample
        wins = {m: 0 for m in modes}
        for _, row in piv.iterrows():
            vals = {m: row[m] for m in modes if np.isfinite(row[m])}
            best = (min if low else max)(vals.values())
            for m, v in vals.items():
                if abs(v - best) < 1e-9:
                    wins[m] += 1
        print(f"  per-sample best count: " +
              ", ".join(f"{m} {wins[m]}/{len(piv)}" for m in modes))
        # paired tests
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                x, y = piv[modes[i]].to_numpy(), piv[modes[j]].to_numpy()
                ok = np.isfinite(x) & np.isfinite(y)
                d = x[ok] - y[ok]
                if ok.sum() < 5 or np.allclose(d, 0):
                    print(f"  {modes[i]} vs {modes[j]}: n={ok.sum()} (too few / identical)")
                    continue
                try:
                    p = stats.wilcoxon(x[ok], y[ok]).pvalue
                except Exception:
                    p = float("nan")
                better = "lower" if low else "higher"
                won = int((d < 0).sum()) if low else int((d > 0).sum())
                print(f"  {modes[i]} vs {modes[j]}: mean diff {d.mean():+8.3f} "
                      f"(sd {d.std(ddof=1):7.3f}), {modes[i]} better on {won}/{ok.sum()} "
                      f"samples, Wilcoxon p={p:.2e}  [{'significant' if p < 0.05 else 'n.s.'}]")
        print()


if __name__ == "__main__":
    main()
