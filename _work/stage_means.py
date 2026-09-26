"""Per-pair stage averages for the Step 8 report."""
import csv

import numpy as np

P = r"D:\项目\空洞填补\output\step8_eval\summary.csv"
rows = list(csv.DictReader(open(P, encoding="utf-8")))
for r in rows:
    for k, v in list(r.items()):
        try:
            r[k] = float(v)
        except (TypeError, ValueError):
            pass
pairs = []
for r in rows:
    k = (int(r["ref_cam"]), int(r["dst_cam"]))
    if k not in pairs:
        pairs.append(k)

hdr = ["pair", "holes%", "warp", "+crack", "+ghost", "+fill",
       "SSIM_warp", "SSIM_fill", "fill_iters", "sec/frame"]
print(" | ".join(f"{h:>10s}" for h in hdr))
for a, b in pairs:
    s = [r for r in rows if int(r["ref_cam"]) == a and int(r["dst_cam"]) == b]
    m = lambda k: float(np.mean([r[k] for r in s]))
    t = np.mean([sum(r[k] or 0 for k in ("sec_step1", "sec_step2", "sec_step3", "sec_step4"))
                 for r in s])
    vals = [f"cam{a}->cam{b}", f"{m('hole_pct'):.2f}", f"{m('psnr_warp'):.2f}",
            f"{m('psnr_cracks'):.2f}", f"{m('psnr_ghosts'):.2f}", f"{m('psnr_filled'):.2f}",
            f"{m('ssim_warp'):.4f}", f"{m('ssim_filled'):.4f}",
            f"{m('fill_iterations'):.0f}", f"{t:.1f}"]
    print(" | ".join(f"{v:>10s}" for v in vals))

print()
print("all-run means: warp %.2f / cracks %.2f / ghosts %.2f / filled %.2f"
      % (np.mean([r["psnr_warp"] for r in rows]),
         np.mean([r["psnr_cracks"] for r in rows]),
         np.mean([r["psnr_ghosts"] for r in rows]),
         np.mean([r["psnr_filled"] for r in rows])))
g = [r["psnr_ghosts"] - r["psnr_cracks"] for r in rows]
print("ghost-step frame gain: mean %.3f dB, max %.3f dB" % (np.mean(g), np.max(g)))
f = [r["psnr_filled"] - r["psnr_ghosts"] for r in rows]
print("fill-step frame gain:  mean %.3f dB, min %.3f dB" % (np.mean(f), np.min(f)))
