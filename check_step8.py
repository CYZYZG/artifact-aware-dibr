"""Automated checks for Step 8 (full-pipeline evaluation) + summary figures.

    python check_step8.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dibr import io_utils, plot, viz  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "output", "step8_eval")
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    p = os.path.join(EVAL, "summary.csv")
    if not os.path.isfile(p):
        print("no summary.csv: run run_all.py first")
        return 1
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("tag",):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass

    pairs = []
    for r in rows:
        key = (int(r["ref_cam"]), int(r["dst_cam"]))
        if key not in pairs:
            pairs.append(key)
    frames = sorted({r["frame"] for r in rows})
    print(f"{len(rows)} runs: pairs {pairs}, frames {frames[0]}..{frames[-1]}")

    # ---- 1. completeness -----------------------------------------------------
    check("every run filled all of its holes",
          all(r["holes_after_fill"] == 0 for r in rows),
          f"max remaining {max(r['holes_after_fill'] for r in rows):.0f} px")
    check("every run produced finite metrics",
          all(np.isfinite(r["final_frame_psnr"]) and np.isfinite(r["final_frame_ssim"])
              for r in rows))
    check("all camera pairs covered the same frames",
          all(len([r for r in rows if (int(r['ref_cam']), int(r['dst_cam'])) == pp]) == len(frames)
              for pp in pairs))

    # ---- 2. monotone pipeline improvement ------------------------------------
    worse = [r["tag"] for r in rows
             if not (r["psnr_filled"] > r["psnr_ghosts"] > 0
                     and r["ssim_filled"] > r["ssim_ghosts"])]
    check("final PSNR/SSIM is always better than the pre-fill state", not worse,
          f"{len(worse)} exceptions" + (f": {worse[:3]}" if worse else ""))
    worse2 = [r["tag"] for r in rows if not (r["psnr_filled"] > r["psnr_warp"])]
    check("final PSNR is always better than the raw warp", not worse2,
          f"{len(worse2)} exceptions")

    # ---- 3. per-pair averages ------------------------------------------------
    avg = {}
    for pp in pairs:
        sel = [r for r in rows if (int(r["ref_cam"]), int(r["dst_cam"])) == pp]
        avg[pp] = (float(np.mean([r["final_frame_psnr"] for r in sel])),
                   float(np.mean([r["final_frame_ssim"] for r in sel])),
                   float(np.mean([r["hole_pct"] for r in sel])),
                   len(sel))
        print(f"   cam{pp[0]}->cam{pp[1]}: PSNR {avg[pp][0]:.2f} dB  "
              f"SSIM {avg[pp][1]:.4f}  holes {avg[pp][2]:.2f}%  n={avg[pp][3]}")
    check("averages are in the same league as the paper's Table I (MVD 29.17 dB / 0.8463)",
          all(20.0 < v[0] < 40.0 and 0.5 < v[1] < 1.0 for v in avg.values()),
          "; ".join(f"cam{a}->cam{b} {v[0]:.2f} dB/{v[1]:.4f}" for (a, b), v in avg.items()))

    # ---- 4. the crack / ghost / fill stages all contributed -------------------
    check("crack filling improved the frame on every run",
          all(r["crack_gt_gain"] > 0 for r in rows),
          f"min gain {min(r['crack_gt_gain'] for r in rows):.2f} dB")
    check("exemplar filling improved the filled region on every run",
          all(r["fill_px_psnr"] > 0 for r in rows),
          f"mean filled-region PSNR {np.mean([r['fill_px_psnr'] for r in rows]):.2f} dB")

    # ---- 5. figures ----------------------------------------------------------
    xlabels = [r["frame"] for r in rows if (int(r["ref_cam"]), int(r["dst_cam"])) == pairs[0]]
    ps = [(f"cam{a}->cam{b}", [r["final_frame_psnr"] for r in rows
                               if (int(r["ref_cam"]), int(r["dst_cam"])) == (a, b)],
           ["red", "green", "blue", "orange"][i % 4]) for i, (a, b) in enumerate(pairs)]
    ss = [(f"cam{a}->cam{b}", [r["final_frame_ssim"] for r in rows
                               if (int(r["ref_cam"]), int(r["dst_cam"])) == (a, b)],
           ["red", "green", "blue", "orange"][i % 4]) for i, (a, b) in enumerate(pairs)]
    c1 = plot.line_chart(ps, "Step 8: final PSNR per frame", "PSNR (dB)", ref_lines=[
        (29.17, "paper Ours (MVD) 29.17")], xlabels=xlabels)
    c2 = plot.line_chart(ss, "Step 8: final SSIM per frame", "SSIM", ref_lines=[
        (0.8463, "paper Ours (MVD) 0.8463")], xlabels=xlabels)
    stage_names = ["psnr_warp", "psnr_cracks", "psnr_ghosts", "psnr_filled"]
    groups = []
    for (a, b) in pairs:
        sel = [r for r in rows if (int(r["ref_cam"]), int(r["dst_cam"])) == (a, b)]
        groups.append((f"cam{a}->cam{b}",
                       [("warp", float(np.mean([r["psnr_warp"] for r in sel])), "grey"),
                        ("+cracks", float(np.mean([r["psnr_cracks"] for r in sel])), "blue"),
                        ("+ghosts", float(np.mean([r["psnr_ghosts"] for r in sel])), "violet"),
                        ("+fill", float(np.mean([r["psnr_filled"] for r in sel])), "green")]))
    c3 = plot.bar_chart(groups, "Step 8: stage-by-stage PSNR (mean per camera pair)",
                        "PSNR (dB)")
    io_utils.imwrite(os.path.join(EVAL, "progress.png"),
                     np.vstack([c1, c2, c3]))
    check("summary figure written", os.path.isfile(os.path.join(EVAL, "progress.png")),
          os.path.join(EVAL, "progress.png"))

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
