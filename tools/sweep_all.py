"""Full sweep: 4 camera pairs x 10 frames x depth_dilate in {3, auto5, auto7}.

Writes one row per run to output/sweep_all.csv and SKIPS rows already present, so it can be
stopped and restarted.  Geometry per pair is the calibrated displacement field from
dibr.calib (same as the reproduction's Step 1), recomputed from the depth map after the
depth_dilate pre-processing, so the comparison across modes is paired and fair.

    python _work/sweep_all.py                 # everything (long)
    python _work/sweep_all.py --limit 1       # one frame per pair, for a timing probe
    python _work/sweep_all.py --pairs 6-7     # one pair only
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = r"D:\项目\空洞填补"
sys.path.insert(0, HERE)
from dibr import calib, io_utils, viz, warp as W  # noqa: E402
from viewfill import FillConfig, io as vio  # noqa: E402
from viewfill.pipeline import _prep_depth, fill_warped  # noqa: E402

CSV = os.path.join(HERE, "output", "sweep_all.csv")
PAIRS = [(6, 7), (6, 5), (3, 0), (3, 2)]
MODES = (3, "auto5", "auto7")
K = 9


def seam(I_f, I_w, hole):
    g_f = 0.299 * I_f[..., 0] + 0.587 * I_f[..., 1] + 0.114 * I_f[..., 2]
    g_w = 0.299 * I_w[..., 0] + 0.587 * I_w[..., 1] + 0.114 * I_w[..., 2]
    valid = ~hole

    def loc(img, m):
        mm = m.astype(np.float32)
        num = cv2.boxFilter(img * mm, -1, (K, K), normalize=False,
                            borderType=cv2.BORDER_REPLICATE)
        den = cv2.boxFilter(mm, -1, (K, K), normalize=False,
                            borderType=cv2.BORDER_REPLICATE)
        return num / np.maximum(den, 1e-6)

    bnd = hole & cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not bnd.any():
        return float("nan"), float("nan")
    j = np.abs(loc(g_f, hole)[bnd] - loc(g_w, valid)[bnd])
    return float(j.mean()), float(np.percentile(j, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--modes", default="3,auto5,auto7")
    a = ap.parse_args()
    pairs = [p for p in PAIRS
             if not a.pairs or "-".join(map(str, p)) in a.pairs]
    modes = [int(m) if m.isdigit() else m for m in a.modes.split(",")]
    cams = calib.load_calib(io_utils.DATASET_ROOT_DEFAULT)
    y_org = "bottom"        # validated in Step 1 (residual flow 2.91 vs 7.78 px)

    done = set()
    if os.path.exists(CSV):
        with open(CSV, encoding="utf-8") as fh:
            for ln in fh.read().splitlines()[1:]:
                if ln.strip():
                    p = ln.split(",")
                    done.add((p[0], p[1], p[2]))
    else:
        with open(CSV, "w", encoding="utf-8") as fh:
            fh.write("pair,frame,mode,seam,p90,cracks,holes,gt_psnr,gt_ssim,seconds,"
                     "hole_pct\n")
    print(f"resume: {len(done)} rows already done -> {CSV}", flush=True)

    for src, dst in pairs:
        for i in range(a.limit):
            fr = f"f{i:03d}"
            view = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, src, fr)
            gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, fr)["color"]
            rgb = view["color"]
            P = np.asarray(view["depth"], np.float32).copy()   # 0..255, near = large
            inv = P / 255.0
            for mode in modes:
                key = (f"{src}-{dst}", fr, str(mode))
                if key in done:
                    continue
                t = time.time()
                cfg = FillConfig(scale=-44.8, depth_dilate=mode, repair_warp="never")
                Pd = _prep_depth(P, cfg)
                fdx, fdy, *_ = calib.displacement_field(cams, src, dst, Pd, y_org)
                fdx = np.asarray(fdx, np.float32)
                fdy = None if fdy is None else np.asarray(fdy, np.float32)
                I_w, D_w, hole, _ = W.forward_warp(
                    np.asarray(rgb, np.float32), fdx, fdy, z=Pd, hole_depth=-1.0,
                    rule="zbuf", splat="sub")
                res = fill_warped(I_w, hole, D_w, rgb, P, disp=(fdx, fdy), cfg=cfg)
                I_f = res["I_filled"]
                s, p90 = seam(I_f, I_w, hole)
                m = ~res["remaining"]
                ps = viz.psnr(gt, I_f, mask=m)
                ss = viz.ssim(gt, I_f, mask=m)
                dt = time.time() - t
                with open(CSV, "a", encoding="utf-8") as fh:
                    fh.write(f"{src}-{dst},{fr},{mode},{s:.4f},{p90:.4f},"
                             f"{int(res['crack'].sum())},{int(hole.sum())},{ps:.4f},"
                             f"{ss:.6f},{dt:.1f},{hole.mean()*100:.3f}\n")
                    fh.flush()
                print(f"  {src}->{dst} {fr} {str(mode):>5s}: seam {s:5.2f}/{p90:5.2f} "
                      f"holes {hole.mean()*100:5.2f}% GT {ps:5.2f}/{ss:.4f} {dt:5.1f}s",
                      flush=True)
    print("sweep finished", flush=True)


if __name__ == "__main__":
    main()
