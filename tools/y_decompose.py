"""Why does the unconstrained search look best?  Decompose the error.

In a disocclusion the correct content is background that is NOT present in the same row of
the reference (that is why it is a hole), so forcing the same row gives a badly matched
patch; the full window can borrow the same background surface from another row, which looks
better everywhere except where that surface has horizontal structure (the barres).

Reports, per mode: GT PSNR/SSIM over all valid pixels, restricted to the FILLED BAND, and
restricted to the BARRE rows inside the band (|y-rail| <= 15), plus the barre rows' share of
the band.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from dibr import calib, io_utils, viz, warp as W  # noqa: E402
from viewfill import FillConfig  # noqa: E402
from viewfill.pipeline import _prep_depth, fill_warped  # noqa: E402

cams = calib.load_calib(io_utils.DATASET_ROOT_DEFAULT)


def run(fr, epi):
    view = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 6, fr)
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, fr)["color"]
    rgb = view["color"]
    P = np.asarray(view["depth"], np.float32).copy()
    cfg = FillConfig(scale=-44.8, epipolar=epi, repair_warp="never")
    Pd = _prep_depth(P, cfg)
    fdx, fdy, *_ = calib.displacement_field(cams, 6, 7, Pd, "bottom")
    fdx = np.asarray(fdx, np.float32)
    fdy = None if fdy is None else np.asarray(fdy, np.float32)
    I_w, D_w, hole, _ = W.forward_warp(np.asarray(rgb, np.float32), fdx, fdy, z=Pd,
                                       hole_depth=-1.0, rule="zbuf", splat="sub")
    res = fill_warped(I_w, hole, D_w, rgb, P, disp=(fdx, fdy), cfg=cfg)
    I_f = np.asarray(res["I_filled"], np.float32)
    g = np.asarray(gt, np.float32)
    gy = np.abs(np.diff(0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2], axis=0))
    rail = int(np.median(np.argsort(-gy.mean(axis=1))[:60]))
    rows = np.zeros_like(hole)
    rows[max(0, rail - 15):rail + 16] = True
    band = hole | rows & False
    valid = ~res["remaining"]
    return dict(
        all_psnr=viz.psnr(gt, I_f, mask=valid), all_ssim=viz.ssim(gt, I_f, mask=valid),
        band_psnr=viz.psnr(gt, I_f, mask=hole), band_ssim=viz.ssim(gt, I_f, mask=hole),
        rail_psnr=viz.psnr(gt, I_f, mask=hole & rows),
        rail_ssim=viz.ssim(gt, I_f, mask=hole & rows),
        rail_share=(hole & rows).sum() / max(1, hole.sum()) * 100, rail=rail)


print(f"{'frame':>6s} {'epi':>5s} | {'all PSNR':>8s} {'all SSIM':>8s} | {'BAND PSNR':>9s} "
      f"{'BAND SSIM':>9s} | {'BARRE PSNR':>10s} {'BARRE SSIM':>10s} | {'barre % of band':>15s}")
tot = {}
for fr in ("f000", "f004"):
    for epi in (None, 0, 2):
        r = run(fr, epi)
        tot.setdefault(str(epi), []).append(r)
        print(f"{fr:>6s} {str(epi):>5s} | {r['all_psnr']:>8.2f} {r['all_ssim']:>8.4f} | "
              f"{r['band_psnr']:>9.2f} {r['band_ssim']:>9.4f} | {r['rail_psnr']:>10.2f} "
              f"{r['rail_ssim']:>10.4f} | {r['rail_share']:>14.1f}%", flush=True)
print()
for k, rows in tot.items():
    a = {m: np.mean([x[m] for x in rows]) for m in rows[0] if m != "rail"}
    print(f"mean epi={k:>4s}: all {a['all_psnr']:.2f}/{a['all_ssim']:.4f} | "
          f"band {a['band_psnr']:.2f}/{a['band_ssim']:.4f} | "
          f"barre {a['rail_psnr']:.2f}/{a['rail_ssim']:.4f} | barre {a['rail_share']:.1f}%")
