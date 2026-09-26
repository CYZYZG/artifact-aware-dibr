"""Effect test for the viewfill package, following the 2D->3D flow.

    1) convention        : disparity == inverse_depth * (-44.8), holes appear, OOFA on the
                           edge the projection vacates
    2) end-to-end        : warp -> cracks -> ghosts -> fill, no hole may survive
    3) external warp     : feed the filler the output of the PROVIDED forward warp
                           (warping.scatter_image) instead of our own warp
    4) real ground truth : stress test against a real neighbouring camera (Ballet cam7)
    5) scale robustness  : -20 / -44.8 / -80 px
    6) API consistency   : warp_and_fill == fill_warped on the same warp, and determinism

Run:  python test_viewfill.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dibr import io_utils, viz                      # noqa: E402
from viewfill import FillConfig, io as vio, report  # noqa: E402
from viewfill.pipeline import fill_warped, warp_and_fill  # noqa: E402

OUT = os.path.join(HERE, "output", "viewfill_test")
IMG = os.path.join(HERE, "input", "color-cam6-f000.jpg")
DEP = os.path.join(HERE, "input", "depth-cam6-f000.png")
ROOT = io_utils.DATASET_ROOT_DEFAULT
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rgb, inv = vio.load_pair(IMG, DEP)
    print(f"input: {os.path.basename(IMG)} {rgb.shape} + {os.path.basename(DEP)} "
          f"inverse depth [{inv.min():.3f}, {inv.max():.3f}]")

    # ---------------------------------------------------------------- 1) convention
    cfg = FillConfig(scale=-44.8)
    from viewfill.pipeline import disparity_from_depth
    disp = disparity_from_depth(inv * 255.0, cfg.scale)
    expected = inv * cfg.scale
    check("disparity == inverse_depth * (-44.8)",
          float(np.abs(disp - expected).max()) < 1e-3,
          f"max |dx - inv*scale| = {np.abs(disp - expected).max():.2e}, "
          f"dx range [{disp.min():.2f}, {disp.max():.2f}] px")

    res = warp_and_fill(rgb, inv, cfg, log=lambda s: print(s))
    hole = res["hole_mask"]
    check("the warp really produces holes", 0.005 < hole.mean() < 0.5,
          f"{int(hole.sum())} px ({hole.mean()*100:.2f}%)")
    check("OOFA sits on the edge the projection vacates (right for a left shift)",
          int(hole[:, -20:].sum()) > 10 * max(1, int(hole[:, :20].sum())),
          f"left 20 cols {int(hole[:, :20].sum())} vs right 20 cols {int(hole[:, -20:].sum())}")

    # ---------------------------------------------------------------- 2) end-to-end
    check("every hole is filled", int(res["remaining"].sum()) == 0,
          f"{int(res['stats']['holes_before'])} px before, "
          f"{int(res['remaining'].sum())} px left, "
          f"{res['stats']['seconds_fill']}s")
    I = res["I_filled"]
    check("result is finite and non-degenerate",
          bool(np.isfinite(I).all()) and float(I[hole].std()) > 5.0,
          f"filled-region std {I[hole].std():.2f}")
    check("the filled pixels are real content, not black",
          float(I[hole].mean()) > 20.0, f"mean {I[hole].mean():.1f}")
    bp_b = res["stats"].get("back_proj_psnr_before", float("nan"))
    bp_a = res["stats"].get("back_proj_psnr_after", float("nan"))
    check("back-projection consistency does not degrade",
          bp_a >= bp_b - 0.05,
          f"{bp_b:.2f} -> {bp_a:.2f} dB (round trip to the reference viewpoint)")
    st = report.save(res, cfg, os.path.join(OUT, "01_end_to_end"), ref_rgb=rgb)
    print(f"  -> {os.path.join(OUT, '01_end_to_end')}")

    # ------------------------------------------------------- 3) external warp (yours)
    sys.path.insert(0, HERE)
    from warping import scatter_image
    # NOTE: inverse_ordering must be False.  The provided splat has no depth test, and
    # with inverse_ordering=True the FAR sample wins the collision, replacing foreground
    # texture with background (the subject looks cut up).  See case 3b below.
    w, hm, dep = scatter_image(rgb.astype(np.float32), inv, direction=-1,
                              scale_factor=44.8, inverse_ordering=False,
                              reproject_depth=True)
    hole_ext = hm > 0
    dw = np.where(hole_ext, 0.0, 255.0 / (dep + 1e-6)).astype(np.float32)
    t = time.time()
    res_ext = fill_warped(np.clip(w, 0, 255).astype(np.float32), hole_ext, dw, rgb, inv,
                          cfg=cfg, log=lambda s: print("[external warp] " + s))
    check("fills the output of the PROVIDED forward warp (warping.scatter_image, "
          "inverse_ordering=False)",
          int(res_ext["remaining"].sum()) == 0,
          f"{int(hole_ext.sum())} hole px -> 0, {time.time()-t:.1f}s, "
          f"deviation from a Z-buffer warp {res_ext['stats']['warp_deviation_pct']:.2f}%")
    check("a correctly warped input is left untouched (no repair triggered)",
          res_ext["stats"]["warp_repaired"] is False and
          res_ext["stats"]["warp_deviation_pct"] < 0.5,
          f"deviation {res_ext['stats']['warp_deviation_pct']:.2f}%")

    # ------------------------------- 3b) a broken warp is detected and repaired
    w_bad, hm_bad, dep_bad = scatter_image(rgb.astype(np.float32), inv, direction=-1,
                                           scale_factor=44.8, inverse_ordering=True,
                                           reproject_depth=True)
    hole_bad = hm_bad > 0
    dw_bad = np.where(hole_bad, 0.0, 255.0 / (dep_bad + 1e-6)).astype(np.float32)
    res_bad = fill_warped(np.clip(w_bad, 0, 255).astype(np.float32), hole_bad, dw_bad,
                          rgb, inv, cfg=FillConfig(scale=-44.8, repair_warp="never"))
    check("the foreground damage of inverse_ordering=True is detected",
          res_bad["stats"]["warp_deviation_pct"] > 1.0,
          f"deviation {res_bad['stats']['warp_deviation_pct']:.2f}% of valid px "
          f"(foreground {res_bad['stats']['fg_deviation_pct']:.2f}%)")
    res_fix = fill_warped(np.clip(w_bad, 0, 255).astype(np.float32), hole_bad, dw_bad,
                          rgb, inv, cfg=FillConfig(scale=-44.8, repair_warp="auto"))
    same_fix = float(np.abs(res_fix["I_filled"] - res["I_filled"]).max())
    check("auto-repair re-warps with the Z-buffer and gives the clean result",
          res_fix["stats"]["warp_repaired"] is True and same_fix == 0.0 and
          int(res_fix["remaining"].sum()) == 0,
          f"repaired=True, max diff vs the clean pipeline {same_fix:g}, "
          f"residual {int(res_fix['remaining'].sum())}")
    report.save(res_ext, cfg, os.path.join(OUT, "02_external_warp"), ref_rgb=rgb)
    print(f"  -> {os.path.join(OUT, '02_external_warp')}")

    # ------------------------------------------------------------- 4) real GT stress
    gt_ok = True
    try:
        gt = io_utils.load_view(ROOT, 7, "f000")["color"]
    except FileNotFoundError:
        gt, gt_ok = None, False
    if gt is not None:
        m = ~res["remaining"]
        st_gt = report.save(res, cfg, os.path.join(OUT, "03_real_gt"), ref_rgb=rgb, gt=gt)
        check("real-camera stress test ran (Ballet cam6 -> cam7 with the 1D flow)",
              st_gt["gt_psnr_frame_filled"] > st_gt["gt_psnr_frame_warped"],
              f"whole frame {st_gt['gt_psnr_frame_warped']:.2f} -> "
              f"{st_gt['gt_psnr_frame_filled']:.2f} dB, filled holes "
              f"{st_gt.get('gt_psnr_in_filled_holes', float('nan')):.2f} dB, "
              f"SSIM {st_gt['gt_ssim_frame_warped']:.4f} -> "
              f"{st_gt['gt_ssim_frame_filled']:.4f}")
        print("  note: the Ballet rig is toed-in, so a pure 1D shift cannot match a real "
              "camera exactly; this is a stress test, not the best case (the calibrated "
              "warp reaches 27.53 dB, see 复现方案.md).")

    # ------------------------------------------------------------ 5) scale robustness
    rows = []
    for s in (-20.0, -44.8, -80.0):
        c = FillConfig(scale=s)
        r = warp_and_fill(rgb, inv, c)
        m = ~r["remaining"]
        row = dict(scale=s, hole_pct=round(float(r["hole_mask"].mean() * 100), 2),
                   residual=int(r["remaining"].sum()),
                   back_before=r["stats"].get("back_proj_psnr_before"),
                   back_after=r["stats"].get("back_proj_psnr_after"),
                   sec=r["stats"]["seconds_fill"])
        if gt is not None:
            row["gt_before"] = round(viz.psnr(gt, r["I_w"], mask=m), 2)
            row["gt_after"] = round(viz.psnr(gt, r["I_filled"], mask=m), 2)
        rows.append(row)
        print(f"  scale {s:+7.1f}: holes {row['hole_pct']:5.2f}% -> residual "
              f"{row['residual']}, back-proj {row['back_before']:.2f} -> "
              f"{row['back_after']:.2f} dB"
              + (f", GT {row['gt_before']:.2f} -> {row['gt_after']:.2f} dB"
                 if gt is not None else ""))
    check("fills cleanly at every tested disparity scale", all(r["residual"] == 0 for r in rows))
    check("hole area grows with the disparity scale",
          rows[0]["hole_pct"] < rows[1]["hole_pct"] < rows[2]["hole_pct"],
          " < ".join(f"{r['hole_pct']:.2f}%" for r in rows))

    # ------------------------------------------------------------ 6) API consistency
    res_b = fill_warped(res["warped_float"], hole,
                        np.asarray(res["warped_depth"], np.float32), rgb, inv, cfg=cfg)
    same = float(np.abs(res_b["I_filled"] - res["I_filled"]).max())
    diff_px = int((np.abs(res_b["I_filled"] - res["I_filled"]).max(axis=2) > 1e-3).sum())
    check("fill_warped reproduces warp_and_fill on the same warp (exact, no quantisation)",
          int(res_b["remaining"].sum()) == 0 and same == 0.0,
          f"max pixel difference {same:g} over {diff_px} px")
    # Through the uint8 round trip the result is NOT pixel-identical: the greedy fill
    # order is chaotic (a 0.5 gray-level change can swap two frontier priorities and
    # divert the whole trajectory).  What must hold is that the two results are of equal
    # quality, which is what we check here.
    res_q = fill_warped(res["warped"].astype(np.float32), hole,
                        np.asarray(res["warped_depth"], np.float32), rgb, inv, cfg=cfg)
    diff = np.abs(res_q["I_filled"] - res["I_filled"])
    bp_a_q = res_q["stats"].get("back_proj_psnr_after", float("nan"))
    bp_a = res["stats"].get("back_proj_psnr_after", float("nan"))
    check("uint8 re-feeding of the same warp gives an equally good result "
          "(greedy order is perturbation sensitive, quality is not)",
          int(res_q["remaining"].sum()) == 0 and abs(bp_a_q - bp_a) < 0.5
          and float(diff.mean()) < 8.0,
          f"both residual 0 | back-proj {bp_a:.2f} vs {bp_a_q:.2f} dB | "
          f"mean |diff| {diff.mean():.2f}, max {diff.max():.0f} gray levels over "
          f"{(diff.max(axis=2) > 1e-3).mean()*100:.1f}% of pixels")
    res_c = warp_and_fill(rgb, inv, FillConfig(scale=-44.8))
    same2 = float(np.abs(res_c["I_filled"] - res["I_filled"]).max())
    check("the pipeline is deterministic", same2 == 0.0, f"max diff {same2:g}")

    # ---------------------------------------------------------------- summary
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed   "
          f"(total {time.time()-t0:.1f}s)")
    print(f"artefacts under {OUT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
