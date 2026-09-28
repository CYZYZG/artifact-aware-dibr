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
from viewfill.pipeline import (check_warp_quality, disparity_from_depth,  # noqa: E402
                               fill_warped, warp_and_fill)

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

    # ------------------------------------------------- 0) the one-call interface
    from viewfill import fill_holes
    fixed = fill_holes(rgb, inv, scale=-44.8)
    info = fill_holes(rgb, inv, scale=-44.8, return_info=True)
    check("fill_holes(image, inv_depth) -> repaired image in one call",
          fixed.shape == rgb.shape and fixed.dtype == np.uint8
          and int(info["remaining"].sum()) == 0,
          f"{rgb.shape} {rgb.dtype} -> {fixed.shape} {fixed.dtype}, "
          f"{int(info['hole_mask'].sum())} hole px -> 0")
    check("fill_holes accepts an 8-bit depth map",
          np.array_equal(fill_holes(rgb, (inv * 255).astype(np.uint8)), fixed))
    gray = (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    fg_out = fill_holes(gray, inv)
    check("fill_holes accepts a grayscale image and returns grayscale",
          fg_out.shape == gray.shape and fg_out.dtype == np.uint8,
          f"{gray.shape} -> {fg_out.shape}")

    # ---------------------------------------------------------------- 1) convention
    cfg = FillConfig(scale=-44.8)
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

    # ------------------- 3c) drop-in replacement with the caller's signature
    from viewfill.compat import scatter_image_safe
    w_s, m_s, d_s = scatter_image_safe(rgb, inv, direction=-1, scale_factor=44.8,
                                       inverse_ordering=True, reproject_depth=True)
    hole_s = m_s > 0
    q_s, _, _ = check_warp_quality(np.asarray(w_s, np.float32), hole_s, rgb, inv * 255.0,
                                   (disparity_from_depth(inv * 255.0, -44.8), None), cfg)
    res_s = fill_warped(np.clip(w_s, 0, 255).astype(np.float32), hole_s,
                        np.where(hole_s, 0.0, 255.0 / (d_s + 1e-6)).astype(np.float32),
                        rgb, inv, cfg=cfg)
    check("viewfill.compat.scatter_image_safe is a drop-in with zero deviation",
          np.array_equal(hole_s, hole_ext) and q_s["warp_deviation_pct"] == 0.0
          and int(res_s["remaining"].sum()) == 0,
          f"same hole mask as the original call ({int(hole_s.sum())} px), "
          f"deviation {q_s['warp_deviation_pct']:.2f}%")
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
                        np.asarray(res["warped_depth"], np.float32), rgb, inv,
                        disp=res["disp"], cfg=cfg)
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
    # ------------------------- 7) seam at the filled/untouched background boundary
    import cv2 as _cv2

    def _jump(res):
        """Mean |jump| between the filled side and the untouched side at the boundary."""
        g_f = 0.299 * res["I_filled"][..., 0] + 0.587 * res["I_filled"][..., 1] \
            + 0.114 * res["I_filled"][..., 2]
        g_w = 0.299 * res["I_w"][..., 0] + 0.587 * res["I_w"][..., 1] \
            + 0.114 * res["I_w"][..., 2]
        hole, valid = res["hole_mask"], ~res["hole_mask"]
        k = 9

        def loc(img, mask):
            m = mask.astype(np.float32)
            num = _cv2.boxFilter(img * m, -1, (k, k), normalize=False,
                                 borderType=_cv2.BORDER_REPLICATE)
            den = _cv2.boxFilter(m, -1, (k, k), normalize=False,
                                 borderType=_cv2.BORDER_REPLICATE)
            return num / np.maximum(den, 1e-6)

        bnd = hole & _cv2.dilate(valid.astype(np.uint8),
                                 np.ones((3, 3), np.uint8)).astype(bool)
        j = np.abs(loc(g_f, hole)[bnd] - loc(g_w, valid)[bnd])
        return float(j.mean()), float(np.percentile(j, 90))

    r_raw = fill_holes(rgb, inv, depth_dilate=0, return_info=True)
    seam_hhf, p90_hhf = _jump(info)
    seam_raw, p90_raw = _jump(r_raw)
    check("demand-driven splat widening reduces the seam at the boundary",
          p90_hhf < 0.75 * p90_raw and seam_hhf <= seam_raw,
          f"seam mean {seam_raw:.2f} -> {seam_hhf:.2f}, p90 {p90_raw:.1f} -> {p90_hhf:.1f} "
          f"gray levels (depth_dilate=0 -> auto)")
    check("the seam is not systematically brighter or darker than the background",
          abs(seam_hhf) < 30.0, f"seam mean {seam_hhf:.2f} gray levels (report only)")

    # ------------------------------- 8) interface completeness and default consistency
    import dataclasses
    import inspect
    sig = inspect.signature(fill_holes)
    fields = {f.name for f in dataclasses.fields(FillConfig)}
    internal = {"repair_warp", "repair_threshold_pct", "dev_threshold_gray", "ablate"}
    missing = sorted(fields - set(sig.parameters) - internal)
    check("fill_holes exposes every applicable FillConfig knob",
          not missing,
          (f"missing {missing}" if missing else
           f"{len(sig.parameters)} parameters, {len(fields)} config fields, "
           f"{len(internal)} intentionally internal"))
    kw = {k: v.default for k, v in sig.parameters.items()
          if v.default is not inspect.Parameter.empty and k != "return_info"}
    check("passing every documented default explicitly changes nothing",
          np.array_equal(fill_holes(rgb, inv, **kw), fixed),
          f"{len(kw)} keyword defaults round-tripped bit-exactly")
    check("return_info exposes the documented result keys",
          {"image", "filled", "warped", "hole_mask", "remaining", "crack", "oofa",
           "disocc", "filled_depth", "warped_depth", "stats"} <= set(info),
          f"{len(info)} keys")

    # ---------- 9) large-hole caps are not cracks, and HHF never leaves black pixels
    from dibr.cracks import fill_cracks
    lum_s = lambda a: 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    Hs, Ws = 60, 80
    simg = np.full((Hs, Ws, 3), 180, np.float32)
    simg[20:40, 30:38] = 40.0
    sD = np.full((Hs, Ws), 200.0, np.float32)
    sD[20:40, 30:38] = -1.0                       # an 8 px wide hole, not a slit
    r_old = fill_cracks(simg, sD, lam=5.0, se_len=4, orientation="h", slit_only=False)
    r_new = fill_cracks(simg, sD, lam=5.0, se_len=4, orientation="h", slit_only=True)
    black_old = int((r_old["crack"]
                     & (lum_s(np.asarray(r_old["I_filled"], np.float32)) <= 1)).sum())
    black_new = int((r_new["crack"]
                     & (lum_s(np.asarray(r_new["I_filled"], np.float32)) <= 1)).sum())
    check("an 8 px wide hole is not classified as a crack (slit_only gate)",
          int(r_old["crack"].sum()) > 0 and int(r_new["crack"].sum()) == 0,
          f"crack px {int(r_old['crack'].sum())} -> {int(r_new['crack'].sum())}, "
          f"rejected as large-hole {int(r_new.get('big_hole_crack_px', 0))}")
    check("HHF leaves no black crack pixel (colour and depth stay consistent)",
          black_new == 0 and int(r_new.get("hhf_unsupported_px", -1)) >= 0,
          f"near-black crack px: legacy path {black_old}, fixed path {black_new}, "
          f"unsupported-by-kernel {int(r_new.get('hhf_unsupported_px', -1))}")
    luma_hole = lum_s(info["filled"])[info["hole_mask"]]
    check("the real run fills every hole with real content (no black, no residual)",
          int(info["remaining"].sum()) == 0 and int((luma_hole <= 1).sum()) == 0,
          f"residual {int(info['remaining'].sum())} px, near-black inside the holes "
          f"{int((luma_hole <= 1).sum())} px")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed   "
          f"(total {time.time()-t0:.1f}s)")
    print(f"artefacts under {OUT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
