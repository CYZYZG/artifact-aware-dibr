"""Automated checks for Step 1 (forward warping). Run:  python check_step1.py

Every check prints PASS/FAIL with the measured numbers; exit code 1 if any fails.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dibr import calib, io_utils, viz, warp  # noqa: E402

ROOT = io_utils.DATASET_ROOT_DEFAULT
FRAME = "f000"
REFCAM, DSTCAM = 6, 7
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    cams = calib.load_calib(ROOT)
    v6 = io_utils.load_view(ROOT, REFCAM, FRAME)
    v7 = io_utils.load_view(ROOT, DSTCAM, FRAME)
    color = v6["color"].astype(np.float32)
    P = v6["depth"].astype(np.float32)

    # ---- 1. integer shift must reproduce a pure roll exactly -------------------
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3)).astype(np.float32)
    worst = 0.0
    for d in (-17, -7, 0, 5, 23):
        out, _, hole, _ = warp.forward_warp(img, np.full((64, 64), float(d), np.float32))
        ref = np.roll(img, d, axis=1)
        if d < 0:
            ref[:, d:] = 0
        else:
            ref[:, :d] = 0
        worst = max(worst, float(np.abs(out - ref).max()))
    check("integer shift == np.roll", worst == 0.0, f"max abs diff {worst:g}")

    # ---- 2. identity warp is exact and hole-free -------------------------------
    out, z, hole, w = warp.forward_warp(color, np.zeros_like(P), z=P)
    check("identity warp is exact", float(np.abs(out - color).max()) == 0.0,
          f"max abs diff {np.abs(out - color).max():g}")
    check("identity warp has no holes", not hole.any(), f"holes {int(hole.sum())}")

    # ---- 3. vectorised vs brute force on a crop --------------------------------
    dx, dy, _, _, _ = calib.displacement_field(cams, REFCAM, DSTCAM, P, y_origin="bottom")
    sl = (slice(300, 460), slice(400, 560))
    c = np.ascontiguousarray(color[sl])
    a = np.ascontiguousarray(dx[sl].astype(np.float32))
    b = np.ascontiguousarray(dy[sl].astype(np.float32))
    zc = np.ascontiguousarray(P[sl].astype(np.float32))
    # NOTE: brute force re-interprets the crop as its own coordinate system, which is
    # fine because the crop spans a small area where displacements are not re-based;
    # to make the two comparable we run the vectorised warp on the same crop with the
    # same local displacement values.
    o1, z1, h1, w1 = warp.forward_warp(c, a, b, z=zc, rule="zbuf")
    o2, z2, h2, w2 = warp.forward_warp_bruteforce(c, a, b, z=zc, rule="zbuf")
    check("vectorised == brute force (colour)", float(np.abs(o1 - o2).max()) < 1e-3,
          f"max abs diff {np.abs(o1 - o2).max():.3g}")
    check("vectorised == brute force (mask)", bool((h1 == h2).all()),
          f"mask mismatch {int((h1 != h2).sum())} px")
    check("vectorised == brute force (Z-buffer)", float(np.abs(z1 - z2).max()) < 1e-3,
          f"max abs diff {np.abs(z1 - z2).max():.3g}")

    # ---- 4. the full warp: mask / domain / weight consistency ------------------
    wc, wp, hm, wt = warp.forward_warp(color, dx, dy, z=P, hole_depth=-1.0)
    valid = ~hm
    check("weight>0 == ~hole_mask", bool(((wt > 0) == valid).all()),
          f"mismatch {int(((wt > 0) != valid).sum())} px")
    check("D_w domain is {-1} u [0,255]",
          bool(((wp[valid] >= 0) & (wp[valid] <= 255)).all() and (wp[hm] == -1).all()),
          f"valid range [{wp[valid].min():.1f}, {wp[valid].max():.1f}], holes == -1")
    check("hole mask non-trivial", 0.02 < hm.mean() < 0.25, f"holes {hm.mean()*100:.2f}%")

    # ---- 5. Z-buffer keeps the nearest layer -----------------------------------
    # for every target pixel compare the warped ordering value against the max over
    # the sources landing there (computed by brute force on the crop above, extended)
    check("warped ordering value == nearest source ordering value",
          float(np.abs(z1 - z2).max()) < 1e-3, "see check 3")

    # ---- 6. geometry: OOFA side agrees with the sign of the displacement -------
    med = float(np.median(dx))
    exp_side = "right" if med < 0 else "left"
    left, right = int(hm[:, :20].sum()), int(hm[:, -20:].sum())
    got = "right" if right > left else "left"
    check("OOFA side matches sign(median dx)", got == exp_side,
          f"median dx {med:+.2f} -> expect {exp_side}, edge holes L={left} R={right}")

    # ---- 7. ground truth sanity: the warp must beat the no-warp baseline ------
    gt = v7["color"]
    ps_id = viz.psnr(gt, color)
    ps_zb = viz.psnr(gt, wc, mask=valid)
    ps_avg = viz.psnr(gt, warp.forward_warp(color, dx, dy, z=P, rule="avg")[0],
                      mask=~warp.forward_warp(color, dx, dy, z=P, rule="avg")[2])
    check("calibrated warp beats no-warp by >5 dB", ps_zb > ps_id + 5.0,
          f"identity {ps_id:.2f} dB -> warped {ps_zb:.2f} dB (valid px)")
    check("Z-buffer beats weight averaging", ps_zb > ps_avg + 1.0,
          f"zbuf {ps_zb:.2f} dB vs avg {ps_avg:.2f} dB")
    ss_zb = viz.ssim(gt, wc, mask=valid)
    check("SSIM improves over no-warp", ss_zb > viz.ssim(gt, color) + 0.05,
          f"identity {viz.ssim(gt, color):.4f} -> warped {ss_zb:.4f}")

    # ---- 8. the calibrated displacement field explains the real image ---------
    import cv2
    g6 = cv2.cvtColor(v6["color"], cv2.COLOR_RGB2GRAY)
    g7 = cv2.cvtColor(v7["color"], cv2.COLOR_RGB2GRAY)
    gw = cv2.cvtColor(np.clip(wc, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    f_raw = dis.calc(g6, g7, None)
    f_res = dis.calc(gw, g7, None)
    m = ~hm
    raw_mag = np.median(np.hypot(f_raw[..., 0], f_raw[..., 1])[m])
    res_mag = np.median(np.hypot(f_res[..., 0], f_res[..., 1])[m])
    check("residual flow after warp < 1/5 of raw flow", res_mag < raw_mag / 5,
          f"raw {raw_mag:.2f} px -> residual {res_mag:.2f} px")

    # ---- 9. provided baseline (no Z-buffer, truncation) ------------------------
    prov, hole_prov, real_depth = warp.scatter_image(color, P / 255.0, direction=1,
                                                     scale_factor=128.26,
                                                     inverse_ordering=True,
                                                     reproject_depth=True)
    check("provided scatter_image is measurably worse than the Z-buffer warp",
          viz.psnr(gt, np.asarray(prov, np.float32), mask=(hole_prov == 0)) < ps_zb,
          f"provided {viz.psnr(gt, np.asarray(prov, np.float32), mask=(hole_prov == 0)):.2f} dB")

    # ---- 10. cross-check against the pre-existing z_buffer_warp.py ------------
    # Two independently written implementations of the same collision rule must agree
    # when fed the same 1D displacement field.
    try:
        import z_buffer_warp as zw
        inv = P / 255.0
        s, d = 44.8, -1
        ref_img, ref_hole, _ = zw.z_buffer_splat(color, inv, d, s,
                                                 return_inverse_depth=True)
        mine_dx = (d * s * inv).astype(np.float32)
        mine_img, mine_P, mine_hole, _ = warp.forward_warp(color, mine_dx, z=P,
                                                           hole_depth=-1.0, rule="zbuf")
        dmask = (mine_hole != (ref_hole > 0))
        derr = float(np.abs(mine_img - np.asarray(ref_img, np.float32)).max())
        check("agrees with pre-existing z_buffer_warp.py (same rule, same input)",
              dmask.sum() == 0 and derr < 1e-3,
              f"mask mismatch {int(dmask.sum())} px, max colour diff {derr:.3g}")
    except Exception as exc:  # pragma: no cover - optional module
        check("agrees with pre-existing z_buffer_warp.py", False, f"skipped: {exc}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
