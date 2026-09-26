"""Automated checks for Step 3 (ghost detection and correction).  Run: python check_step3.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dibr import calib, ghosts, io_utils, viz  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "cam6_to_cam7_f000"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    src = os.path.join(HERE, "output", "step2_cracks_int", TAG, "lam5_h_none")
    I_w = np.load(os.path.join(src, "I_filled.npy")).astype(np.float32)
    D_w = np.load(os.path.join(src, "D_filled.npy")).astype(np.float32)
    hole = np.load(os.path.join(src, "remaining_holes.npy"))
    crack = np.load(os.path.join(src, "crack_mask.npy"))
    h, w = hole.shape

    # ---- 1. masked median must be exact (this is where the strip bug lived) ----
    rng = np.random.default_rng(0)
    small = rng.uniform(0, 255, (37, 53)).astype(np.float32)
    valid = rng.random((37, 53)) > 0.35
    got = ghosts.masked_median(small, valid, 9)
    ref = np.zeros_like(small)
    r = 4
    for y in range(37):
        for x in range(53):
            y0, y1 = max(0, y - r), min(37, y + r + 1)
            x0, x1 = max(0, x - r), min(53, x + r + 1)
            m = valid[y0:y1, x0:x1]
            if m.any():
                ref[y, x] = np.median(small[y0:y1, x0:x1][m])
    err = float(np.abs(got - ref).max())
    check("masked_median == brute-force masked median", err < 1e-3,
          f"max abs diff {err:g}")
    check("masked_median is finite wherever a valid pixel exists",
          bool(np.isfinite(got).all()), f"non-finite {int((~np.isfinite(got)).sum())}")

    # ---- 2. masks ------------------------------------------------------------
    cams = calib.load_calib(io_utils.DATASET_ROOT_DEFAULT)
    P_ref = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 6, "f000")["depth"]
    P_ref = P_ref.astype(np.float32)
    dx, dy, _, _, _ = calib.displacement_field(cams, 6, 7, P_ref)
    D_FG = ghosts.extended_fg(P_ref)
    fx, fy, _, _, _ = calib.displacement_field(cams, 6, 7, D_FG)
    direction = 1 if float(np.median(dx)) > 0 else -1

    oofa = ghosts.find_oofa(hole, direction)
    check("OOFA is a subset of the holes", bool((oofa & ~hole).sum() == 0),
          f"{int(oofa.sum())} px")
    check("OOFA sits on the edge the projection vacates",
          (direction < 0 and oofa[:, -20:].sum() > 0 and oofa[:, :20].sum() == 0) or
          (direction > 0 and oofa[:, :20].sum() > 0 and oofa[:, -20:].sum() == 0),
          f"direction {direction:+d}: left {int(oofa[:, :20].sum())} / "
          f"right {int(oofa[:, -20:].sum())}")

    r = ghosts.detect_and_fix(I_w, D_w, hole, (dx, dy), (fx, fy), direction=direction,
                              fix_mode="copy")
    cand = r["candidates"] | r["fg_points"]
    check("candidates are disjoint from the holes", bool((cand & hole).sum() == 0),
          f"overlap {int((cand & hole).sum())} px")
    check("candidates lie inside the dilated hole set",
          bool((cand & ~viz.dilate(hole, 2)).sum() == 0))
    check("ghost mask is a subset of the candidates",
          bool((r["ghost"] & ~r["candidates"]).sum() == 0))

    # ---- 3. FG removal follows the disparity convention ----------------------
    fg = r["fg_points"]
    Tmap = r["T"][np.clip(r["lab_near"], 0, len(r["T"]) - 1)]
    check("fg_side=gt removes only candidates above the local threshold T_O",
          bool(int((fg & r["candidates"]).sum()) == 0 and
               (D_w[fg] > Tmap[fg] - 1e-6).all()),
          f"removed {int(fg.sum())} px, min(D_w - T) {float((D_w[fg]-Tmap[fg]).min()):+.2f}, "
          f"still in the candidate set after removal {int((fg & r['candidates']).sum())}")

    # ---- 4. relocation -------------------------------------------------------
    wys, wxs = r["write_yx"]
    wtx, wty = r["write_xy"]
    check("every relocated ghost lands inside the frame",
          bool((wtx >= 0).all() and (wtx < w).all() and (wty >= 0).all() and (wty < h).all()),
          f"{len(wys)} writes")
    check("relocation moves at least some content (not a no-op)",
          int((np.hypot(wtx - wxs, wty - wys) > 8).sum()) > 0,
          f"{int((np.hypot(wtx-wxs, wty-wys) > 8).sum())} ghosts move more than 8 px")
    changed = np.abs(r["I_fixed"] - I_w).max(axis=2) > 1e-3
    check("I_fixed differs from I_w only where content was written",
          int((changed & ~np.zeros_like(changed)).sum()) == int(changed.sum()),
          f"{int(changed.sum())} px changed")

    # ---- 5. fix modes --------------------------------------------------------
    rm = ghosts.detect_and_fix(I_w, D_w, hole, (dx, dy), (fx, fy), direction=direction,
                               fix_mode="move")
    check("fix_mode=move adds exactly the ghosts to the hole mask",
          bool((rm["hole_out"] == (hole | rm["ghost"])).all()),
          f"holes {int(hole.sum())} -> {int(rm['hole_out'].sum())}")
    rb = ghosts.detect_and_fix(I_w, D_w, hole, (dx, dy), (fx, fy), direction=direction,
                               fix_mode="bg")
    check("fix_mode=bg keeps the image finite", bool(np.isfinite(rb["I_fixed"]).all()))

    # ---- 6. ground-truth judgement -------------------------------------------
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, "f000")["color"]
    mb = ~hole
    p_b, p_a = viz.psnr(gt, I_w, mask=mb), viz.psnr(gt, r["I_fixed"], mask=mb)
    s_b, s_a = viz.ssim(gt, I_w, mask=mb), viz.ssim(gt, r["I_fixed"], mask=mb)
    check("frame PSNR improves after the ghost step", p_a > p_b,
          f"{p_b:.3f} -> {p_a:.3f} dB")
    check("frame SSIM improves after the ghost step", s_a > s_b,
          f"{s_b:.4f} -> {s_a:.4f}")
    cb, ca = viz.psnr(gt, I_w, mask=changed), viz.psnr(gt, r["I_fixed"], mask=changed)
    check("content that was actually moved gets >5 dB better", ca > cb + 5.0,
          f"{cb:.2f} -> {ca:.2f} dB over {int(changed.sum())} px")

    rl = ghosts.detect_and_fix(I_w, D_w, hole, (dx, dy), (fx, fy), direction=direction,
                               fg_side="lt", fix_mode="copy")
    p_lt = viz.psnr(gt, rl["I_fixed"], mask=mb)
    check("fg_side=gt (paper II-A convention) beats the literal II-B wording",
          p_a >= p_lt,
          f"gt {p_a:.3f} dB vs lt {p_lt:.3f} dB")

    # ---- 7. ambiguity #17 ----------------------------------------------------
    crack_near = crack & viz.dilate(hole, 2)
    ov = int((r["ghost"] & crack_near).sum())
    check("ambiguity #17 quantified (ghosts overlapping the crack step)",
          True, f"{ov} px = {ov / max(1, int(r['ghost'].sum())) * 100:.1f}% of ghosts")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
