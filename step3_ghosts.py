"""Step 3 - ghost detection and correction (paper II-B), with an overlap report against
the Step 2 crack mask (paper ambiguity #17) and a ground-truth judgement.

    python step3_ghosts.py --ref_cam 6 --dst_cam 7 --frame f000
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dibr import calib, cracks, ghosts, io_utils, viz, warp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser(description="Step 3: ghost detection and correction")
    ap.add_argument("--ref_cam", type=int, default=6)
    ap.add_argument("--dst_cam", type=int, default=7)
    ap.add_argument("--frame", default="f000")
    ap.add_argument("--dataset_root", default=io_utils.DATASET_ROOT_DEFAULT)
    ap.add_argument("--step2_dir", default=os.path.join(HERE, "output", "step2_cracks_int"))
    ap.add_argument("--lam_subdir", default="lam5_h_none")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "output", "step3_ghosts"))
    ap.add_argument("--band_radius", type=int, default=2)
    ap.add_argument("--alpha_trim", type=float, default=0.10)
    ap.add_argument("--alpha_sim", type=float, default=11.0)
    ap.add_argument("--fg_side", choices=["gt", "lt"], default="gt",
                    help="gt: remove candidates with disparity ABOVE T (larger=nearer=FG, "
                         "follows paper II-A); lt: the literal wording of II-B")
    ap.add_argument("--ksize", type=int, default=9)
    ap.add_argument("--fix_mode", choices=["copy", "move", "bg"], default="copy")
    ap.add_argument("--splat", choices=["sub", "floor", "round"], default="floor")
    ap.add_argument("--no_gt", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    tag = f"cam{args.ref_cam}_to_cam{args.dst_cam}_{args.frame}"
    src = os.path.join(args.step2_dir, tag, args.lam_subdir)
    I_w = np.load(os.path.join(src, "I_filled.npy")).astype(np.float32)
    D_w = np.load(os.path.join(src, "D_filled.npy")).astype(np.float32)
    hole = np.load(os.path.join(src, "remaining_holes.npy"))
    crack_mask = np.load(os.path.join(src, "crack_mask.npy"))
    h, w = hole.shape
    out = os.path.join(args.out_dir, tag,
                       f"band{args.band_radius}_{args.fg_side}_a{args.alpha_sim:g}_{args.fix_mode}")
    os.makedirs(out, exist_ok=True)

    # ---------------- displacement fields ----------------
    cams = calib.load_calib(args.dataset_root)
    view = io_utils.load_view(args.dataset_root, args.ref_cam, args.frame)
    P_ref = view["depth"].astype(np.float32)
    dx, dy, _, _, _ = calib.displacement_field(cams, args.ref_cam, args.dst_cam, P_ref,
                                               y_origin="bottom")
    D_FG = ghosts.extended_fg(P_ref)
    fx, fy, _, _, _ = calib.displacement_field(cams, args.ref_cam, args.dst_cam, D_FG,
                                               y_origin="bottom")
    direction = 1 if float(np.median(dx)) > 0 else -1

    # ---------------- ghost detection + correction ----------------
    r = ghosts.detect_and_fix(I_w, D_w, hole, (dx, dy), (fx, fy), direction=direction,
                              band_radius=args.band_radius, alpha_trim=args.alpha_trim,
                              alpha_sim=args.alpha_sim, fg_side=args.fg_side,
                              ksize=args.ksize, fix_mode=args.fix_mode)

    ghost = r["ghost"]
    # ---------------- ambiguity #17: overlap with the crack step ----------------
    crack_touch = crack_mask & viz.dilate(hole, 2)

    stats = dict(
        frame=args.frame, ref_cam=args.ref_cam, dst_cam=args.dst_cam,
        band_radius=args.band_radius, alpha_trim=args.alpha_trim,
        alpha_sim=args.alpha_sim, fg_side=args.fg_side, ksize=args.ksize,
        fix_mode=args.fix_mode, direction=direction,
        hole_px=int(hole.sum()), oofa_px=int(r["oofa"].sum()),
        oofa_side=("right" if r["oofa"][:, -20:].sum() > r["oofa"][:, :20].sum() else "left"),
        candidates_px=int(r["candidates"].sum() + r["fg_points"].sum()),
        fg_removed_px=int(r["fg_points"].sum()),
        candidates_after_fg_px=int(r["candidates"].sum()),
        ghost_px=int(ghost.sum()),
        ghost_pct_of_candidates=round(float(ghost.sum() / max(1, r["candidates"].sum()) * 100), 2),
        ghost_pct_of_frame=round(float(ghost.mean() * 100), 4),
        step2_crack_px=int(crack_mask.sum()),
        step2_crack_near_holes_px=int(crack_touch.sum()),
        ghost_overlap_step2_crack_px=int((ghost & crack_mask).sum()),
        ghost_overlap_step2_crack_near_holes_px=int((ghost & crack_touch).sum()),
        ghost_overlap_pct_of_ghost=round(
            float((ghost & crack_touch).sum() / max(1, ghost.sum()) * 100), 2),
        p_fg_inside_px=int(np.sum((r["p_fg_xy"][0] >= 0) & (r["p_fg_xy"][0] < w) &
                                 (r["p_fg_xy"][1] >= 0) & (r["p_fg_xy"][1] < h))),
        p_fg_on_valid_px=int(np.sum(~hole[np.clip(np.rint(r["p_fg_xy"][1]).astype(int), 0, h - 1),
                                          np.clip(np.rint(r["p_fg_xy"][0]).astype(int), 0, w - 1)])),
    )

    # ---------------- ground-truth judgement ----------------
    gt = None
    if not args.no_gt:
        try:
            gt = io_utils.load_view(args.dataset_root, args.dst_cam, args.frame)["color"]
        except FileNotFoundError:
            gt = None
    if gt is not None:
        for name, m in (("ghost", ghost), ("candidates", r["candidates"]),
                        ("band", viz.dilate(hole, args.band_radius) & ~hole)):
            if m.sum() == 0:
                continue
            stats[f"gt_psnr_{name}_before"] = round(viz.psnr(gt, I_w, mask=m), 3)
            stats[f"gt_psnr_{name}_after"] = round(viz.psnr(gt, r["I_fixed"], mask=m), 3)
        mb = ~hole
        stats["gt_psnr_frame_before"] = round(viz.psnr(gt, I_w, mask=mb), 3)
        stats["gt_psnr_frame_after"] = round(viz.psnr(gt, r["I_fixed"], mask=mb), 3)
        stats["gt_ssim_frame_before"] = round(viz.ssim(gt, I_w, mask=mb), 5)
        stats["gt_ssim_frame_after"] = round(viz.ssim(gt, r["I_fixed"], mask=mb), 5)
        # how much did the correction change the frame at all
        changed = np.abs(r["I_fixed"] - I_w).max(axis=2) > 1e-3
        stats["changed_px"] = int(changed.sum())
        if changed.sum():
            stats["gt_psnr_changed_before"] = round(viz.psnr(gt, I_w, mask=changed), 3)
            stats["gt_psnr_changed_after"] = round(viz.psnr(gt, r["I_fixed"], mask=changed), 3)

    with open(os.path.join(out, "stats.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(stats.keys()))
        wr.writeheader()
        wr.writerow(stats)

    # ---------------- artefacts ----------------
    vis = viz.to_u8(I_w).copy()
    vis[viz.dilate(r["candidates"])] = (60, 60, 200)      # candidates: dark red
    vis[viz.dilate(ghost)] = (255, 40, 40)                # ghosts: red
    io_utils.imwrite(os.path.join(out, "ghost_overlay.png"), vis)
    io_utils.imwrite(os.path.join(out, "ghost_mask.png"), (ghost * 255).astype(np.uint8))
    io_utils.imwrite(os.path.join(out, "candidate_mask.png"),
                     (r["candidates"] * 255).astype(np.uint8))
    io_utils.imwrite(os.path.join(out, "fg_removed_mask.png"),
                     (r["fg_points"] * 255).astype(np.uint8))
    io_utils.imwrite(os.path.join(out, "oofa_mask.png"), (r["oofa"] * 255).astype(np.uint8))
    io_utils.imwrite(os.path.join(out, "fixed_color.png"), viz.to_u8(r["I_fixed"]))
    viz.imwrite_gray(os.path.join(out, "mu_bg.png"),
                     _scatter_field(ghost.shape, r["cand_yx"], r["cand_mu_bg"]))
    viz.imwrite_gray(os.path.join(out, "d_bg.png"),
                     _scatter_field(ghost.shape, r["cand_yx"], r["d_bg"]), vmin=0, vmax=60)
    viz.imwrite_gray(os.path.join(out, "d_fg.png"),
                     _scatter_field(ghost.shape, r["cand_yx"], r["d_fg"]), vmin=0, vmax=60)

    import cv2
    tiles = [
        viz.label(viz.to_u8(I_w), "1) I_w after the crack step"),
        viz.label(vis, "2) candidates(dark red) + ghosts(red)"),
        viz.label(viz.to_u8(r["I_fixed"]), f"3) I after ghost fix ({args.fix_mode})"),
        viz.label(viz.colorize(viz.to_u8(np.abs(r["I_fixed"] - I_w)).max(axis=2),
                               vmin=0, vmax=60), "4) |change|"),
        viz.label((r["oofa"] * 255).astype(np.uint8)[:, :, None].repeat(3, 2),
                  f"5) OOFA ({stats['oofa_side']})"),
        viz.label((r["candidates"] * 255).astype(np.uint8)[:, :, None].repeat(3, 2),
                  f"6) candidates kept after FG removal ({int(r['candidates'].sum())})"),
        viz.label((r["fg_points"] * 255).astype(np.uint8)[:, :, None].repeat(3, 2),
                  "7) FG points removed from candidates"),
        viz.label(viz.colorize(_scatter_field(ghost.shape, r["cand_yx"], r["d_bg"]),
                               vmin=0, vmax=60), "8) d_BG"),
    ]
    ys, xs = np.nonzero(viz.dilate(ghost, 2))
    if ys.size:
        n_lab, _, ccs, cents = cv2.connectedComponentsWithStats(
            viz.dilate(ghost, 2).astype(np.uint8), 8)
        k = 1 + int(np.argmax(ccs[1:, cv2.CC_STAT_AREA]))
        cy, cx = int(cents[k][1]), int(cents[k][0])
        tiles += [viz.label(viz.zoom(viz.to_u8(I_w), cx, cy, 130), "9) zoom before"),
                  viz.label(viz.zoom(viz.to_u8(r["I_fixed"]), cx, cy, 130), "10) zoom after"),
                  viz.label(viz.zoom(vis, cx, cy, 130), "11) zoom masks")]
    panel = viz.grid(tiles, cols=4)
    lines = [
        f"Step 3 ghosts | cam{args.ref_cam}->cam{args.dst_cam} {args.frame} | "
        f"band r={args.band_radius} alpha_sim={args.alpha_sim:g} fg_side={args.fg_side} "
        f"fix={args.fix_mode}",
        f"holes {stats['hole_px']} | OOFA {stats['oofa_px']} ({stats['oofa_side']}) | "
        f"candidates {stats['candidates_px']} -> {stats['candidates_after_fg_px']} "
        f"after removing {stats['fg_removed_px']} FG px",
        f"ghosts {stats['ghost_px']} px ({stats['ghost_pct_of_frame']:.3f}% of frame, "
        f"{stats['ghost_pct_of_candidates']:.1f}% of candidates)",
        f"p_FG inside the frame for {stats['p_fg_inside_px']} ghosts, on valid content for "
        f"{stats['p_fg_on_valid_px']}",
        "",
        "ambiguity #17 (crack step vs ghost candidates):",
        f"  Step 2 cracks near holes: {stats['step2_crack_near_holes_px']} px; "
        f"ghosts overlapping them: {stats['ghost_overlap_step2_crack_near_holes_px']} px "
        f"({stats['ghost_overlap_pct_of_ghost']:.1f}% of all ghosts)",
    ]
    if gt is not None:
        lines += ["", "ground-truth judgement (vs the real target camera):",
                  f"  ghost px      PSNR {stats['gt_psnr_ghost_before']:.2f} -> "
                  f"{stats['gt_psnr_ghost_after']:.2f} dB",
                  f"  band px        PSNR {stats.get('gt_psnr_band_before', float('nan')):.2f} -> "
                  f"{stats.get('gt_psnr_band_after', float('nan')):.2f} dB",
                  f"  changed px     PSNR {stats.get('gt_psnr_changed_before', float('nan')):.2f} -> "
                  f"{stats.get('gt_psnr_changed_after', float('nan')):.2f} dB",
                  f"  whole frame    PSNR {stats['gt_psnr_frame_before']:.2f} -> "
                  f"{stats['gt_psnr_frame_after']:.2f} dB   SSIM "
                  f"{stats['gt_ssim_frame_before']:.4f} -> {stats['gt_ssim_frame_after']:.4f}"]
    card = viz.text_card(lines, width=1000)
    hh = max(panel.shape[0], card.shape[0])
    p2 = np.full((hh, panel.shape[1], 3), 12, np.uint8)
    p2[:panel.shape[0]] = panel
    c2 = np.full((hh, card.shape[1], 3), 20, np.uint8)
    c2[:card.shape[0]] = card
    io_utils.imwrite(os.path.join(out, "panel.png"),
                     np.hstack([p2, np.full((hh, 8, 3), 12, np.uint8), c2]))
    np.save(os.path.join(out, "ghost_mask.npy"), ghost)
    np.save(os.path.join(out, "I_fixed.npy"), r["I_fixed"].astype(np.float32))
    np.save(os.path.join(out, "D_fixed.npy"), r["D_fixed"].astype(np.float32))
    np.save(os.path.join(out, "hole_out.npy"), r["hole_out"])

    print(f"--- Step 3 ghosts | cam{args.ref_cam}->cam{args.dst_cam} {args.frame} | "
          f"fg_side={args.fg_side} fix={args.fix_mode} ---")
    for k, v in stats.items():
        print(f"  {k:38s} {v}")
    print(f"artefacts: {out}")
    return 0


def _scatter_field(shape, yx, vals):
    f = np.zeros(shape, np.float32)
    f[yx[0], yx[1]] = vals
    return f


if __name__ == "__main__":
    raise SystemExit(main())
