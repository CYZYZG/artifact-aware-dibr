"""Step 1 - forward 3D warping (DIBR) of one image pair: I_w, D_w and the hole mask.

    # paper model: 1D disparity D = s * (P/255), direction = +-1
    python step1_warp.py --ref_cam 6 --dst_cam 7 --frame f000 --mode disp --fit_from_calib

    # exact calibrated projection (also produces dy); comparable against the real
    # camera image, which gives a ground-truth PSNR for this step
    python step1_warp.py --ref_cam 6 --dst_cam 7 --frame f000 --mode calib
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dibr import calib, io_utils, viz, warp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser(description="Step 1: forward warping / hole generation")
    ap.add_argument("--dataset_root", default=io_utils.DATASET_ROOT_DEFAULT)
    ap.add_argument("--input_dir", default=os.path.join(HERE, "input"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "output", "step1_warp"))
    ap.add_argument("--ref_cam", type=int, default=6)
    ap.add_argument("--dst_cam", type=int, default=7)
    ap.add_argument("--frame", default="f000")
    ap.add_argument("--mode", choices=["disp", "calib", "identity", "provided"], default="calib",
                    help="disp = 1D disparity model (paper); calib = exact projection; "
                         "identity = dx=0 baseline; provided = warping.scatter_image")
    ap.add_argument("--scale", type=float, default=44.8,
                    help="disparity scale s in px for --mode disp")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="constant disparity offset c in px for --mode disp")
    ap.add_argument("--direction", type=int, default=-1, choices=[-1, 1])
    ap.add_argument("--fit_from_calib", action="store_true",
                    help="derive s, c and direction from the calibrated displacement field")
    ap.add_argument("--use_dv", action="store_true", default=True,
                    help="calib mode: include the vertical displacement (default on)")
    ap.add_argument("--no_dv", dest="use_dv", action="store_false")
    ap.add_argument("--y_origin", default="bottom", choices=["bottom", "top"])
    ap.add_argument("--rule", choices=["zbuf", "avg"], default="zbuf",
                    help="zbuf = nearest source layer wins (default); "
                         "avg = accumulate every sample (behaviour of warping.scatter_image)")
    ap.add_argument("--splat", choices=["sub", "floor", "round"], default="sub",
                    help="sub = sub-pixel 2-tap splat (our default); "
                         "floor/round = integer single-tap splat, which is what produces "
                         "the classical 1-2 px crack artifact (paper-faithful baseline)")
    ap.add_argument("--no_gt", action="store_true", help="skip the ground-truth comparison")
    return ap.parse_args()


def main():
    args = parse_args()
    cams = calib.load_calib(args.dataset_root)
    try:
        view = io_utils.load_view(args.dataset_root, args.ref_cam, args.frame)
        root = args.dataset_root
    except FileNotFoundError:
        view = io_utils.load_view(args.input_dir, args.ref_cam, args.frame)
        root = args.input_dir
    color = view["color"].astype(np.float32)
    P = view["depth"].astype(np.float32)
    h, w = P.shape
    out = os.path.join(args.out_dir, f"cam{args.ref_cam}_to_cam{args.dst_cam}_{args.frame}")
    os.makedirs(out, exist_ok=True)

    # ---------------- displacement field ----------------
    fit_info = None
    dx = dy = None
    if args.mode in ("calib", "disp") and args.fit_from_calib:
        f = calib.displacement_field(cams, args.ref_cam, args.dst_cam, P, args.y_origin)
        d = P / 255.0
        A = np.stack([d[f[4]], np.ones(int(f[4].sum()))], 1)
        coef, *_ = np.linalg.lstsq(A, f[0][f[4]], rcond=None)
        args.scale = float(coef[0])
        args.offset = float(coef[1])
        args.direction = 1 if coef[0] >= 0 else -1
        args.scale = abs(args.scale)
        fit_info = dict(scale=args.scale, offset=args.offset, direction=args.direction,
                        residual_median=float(np.median(np.abs(f[0][f[4]] - A @ coef))),
                        dy_median=float(np.median(np.abs(f[1][f[4]]))))

    if args.mode == "identity":
        dx = np.zeros((h, w), np.float32)
    elif args.mode == "disp":
        dx = (args.direction * (args.scale * (P / 255.0) + args.offset)).astype(np.float32)
    elif args.mode == "calib":
        fdx, fdy, _, _, val = calib.displacement_field(cams, args.ref_cam, args.dst_cam, P,
                                                       args.y_origin)
        dx = fdx.astype(np.float32)
        dy = fdy.astype(np.float32) if args.use_dv else None

    # ---------------- warp ----------------
    if args.mode == "provided":
        warped, hole, real_depth = warp.scatter_image(
            color, P / 255.0, direction=args.direction, scale_factor=args.scale,
            inverse_ordering=True, reproject_depth=True)
        warped = np.asarray(warped, np.float32)
        hole_mask = hole > 0
        # scatter_image returns the warp of 1/(P/255) = 255/P, so invert it back to P
        warped_P = np.where(hole_mask, -1.0, 255.0 / (real_depth + 1e-6)).astype(np.float32)
        weight = (~hole_mask).astype(np.float32)
        dx = (args.direction * args.scale * (P / 255.0)).astype(np.float32)
    else:
        warped, warped_P, hole_mask, weight = warp.forward_warp(
            color, dx, dy, z=P, hole_depth=-1.0, rule=args.rule, splat=args.splat)
        warped_P = warped_P.astype(np.float32)

    # ---------------- ground truth ----------------
    gt = None
    stats = {}
    if not args.no_gt:
        try:
            gt = io_utils.load_view(args.dataset_root, args.dst_cam, args.frame)["color"]
        except FileNotFoundError:
            gt = None

    valid = ~hole_mask
    # identity (no-shift) baseline for reference
    if gt is not None:
        stats["psnr_identity_all"] = viz.psnr(gt, color)
        stats["psnr_warped_all"] = viz.psnr(gt, warped)
        stats["psnr_warped_valid"] = viz.psnr(gt, warped, mask=valid)
        stats["ssim_identity_all"] = viz.ssim(gt, color)
        stats["ssim_warped_all"] = viz.ssim(gt, warped)
        stats["ssim_warped_valid"] = viz.ssim(gt, warped, mask=valid)

    stats.update(
        frame=args.frame, ref_cam=args.ref_cam, dst_cam=args.dst_cam, mode=args.mode,
        rule=args.rule, splat=args.splat,
        width=w, height=h, scale=args.scale, offset=args.offset, direction=args.direction,
        use_dv=bool(dy is not None),
        dx_min=float(dx.min()), dx_max=float(dx.max()),
        dy_min=float(dy.min()) if dy is not None else 0.0,
        dy_max=float(dy.max()) if dy is not None else 0.0,
        hole_pixels=int(hole_mask.sum()), hole_ratio_pct=round(float(hole_mask.mean() * 100), 4),
        warped_P_min=float(warped_P[valid].min()) if valid.any() else -1.0,
        warped_P_max=float(warped_P[valid].max()) if valid.any() else -1.0,
    )
    import cv2
    n_lab, _, cc, _ = cv2.connectedComponentsWithStats(hole_mask.astype(np.uint8), 8)
    areas = cc[1:, cv2.CC_STAT_AREA] if n_lab > 1 else np.array([], np.int32)
    stats["hole_components"] = int(n_lab - 1)
    stats["hole_components_ge50"] = int((areas >= 50).sum())
    # thin (crack-like) components: bounding box thin in one direction and elongated
    thin = 0
    if n_lab > 1:
        bw, bh = cc[1:, cv2.CC_STAT_WIDTH], cc[1:, cv2.CC_STAT_HEIGHT]
        thin = int(((np.minimum(bw, bh) <= 3) & (np.maximum(bw, bh) >= 8)).sum())
    stats["hole_components_thin"] = thin
    # OOFA side by edge scanning
    left_hole = int(hole_mask[:, :20].sum())
    right_hole = int(hole_mask[:, -20:].sum())
    stats["holes_left_edge20"] = left_hole
    stats["holes_right_edge20"] = right_hole
    stats["oofa_side"] = "right" if right_hole > left_hole else "left"

    # ---------------- artefacts ----------------
    warped_u8 = viz.to_u8(warped)
    io_utils.imwrite(os.path.join(out, "warped_color.png"), warped_u8)
    io_utils.imwrite(os.path.join(out, "hole_mask.png"),
                     (hole_mask * 255).astype(np.uint8))
    io_utils.imwrite(os.path.join(out, "overlay_holes_red.png"),
                     viz.overlay_mask(warped_u8, hole_mask, (255, 0, 0)))
    np.save(os.path.join(out, "warped_P.npy"), warped_P)
    np.save(os.path.join(out, "warped_color.npy"), warped.astype(np.float32))
    np.save(os.path.join(out, "dx.npy"), dx)
    if dy is not None:
        np.save(os.path.join(out, "dy.npy"), dy)
    viz.imwrite_gray(os.path.join(out, "warped_P_color.png"), warped_P, valid_mask=valid,
                     vmin=0, vmax=255)
    viz.imwrite_gray(os.path.join(out, "input_P_color.png"), P, vmin=0, vmax=255)
    if not args.mode == "identity":
        viz.imwrite_gray(os.path.join(out, "dx_color.png"), dx)
        if dy is not None:
            viz.imwrite_gray(os.path.join(out, "dy_color.png"), dy)

    # ---------------- inspection panel ----------------
    tiles = [viz.label(viz.to_u8(color), f"1) source cam{args.ref_cam} {args.frame}"),
             viz.label(viz.to_u8(gt) if gt is not None else np.zeros_like(warped_u8),
                       f"2) GT cam{args.dst_cam}" if gt is not None else "2) GT unavailable"),
             viz.label(warped_u8, f"3) warped I_w  ({args.mode})"),
             viz.label(viz.overlay_mask(warped_u8, hole_mask, (255, 0, 0)),
                       f"4) holes red  {stats['hole_ratio_pct']:.2f}%"),
             viz.label(viz.colorize(P, vmin=0, vmax=255), "5) source inverse depth P"),
             viz.label(viz.colorize(warped_P, valid_mask=valid, vmin=0, vmax=255),
                       f"6) warped P_w  holes={int(hole_mask.sum())}"),
             viz.label((hole_mask * 255).astype(np.uint8)[:, :, None].repeat(3, 2),
                       "7) hole mask (white=hole)"),
             viz.label(np.abs(viz.to_u8(warped).astype(np.int16) -
                              (viz.to_u8(gt).astype(np.int16) if gt is not None else 0))
                       .astype(np.uint8) if gt is not None else np.zeros_like(warped_u8),
                       "8) |I_w - GT|")]

    # zoom on the largest interior hole component
    ys, xs = np.where(hole_mask)
    if ys.size:
        n2, _, cc2, cent = cv2.connectedComponentsWithStats(hole_mask.astype(np.uint8), 8)
        if n2 > 1:
            k = 1 + int(np.argmax(cc2[1:, cv2.CC_STAT_AREA]))
            cy, cx = int(cent[k][1]), int(cent[k][0])
            tiles.append(viz.label(np.hstack([viz.zoom(viz.to_u8(color), cx, cy),
                                              viz.zoom(viz.overlay_mask(warped_u8, hole_mask,
                                                                        (255, 0, 0)), cx, cy)]),
                                   "9) zoom largest hole: source | warped+holes"))
    panel = viz.grid(tiles, cols=4)

    lines = [f"Step 1 forward warp | mode={args.mode} | cam{args.ref_cam} -> cam{args.dst_cam} {args.frame}",
             f"size {w}x{h} | displacement dx [{dx.min():+.2f}, {dx.max():+.2f}] px"
             + (f" | dy [{dy.min():+.2f}, {dy.max():+.2f}] px" if dy is not None else " | dy = 0"),
             f"holes {stats['hole_pixels']} px ({stats['hole_ratio_pct']:.3f}%) | "
             f"components {stats['hole_components']} (>=50px {stats['hole_components_ge50']}, "
             f"thin {stats['hole_components_thin']})",
             f"edge holes: left20 {left_hole} / right20 {right_hole} -> OOFA side {stats['oofa_side']}",
             f"warped P range (valid) [{stats['warped_P_min']:.1f}, {stats['warped_P_max']:.1f}], "
             f"holes set to -1"]
    if fit_info:
        lines.append(f"scale fitted from calibration: s={fit_info['scale']:.3f} px, "
                     f"offset={fit_info['offset']:+.3f} px, direction={fit_info['direction']:+d} "
                     f"(residual median {fit_info['residual_median']:.2f} px, "
                     f"|dy| median {fit_info['dy_median']:.2f} px)")
    if gt is not None:
        lines += ["ground truth (real cam%d image, grayscale, all pixels / valid only):" % args.dst_cam,
                  f"  no warp     : PSNR {stats['psnr_identity_all']:6.2f} dB  SSIM {stats['ssim_identity_all']:.4f}",
                  f"  this warp   : PSNR {stats['psnr_warped_all']:6.2f} dB  "
                  f"(valid {stats['psnr_warped_valid']:6.2f})  SSIM {stats['ssim_warped_all']:.4f} "
                  f"(valid {stats['ssim_warped_valid']:.4f})"]
    if args.mode == "disp":
        lines.append("NOTE: pure 1D disparity model; dy is ignored by construction.")
    card = viz.text_card(lines, width=900)
    hh = max(panel.shape[0], card.shape[0])
    panel_p = np.full((hh, panel.shape[1], 3), 12, np.uint8)
    panel_p[:panel.shape[0]] = panel
    card_p = np.full((hh, card.shape[1], 3), 20, np.uint8)
    card_p[:card.shape[0]] = card
    io_utils.imwrite(os.path.join(out, "panel.png"),
                     np.hstack([panel_p, np.full((hh, 8, 3), 12, np.uint8), card_p]))

    with open(os.path.join(out, "stats.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(stats.keys()))
        wr.writeheader()
        wr.writerow(stats)
    print(f"--- Step 1 ({args.mode}) cam{args.ref_cam} -> cam{args.dst_cam} {args.frame} ---")
    for k, v in stats.items():
        print(f"  {k:24s} {v}")
    if fit_info:
        print(f"  fitted scale/offset/direction: {fit_info}")
    print(f"artefacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
