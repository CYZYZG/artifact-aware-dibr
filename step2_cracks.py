"""Step 2 - crack detection and filling (paper II-A), plus the HHF leave-one-out test.

    python step2_cracks.py --ref_cam 6 --dst_cam 7 --frame f000
    python step2_cracks.py --ref_cam 6 --dst_cam 7 --frame f000 --crack_shape thin
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dibr import cracks, io_utils, viz  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser(description="Step 2: crack detection and filling")
    ap.add_argument("--ref_cam", type=int, default=6)
    ap.add_argument("--dst_cam", type=int, default=7)
    ap.add_argument("--frame", default="f000")
    ap.add_argument("--step1_dir", default=os.path.join(HERE, "output", "step1_warp"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "output", "step2_cracks"))
    ap.add_argument("--lam", type=float, default=5.0, help="paper lambda = 5")
    ap.add_argument("--se_len", type=int, default=4, help="paper H = [1 1 1 1]^T")
    ap.add_argument("--orientation", choices=["v", "h", "both", "auto"], default="auto",
                    help="SE orientation.  The SE must be PERPENDICULAR to the slit: a "
                         "horizontal projection opens vertical slits, which need H^T. "
                         "auto picks the SE perpendicular to the dominant projection "
                         "(from the saved dx/dy): measured 97.8%% vs 78.9%% thin-crack "
                         "recovery on this data.")
    ap.add_argument("--crack_shape", choices=["none", "thin"], default="none",
                    help="none = paper-faithful detection; thin = restrict empty cracks "
                         "to hole components thinner than --max_thickness")
    ap.add_argument("--max_thickness", type=float, default=3.0)
    ap.add_argument("--hhf_sigma", type=float, default=1.0)
    ap.add_argument("--hhf_ksize", type=int, default=5)
    ap.add_argument("--n_synth", type=int, default=60)
    ap.add_argument("--dataset_root", default=io_utils.DATASET_ROOT_DEFAULT)
    ap.add_argument("--no_gt", action="store_true",
                    help="skip the ground-truth judgement of the crack filling")
    return ap.parse_args()


def main():
    args = parse_args()
    tag = f"cam{args.ref_cam}_to_cam{args.dst_cam}_{args.frame}"
    src = os.path.join(args.step1_dir, tag)
    D_w = np.load(os.path.join(src, "warped_P.npy")).astype(np.float32)
    I_w = np.load(os.path.join(src, "warped_color.npy")).astype(np.float32)
    hole = D_w < 0
    h, w = D_w.shape
    if args.orientation == "auto":
        try:
            ddx = np.load(os.path.join(src, "dx.npy"))
            ddy_path = os.path.join(src, "dy.npy")
            ddy = np.load(ddy_path) if os.path.isfile(ddy_path) else np.zeros_like(ddx)
            mx, my = float(np.median(np.abs(ddx))), float(np.median(np.abs(ddy)))
        except FileNotFoundError:
            mx, my = 1.0, 0.0
        args.orientation = "h" if mx >= my else "v"
        print(f"[auto] median|dx|={mx:.2f} px, median|dy|={my:.2f} px -> SE orientation "
              f"'{args.orientation}'")
    out = os.path.join(args.out_dir, tag, f"lam{args.lam:g}_{args.orientation}_{args.crack_shape}")
    os.makedirs(out, exist_ok=True)

    # ---------------- hole morphology (what is actually there?) ----------------
    rows, lab = cracks.component_stats(hole)
    areas = np.array([r["area"] for r in rows]) if rows else np.array([0])
    thick = np.array([r["thickness"] for r in rows]) if rows else np.array([0.0])
    big_labels = [r["label"] for r in rows if r["area"] > 200]
    big_mask = np.isin(lab, big_labels) if big_labels else np.zeros_like(hole)
    thin_mask = np.isin(lab, [r["label"] for r in rows if r["thickness"] <= 2])

    # ---------------- detection + filling ----------------
    res = cracks.fill_cracks(I_w, D_w, lam=args.lam, se_len=args.se_len,
                             orientation=args.orientation, hhf_sigma=args.hhf_sigma,
                             hhf_ksize=args.hhf_ksize, shape_filter=args.crack_shape,
                             max_thickness=args.max_thickness)
    crack = res["crack"]
    crows, _ = cracks.component_stats(crack)
    cthick = np.array([r["thickness"] for r in crows]) if crows else np.array([0.0])
    careas = np.array([r["area"] for r in crows]) if crows else np.array([0])

    # ---------------- lambda sweep ----------------
    sweep = []
    for lam in (1, 2, 3, 5, 8, 12, 20):
        if args.orientation == "both":
            c = cracks.detect_cracks_both(D_w, lam=lam, se_len=args.se_len)[0]
        else:
            c, _, _ = cracks.detect_cracks(D_w, lam=lam, se_len=args.se_len,
                                           orientation=args.orientation)
        sweep.append(dict(lam=lam, crack_px=int(c.sum()),
                          empty_px=int((c & hole).sum()),
                          translucent_px=int((c & ~hole).sum()),
                          in_big_holes_px=int((c & big_mask).sum())))
    with open(os.path.join(out, "lambda_sweep.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        wr.writeheader()
        wr.writerows(sweep)

    # ---------------- HHF leave-one-out validation ----------------
    val = cracks.validate_hhf(I_w, D_w, hole, n=args.n_synth)
    if val is not None:
        with open(os.path.join(out, "hhf_validation.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            keys = [k for k in val if not isinstance(val[k], np.ndarray)]
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerow({k: val[k] for k in keys})

    # ---------------- statistics ----------------
    stats = dict(
        frame=args.frame, ref_cam=args.ref_cam, dst_cam=args.dst_cam,
        lam=args.lam, se_len=args.se_len, orientation=args.orientation,
        crack_shape=args.crack_shape,
        hole_px=int(hole.sum()), hole_pct=round(float(hole.mean() * 100), 4),
        hole_components=len(rows),
        hole_thin_components=int((thick <= 2).sum()),
        hole_thin_px=int(areas[thick <= 2].sum()) if len(areas) else 0,
        hole_big_components=len(big_labels),
        hole_big_px=int(areas[areas > 200].sum()) if len(areas) else 0,
        crack_px=int(crack.sum()), crack_pct_of_frame=round(float(crack.mean() * 100), 4),
        crack_pct_of_holes=round(float(crack.sum() / max(1, hole.sum()) * 100), 3),
        crack_empty_px=int(res["empty_crack"].sum()),
        crack_translucent_px=int(res["translucent_crack"].sum()),
        crack_in_big_holes_px=int((crack & big_mask).sum()),
        crack_of_thin_px=int((crack & thin_mask).sum()),
        thin_crack_recovery_pct=round(
            float((crack & thin_mask).sum() / max(1, thin_mask.sum()) * 100), 2),
        crack_components=len(crows),
        crack_thin_components=int((cthick <= 3).sum()),
        crack_max_thickness=float(cthick.max()) if len(cthick) else 0.0,
        crack_max_area=int(careas.max()) if len(careas) else 0,
        remaining_holes_px=int(res["remaining_holes"].sum()),
        remaining_holes_pct_of_holes=round(
            float(res["remaining_holes"].sum() / max(1, hole.sum()) * 100), 3),
        D_filled_has_negative_at_cracks=int((res["D_filled"][crack] < 0).sum()),
        color_changed_px=int((np.abs(res["I_filled"] - I_w).max(axis=2) > 1e-3).sum()),
    )
    if val is not None:
        stats.update(hhf_val_slivers=val["n_slivers"], hhf_val_px=val["sliver_px"],
                     hhf_val_mae=round(val["hhf_mae"], 3),
                     hhf_val_rmse=round(val["hhf_rmse"], 3),
                     hhf_val_psnr=round(val["hhf_psnr"], 3),
                     lin_val_mae=round(val["lin_mae"], 3),
                     lin_val_psnr=round(val["lin_psnr"], 3))

    # ---------------- ground-truth judgement of the crack treatment ----------------
    # I_w is already aligned with the real target camera (Step 1: 29.1 dB), so it is a
    # fair reference: does replacing the detected pixels by HHF output move I_w closer
    # to the truth or further away?
    gt_psnr = {}
    if not args.no_gt:
        try:
            gt = io_utils.load_view(args.dataset_root, args.dst_cam, args.frame)["color"]
        except FileNotFoundError:
            gt = None
        if gt is not None:
            validate = ~res["remaining_holes"]
            for name, m in (("all_crack", crack), ("empty", res["empty_crack"]),
                            ("translucent", res["translucent_crack"]),
                            ("frame_valid", validate)):
                if m.sum() == 0:
                    continue
                gt_psnr[f"gt_psnr_{name}_before"] = round(viz.psnr(gt, I_w, mask=m), 3)
                gt_psnr[f"gt_psnr_{name}_after"] = round(viz.psnr(gt, res["I_filled"], mask=m), 3)
            gt_psnr["gt_ssim_frame_before"] = round(viz.ssim(gt, I_w, mask=validate), 5)
            gt_psnr["gt_ssim_frame_after"] = round(viz.ssim(gt, res["I_filled"], mask=validate), 5)
            stats.update(gt_psnr)
    with open(os.path.join(out, "stats.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(stats.keys()))
        wr.writeheader()
        wr.writerow(stats)

    # ---------------- artefacts ----------------
    # Fig. 1(c) colour convention: empty cracks blue, translucent cracks red
    vis = viz.to_u8(I_w).copy()
    # dilate by 1 px purely for visibility in the figure
    vis[viz.dilate(res["empty_crack"])] = (0, 80, 255)          # blue (RGB)
    vis[viz.dilate(res["translucent_crack"])] = (255, 40, 40)   # red
    io_utils.imwrite(os.path.join(out, "crack_overlay.png"), vis)
    io_utils.imwrite(os.path.join(out, "crack_mask.png"), (crack * 255).astype(np.uint8))
    viz.imwrite_gray(os.path.join(out, "D_w_diff.png"), res["diff"], vmin=0, vmax=40)
    io_utils.imwrite(os.path.join(out, "filled_color.png"), viz.to_u8(res["I_filled"]))
    viz.imwrite_gray(os.path.join(out, "filled_P.png"), res["D_filled"],
                     valid_mask=res["remaining_holes"] == 0, vmin=0, vmax=255)
    io_utils.imwrite(os.path.join(out, "remaining_holes.png"),
                     (res["remaining_holes"] * 255).astype(np.uint8))
    # raw arrays for the next step (avoid a lossy PNG round trip)
    np.save(os.path.join(out, "I_filled.npy"), res["I_filled"].astype(np.float32))
    np.save(os.path.join(out, "D_filled.npy"), res["D_filled"].astype(np.float32))
    np.save(os.path.join(out, "remaining_holes.npy"), res["remaining_holes"])
    np.save(os.path.join(out, "crack_mask.npy"), crack)

    # zoom on the largest crack clump
    import cv2
    crack_dil = viz.dilate(crack, 3)
    n_lab, _, ccs, cents = cv2.connectedComponentsWithStats(crack_dil.astype(np.uint8), 8)
    tiles = [
        viz.label(viz.to_u8(I_w), "1) warped I_w"),
        viz.label(vis, "2) cracks: blue=empty, red=translucent (1px dilated)"),
        viz.label(viz.colorize(res["diff"], vmin=0, vmax=40), "3) D_hat - D_w"),
        viz.label(viz.colorize(D_w, valid_mask=~hole, vmin=0, vmax=255), "4) D_w (grey=hole)"),
        viz.label(viz.colorize(res["D_filled"], valid_mask=res["remaining_holes"] == 0,
                               vmin=0, vmax=255), "5) D_filled (cracks repaired)"),
        viz.label((res["remaining_holes"] * 255).astype(np.uint8)[:, :, None].repeat(3, 2),
                  f"6) holes left: {int(res['remaining_holes'].sum())}"),
        viz.label(viz.to_u8(res["I_filled"]), "7) I_w after HHF crack fill"),
        viz.label(viz.to_u8(np.abs(res["I_filled"] - I_w)),
                  "8) |change| from HHF"),
    ]
    if n_lab > 1:
        k = 1 + int(np.argmax(ccs[1:, cv2.CC_STAT_AREA]))
        cy, cx = int(cents[k][1]), int(cents[k][0])
        tiles += [viz.label(viz.zoom(viz.to_u8(I_w), cx, cy, 120), "9) zoom source"),
                  viz.label(viz.zoom(vis, cx, cy, 120), "10) zoom cracks"),
                  viz.label(viz.zoom(viz.to_u8(res["I_filled"]), cx, cy, 120), "11) zoom filled")]
    if val is not None:
        tiles += [viz.label(viz.to_u8(np.abs(val["filled"] - I_w) * 4), "12) HHF LOO err x4"),
                  viz.label(viz.to_u8(np.abs(val["lin"] - I_w) * 4), "13) linear LOO err x4")]
    panel = viz.grid(tiles, cols=4)

    lines = [
        f"Step 2 cracks | cam{args.ref_cam}->cam{args.dst_cam} {args.frame} | "
        f"lambda={args.lam:g} H(len {args.se_len},{args.orientation}) shape={args.crack_shape}",
        f"holes {stats['hole_px']} px ({stats['hole_pct']:.2f}%) in {stats['hole_components']} "
        f"components | thin(<=2px) {stats['hole_thin_components']} comps / "
        f"{stats['hole_thin_px']} px | big(>200px) {stats['hole_big_components']} comps / "
        f"{stats['hole_big_px']} px",
        f"cracks {stats['crack_px']} px ({stats['crack_pct_of_frame']:.3f}% of frame, "
        f"{stats['crack_pct_of_holes']:.1f}% of holes) in {stats['crack_components']} comps",
        f"  empty {stats['crack_empty_px']} px | translucent {stats['crack_translucent_px']} px "
        f"| inside big holes {stats['crack_in_big_holes_px']} px",
        f"  crack thickness: max {stats['crack_max_thickness']:.1f} px, "
        f"{stats['crack_thin_components']} comps <=3 px thick",
        f"after fill: holes left {stats['remaining_holes_px']} px "
        f"({stats['remaining_holes_pct_of_holes']:.1f}% of original holes); "
        f"colour changed at {stats['color_changed_px']} px",
    ]
    if val is not None:
        lines += ["HHF leave-one-out on synthetic thin slivers in uniform-disparity areas:",
                  f"  HHF     MAE {val['hhf_mae']:.2f}  PSNR {val['hhf_psnr']:.2f} dB",
                  f"  linear  MAE {val['lin_mae']:.2f}  PSNR {val['lin_psnr']:.2f} dB  (baseline)"]
    lines += ["", "lambda sweep (crack px): " + ", ".join(
        f"{r['lam']}:{r['crack_px']}" for r in sweep)]
    if gt_psnr:
        lines += ["", "ground-truth judgement (vs the real target camera image):"]
        for name in ("all_crack", "empty", "translucent"):
            b, a = gt_psnr.get(f"gt_psnr_{name}_before"), gt_psnr.get(f"gt_psnr_{name}_after")
            if b is not None:
                verdict = "better" if a > b else "WORSE"
                lines.append(f"  {name:12s} PSNR before {b:6.2f} -> after {a:6.2f} dB  ({verdict})")
        lines.append(f"  whole frame SSIM {gt_psnr['gt_ssim_frame_before']:.4f} -> "
                     f"{gt_psnr['gt_ssim_frame_after']:.4f}")
    card = viz.text_card(lines, width=1000)
    hh = max(panel.shape[0], card.shape[0])
    p2 = np.full((hh, panel.shape[1], 3), 12, np.uint8)
    p2[:panel.shape[0]] = panel
    c2 = np.full((hh, card.shape[1], 3), 20, np.uint8)
    c2[:card.shape[0]] = card
    io_utils.imwrite(os.path.join(out, "panel.png"),
                     np.hstack([p2, np.full((hh, 8, 3), 12, np.uint8), c2]))

    print(f"--- Step 2 cracks | cam{args.ref_cam}->cam{args.dst_cam} {args.frame} | "
          f"lam={args.lam:g} orient={args.orientation} shape={args.crack_shape} ---")
    for k, v in stats.items():
        print(f"  {k:32s} {v}")
    print(f"artefacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
