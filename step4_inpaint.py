"""Steps 4-7 - hole classification (OOFA / disocclusion) and exemplar-based filling.

    python step4_inpaint.py --ref_cam 6 --dst_cam 7 --frame f000
    python step4_inpaint.py --ablate nob|nodepth|nocd|noconf      # ablations

Step 4 = classify, Steps 5-7 = priority + patch matching + iterative fill (paper II-C).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dibr import calib, ghosts, holes, inpaint, io_utils, viz  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser(description="Steps 4-7: hole filling (paper II-C)")
    ap.add_argument("--ref_cam", type=int, default=6)
    ap.add_argument("--dst_cam", type=int, default=7)
    ap.add_argument("--frame", default="f000")
    ap.add_argument("--dataset_root", default=io_utils.DATASET_ROOT_DEFAULT)
    ap.add_argument("--step3_dir", default=os.path.join(HERE, "output", "step3_ghosts"))
    ap.add_argument("--step3_subdir", default="band2_gt_a11_copy")
    ap.add_argument("--from_step", choices=["step3", "step2"], default="step3",
                    help="step3 = use the ghost-corrected input (default); "
                         "step2 = skip the ghost step (ablation), take the crack output")
    ap.add_argument("--step1_dir", default=os.path.join(HERE, "output", "step1_warp_int"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "output", "step4_inpaint"))
    ap.add_argument("--n_window", type=int, default=69, help="paper N = 69")
    ap.add_argument("--sizes", default="9,7,5,3", help="adaptive patch sizes (paper 9->3 step 2)")
    ap.add_argument("--beta", type=float, default=35.0, help="paper beta = 35")
    ap.add_argument("--beta_mode", choices=["mean", "sum"], default="mean",
                    help="mean = SSD/(3*valid px) so that beta is a mean squared "
                         "per-channel error (the paper's 35 is only reachable on this "
                         "scale); sum = the paper's literal SSD sum")
    ap.add_argument("--max_iter", type=int, default=400000)
    ap.add_argument("--ablate", choices=["none", "nob", "nodepth", "nocd", "noconf",
                                         "search_iw", "nobgsearch"],
                    default="none")
    ap.add_argument("--max_components", type=int, default=0,
                    help="debug: only fill the N largest components (0 = all)")
    ap.add_argument("--no_gt", action="store_true")
    return ap.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    tag = f"cam{args.ref_cam}_to_cam{args.dst_cam}_{args.frame}"
    s3 = os.path.join(args.step3_dir, tag, args.step3_subdir)
    if args.from_step == "step2":
        s3 = os.path.join(HERE, "output", "step2_cracks_int", tag, "lam5_h_none")
        names = ("I_filled.npy", "D_filled.npy", "remaining_holes.npy")
    else:
        names = ("I_fixed.npy", "D_fixed.npy", "hole_out.npy")
    I_w = np.load(os.path.join(s3, names[0])).astype(np.float32)
    D_w = np.load(os.path.join(s3, names[1])).astype(np.float32)
    hole = np.load(os.path.join(s3, names[2])).astype(bool)
    h, w = hole.shape
    out = os.path.join(args.out_dir, tag, args.ablate
                       + (f"_comp{args.max_components}" if args.max_components else ""))
    os.makedirs(out, exist_ok=True)

    cams = calib.load_calib(args.dataset_root)
    ref = io_utils.load_view(args.dataset_root, args.ref_cam, args.frame)
    ref_color = ref["color"].astype(np.float32)
    ref_depth = ref["depth"].astype(np.float32)
    dx, dy, _, _, _ = calib.displacement_field(cams, args.ref_cam, args.dst_cam, ref_depth)
    direction = 1 if float(np.median(dx)) > 0 else -1

    # ---------------- Steps 4-7: classify, prioritise, match, fill ----------------
    params = dict(n_window=args.n_window,
                  sizes=tuple(int(s) for s in args.sizes.split(",")),
                  beta=args.beta, beta_mode=args.beta_mode, max_iter=args.max_iter)
    t1 = time.time()

    def _progress(i, n, logs):
        done = sum(l["iterations"] for l in logs.values())
        left = sum(l["rem_left"] for l in logs.values())
        print(f"  [{i}/{n}] components, {done} patch iterations, "
              f"{left} px left in touched comps, elapsed {time.time()-t1:.1f}s", flush=True)

    res = inpaint.fill_all(I_w, D_w, hole, ref_color, ref_depth, (dx, dy),
                           src_of=None, direction=direction, params=params,
                           ablate=args.ablate, progress=_progress)
    t2 = time.time()
    remaining = res["remaining"]
    I_w = res["I_filled"]
    D_w = res["D_filled"]
    oofa, disocc, lab, comp_type = res["oofa"], res["disocc"], res["lab"], res["comp_type"]
    info = dict(oofa_px=res["stats"]["oofa_px"], disocc_px=res["stats"]["disocc_px"],
                oofa_components=res["stats"]["oofa_components"],
                disocc_components=res["stats"]["disocc_components"])
    logs = res["logs"]
    if args.max_components:
        pass          # already processed; kept for CLI compatibility
    E_range = (res["stats"]["E_lo"], res["stats"]["E_hi"])

    # ---------------- stats ----------------
    stats = dict(tag=tag, ablate=args.ablate, n_window=args.n_window,
                 sizes=args.sizes, beta=args.beta,
                 holes_before=int(hole.sum()),
                 holes_after=int(remaining.sum()),
                 filled_px=int(hole.sum() - remaining.sum()),
                 oofa_px=info["oofa_px"], disocc_px=info["disocc_px"],
                 oofa_components=info["oofa_components"],
                 disocc_components=info["disocc_components"],
                 components_filled=len(logs),
                 iterations=int(sum(l["iterations"] for l in logs.values())),
                 failed_components=int(sum(1 for l in logs.values() if l["failed"])),
                 fail_no_frontier=int(sum(1 for l in logs.values()
                                         if l.get("fail_reason") == "no frontier")),
                 fail_no_patch=int(sum(1 for l in logs.values()
                                       if l.get("fail_reason") == "no patch")),
                 px_left_in_failed=int(sum(l["rem_left"] for l in logs.values()
                                           if l["failed"])),
                 bg_search_used=int(sum(l["bg_search"] for l in logs.values())),
                 seconds_fill=round(t2 - t1, 2), seconds_total=round(time.time() - t0, 2),
                 E_lo=E_range[0], E_hi=E_range[1])
    sizes_used = {}
    for l in logs.values():
        for k, v in l["sizes"].items():
            sizes_used[k] = sizes_used.get(k, 0) + v
    for k in sorted(sizes_used):
        stats[f"patchsize_{k}"] = sizes_used[k]
    costs = np.array([c for l in logs.values() for c in l["costs"]], np.float64)
    if costs.size:
        stats.update(cost_min=float(costs.min()), cost_p25=float(np.percentile(costs, 25)),
                     cost_median=float(np.median(costs)),
                     cost_p75=float(np.percentile(costs, 75)), cost_max=float(costs.max()),
                     cost_frac_below_beta=round(float((costs <= args.beta).mean()), 4))
    if not args.no_gt:
        try:
            gt = io_utils.load_view(args.dataset_root, args.dst_cam, args.frame)["color"]
        except FileNotFoundError:
            gt = None
        if gt is not None:
            prev = np.load(os.path.join(s3, names[0])).astype(np.float32)
            m = ~remaining
            stats["gt_psnr_before"] = round(viz.psnr(gt, prev, mask=m), 3)
            stats["gt_psnr_after"] = round(viz.psnr(gt, I_w, mask=m), 3)
            stats["gt_ssim_before"] = round(viz.ssim(gt, prev, mask=m), 5)
            stats["gt_ssim_after"] = round(viz.ssim(gt, I_w, mask=m), 5)
            # full-frame, holes counted as black
            stats["gt_psnr_frame_before"] = round(viz.psnr(gt, prev), 3)
            stats["gt_psnr_frame_after"] = round(viz.psnr(gt, I_w), 3)
            stats["gt_ssim_frame_before"] = round(viz.ssim(gt, prev), 5)
            stats["gt_ssim_frame_after"] = round(viz.ssim(gt, I_w), 5)
            if hole.sum():
                stats["gt_psnr_in_holes_before"] = round(viz.psnr(gt, prev, mask=hole), 3)
                stats["gt_psnr_in_holes_after"] = round(viz.psnr(gt, I_w, mask=hole), 3)

    with open(os.path.join(out, "stats.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(stats.keys()))
        wr.writeheader()
        wr.writerow(stats)
    with open(os.path.join(out, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)

    # ---------------- artefacts ----------------
    io_utils.imwrite(os.path.join(out, "final_color.png"), viz.to_u8(I_w))
    viz.imwrite_gray(os.path.join(out, "final_depth.png"), D_w,
                     valid_mask=~remaining, vmin=0, vmax=255)
    io_utils.imwrite(os.path.join(out, "remaining_holes.png"),
                     (remaining * 255).astype(np.uint8))
    cls = np.zeros((h, w, 3), np.uint8)
    cls[disocc] = (255, 200, 0)
    cls[oofa] = (0, 90, 255)
    io_utils.imwrite(os.path.join(out, "classify_overlay.png"),
                     np.hstack([viz.label(viz.to_u8(I_w), "1) input to step 4"),
                                viz.label(cls, "2) blue=OOFA, yellow=disocclusion")]))
    tiles = [viz.label(viz.to_u8(np.load(os.path.join(s3, names[0]))), "1) step-3/2 input"),
             viz.label(cls, "2) OOFA(blue) / disocc(yellow)"),
             viz.label(viz.to_u8(I_w), "3) after filling"),
             viz.label((remaining * 255).astype(np.uint8)[:, :, None].repeat(3, 2),
                       f"4) holes left {int(remaining.sum())}")]
    ys, xs = np.nonzero(viz.dilate(hole, 2))
    if ys.size:
        import cv2
        n_lab, _, ccs, cents = cv2.connectedComponentsWithStats(
            viz.dilate(hole, 2).astype(np.uint8), 8)
        k = 1 + int(np.argmax(ccs[1:, cv2.CC_STAT_AREA]))
        cy, cx = int(cents[k][1]), int(cents[k][0])
        prev = np.load(os.path.join(s3, names[0])).astype(np.float32)
        tiles += [viz.label(viz.zoom(viz.to_u8(prev), cx, cy, 150), "5) zoom before"),
                  viz.label(viz.zoom(viz.to_u8(I_w), cx, cy, 150), "6) zoom after")]
        if not args.no_gt:
            try:
                gt = io_utils.load_view(args.dataset_root, args.dst_cam, args.frame)["color"]
                tiles.append(viz.label(viz.zoom(viz.to_u8(gt), cx, cy, 150), "7) zoom GT"))
            except FileNotFoundError:
                pass
    panel = viz.grid(tiles, cols=4)
    lines = [
        f"Steps 4-7 | {tag} | ablate={args.ablate} | N={args.n_window} beta={args.beta:g} "
        f"sizes={args.sizes}",
        f"holes {stats['holes_before']} -> {stats['holes_after']} "
        f"(filled {stats['filled_px']} px, {100*stats['filled_px']/max(1,stats['holes_before']):.1f}%)",
        f"OOFA {stats['oofa_px']} px in {stats['oofa_components']} comps | "
        f"disocclusion {stats['disocc_px']} px in {stats['disocc_components']} comps",
        f"components filled {stats['components_filled']}, iterations {stats['iterations']}, "
        f"failed {stats['failed_components']}, time {stats['seconds_fill']:.1f}s",
        f"patch sizes used: " + ", ".join(f"{k}:{v}" for k, v in sizes_used.items()),
    ]
    if costs.size:
        lines.append(f"patch cost (SSD sum): median {stats['cost_median']:.1f} "
                     f"p75 {stats['cost_p75']:.1f} max {stats['cost_max']:.1f} | "
                     f"below beta {stats['cost_frac_below_beta']*100:.1f}%")
    if not args.no_gt and "gt_psnr_after" in stats:
        lines += ["", "ground truth (vs the real target camera):",
                  f"  filled px      {stats['gt_psnr_in_holes_before']:.2f} -> "
                  f"{stats['gt_psnr_in_holes_after']:.2f} dB",
                  f"  whole frame    {stats['gt_psnr_frame_before']:.2f} -> "
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
    np.save(os.path.join(out, "final_color.npy"), I_w.astype(np.float32))
    np.save(os.path.join(out, "final_depth.npy"), D_w.astype(np.float32))
    np.save(os.path.join(out, "remaining_holes.npy"), remaining)

    print(f"--- Steps 4-7 | {tag} | ablate={args.ablate} ---")
    for k, v in stats.items():
        print(f"  {k:28s} {v}")
    print(f"artefacts: {out}")
    return 0


def _threshold(D_w, lab, k):
    """T_O = trimmed mean (alpha 10%) of the valid disparity along the hole boundary."""
    from dibr.ghosts import trimmed_mean
    import scipy.ndimage as ndi
    m = lab == k
    b = ndi.binary_dilation(m, np.ones((3, 3), bool)) & ~m
    v = D_w[b]
    v = v[v >= 0]
    return trimmed_mean(v, 0.10) if v.size else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
