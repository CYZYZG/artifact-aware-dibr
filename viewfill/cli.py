"""viewfill CLI.

    python -m viewfill --image color.jpg --depth depth.png --scale -44.8 --out out_dir
    python -m viewfill --warped w.png --hole h.png --depth-warped dw.png `
                       --image color.jpg --depth depth.png --out out_dir
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viewfill import FillConfig, io as vio, report  # noqa: E402
from viewfill.pipeline import fill_warped, warp_and_fill  # noqa: E402


def build_config(a):
    return FillConfig(
        scale=a.scale, splat=a.splat, rule=a.rule, lam=a.lam, se_len=a.se_len,
        se_orientation=a.orientation, crack_shape=a.crack_shape,
        band_radius=a.band_radius, alpha_sim=a.alpha_sim, fg_side=a.fg_side,
        fix_mode=a.fix_mode, skip_ghosts=a.skip_ghosts, n_window=a.n_window,
        sizes=tuple(int(s) for s in a.sizes.split(",")), beta=a.beta,
        beta_mode=a.beta_mode, ablate=a.ablate)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="viewfill", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="reference colour image")
    ap.add_argument("--depth", required=True, help="reference inverse depth map")
    ap.add_argument("--depth-mode", default="auto", choices=["auto", "01", "255"])
    ap.add_argument("--scale", type=float, default=-44.8,
                    help="disparity = inverse_depth * scale, in pixels (default -44.8)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--gt", default=None, help="optional ground-truth view for metrics")
    # already-warped mode
    ap.add_argument("--warped", default=None, help="skip our warp: input synthetic view")
    ap.add_argument("--hole", default=None, help="skip our warp: hole mask (white = hole)")
    ap.add_argument("--depth-warped", default=None,
                    help="skip our warp: depth map of the synthetic view")
    # algorithm knobs
    ap.add_argument("--splat", default="sub", choices=["sub", "floor", "round"])
    ap.add_argument("--rule", default="zbuf", choices=["zbuf", "avg"])
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--se-len", type=int, default=4)
    ap.add_argument("--orientation", default="auto", choices=["auto", "v", "h", "both"])
    ap.add_argument("--crack-shape", default="none", choices=["none", "thin"])
    ap.add_argument("--band-radius", type=int, default=2)
    ap.add_argument("--alpha-sim", type=float, default=11.0)
    ap.add_argument("--fg-side", default="gt", choices=["gt", "lt"])
    ap.add_argument("--fix-mode", default="copy", choices=["copy", "move", "bg"])
    ap.add_argument("--skip-ghosts", action="store_true")
    ap.add_argument("--n-window", type=int, default=69)
    ap.add_argument("--sizes", default="9,7,5,3")
    ap.add_argument("--beta", type=float, default=150.0)
    ap.add_argument("--beta-mode", default="mean", choices=["mean", "sum"])
    ap.add_argument("--ablate", default="none",
                    choices=["none", "nob", "nodepth", "nocd", "noconf"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    cfg = build_config(args)
    log = (lambda s: None) if args.quiet else (lambda s: print(s, flush=True))

    rgb, inv = vio.load_pair(args.image, args.depth, args.depth_mode)
    gt = vio.load_image(args.gt) if args.gt else None

    if args.warped:
        if not args.hole or not args.depth_warped:
            ap.error("--warped requires --hole and --depth-warped")
        warped = vio.load_image(args.warped)
        hole = np.asarray(vio.load_depth(args.hole, "255"), np.float32) > 127
        dw = vio.load_depth(args.depth_warped, args.depth_mode)
        res = fill_warped(warped, hole, dw, rgb, inv, cfg=cfg, log=log)
        out = args.out
    else:
        res = warp_and_fill(rgb, inv, cfg, reference=(rgb, inv), log=log)
        out = os.path.join(args.out)

    st = report.save(res, cfg, out, ref_rgb=rgb, gt=gt)
    print(f"\n=== viewfill === {os.path.abspath(out)}")
    for k in ("scale", "hole_pct", "crack_px", "ghost_px", "holes_before",
              "holes_after", "components", "iterations", "seconds_fill",
              "back_proj_psnr_before", "back_proj_psnr_after",
              "gt_psnr_frame_warped", "gt_psnr_frame_filled"):
        if k in st:
            print(f"  {k:26s} {st[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
