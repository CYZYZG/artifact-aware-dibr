"""Step 8 - run the whole pipeline over frames and camera pairs and aggregate the metrics.

    python run_all.py --pairs 6:7 --frames f000,f001        # two frame pairs
    python run_all.py --pairs 6:7,6:5 --frames all          # all 10 frames
    python run_all.py --pairs 3:0,3:2 --frames f000,f004     # the paper's Ballet setup

Each run chains step1_warp -> step2_cracks -> step3_ghosts -> step4_inpaint and writes
`output/step8_eval/summary.csv` plus a markdown report.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dibr import io_utils, viz  # noqa: E402

PY = sys.executable


def paths(ref, dst, frame):
    tag = f"cam{ref}_to_cam{dst}_{frame}"
    return dict(
        s1=os.path.join(HERE, "output", "step1_warp_int", tag),
        s2=os.path.join(HERE, "output", "step2_cracks_int", tag, "lam5_h_none"),
        s3=os.path.join(HERE, "output", "step3_ghosts", tag, "band2_gt_a11_copy"),
        s4=os.path.join(HERE, "output", "step4_inpaint", tag, "none"),
    )


def run_step(name, cmd, logdir):
    os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, name + ".log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run([PY] + cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    return time.time() - t0, p.returncode, log


def read_stats(path, kind="csv"):
    import json
    p = os.path.join(path, "stats.json" if kind == "json" else "stats.csv")
    if not os.path.isfile(p):
        return None
    if kind == "json":
        return json.load(open(p, encoding="utf-8"))
    with open(p, encoding="utf-8") as fh:
        return next(csv.DictReader(fh))


def fnum(d, k, default=float("nan")):
    try:
        return float(d[k])
    except (KeyError, TypeError, ValueError):
        return default


def run_pair(ref, dst, frame, args, root):
    P = paths(ref, dst, frame)
    tag = f"cam{ref}_to_cam{dst}_{frame}"
    logdir = os.path.join(root, "logs", tag)
    timings = {}
    done_marker = os.path.join(P["s4"], "stats.json")
    if args.skip_existing and os.path.isfile(done_marker):
        print(f"  [skip] {tag} (already complete)")
        return collect(ref, dst, frame, P, {})
    steps = [
        ("step1", ["step1_warp.py", "--ref_cam", str(ref), "--dst_cam", str(dst),
                   "--frame", frame, "--mode", "calib", "--use_dv", "--rule", "zbuf",
                   "--splat", args.splat, "--out_dir",
                   os.path.join(HERE, "output", "step1_warp_int")]),
        ("step2", ["step2_cracks.py", "--ref_cam", str(ref), "--dst_cam", str(dst),
                   "--frame", frame, "--step1_dir",
                   os.path.join(HERE, "output", "step1_warp_int"),
                   "--out_dir", os.path.join(HERE, "output", "step2_cracks_int"),
                   "--orientation", "auto", "--lam", str(args.lam)]),
        ("step3", ["step3_ghosts.py", "--ref_cam", str(ref), "--dst_cam", str(dst),
                   "--frame", frame, "--step2_dir",
                   os.path.join(HERE, "output", "step2_cracks_int"),
                   "--lam_subdir", "lam5_h_none", "--out_dir",
                   os.path.join(HERE, "output", "step3_ghosts")]),
        ("step4", ["step4_inpaint.py", "--ref_cam", str(ref), "--dst_cam", str(dst),
                   "--frame", frame, "--step1_dir",
                   os.path.join(HERE, "output", "step1_warp_int"),
                   "--step3_dir", os.path.join(HERE, "output", "step3_ghosts"),
                   "--step3_subdir", "band2_gt_a11_copy", "--beta", str(args.beta),
                   "--beta_mode", "mean", "--out_dir",
                   os.path.join(HERE, "output", "step4_inpaint")]),
    ]
    for name, cmd in steps:
        dt, rc, log = run_step(name, cmd, logdir)
        timings[name] = round(dt, 1)
        if rc != 0:
            print(f"  [FAIL] {tag} {name} (rc={rc}) see {log}")
            return None
    return collect(ref, dst, frame, P, timings)


def collect(ref, dst, frame, P, timings):
    """Gather the per-step metrics plus the whole-frame progression of one run."""
    s1 = read_stats(P["s1"], "csv")
    s2 = read_stats(P["s2"], "csv")
    s3 = read_stats(P["s3"], "csv")
    s4 = read_stats(P["s4"], "json")
    try:
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
    except FileNotFoundError:
        gt = None
    stage = {}
    if gt is not None:
        for name, path in (("warp", os.path.join(P["s1"], "warped_color.npy")),
                           ("cracks", os.path.join(P["s2"], "I_filled.npy")),
                           ("ghosts", os.path.join(P["s3"], "I_fixed.npy")),
                           ("filled", os.path.join(P["s4"], "final_color.npy"))):
            if os.path.isfile(path):
                img = np.load(path).astype(np.float32)
                stage[f"psnr_{name}"] = round(viz.psnr(gt, img), 3)
                stage[f"ssim_{name}"] = round(viz.ssim(gt, img), 5)
    row = dict(tag=f"cam{ref}_to_cam{dst}_{frame}", ref_cam=ref, dst_cam=dst, frame=frame,
               hole_pct=fnum(s1, "hole_ratio_pct"),
               warp_psnr_valid=fnum(s1, "psnr_warped_valid"),
               warp_ssim_valid=fnum(s1, "ssim_warped_valid"),
               crack_px=fnum(s2, "crack_px"),
               crack_gt_gain=round(fnum(s2, "gt_psnr_all_crack_after")
                                   - fnum(s2, "gt_psnr_all_crack_before"), 3),
               ghost_px=fnum(s3, "ghost_px"),
               holes_before_fill=fnum(s4, "holes_before"),
               holes_after_fill=fnum(s4, "holes_after"),
               fill_iterations=fnum(s4, "iterations"),
               fill_px_psnr=fnum(s4, "gt_psnr_in_holes_after"),
               final_frame_psnr=fnum(s4, "gt_psnr_frame_after"),
               final_frame_ssim=fnum(s4, "gt_ssim_frame_after"),
               sec_step1=timings.get("step1"), sec_step2=timings.get("step2"),
               sec_step3=timings.get("step3"), sec_step4=timings.get("step4"))
    row.update(stage)
    return row


def main():
    ap = argparse.ArgumentParser(description="Step 8: full pipeline + evaluation")
    ap.add_argument("--pairs", default="6:7", help="comma separated ref:dst camera pairs")
    ap.add_argument("--frames", default="all", help="'all' (f000..f099 as available), "
                                                    "or a comma separated list")
    ap.add_argument("--n_frames", type=int, default=10,
                    help="how many frames when --frames all")
    ap.add_argument("--splat", default="floor", choices=["floor", "round", "sub"])
    ap.add_argument("--beta", type=float, default=150.0)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "output", "step8_eval"))
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    pairs = [tuple(int(v) for v in p.split(":")) for p in args.pairs.split(",")]
    if args.frames.strip().lower() == "all":
        frames = [f"f{i:03d}" for i in range(args.n_frames)]
    else:
        frames = [f.strip() for f in args.frames.split(",") if f.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    t0 = time.time()
    for ref, dst in pairs:
        for fr in frames:
            print(f"=== cam{ref} -> cam{dst} {fr} ===", flush=True)
            r = run_pair(ref, dst, fr, args, args.out_dir)
            if r:
                rows.append(r)
                print(f"  final PSNR {r['final_frame_psnr']:.2f} dB  "
                      f"SSIM {r['final_frame_ssim']:.4f}  "
                      f"holes {int(r['holes_after_fill'])}  "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
    if not rows:
        print("nothing collected")
        return 1

    keys = list(rows[0].keys())
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    # ---- markdown report -----------------------------------------------------
    lines = ["# Step 8 - full pipeline evaluation", "",
             f"configuration: splat={args.splat}, lambda={args.lam:g}, beta={args.beta:g}, "
             f"SE orientation=auto", "",
             "## per frame", "",
             "| pair | frame | holes % | cracks px | ghosts px | fill iters | filled-hole "
             "PSNR | final PSNR | final SSIM | warped→cracked→ghosted→filled PSNR |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(
            f"| cam{r['ref_cam']}→cam{r['dst_cam']} | {r['frame']} | {r['hole_pct']:.2f} | "
            f"{int(r['crack_px'])} | {int(r['ghost_px'])} | {int(r['fill_iterations'])} | "
            f"{r['fill_px_psnr']:.2f} | {r['final_frame_psnr']:.2f} | "
            f"{r['final_frame_ssim']:.4f} | "
            f"{r.get('psnr_warp', float('nan')):.2f}→{r.get('psnr_cracks', float('nan')):.2f}→"
            f"{r.get('psnr_ghosts', float('nan')):.2f}→{r.get('psnr_filled', float('nan')):.2f} |")
    lines += ["", "## averages per camera pair", "",
              "| pair | frames | final PSNR | final SSIM | holes % | time/frame (s) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for ref, dst in pairs:
        sel = [r for r in rows if r["ref_cam"] == ref and r["dst_cam"] == dst]
        if not sel:
            continue
        secs = np.mean([sum(r[k] or 0 for k in
                            ("sec_step1", "sec_step2", "sec_step3", "sec_step4"))
                        for r in sel])
        lines.append(f"| cam{ref}→cam{dst} | {len(sel)} | "
                     f"{np.mean([r['final_frame_psnr'] for r in sel]):.2f} | "
                     f"{np.mean([r['final_frame_ssim'] for r in sel]):.4f} | "
                     f"{np.mean([r['hole_pct'] for r in sel]):.2f} | {secs:.1f} |")
    lines += ["", "## reference: paper Table I (PSNR / SSIM)",
              "",
              "| Test | [17] | [26] | [6] | [16] | [9] | [2] | Ours |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |",
              "| Still Image | 28.67 / 9.271 | 28.90 / 9.382 | 27.55 / 9.197 | "
              "27.82 / 9.370 | 28.43 / 9.322 | 30.57 / 9.384 | 31.74 / 9.464 |",
              "| MVD | 27.57 / 8.271 | 27.75 / 8.391 | 28.39 / 8.303 | 25.58 / 8.088 | "
              "28.35 / 8.276 | 28.90 / 8.411 | 29.17 / 8.463 |",
              "",
              "(SSIM x 10^-1; the paper's Ballet setup is view 4 -> views 1 and 3, i.e. "
              "cam3 -> cam0 and cam3 -> cam2 in the 0-based naming of this dataset.)"]
    rep = os.path.join(args.out_dir, "report.md")
    open(rep, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nwrote {csv_path}\nwrote {rep}")
    print(f"total wall time {time.time()-t0:.0f}s for {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
