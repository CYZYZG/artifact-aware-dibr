"""Run the Step 3 option matrix and tabulate the ground-truth outcomes."""
import csv
import os
import sys

sys.path.insert(0, r"D:\项目\空洞填补")
import step3_ghosts  # noqa: E402

OUT = r"D:\项目\空洞填补\output\step3_sweep"
VARIANTS = [
    ("gt/copy/a11", ["--fg_side", "gt", "--fix_mode", "copy", "--alpha_sim", "11"]),
    ("gt/bg/a11", ["--fg_side", "gt", "--fix_mode", "bg", "--alpha_sim", "11"]),
    ("gt/move/a11", ["--fg_side", "gt", "--fix_mode", "move", "--alpha_sim", "11"]),
    ("lt/copy/a11", ["--fg_side", "lt", "--fix_mode", "copy", "--alpha_sim", "11"]),
    ("gt/copy/a5", ["--fg_side", "gt", "--fix_mode", "copy", "--alpha_sim", "5"]),
    ("gt/copy/a25", ["--fg_side", "gt", "--fix_mode", "copy", "--alpha_sim", "25"]),
    ("gt/bg/a5", ["--fg_side", "gt", "--fix_mode", "bg", "--alpha_sim", "5"]),
]
base = ["--ref_cam", "6", "--dst_cam", "7", "--frame", "f000", "--out_dir", OUT]

rows = []
for name, extra in VARIANTS:
    sys.argv = ["step3_ghosts.py"] + base + extra
    try:
        step3_ghosts.main()
    except SystemExit:
        pass
    # locate the stats.csv just written
    tag = "cam6_to_cam7_f000"
    sub = [d for d in os.listdir(os.path.join(OUT, tag))
           if d.startswith("band2_")]
    d = None
    for s in sub:
        p = os.path.join(OUT, tag, s, "stats.csv")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                r = next(csv.DictReader(fh))
            if (r["fg_side"] == extra[1] and r["fix_mode"] == extra[3]
                    and abs(float(r["alpha_sim"]) - float(extra[5])) < 1e-9):
                d = r
    if d:
        rows.append((name, d))

hdr = ["variant", "ghost", "fg_removed", "frame_PSNR", "frame_SSIM", "changed",
       "chg_PSNR_before", "chg_PSNR_after", "band_before", "band_after"]
print("\n" + " | ".join(f"{h:>15s}" for h in hdr))
for name, d in rows:
    vals = [name, d["ghost_px"], d["fg_removed_px"],
            f"{float(d['gt_psnr_frame_before']):.2f}->{float(d['gt_psnr_frame_after']):.2f}",
            f"{float(d['gt_ssim_frame_before']):.4f}->{float(d['gt_ssim_frame_after']):.4f}",
            d.get("changed_px", "-"),
            d.get("gt_psnr_changed_before", "-"), d.get("gt_psnr_changed_after", "-"),
            d.get("gt_psnr_band_before", "-"), d.get("gt_psnr_band_after", "-")]
    print(" | ".join(f"{str(v):>15s}" for v in vals))
