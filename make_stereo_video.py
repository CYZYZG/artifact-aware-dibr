"""2D -> 3D: center view + inverse depth  ->  left/right views  ->  stereo video.

    python make_stereo_video.py --images input --depths input --out out/sbs.mp4
    python make_stereo_video.py --images mid/ --depths dep/ --out sbs.mp4 --layout half-sbs
                                 --fps 30 --total-pct 3 --near-pct 1 --limit 20

Pairing: images and depths are matched by the LAST number in the file name, so both
`color-cam6-f000.jpg` + `depth-cam6-f000.png` and `0001.png` + `0001_depth.png` work; if the
two sets share no number the files are paired in sorted order.

Disparity model (逆深度转视差.md):  Delta = (far% - total% * inv) * W ,  d_R = +Delta/2 ,
d_L = -Delta/2 , far% = total% - near% .  Defaults: total 3 %, near 1 % (=> far 2 %), i.e.
nearest -> right eye -0.5 %W / left eye +0.5 %W, farthest -> right +1 %W / left -1 %W.
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viewfill import io as vio                        # noqa: E402
from viewfill.config import FillConfig                # noqa: E402
from viewfill.stereo import make_sbs, stereo_pair     # noqa: E402
from viewfill.video import VideoWriter, video_info    # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
# 同一目录混放时，用文件名关键字区分"哪张是逆深度"
DEPTH_HINT = re.compile(r"(depth|disp|disparity|inv|dpt)", re.I)


def frame_key(name):
    """Frame id = the LAST run of digits in the file name (zero padded for sorting)."""
    nums = re.findall(r"\d+", os.path.splitext(os.path.basename(name))[0])
    return nums[-1].zfill(6) if nums else os.path.splitext(os.path.basename(name))[0]


def collect(dirs, exts=IMG_EXT):
    files = []
    for d in dirs:
        if os.path.isdir(d):
            files += [p for p in glob.glob(os.path.join(d, "*"))
                      if os.path.splitext(p)[1].lower() in exts]
        elif os.path.isfile(d) and os.path.splitext(d)[1].lower() in exts:
            files.append(d)
    return sorted(files, key=lambda p: (frame_key(p), p))


def pair_files(images, depths):
    """(pairs, counts) - pairs = [(image, depth)], counted by shared frame id."""
    di = {}
    for p in depths:
        di.setdefault(frame_key(p), p)
    pairs = [(p, di[frame_key(p)]) for p in images if frame_key(p) in di]
    if not pairs:                                  # no shared numbering -> sorted order
        pairs = list(zip(images, depths))
    return pairs, len(images), len(depths)


def main():
    ap = argparse.ArgumentParser(description="center view + inverse depth -> stereo video")
    ap.add_argument("--images", nargs="+", required=True, help="图像目录或文件")
    ap.add_argument("--depths", nargs="+", required=True, help="逆深度目录或文件")
    ap.add_argument("--out", default="out/sbs.mp4", help="输出视频（full-SBS = 2W x H）")
    ap.add_argument("--layout", default="full-sbs", choices=["full-sbs", "half-sbs", "tab"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--total-pct", type=float, default=3.0, help="总视差占宽度百分比")
    ap.add_argument("--near-pct", type=float, default=1.0, help="出屏最大百分比（其余为入屏）")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 帧（0 = 全部）")
    ap.add_argument("--start", type=int, default=0, help="从第 N 帧开始")
    ap.add_argument("--frames-dir", default="", help="每帧左右眼 PNG 的输出目录")
    ap.add_argument("--no-frames", action="store_true", help="不保存每帧 PNG")
    ap.add_argument("--panel-every", type=int, default=10, help="每 N 帧存一张三联对照图")
    ap.add_argument("--ghosts", action="store_true",
                    help="启用论文 §II-B 的鬼影搬移（默认关闭：它会改写前景内完好像素）")
    ap.add_argument("--splat", default="sub", choices=["sub", "floor", "round"])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    images, depths = collect(a.images), collect(a.depths)
    if set(images) & set(depths):          # 同一目录给了两次（常见：图与深度混放）
        allf = sorted(set(images) | set(depths), key=lambda p: (frame_key(p), p))
        depths = [p for p in allf if DEPTH_HINT.search(os.path.basename(p))]
        images = [p for p in allf if p not in depths]
        print("提示：图像与逆深度在同一目录，已按文件名关键字自动区分"
              "（图像 %d / 深度 %d；若识别有误请用 --images/--depths 分别指定目录）"
              % (len(images), len(depths)))
    pairs, ni, nd = pair_files(images, depths)
    if a.start:
        pairs = pairs[a.start:]
    if a.limit:
        pairs = pairs[:a.limit]
    if not pairs:
        sys.exit("没有可配对的 图像/逆深度（用 --images/--depths 指定目录或文件）")

    out_dir = os.path.dirname(os.path.abspath(a.out))
    fdir = a.frames_dir or os.path.join(out_dir, "frames")
    os.makedirs(fdir, exist_ok=True)
    log = None if a.quiet else (lambda s: print("     " + s, flush=True))
    cfg = FillConfig(splat=a.splat)
    total, near = a.total_pct / 100.0, a.near_pct / 100.0

    print("配对 %d 帧（图像 %d / 深度 %d）" % (len(pairs), ni, nd))
    print("视差：总 %.2f%%W，出屏 %.2f%%W，入屏 %.2f%%W | 布局 %s | %.0f fps"
          % (a.total_pct, a.near_pct, a.total_pct - a.near_pct, a.layout, a.fps))
    if not a.no_frames:
        print("每帧左右眼 PNG -> %s" % fdir)

    rows = []
    vw = VideoWriter(a.out, fps=a.fps, crf=a.crf, quiet=a.quiet)
    for i, (ip, dp) in enumerate(pairs):
        t = time.time()
        rgb, inv = vio.load_pair(ip, dp)
        res = stereo_pair(rgb, inv, cfg=cfg, total_pct=total, near_pct=near, log=log,
                          skip_ghosts=not a.ghosts)
        sbs = make_sbs(res["left"], res["right"], a.layout)
        vw.append(sbs)
        tag = frame_key(ip)
        if not a.no_frames:
            os.makedirs(os.path.join(fdir, "left"), exist_ok=True)
            os.makedirs(os.path.join(fdir, "right"), exist_ok=True)
            vio.save_image(os.path.join(fdir, "left", tag + ".png"), res["left"])
            vio.save_image(os.path.join(fdir, "right", tag + ".png"), res["right"])
            vio.save_image(os.path.join(fdir, "sbs_%06d.png" % i), sbs)
            if a.panel_every and i % a.panel_every == 0:
                mid = np.clip(rgb, 0, 255).astype(np.uint8)
                vio.save_image(os.path.join(fdir, "panel_%06d.png" % i),
                               np.hstack([mid, res["left"], res["right"]]))
        s = res["stats"]
        rows.append(dict(tag=tag, hl=s["left"]["hole_pct"], hr=s["right"]["hole_pct"],
                         res=s["left"]["residual"] + s["right"]["residual"],
                         sec=time.time() - t, bl=s["left"]["back_proj_after"],
                         br=s["right"]["back_proj_after"]))
        if not a.quiet:
            print("  [%3d/%3d] %s  空洞 L/R %.2f%%/%.2f%%  残留 %d  回投 %.1f/%.1f dB  %.1fs"
                  % (i + 1, len(pairs), tag, rows[-1]["hl"], rows[-1]["hr"], rows[-1]["res"],
                     rows[-1]["bl"] or 0, rows[-1]["br"] or 0, rows[-1]["sec"]))
    info = vw.close()
    n, w, h, fps = video_info(a.out)
    print("\n完成：%s" % a.out)
    print("  编码器 %s | 帧数 %d | 尺寸 %dx%d | %.1f fps | %.1f MB"
          % (info["encoder"], n, w, h, fps, os.path.getsize(a.out) / 1e6))
    print("  平均空洞率 左 %.2f%% / 右 %.2f%% | 残留空洞合计 %d px | 平均 %.2f s/帧"
          % (np.mean([r["hl"] for r in rows]), np.mean([r["hr"] for r in rows]),
             sum(r["res"] for r in rows), np.mean([r["sec"] for r in rows])))
    if not a.no_frames:
        print("  每帧图：%s\\{left,right,sbs_*.png}" % fdir)


if __name__ == "__main__":
    main()
