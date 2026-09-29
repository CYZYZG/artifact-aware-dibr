"""Checks for the 2D -> 3D stereo pipeline (formula, warping, filling, video output).

    python test_stereo.py
"""
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from viewfill import io as vio                                   # noqa: E402
from viewfill.stereo import disparity_fields, make_sbs, stereo_pair  # noqa: E402
from viewfill.video import VideoWriter, video_info                # noqa: E402

IMG = os.path.join(HERE, "input", "color-cam6-f000.jpg")
DEP = os.path.join(HERE, "input", "depth-cam6-f000.png")
OUT = os.path.join(HERE, "output", "stereo_test")
W, H = 1024, 768
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name, ("   " + detail) if detail else ""))


def main():
    t0 = time.time()
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)

    # ---------------------------------------------------------------- 1) formula
    print("--- 1) 视差公式（对照 逆深度转视差.md 的数值表，W = %d）---" % W)
    dx_l, dx_r, delta = disparity_fields(np.array([[0.0, 2 / 3, 1.0]], np.float32), W)
    exp = {0.0: (0.02 * W, 0.01 * W), 2 / 3: (0.0, 0.0), 1.0: (-0.01 * W, -0.005 * W)}
    ok = True
    for k, (d_exp, r_exp) in exp.items():
        i = [0.0, 2 / 3, 1.0].index(k)
        d, r, l = float(delta[0, i]), float(dx_r[0, i]), float(dx_l[0, i])
        good = abs(d - d_exp) < 1e-3 and abs(r - r_exp) < 1e-3 and abs(l + r) < 1e-6
        ok = ok and good
        print("    inv=%.3f: 总视差 %+7.2f px (期望 %+7.2f) | 右眼 %+7.2f | 左眼 %+7.2f  %s"
              % (k, d, d_exp, r, l, "OK" if good else "**错误**"))
    check("总视差 delta=(0.02-0.03·inv)·W，d_R=delta/2，d_L=-delta/2（inv=0/2/3/1 三档与文档一致）", ok,
          "出屏 1%%W=%.2f px，入屏 2%%W=%.2f px，总跨度 3%%W=%.2f px"
          % (0.01 * W, 0.02 * W, 0.03 * W))
    dx_l2, dx_r2, _ = disparity_fields(np.linspace(0, 1, 11).reshape(1, -1), W)
    check("全程 d_L = -d_R（左右眼严格对称反向）",
          bool(np.allclose(dx_l2, -dx_r2)), "11 个 inv 采样点")

    # ---------------------------------------------------------------- 2) one frame
    print("\n--- 2) 单帧：warp + 填洞 ---")
    rgb, inv = vio.load_pair(IMG, DEP)
    t = time.time()
    res = stereo_pair(rgb, inv)
    dt = time.time() - t
    L, R = res["left"], res["right"]
    check("左右眼尺寸/类型正确", L.shape == (H, W, 3) and R.shape == (H, W, 3)
          and L.dtype == np.uint8 and R.dtype == np.uint8,
          "left %s right %s %s" % (L.shape, R.shape, L.dtype))
    check("左右眼空洞全部填满（残留 = 0）",
          res["stats"]["left"]["residual"] == 0 and res["stats"]["right"]["residual"] == 0,
          "空洞 L %.2f%% / R %.2f%% | %.1f s"
          % (res["stats"]["left"]["hole_pct"], res["stats"]["right"]["hole_pct"], dt))
    diso = int(res["stats"]["left"].get("disocclusion_px", 0))
    check("warp 产生空洞，且遮挡带已被背景侧延展预填（不是空跑）",
          res["stats"]["left"]["hole_pct"] > 0.1 and res["stats"]["right"]["hole_pct"] > 0.1
          and diso > 0,
          "左右剩余空洞率 %.2f%% / %.2f%% | disocclusion %d px（预填）"
          % (res["stats"]["left"]["hole_pct"], res["stats"]["right"]["hole_pct"], diso))
    check("填洞后左右眼没有近黑像素（无黑洞）",
          int((L.astype(np.float32).mean(2) < 1).sum()) == 0
          and int((R.astype(np.float32).mean(2) < 1).sum()) == 0,
          "近黑像素 0")
    check("左右眼不同（确实做了方向相反的位移）",
          float(np.abs(L.astype(np.int16) - R.astype(np.int16)).mean()) > 1.0,
          "平均像素差 %.1f" % float(np.abs(L.astype(np.int16) - R.astype(np.int16)).mean()))
    res2 = stereo_pair(rgb, inv)
    check("同一输入两次运行逐位一致（确定性）",
          np.array_equal(res["left"], res2["left"]) and np.array_equal(res["right"], res2["right"]))

    # ---------------------------------------------------------------- 3) SBS + video
    print("\n--- 3) full-SBS 合成与视频编码 ---")
    sbs = make_sbs(L, R, "full-sbs")
    check("full-SBS = 2W x H", sbs.shape == (H, 2 * W, 3), "%s" % (sbs.shape,))
    check("half-SBS = W x H", make_sbs(L, R, "half-sbs").shape == (H, W, 3))
    check("上下布局 = W x 2H", make_sbs(L, R, "tab").shape == (2 * H, W, 3))

    vid = os.path.join(OUT, "smoke.mp4")
    with VideoWriter(vid, fps=30, quiet=True) as vw:
        for _ in range(3):
            vw.append(sbs)
        n_enc = vw.n
        kind = vw.kind
    n, w, h, fps = video_info(vid)
    check("视频写出成功且可读回（3 帧 / 2048x768 / 30fps）",
          n == 3 and (w, h) == (2 * W, H) and abs(fps - 30) < 0.5,
          "编码器 %s | 读回 %d 帧 %dx%d %.1f fps | %.2f MB"
          % (kind, n, w, h, fps, os.path.getsize(vid) / 1e6))
    check("编码器不是本地 ffmpeg（使用 Python 库 imageio/imageio-ffmpeg）",
          str(kind).startswith("imageio"), "实际: %s" % kind)

    # ---------------------------------------------------------------- 4) CLI end to end
    print("\n--- 4) 命令行端到端（3 帧 + 保存左右眼 PNG）---")
    import subprocess
    out_dir = os.path.join(OUT, "cli")
    cmd = [sys.executable, os.path.join(HERE, "make_stereo_video.py"),
           "--images", os.path.join(HERE, "input"), "--depths", os.path.join(HERE, "input"),
           "--out", os.path.join(out_dir, "sbs.mp4"), "--limit", "3", "--fps", "30",
           "--quiet"]
    logf = os.path.join(OUT, "cli.log")          # 不用管道（此环境不支持捕获子进程管道）
    with open(logf, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    tail = open(logf, encoding="utf-8", errors="replace").read().strip().split("\n")
    ok = r.returncode == 0
    got = os.path.join(out_dir, "sbs.mp4")
    check("make_stereo_video.py 端到端成功", ok and os.path.exists(got),
          (" | ".join(tail[-3:]))[:220] if ok else (" | ".join(tail[-6:]))[:300])
    if ok:
        n2, w2, h2, _ = video_info(got)
        check("CLI 输出视频帧数/尺寸正确", n2 == 3 and (w2, h2) == (2 * W, H),
              "%d 帧 %dx%d" % (n2, w2, h2))
        fl = os.path.join(out_dir, "frames", "left")
        fr = os.path.join(out_dir, "frames", "right")
        check("每帧左右眼 PNG 已保存",
              len(os.listdir(fl)) == 3 and len(os.listdir(fr)) == 3,
              "%s: %d 张, %s: %d 张" % ("left", len(os.listdir(fl)), "right", len(os.listdir(fr))))

    n_fail = sum(1 for _, ok_, _ in results if not ok_)
    print("\n%d/%d checks passed   (total %.1fs)" % (len(results) - n_fail, len(results),
                                                     time.time() - t0))
    print("artefacts under %s" % OUT)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
