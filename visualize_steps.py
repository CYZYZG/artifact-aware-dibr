"""Visualise every stage of the warp+fill pipeline for one frame / one eye.

    python visualize_steps.py --frame f003 --eye right --x 332 --y 197 --zoom 90

Writes to output/steps/<frame>_<eye>/:
  NN_<stage>.png         full-frame stage images
  contact_full.png       all full-frame stages in a grid
  contact_zoom.png       the same stages cropped around (--x, --y) at 3x
  log.txt                per-stage numbers

Stages
  01 mid            中间图（参考）
  02 depth          逆深度（伪彩）+ 该行剖面
  03 dx             右/左眼位移场（伪彩）
  04 dx_grad        位移场沿 x 的梯度，>1 px/px 的地方必然拉缝（蓝=拉缝区）
  05 warp           前向 warp 结果（未填补）+ 空洞（红）
  06 hole           空洞掩码单独看
  07 crack          裂纹检测：空裂纹(品红) / 半透明裂纹(青)
  08 hhf            裂纹/HHF 填补后 + HHF 写入处(绿)
  09 ghost_delta    鬼影步改动（默认 vs skip_ghosts），前景内被改写处(橙)
  10 final          示例块填补后的最终结果 + 示例块写入处(黄)
  11 residual       与"按位移对齐的中间图"的残差热图
"""
import argparse
import os
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from viewfill import io as vio                                    # noqa: E402
from viewfill.config import FillConfig                             # noqa: E402
from viewfill.pipeline import depth_to_scale255, fill_warped, auto_se_orientation  # noqa: E402
from dibr import warp as W                                        # noqa: E402
from dibr import cracks as C                                      # noqa: E402

GRID_COLS = 3


def u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def overlay(base, mask, color, alpha=0.55):
    out = base.copy()
    m = mask.astype(bool)
    if m.ndim == 3:
        m = m.any(2)
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color, np.float32)).astype(np.uint8)
    return out


def colorize(v):
    v = v.astype(np.float32)
    rng = float(v.max() - v.min()) or 1.0
    z = ((v - v.min()) / rng * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(z, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def label(img, text, scale=0.7):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 26 + 11 * len(text)), 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2,
                cv2.LINE_AA)
    return out


def grid(tiles, cols=GRID_COLS, pad=6):
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[pad + r * (h + pad):pad + r * (h + pad) + h,
               pad + c * (w + pad):pad + c * (w + pad) + w] = t
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="f003")
    ap.add_argument("--cam", default="cam6")
    ap.add_argument("--eye", default="right", choices=["right", "left"])
    ap.add_argument("--x", type=int, default=332)
    ap.add_argument("--y", type=int, default=197)
    ap.add_argument("--zoom", type=int, default=90, help="裁剪半径")
    ap.add_argument("--zoom-scale", type=int, default=3)
    ap.add_argument("--total-pct", type=float, default=3.0)
    ap.add_argument("--near-pct", type=float, default=1.0)
    ap.add_argument("--slit-only", action="store_true", default=True,
                    help="与立体默认一致：只把 1-2 px 细缝判为裂纹")
    ap.add_argument("--no-slit-only", dest="slit_only", action="store_false")
    ap.add_argument("--ghosts", action="store_true")
    ap.add_argument("--same-surface", action="store_true", default=True,
                    help="启用单表面约束（立体默认开）")
    ap.add_argument("--no-same-surface", dest="same_surface", action="store_false")
    ap.add_argument("--surf-tol", type=float, default=20.0)
    a = ap.parse_args()

    out = os.path.join(HERE, "output", "steps", "%s_%s" % (a.frame, a.eye))
    os.makedirs(out, exist_ok=True)
    img_p = os.path.join(HERE, "input", "color-%s-%s.jpg" % (a.cam, a.frame))
    dep_p = os.path.join(HERE, "input", "depth-%s-%s.png" % (a.cam, a.frame))
    rgb, inv = vio.load_pair(img_p, dep_p)
    mid = u8(rgb)
    P = depth_to_scale255(inv)
    H, Wd = P.shape
    total, near = a.total_pct / 100.0, a.near_pct / 100.0
    inv01 = P / 255.0
    delta = ((total - near) - total * inv01) * Wd                  # + = 入屏
    dx = np.ascontiguousarray((delta / 2.0) if a.eye == "right" else (-delta / 2.0))
    dydx = np.zeros_like(dx)
    dydx[:, :-1] = np.diff(dx, axis=1)

    # ---- stages -------------------------------------------------------------
    I_w, D_w, hole, wt = W.forward_warp(rgb.astype(np.float32), dx, None, z=P,
                                        hole_depth=-1.0, rule="zbuf", splat="sub")
    orient = auto_se_orientation(dx, dx * 0)
    res1 = C.fill_cracks(I_w, D_w, lam=5.0, se_len=4, orientation=orient, hhf_sigma=1.0,
                         hhf_ksize=5, shape_filter="none", max_thickness=3.0,
                         slit_only=a.slit_only, translucent="keep",
                         same_surface=a.same_surface, surf_tol=a.surf_tol)
    diso = C.disocclusion_mask(D_w, hole, surf_tol=a.surf_tol, orientation=orient)
    crack_kept = res1["crack"]
    crack_dropped = (hole & diso)
    I_hhf, D_hhf = res1["I_filled"], res1["D_filled"]
    cfg = FillConfig(crack_translucent="keep", skip_ghosts=not a.ghosts,
                     slit_only=a.slit_only)
    res_final = fill_warped(I_w, hole, D_w, rgb, P, disp=(dx, None), cfg=cfg)
    final = res_final["filled"]
    # 混合比 t: 0=纯背景(好) 0.5=前后景各半(重影) 1=纯前景(坏)
    from dibr.cracks import _nearest_valid
    gW = cv2.cvtColor(u8(I_w), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lo, ro = _nearest_valid(~hole)
    cols = np.arange(Wd)[None, :].repeat(H, 0)
    xl = np.clip(lo, 0, Wd - 1).astype(np.int32)
    xr = np.clip(ro, 0, Wd - 1).astype(np.int32)
    rows = np.arange(H)[:, None]
    Lc, Rc = gW[rows, xl], gW[rows, xr]
    den = Lc - Rc
    tmap = np.zeros((H, Wd), np.float32)
    ok = hole & (np.abs(den) > 12)
    tmap[ok] = np.clip((cv2.cvtColor(u8(final), cv2.COLOR_RGB2GRAY).astype(np.float32)[ok]
                        - Rc[ok]) / den[ok], -0.5, 1.5)
    final_gh = fill_warped(I_w, hole, D_w, rgb, P, disp=(dx, None),
                           cfg=FillConfig(crack_translucent="keep", skip_ghosts=True,
                                          slit_only=a.slit_only))["filled"]

    hole_mask = hole
    crack_empty = res1["empty_crack"]
    crack_tr = res1["translucent_crack"]
    hhf_wrote = np.abs(I_hhf.astype(np.int16) - I_w.astype(np.int16)).max(2) > 0
    exem_wrote = (np.abs(final.astype(np.int16) - I_hhf.astype(np.int16)).max(2) > 0) \
        & ~hhf_wrote
    ghost_px = (np.abs(final.astype(np.int16) - final_gh.astype(np.int16)).max(2) > 0)
    ghost_nohole = ghost_px & ~hole
    gy, gx = np.mgrid[0:H, 0:Wd]
    sx = np.clip(np.round(gx - dx).astype(np.int32), 0, Wd - 1)
    resid = np.abs(final.astype(np.int16) - mid[gy, sx].astype(np.int16)).mean(2)

    stages = [
        ("01 mid 中间图", mid),
        ("02 depth 逆深度", colorize(P)),
        ("03 dx %s 位移场" % a.eye, colorize(dx)),
        ("04 dx_grad |ddx/dx| (蓝=>1px 拉缝)", overlay(mid, np.abs(dydx) > 1.0, (0, 0, 255))),
        ("05 warp 未填补 (红=空洞)", overlay(u8(I_w), hole_mask, (255, 0, 0))),
        ("06 hole 空洞掩码", overlay(mid, hole_mask, (255, 0, 0), 0.85)),
        ("07 crack 裂纹 (品红=空 青=半透明)", overlay(overlay(u8(I_w), crack_empty, (255, 0, 255), 0.8),
                                                      crack_tr, (0, 255, 255), 0.8)),
        ("08 hhf 填补后 (绿=HHF写入)", overlay(u8(I_hhf), hhf_wrote, (0, 255, 0), 0.7)),
        ("09 ghost 鬼影步改动 (橙=前景内被改写)", overlay(final, ghost_nohole, (255, 128, 0), 0.85)),
        ("10 final 示例块写入 (黄)", overlay(final, exem_wrote, (255, 255, 0), 0.7)),
        ("11 residual |final - mid(对齐)| x4", cv2.cvtColor(
            cv2.applyColorMap(np.clip(resid * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB)),
        ("12 diso 跨表面遮挡带(蓝)", overlay(mid, diso, (0, 0, 255), 0.8)),
        ("13 crack_drop 被移出裂纹类(橙)", overlay(u8(I_w), crack_dropped & ~crack_kept,
                                                  (255, 128, 0), 0.9)),
        ("14 t 混合比 蓝=背景 绿=各半重影 红=前景", cv2.cvtColor(
            np.stack([np.clip(tmap, 0, 1) * 255,
                      np.clip(1 - 2 * np.abs(tmap - 0.5), 0, 1) * 255,
                      np.clip(1 - tmap, 0, 1) * 255], -1).astype(np.uint8), cv2.COLOR_BGR2RGB)),
    ]
    for i, (name, img) in enumerate(stages):
        vio.save_image(os.path.join(out, "%02d_%s.png" % (i + 1, name.split()[1])), img)

    r = a.zoom
    x0, x1 = max(0, a.x - r), min(Wd, a.x + r)
    y0, y1 = max(0, a.y - r), min(H, a.y + r)
    Z = a.zoom_scale
    zoom = [label(cv2.resize(s[1][y0:y1, x0:x1],
                             ((x1 - x0) * Z, (y1 - y0) * Z), interpolation=cv2.INTER_NEAREST),
                  s[0].split()[0], 1.2) for s in stages]
    vio.save_image(os.path.join(out, "contact_full.png"),
                   grid([label(s[1], s[0]) for s in stages]))
    vio.save_image(os.path.join(out, "contact_zoom.png"), grid(zoom, cols=4))

    # ---- log ---------------------------------------------------------------
    lines = []
    lines.append("frame %s  eye %s  视差 总 %.2f%%W 出屏 %.2f%%W  (布局: 位移 = %s)"
                 % (a.frame, a.eye, a.total_pct, a.near_pct, a.eye))
    lines.append("裁剪区域 x∈[%d,%d) y∈[%d,%d)  放大 %dx" % (x0, x1, y0, y1, Z))
    lines.append("")
    lines.append("位移场: d 范围 [%.2f, %.2f] px   |dd/dx|>1 的像素 %d (%.2f%%)"
                 % (dx.min(), dx.max(), int((np.abs(dydx) > 1).sum()),
                    (np.abs(dydx) > 1).mean() * 100))
    t = (slice(y0, y1), slice(x0, x1))
    lines.append("  [裁剪区] d 范围 [%.2f, %.2f]  |dd/dx|>1 像素 %d"
                 % (dx[t].min(), dx[t].max(), int((np.abs(dydx[t]) > 1).sum())))
    lines.append("")
    lines.append("05 warp: 空洞 %d px (%.2f%%)  [裁剪区 %d px]"
                 % (hole_mask.sum(), hole_mask.mean() * 100, int(hole_mask[t].sum())))
    lines.append("07 crack: 空裂纹 %d px / 半透明裂纹 %d px  [裁剪区 %d / %d]"
                 % (crack_empty.sum(), crack_tr.sum(), int(crack_empty[t].sum()),
                    int(crack_tr[t].sum())))
    lines.append("   HHF 无支撑像素 %d（应为 0：避免写出黑色）" % res1.get("hhf_unsupported_px", -1))
    lines.append("08 hhf: 写入 %d px（其中裁剪区 %d）" % (hhf_wrote.sum(), int(hhf_wrote[t].sum())))
    lines.append("09 ghost: 改动 %d px，其中**非空洞（前景内）** %d px  [裁剪区非空洞 %d]"
                 % (ghost_px.sum(), ghost_nohole.sum(), int(ghost_nohole[t].sum())))
    lines.append("10 final: 示例块写入 %d px（其中裁剪区 %d）| 残留空洞 %d"
                 % (exem_wrote.sum(), int(exem_wrote[t].sum()),
                    int(res_final["remaining"].sum())))
    lines.append("11 residual: 全图 均值 %.2f | >10 %.2f%% | >25 %.2f%%"
                 % (resid.mean(), (resid > 10).mean() * 100, (resid > 25).mean() * 100))
    lines.append("   [裁剪区] 残差 均值 %.2f | >25 %.2f%%   （裁剪区是问题区，全图均值会被大片正确区域拉低）"
                 % (resid[t].mean(), (resid[t] > 25).mean() * 100))
    lines.append("")
    lines.append("12 diso: 跨表面遮挡带 %d px（占空洞 %.0f%%）"
                 % (diso.sum(), diso.sum() / max(1, hole.sum()) * 100))
    lines.append("13 crack_drop: 因跨表面被移出裂纹类 %d px（原本会被 HHF 横向插值→重影）"
                 % int((crack_dropped & ~crack_kept).sum()))
    tt = tmap[ok]
    lines.append("14 t: HHF写入像素 t 均值 %.2f（0.5=前后景各半的混合）| "
                 "t∈[0.35,0.65] 的混合像素 %d px（越少越好）"
                 % (tmap[ok & crack_kept].mean() if (ok & crack_kept).any() else float("nan"),
                    int(((np.abs(tmap - 0.5) < 0.15) & ok).sum())))
    lines.append("提示: 04 的蓝色列 = 位移场膨胀处（那里必然出现空洞）; 07 的青色 = 被判为'半透明裂纹'"
                 "却并非空洞的像素（HHF 会去改它们）; 09 的橙色 = 鬼影步改写的**非空洞**像素（前景被改写）。")
    txt = "\n".join(lines)
    open(os.path.join(out, "log.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    print("\n输出目录:", out)


if __name__ == "__main__":
    main()
