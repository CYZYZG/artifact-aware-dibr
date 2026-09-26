"""viewfill.report - metrics and inspection figures."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dibr import viz as _viz                # noqa: E402
from dibr import warp as _warp              # noqa: E402


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def back_projection_check(res, ref_rgb, cfg):
    """GT-free quality signal: warp the result back to the reference viewpoint.

    The forward warp is x -> x + d(x); the inverse is a scatter t -> t - d(t) using the
    FILLED depth, so a wrong fill shows up as a disagreement with the reference image.
    The same measurement is taken on the unfilled warp, so the two numbers are paired
    (identical geometry and mask) and directly comparable.
    """
    D = np.asarray(res["D_filled"], np.float32)
    d = (D / 255.0 * float(cfg.scale)).astype(np.float32)
    back_f, _, hole_f, _ = _warp.forward_warp(res["I_filled"], -d, z=D, splat="floor",
                                              rule="zbuf")
    D_in = np.asarray(res.get("D_warp", res["D_w"]), np.float32).copy()
    d_in = (np.maximum(D_in, 0.0) / 255.0 * float(cfg.scale)).astype(np.float32)
    back_w, _, hole_w, _ = _warp.forward_warp(np.asarray(res.get("I_warp", res["I_w"]),
                                                         np.float32),
                                              -d_in, z=np.maximum(D_in, 0.0),
                                              splat="floor", rule="zbuf")
    m = ~hole_f
    out = {}
    if m.sum():
        out["back_proj_mask_px"] = int(m.sum())
        out["back_proj_psnr_before"] = round(_viz.psnr(ref_rgb, back_w, mask=m), 3)
        out["back_proj_psnr_after"] = round(_viz.psnr(ref_rgb, back_f, mask=m), 3)
        out["back_proj_mae_before"] = round(float(np.abs(
            _gray(ref_rgb) - _gray(back_w))[m].mean()), 3)
        out["back_proj_mae_after"] = round(float(np.abs(
            _gray(ref_rgb) - _gray(back_f))[m].mean()), 3)
    out["_back_after"] = back_f
    return out


def _gray(img):
    a = np.asarray(img, np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def metrics(res, cfg, ref_rgb=None, gt=None):
    """Collect the stage metrics into one flat dict."""
    st = dict(res["stats"])
    st["scale"] = cfg.scale
    st["splat"] = cfg.splat
    st["rule"] = cfg.rule
    st["hole_pct"] = round(float(res["hole_mask"].mean() * 100), 4)
    st["filled_px_actual"] = int(res["hole_mask"].sum() - res["remaining"].sum())
    if ref_rgb is not None:
        try:
            st.update({k: v for k, v in back_projection_check(res, ref_rgb, cfg).items()
                       if not k.startswith("_")})
        except Exception as exc:      # pragma: no cover
            st["back_proj_error"] = str(exc)
    if gt is not None:
        m = ~res["remaining"]
        st["gt_psnr_warped"] = round(_viz.psnr(gt, res["I_w"], mask=m), 3)
        st["gt_psnr_filled"] = round(_viz.psnr(gt, res["I_filled"], mask=m), 3)
        st["gt_ssim_warped"] = round(_viz.ssim(gt, res["I_w"], mask=m), 5)
        st["gt_ssim_filled"] = round(_viz.ssim(gt, res["I_filled"], mask=m), 5)
        st["gt_psnr_frame_warped"] = round(_viz.psnr(gt, res["I_w"]), 3)
        st["gt_psnr_frame_filled"] = round(_viz.psnr(gt, res["I_filled"]), 3)
        st["gt_ssim_frame_warped"] = round(_viz.ssim(gt, res["I_w"]), 5)
        st["gt_ssim_frame_filled"] = round(_viz.ssim(gt, res["I_filled"]), 5)
        if res["hole_mask"].sum():
            st["gt_psnr_in_filled_holes"] = round(
                _viz.psnr(gt, res["I_filled"], mask=res["hole_mask"]), 3)
            st["gt_psnr_in_empty_holes"] = round(
                _viz.psnr(gt, res["I_w"], mask=res["hole_mask"]), 3)
    return st


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def panel(res, ref_rgb, cfg, gt=None, title=""):
    I_w = res["warped"]
    I_f = res["filled"]
    hole = res["hole_mask"]
    ov_b = _viz.overlay_mask(I_w, hole, (255, 0, 0))
    ov_a = _viz.overlay_mask(I_f, res["remaining"], (255, 0, 0))
    tiles = [
        _viz.label(_viz.to_u8(ref_rgb), "1) reference image"),
        _viz.label(_viz.to_u8(I_w), f"2) warped (scale {cfg.scale:+g})"),
        _viz.label(ov_b, f"3) holes before (red) {hole.mean()*100:.2f}%"),
        _viz.label(_viz.to_u8(I_f), "4) filled"),
        _viz.label(ov_a, f"5) holes after (red) {res['remaining'].sum()} px"),
        _viz.label(_viz.colorize(np.maximum(np.asarray(res["warped_depth"], np.float32), 0),
                                 valid_mask=~hole, vmin=0, vmax=255),
                   "6) warped depth (grey = hole)"),
    ]
    ys, xs = np.nonzero(hole)
    if ys.size:
        n, _, cc, cent = cv2.connectedComponentsWithStats(hole.astype(np.uint8), 8)
        k = 1 + int(np.argmax(cc[1:, cv2.CC_STAT_AREA]))
        cy, cx = int(cent[k][1]), int(cent[k][0])
        half = 120
        tiles += [_viz.label(_viz.zoom(_viz.to_u8(I_w), cx, cy, half), "7) zoom before"),
                  _viz.label(_viz.zoom(_viz.to_u8(I_f), cx, cy, half), "8) zoom after"),
                  _viz.label(_viz.zoom(_viz.to_u8(ref_rgb), cx, cy, half), "9) zoom reference")]
        if gt is not None:
            tiles.append(_viz.label(_viz.zoom(_viz.to_u8(gt), cx, cy, half), "10) zoom GT"))
    return _viz.grid(tiles, cols=3)


def save(res, cfg, out_dir, ref_rgb=None, gt=None, write_arrays=True):
    """Write the artefacts of one run; returns the metrics dict."""
    os.makedirs(out_dir, exist_ok=True)
    from . import io as vio
    vio.save_image(os.path.join(out_dir, "01_warped.png"), res["warped"])
    vio.save_image(os.path.join(out_dir, "02_hole_mask.png"),
                   np.repeat((res["hole_mask"] * 255).astype(np.uint8)[:, :, None], 3, 2))
    vio.save_image(os.path.join(out_dir, "03_filled.png"), res["filled"])
    vio.save_image(os.path.join(out_dir, "04_overlay_before.png"),
                   _viz.overlay_mask(res["warped"], res["hole_mask"], (255, 0, 0)))
    vio.save_image(os.path.join(out_dir, "05_overlay_after.png"),
                   _viz.overlay_mask(res["filled"], res["remaining"], (255, 0, 0)))
    vio.save_gray(os.path.join(out_dir, "06_warped_depth.png"), res["warped_depth"],
                  valid_mask=~res["hole_mask"], vmin=0, vmax=255)
    vio.save_gray(os.path.join(out_dir, "07_filled_depth.png"), res["D_filled"],
                  valid_mask=~res["remaining"], vmin=0, vmax=255)
    # raw (non-colourised) depth maps so the outputs can be fed back in via
    # --warped/--hole/--depth-warped for a second pass or for downstream use
    vio.save_image(os.path.join(out_dir, "06b_warped_depth_raw.png"),
                   np.repeat(np.clip(np.maximum(np.asarray(res["warped_depth"]), 0), 0, 255)
                             .astype(np.uint8)[:, :, None], 3, 2))
    vio.save_image(os.path.join(out_dir, "07b_filled_depth_raw.png"),
                   np.repeat(np.clip(np.asarray(res["D_filled"]), 0, 255)
                             .astype(np.uint8)[:, :, None], 3, 2))
    st = metrics(res, cfg, ref_rgb=ref_rgb, gt=gt)
    if ref_rgb is not None:
        vio.save_image(os.path.join(out_dir, "08_panel.png"),
                       panel(res, ref_rgb, cfg, gt=gt))
    if write_arrays:
        np.save(os.path.join(out_dir, "filled.npy"), res["I_filled"].astype(np.float32))
        np.save(os.path.join(out_dir, "filled_depth.npy"),
                res["D_filled"].astype(np.float32))
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1, ensure_ascii=False, default=str)
    return st
