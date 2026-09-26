"""viewfill.pipeline - warp (optionally) and fill every hole.

Two entry points

    warp_and_fill(rgb, inv_depth, cfg)          warp + cracks + ghosts + fill
    fill_warped(rgb, hole, depth, ref, ...)     fill an image that is already warped

Both return a result dict; see `_finish` for the keys.
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dibr import cracks as _cracks          # noqa: E402
from dibr import ghosts as _ghosts          # noqa: E402
from dibr import inpaint as _inpaint        # noqa: E402
from dibr import viz as _viz                # noqa: E402
from dibr import warp as _warp              # noqa: E402

from .config import FillConfig               # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def depth_to_scale255(depth):
    """Normalise an inverse-depth map to the internal 0..255 scale used by the paper's
    thresholds (lambda = 5, T_O, the 9x9 medians all assume an 8-bit-like range)."""
    d = np.asarray(depth, np.float32)
    if d.max() > 1.0 + 1e-6:
        return d
    return d * 255.0


def disparity_from_depth(depth255, scale):
    """Signed pixel displacement: dx = (depth/255) * scale (the 2D->3D convention)."""
    return (np.asarray(depth255, np.float32) / 255.0 * float(scale)).astype(np.float32)


def auto_se_orientation(dx, dy=None):
    """The line SE must be perpendicular to the slit: a horizontal projection opens
    vertical slits, which need the HORIZONTAL structuring element."""
    mx = float(np.median(np.abs(dx)))
    my = 0.0 if dy is None else float(np.median(np.abs(dy)))
    return "h" if mx >= my else "v"


# --------------------------------------------------------------------------- #
# warping
# --------------------------------------------------------------------------- #
def warp_view(rgb, depth255, cfg, dy=None):
    """Forward warp; returns (I_w, D_w, hole_mask, weight, dx, dy)."""
    dx = disparity_from_depth(depth255, cfg.scale)
    dya = None if dy is None else np.asarray(dy, np.float32)
    I_w, D_w, hole, weight = _warp.forward_warp(
        np.asarray(rgb, np.float32), dx, dya, z=np.asarray(depth255, np.float32),
        hole_depth=-1.0, rule=cfg.rule, splat=cfg.splat)
    return I_w, D_w, hole, weight, dx, dya


# --------------------------------------------------------------------------- #
# main pipeline
# --------------------------------------------------------------------------- #
def check_warp_quality(I_w, hole, ref_rgb, depth255, disp, cfg):
    """Compare an externally produced warp against a correct Z-buffer warp.

    A forward splat without a depth test decides collisions by traversal order.  The
    provided `warping.scatter_image` with `inverse_ordering=True`, for instance, lets the
    FAR sample win, so foreground texture is replaced by background and the subject looks
    cut up.  Measuring the deviation turns that into a number instead of silently filling
    a damaged image.

    Returns (metrics, correct_warp, correct_hole_mask).
    """
    dya = None if disp[1] is None else np.asarray(disp[1], np.float32)
    I_c, _, hole_c, _ = _warp.forward_warp(
        np.asarray(ref_rgb, np.float32), np.asarray(disp[0], np.float32), dya,
        z=np.asarray(depth255, np.float32), hole_depth=-1.0, rule="zbuf", splat=cfg.splat)
    m = ~np.asarray(hole, bool)
    if not m.any():
        return dict(warp_deviation_pct=0.0, fg_deviation_pct=0.0,
                    warp_deviation_px=0, fg_deviation_px=0), I_c, hole_c
    dev = (np.abs(np.asarray(I_w, np.float32) - I_c).max(axis=2)
           > cfg.dev_threshold_gray) & m
    fg = (np.asarray(depth255, np.float32) / 255.0) > 0.6
    mfg = m & fg
    q = dict(warp_deviation_px=int(dev.sum()),
             warp_deviation_pct=round(float(dev.sum() / max(1, int(m.sum())) * 100), 3),
             fg_deviation_px=int((dev & fg).sum()),
             fg_deviation_pct=round(float((dev & fg).sum() / max(1, int(mfg.sum())) * 100),
                                    3))
    return q, I_c, hole_c


def _run_pipeline(I_w, D_w, hole, ref_rgb, ref_depth255, disp, cfg, log=None):
    """cracks -> ghosts -> classification + exemplar filling.  Mutates/returns arrays."""
    stats = {}
    t0 = time.time()
    # normalise the displacement field to a pair of real 2-D arrays (a 1-D horizontal
    # warp is dy = 0, never None, because the downstream helpers index it)
    dx_s = np.asarray(disp[0], np.float32)
    dy_s = np.zeros_like(dx_s) if disp[1] is None else np.asarray(disp[1], np.float32)
    disp = (dx_s, dy_s)
    # keep the raw warp: the crack/ghost steps modify both images, so reporting the
    # post-step arrays as "warped" would be wrong (and would break re-feeding them)
    I_warp = np.array(I_w, dtype=np.float32, copy=True)
    D_warp = np.array(D_w, dtype=np.float32, copy=True)
    hole_warp = np.array(hole, dtype=bool, copy=True)

    # ---- Step 2: cracks ----
    orient = cfg.se_orientation
    if orient == "auto":
        orient = auto_se_orientation(disp[0], disp[1])
    res1 = _cracks.fill_cracks(
        I_w, D_w, lam=cfg.lam, se_len=cfg.se_len, orientation=orient,
        hhf_sigma=cfg.hhf_sigma, hhf_ksize=cfg.hhf_ksize,
        shape_filter=cfg.crack_shape, max_thickness=cfg.max_thickness,
        fill_mode=getattr(cfg, "crack_fill", "auto"))
    I_w, D_w = res1["I_filled"], res1["D_filled"]
    hole = res1["remaining_holes"]
    stats.update(se_orientation=orient,
                 crack_bg_side_px=int(res1.get("bg_side_px", 0)),
                 crack_px=int(res1["crack"].sum()),
                 crack_empty_px=int(res1["empty_crack"].sum()),
                 crack_translucent_px=int(res1["translucent_crack"].sum()),
                 holes_after_cracks=int(hole.sum()))
    if log:
        log(f"  cracks: {stats['crack_px']} px detected (empty "
            f"{stats['crack_empty_px']} / translucent {stats['crack_translucent_px']}), "
            f"{stats['holes_after_cracks']} px left")

    # ---- Step 3: ghosts ----
    direction = 1 if float(np.median(disp[0])) > 0 else -1
    if cfg.skip_ghosts:
        I2, D2, hole2 = I_w, D_w, hole
        stats.update(ghost_px=0, holes_after_ghosts=int(hole.sum()))
    else:
        D_FG = _ghosts.extended_fg(ref_depth255)
        disp_fg = (disparity_from_depth(D_FG, cfg.scale), np.zeros_like(dx_s))
        res2 = _ghosts.detect_and_fix(
            I_w, D_w, hole, disp, disp_fg, direction=direction,
            band_radius=cfg.band_radius, alpha_trim=cfg.alpha_trim,
            alpha_sim=cfg.alpha_sim, fg_side=cfg.fg_side, ksize=cfg.ksize,
            fix_mode=cfg.fix_mode)
        I2, D2, hole2 = res2["I_fixed"], res2["D_fixed"], res2["hole_out"]
        stats.update(ghost_px=int(res2["ghost"].sum()),
                     ghost_candidates=int(res2["candidates"].sum()),
                     holes_after_ghosts=int(hole2.sum()),
                     oofa_px=int(res2["oofa"].sum()))
        if log:
            log(f"  ghosts: {stats['ghost_px']} px of "
                f"{stats['ghost_candidates']} candidates moved to p_FG")

    # ---- Steps 4-7: classify + fill ----
    res3 = _inpaint.fill_all(I2, D2, hole2, ref_rgb, ref_depth255, disp, src_of=None,
                             direction=direction, params=cfg.patch_params(),
                             ablate=cfg.ablate, oofa_frac=cfg.oofa_frac)
    stats.update(res3["stats"])
    stats["holes_after_fill"] = int(res3["remaining"].sum())
    stats["seconds_fill"] = round(time.time() - t0, 2)
    if log:
        log(f"  fill: {stats['components']} components, {stats['iterations']} patch "
            f"iterations, {stats['holes_after_fill']} px left "
            f"({stats['seconds_fill']}s)")
    return dict(I_w=I_w, D_w=D_w, hole=hole, I_filled=res3["I_filled"],
                D_filled=res3["D_filled"], remaining=res3["remaining"],
                oofa=res3["oofa"], disocc=res3["disocc"], crack=res1["crack"],
                I_warp=I_warp, D_warp=D_warp, hole_warp=hole_warp, stats=stats)


def fill_warped(warped_rgb, hole_mask, warped_depth, ref_rgb, ref_depth,
                disp=None, cfg=None, log=None):
    """Fill an ALREADY warped image.

    warped_rgb   : HxWx3, the synthetic view (holes may hold anything)
    hole_mask    : HxW bool, True where the view has no content
    warped_depth : HxW, nearer-is-larger scalar map of the warped view; holes are
                   ignored (they are set to the -1 sentinel internally)
    ref_rgb      : HxWx3, the artefact-free reference image used as patch source
    ref_depth    : HxW, reference inverse depth (0..1 or 0..255)
    disp         : optional (dx, dy) displacement field that produced the warp; when
                   omitted it is rebuilt from `ref_depth` and cfg.scale

    cfg.repair_warp decides what happens when the supplied warp is detected to be broken
    (foreground texture replaced by background): "auto" repairs it when the deviation
    exceeds cfg.repair_threshold_pct, "always" always re-warps with the Z-buffer,
    "never" keeps the supplied image.
    """
    cfg = cfg or FillConfig()
    I_w = np.asarray(warped_rgb, np.float32)
    hole = np.asarray(hole_mask, bool)
    P_ref = depth_to_scale255(ref_depth)
    ref_rgb = np.asarray(ref_rgb, np.float32)
    if disp is None:
        dx = disparity_from_depth(P_ref, cfg.scale)
        disp = (dx, None)

    mode = (cfg.repair_warp or "never").lower()
    q, I_c, hole_c = check_warp_quality(I_w, hole, ref_rgb, P_ref, disp, cfg)
    repaired = False
    if mode == "always" or (mode == "auto"
                            and q["warp_deviation_pct"] > cfg.repair_threshold_pct):
        if log:
            log(f"  [warp check] supplied warp deviates from a correct Z-buffer warp on "
                f"{q['warp_deviation_pct']:.2f}% of the valid pixels "
                f"(foreground {q['fg_deviation_pct']:.2f}%) -> re-warping with the "
                f"Z-buffer")
        I_w, hole, D_w = I_c, hole_c, None
        repaired = True
    else:
        if log:
            log(f"  [warp check] deviation {q['warp_deviation_pct']:.2f}% of valid px "
                f"(foreground {q['fg_deviation_pct']:.2f}%) -> keeping the supplied warp")
        D_w = np.asarray(warped_depth, np.float32).copy()
        if D_w.max() <= 1.0 + 1e-6:
            D_w = D_w * 255.0
        D_w[hole] = -1.0
    if repaired:
        _, D_w, _, _ = _warp.forward_warp(
            ref_rgb, np.asarray(disp[0], np.float32),
            None if disp[1] is None else np.asarray(disp[1], np.float32),
            z=P_ref, hole_depth=-1.0, rule="zbuf", splat=cfg.splat)
    if log:
        log(f"  input: {int(hole.sum())} hole px ({hole.mean()*100:.2f}%)")
    out = _run_pipeline(I_w, D_w, hole, ref_rgb, P_ref, disp, cfg, log=log)
    out["hole_input"] = hole
    out["disp"] = disp
    return _finish(out, cfg, ref_rgb=ref_rgb,
                   extra=dict(warp_repaired=bool(repaired), **q))


def warp_and_fill(rgb, inv_depth, cfg=None, reference=None, log=None):
    """Warp `rgb` to a virtual viewpoint with disparity = inv_depth * cfg.scale, then
    fill every hole.  `reference` = (ref_rgb, ref_depth) defaults to (rgb, inv_depth)."""
    cfg = cfg or FillConfig()
    rgb = np.asarray(rgb, np.float32)
    P = depth_to_scale255(inv_depth)
    I_w, D_w, hole, weight, dx, dy = warp_view(rgb, P, cfg)
    if reference is None:
        ref_rgb, ref_P = rgb, P
    else:
        ref_rgb = np.asarray(reference[0], np.float32)
        ref_P = depth_to_scale255(reference[1])
    if log:
        log(f"  warp: scale {cfg.scale:+g} ({cfg.splat}/{cfg.rule}), "
            f"|dx| max {np.abs(dx).max():.1f} px, {int(hole.sum())} hole px "
            f"({hole.mean()*100:.2f}%)")
    out = _run_pipeline(I_w, D_w, hole, ref_rgb, ref_P, (dx, dy), cfg, log=log)
    out["hole_input"] = hole
    out["disp"] = (dx, dy)
    return _finish(out, cfg, ref_rgb=ref_rgb)


# --------------------------------------------------------------------------- #
# packaging the result
# --------------------------------------------------------------------------- #
def _finish(res, cfg, ref_rgb=None, extra=None):
    """Add the convenience fields (uint8 images, masks, metrics) to a result dict."""
    I = res["I_filled"]
    out = dict(res)
    out["filled"] = np.clip(I, 0, 255).astype(np.uint8)
    out["warped"] = np.clip(res["I_warp"], 0, 255).astype(np.uint8)
    out["warped_float"] = res["I_warp"]          # un-quantised, for exact re-feeding
    out["hole_mask"] = res["hole_warp"]
    out["warped_depth"] = res["D_warp"]
    out["filled_depth"] = res["D_filled"]
    out["ref_rgb"] = ref_rgb
    out["config"] = cfg.as_dict()
    stats = dict(res["stats"])
    # GT-free quality signal: round trip to the reference viewpoint (needs the source)
    if ref_rgb is not None:
        try:
            from . import report as _report
            stats.update({k: v for k, v in
                          _report.back_projection_check(out, ref_rgb, cfg).items()
                          if not k.startswith("_")})
        except Exception as exc:                       # pragma: no cover
            stats["back_proj_error"] = str(exc)
    if extra:
        stats.update(extra)
    out["stats"] = stats
    return out
