"""viewfill.api - the one-call interface.

    from viewfill import fill_holes

    fixed = fill_holes(image, inv_depth)                     # -> HxWx3 uint8, no holes
    fixed = fill_holes(image, inv_depth, scale=-44.8)         # explicit disparity scale
    fixed = fill_holes(image, inv_depth, struct_pen=8.0)      # tune a stage
    info  = fill_holes(image, inv_depth, return_info=True)    # masks + per-stage stats

    image      : (H, W, 3) RGB/BGR (channel order preserved) or (H, W) grayscale
    inv_depth  : (H, W) inverse depth, NEAR = LARGE, 0..1 float or 0..255 integer
    returns    : the filled image (uint8), or the full result dict with return_info=True

Everything else is optional; the defaults are the values validated in 复现方案.md 9.x.
`FillConfig` + `warp_and_fill` / `fill_warped` remain available for lower-level use.
"""
import numpy as np

from .config import FillConfig
from .pipeline import warp_and_fill

__all__ = ["fill_holes"]

# FillConfig fields that intentionally have no `fill_holes` kwarg and why.
_NOT_EXPOSED = {
    "repair_warp": "only meaningful for fill_warped(), which receives an externally "
                   "produced warp; fill_holes warps itself and forces 'never'",
    "repair_threshold_pct": "same as repair_warp",
    "dev_threshold_gray": "same as repair_warp",
    "ablate": "internal ablation switch used by the reproduction experiments",
}


def fill_holes(image, inv_depth, scale=-44.8, *,
               # ---- input convention -------------------------------------------------
               depth_range="auto",
               # ---- warp --------------------------------------------------------------
               splat="sub", rule="zbuf", depth_dilate="auto7",
               # ---- II-A cracks -------------------------------------------------------
               lam=5.0, se_len=4, se_orientation="auto", crack_shape="none",
               crack_fill="hhf", hhf_sigma=1.0, hhf_ksize=5, max_thickness=3.0,
               # ---- II-B ghosts -------------------------------------------------------
               skip_ghosts=False, fix_mode="copy", band_radius=2, alpha_trim=0.10,
               # ---- II-C classification + patch filling -------------------------------
               fg_side="gt", alpha_sim=11.0, oofa_frac=0.5,
               ksize=9, n_window=69, sizes=(9, 7, 5, 3),
               beta=150.0, beta_mode="mean", max_iter=400000,
               # ---- patch search priors ----------------------------------------------
               struct_pen=8.0, epipolar=None, src_depth_tol=0.0,
               bg_template=True, require_full_bg=False,
               # ---- misc --------------------------------------------------------------
               seed=0, return_info=False, verbose=False):
    """Fill the holes produced by warping `image` with disparity = inv_depth * scale.

    Input
    -----
    image : (H, W, 3) uint8/float (RGB or BGR - the channel order is preserved) or
        (H, W) grayscale.  An alpha channel is dropped.
    inv_depth : (H, W) inverse depth, near = large.  0..1 or 0..255; a map whose maximum
        is <= 1 is treated as normalised, otherwise as 8-bit.
    depth_range : "auto" (default) | "01" | "255" - force the convention when the
        automatic detection is ambiguous.

    Warp
    ----
    scale : disparity in pixels per unit of normalised inverse depth (default -44.8, the
        2D->3D convention used throughout: content shifts LEFT, the virtual view is to the
        RIGHT of the reference, the out-of-field area appears on the right).  The sign picks
        the direction, the magnitude sets how big the holes are.
    splat : "sub" sub-pixel 2-tap (default, few cracks) | "floor"/"round" integer single tap
        (classical DIBR: many cracks, which is what the paper's figures show).
    rule : "zbuf" nearest source wins (default) | "avg" weight average.  "avg" is what a
        splat without a depth test does; it mixes foreground into background.
    depth_dilate : int n = n iterations of a 3x3 max filter on the disparity everywhere, or
        "auto"/"auto5"/"auto7" = widen only where the disparity gradient demands it, with the
        demand map smoothed by a w x w max filter first (default "auto7").  Widening the
        splat footprint closes the 1-2 px crack network that a fast disparity ramp leaves at
        a silhouette - the main source of the visible seam at the filled/background junction.
        Measured (seam mean / p90): 0 = 10.27/39.1, 2 = 7.90/22.3, 3 = 4.49/11.2,
        "auto5" = 5.21/13.1, "auto7" = 4.99/12.4 (best seam and SSIM over 6 frames).

    II-A cracks (paper Fig. 1 / Sec. II-A)
    --------------------------------------
    lam : crack detection threshold on the 0..255 depth scale (paper: 5).
    se_len : length of the line structuring element (paper: 4).
    se_orientation : "auto" (default) | "v" | "h" | "both" - the line SE must cross the slit.
    crack_shape : "none" (default) | "thickness" - extra shape filter for the crack mask.
    crack_fill : "hhf" (default, the paper's isotropic hierarchical fill) | "linear"
        (interpolate across a thin crack) | "bg" (copy the background side).  Both
        alternatives were measured to be no better ("bg" clearly worse).
    hhf_sigma, hhf_ksize : hierarchical fill kernel.

    II-B ghosts (Sec. II-B)
    -----------------------
    skip_ghosts : skip the ghost search/relocation (measured to be within noise on the
        development data; harmless to leave on).
    fix_mode : "copy" (default) | "move" | "bg" - how the ghost is corrected.
    band_radius : half width of the candidate band around the dilated background.
    alpha_trim : trimmed fraction for the T_Omega thresholds (paper: 10 %).

    II-C classification and filling (Sec. II-C)
    -------------------------------------------
    fg_side : "gt" (default) | "dilate" - how the foreground side of the band is taken.
    alpha_sim : foreground/background similarity threshold (paper: 11).
    oofa_frac : fraction of outlier displacements that marks the out-of-field side (0.5).
    ksize : template/patch side used for the terms (paper: 9).
    n_window : side of the square search window (paper: 69).
    sizes : adaptive patch sizes tried in order (paper: 9 -> 3).
    beta : acceptance threshold on the patch cost (default 150 on the per-pixel,
        per-channel mean squared error scale; the paper's 35 sums over 3 channels x pixels
        and is unreachable on 8-bit data).
    beta_mode : "mean" (per-pixel MSE, default) | "sum".
    max_iter : safety cap on fill iterations.

    Patch-search priors (this implementation's additions)
    -----------------------------------------------------
    struct_pen : structure-aware CROSS-ROW penalty (default 8.0).  A source patch may be
        borrowed from another row but pays struct_pen * w * dy**2, where dy is the row
        offset and w grows with the horizontal-structure strength around the hole.  On
        vertically homogeneous background w ~ 0, so good cross-row matches stay free; near a
        rail/fence the penalty stops the vertical shift of horizontal structures.  Measured
        (all / filled-band / barre-row PSNR, row-offset p90 / max):
          0   = 28.40 / 24.66 / 24.64   p90 22.5  max 49
          0.5 = 28.26 / 24.00 / 25.08   p90  4.0  max 21
          8   = 28.41 / 24.74 / 24.42   p90  1.0  max  8   <- default
        0 disables the term.
    epipolar : escape hatch that does NOT help on the development data.  Hard-limit how many
        rows the matcher may deviate from the BACKPROJECTED row; None (default) = no limit.
        Measured (all / band / barre PSNR): None 28.40/24.66/24.64, 2 = 28.33/24.30/24.29,
        0 = 28.12/23.40/19.92.  The row offsets are real (58.8 % of patches, p90 22.7 px) but
        mostly BENEFICIAL: the correct background of a disocclusion is occluded in the
        reference at that very row, so a same-row pool cannot contain it.
    src_depth_tol : reject source patches whose inverse-depth standard deviation exceeds this
        tolerance (0 = off, default).  Measured to have no effect here (95 % of the patches
        already sit inside one depth layer).
    bg_template : mask the template's foreground pixels out of the SSD (default True).
    require_full_bg : demand that every pixel of the source patch is background
        (default False; measured to have no effect here).

    Misc
    ----
    seed : kept for reproducibility of any future stochastic step (default 0).
    return_info : return the full result dict (holes/masks/depth/per-stage stats) with an
        extra "image" key instead of just the image.
    verbose : print the per-stage log.

    Not exposed on purpose
    ----------------------
    repair_warp / repair_threshold_pct / dev_threshold_gray : they only concern
        `fill_warped()`, which fills an externally produced warp and has to decide whether to
        re-warp it; `fill_holes` warps by itself and therefore forces repair_warp="never".
    ablate : internal ablation switch of the reproduction experiments.

    Returns
    -------
    filled : (H, W, 3) uint8, or (H, W) uint8 when the input was grayscale.
        With return_info=True: the `warp_and_fill` result dict, which contains
        "image", "filled", "warped", "hole_mask", "remaining", "crack", "oofa", "disocc",
        "filled_depth", "warped_depth" and "stats".
    """
    img = np.asarray(image)
    if img.ndim == 2:
        squeeze = True
        img3 = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        squeeze = False
        img3 = img[:, :, :3].astype(np.float32)      # drop alpha if present
    else:
        raise ValueError(f"unsupported image shape {img.shape}; expected HxW or HxWx3")

    d = np.asarray(inv_depth, np.float32)
    if d.ndim == 3 and d.shape[2] in (3, 4):
        d = d[:, :, 0]
    if d.ndim != 2:
        raise ValueError(f"unsupported depth shape {d.shape}; expected HxW")
    if d.shape != img3.shape[:2]:
        raise ValueError(f"image {img3.shape[:2]} and depth {d.shape} must have the "
                         f"same height/width")
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    if depth_range == "01":
        if d.max() > 1.0 + 1e-6:
            raise ValueError("depth_range='01' but the map has values > 1")
        d = d * 255.0
    elif depth_range == "255":
        pass
    elif depth_range != "auto":
        raise ValueError("depth_range must be 'auto', '01' or '255'")
    # "auto" is handled by the pipeline: a map whose max is <= 1 is treated as normalised

    cfg = FillConfig(scale=scale, splat=splat, rule=rule, depth_dilate=depth_dilate,
                     lam=lam, se_len=se_len, se_orientation=se_orientation,
                     crack_shape=crack_shape, crack_fill=crack_fill,
                     hhf_sigma=hhf_sigma, hhf_ksize=hhf_ksize,
                     max_thickness=max_thickness,
                     skip_ghosts=skip_ghosts, fix_mode=fix_mode,
                     band_radius=band_radius, alpha_trim=alpha_trim,
                     fg_side=fg_side, alpha_sim=alpha_sim, oofa_frac=oofa_frac,
                     ksize=ksize, n_window=n_window, sizes=tuple(sizes),
                     beta=beta, beta_mode=beta_mode, max_iter=max_iter,
                     struct_pen=struct_pen, epipolar=epipolar,
                     src_depth_tol=src_depth_tol, bg_template=bg_template,
                     require_full_bg=require_full_bg, seed=seed,
                     repair_warp="never")     # we do the warp ourselves, nothing to fix
    log = (lambda s: print(s, flush=True)) if verbose else None
    res = warp_and_fill(img3, d, cfg, reference=(img3, d), log=log)

    out = res["filled"]
    if squeeze:
        out = out[:, :, 0]
    if return_info:
        res = dict(res)
        res["image"] = out
        return res
    return out
