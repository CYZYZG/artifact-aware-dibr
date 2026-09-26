"""viewfill.api - one-call interface.

    from viewfill import fill_holes
    fixed = fill_holes(image, inv_depth)                 # -> HxWx3 uint8, no holes
    fixed = fill_holes(image, inv_depth, scale=-44.8)    # explicit disparity scale
    info  = fill_holes(image, inv_depth, return_info=True)

Input : original image (HxWx3 RGB/BGR, or HxW grayscale) + normalised inverse depth
        (near = large; 0..1 float or 0..255 integer both accepted)
Output: the same image with every hole filled.
"""
import numpy as np

from .config import FillConfig
from .pipeline import warp_and_fill

__all__ = ["fill_holes"]


def fill_holes(image, inv_depth, scale=-44.8, *, depth_range="auto",
               splat="sub", rule="zbuf", lam=5.0, se_orientation="auto",
               crack_shape="none", crack_fill="hhf", depth_dilate="auto7", src_depth_tol=0.0,
               beta=150.0, beta_mode="mean",
               skip_ghosts=False, fix_mode="copy", fg_side="gt",
               band_radius=2, alpha_sim=11.0, hhf_sigma=1.0, ksize=9,
               n_window=69, sizes=(9, 7, 5, 3), max_iter=400000,
               return_info=False, verbose=False):
    """Fill the holes produced by warping `image` with disparity = inv_depth * scale.

    Parameters
    ----------
    image : (H, W, 3) uint8/float, or (H, W) grayscale.  Channel order is preserved.
    inv_depth : (H, W) inverse depth, near = large.  0..1 (normalised) or 0..255.
        The disparity used for the warp is  (inv_depth / 255) * scale  when the map is
        8-bit-like and  inv_depth * scale  when it is already normalised to 0..1.
    scale : disparity in pixels per unit of normalised inverse depth.  The 2D->3D
        convention this was written for is -44.8: content shifts LEFT, the virtual view is
        to the RIGHT of the reference, and the out-of-field area appears on the right.
        The sign picks the direction; only the magnitude changes how big the holes are.
    depth_range : "auto" (default) | "01" | "255" - force the input convention when the
        automatic detection is ambiguous.
    splat : "sub" sub-pixel 2-tap (default; few cracks) | "floor"/"round" integer single
        tap (classical DIBR, many cracks - this is what the paper's figures show).
    rule : "zbuf" nearest source wins (default) | "avg" weight average.  "avg" is what a
        splat without a depth test does and it mixes foreground into background.
    lam : crack detection threshold on the 0..255 depth scale (paper: 5).
    se_orientation : "auto" (default), "v", "h", "both" - the line SE must cross the slit.
    crack_fill : "hhf" (default, the paper's isotropic hierarchical fill) | "linear"
        (interpolate across a thin crack) | "bg" (copy the background side).  Measured to be
        equivalent-to-worse on the seam metric, kept for experiments.
    depth_dilate : widen the splat footprint before warping, so that a fast disparity ramp
        at a silhouette does not leave a 1-2 px crack network - the main source of the
        visible seam at the filled/background junction.  An int n applies n iterations of a
        3x3 max filter everywhere; "auto"/"auto5"/"auto7" widen only where the disparity
        gradient demands it (demand map smoothed with a w x w max filter, so the boundary of
        the widened region does not create new thin holes).  Measured over 6 frames of
        cam6 -> cam7 (seam mean / p90 / crack px / GT PSNR / GT SSIM): 2 =
        7.69/21.65/9351/20.35/0.7103, 3 = 5.52/13.99/9133/20.32/0.7108, auto5 =
        5.21/13.14/10182/20.36/0.7112, auto7 = 4.99/12.42/9936/20.34/0.7114 (default).
        0 = off.
    beta, beta_mode : adaptive patch-size acceptance threshold (default 150 on the
        per-pixel mean squared error scale).
    skip_ghosts : skip the ghost-removal stage (measured to be within noise on the data
        this was developed on; harmless to leave on).
    return_info : also return the full result dict (masks, per-stage stats, depth).
    verbose : print the per-stage log.

    Returns
    -------
    filled : (H, W, 3) uint8 (or (H, W) uint8 if the input was grayscale)
    if return_info: the result dict from `warp_and_fill`, with an extra "image" key.
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
    # "auto" is handled by the pipeline: a map whose max is <= 1 is treated as 0..1

    cfg = FillConfig(scale=scale, splat=splat, rule=rule, lam=lam,
                     se_orientation=se_orientation, crack_shape=crack_shape,
                     crack_fill=crack_fill, depth_dilate=depth_dilate, src_depth_tol=src_depth_tol,
                     beta=beta, beta_mode=beta_mode, skip_ghosts=skip_ghosts,
                     fix_mode=fix_mode, fg_side=fg_side, band_radius=band_radius,
                     alpha_sim=alpha_sim, hhf_sigma=hhf_sigma, ksize=ksize,
                     n_window=n_window, sizes=tuple(sizes), max_iter=max_iter,
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
