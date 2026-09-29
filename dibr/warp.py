"""Forward 3D warping (DIBR): sub-pixel splatting with a proper Z-buffer.

Three implementations
    forward_warp            adopted: vectorised, sub-pixel splat, Z-buffer (nearest wins)
    scatter_image           the provided warping.py, kept only as an A/B baseline
"""
import numpy as np


# --------------------------------------------------------------------------- #
# main implementation
# --------------------------------------------------------------------------- #
def forward_warp(color, dx, dy=None, z=None, hole_depth=-1.0, rule="zbuf", splat="sub"):
    """Splat every source pixel to `src + (dx, dy)` with sub-pixel weights.

    Parameters
    ----------
    color : (H, W, C) float32   source image
    dx    : (H, W) float32      signed horizontal displacement, target = source + d
    dy    : (H, W) or None      signed vertical displacement (None = horizontal only)
    z     : (H, W) float32      ordering value, LARGER = NEARER (e.g. inverse depth).
                                Used for the Z-buffer; defaults to 1 everywhere
                                (then behaviour is pure ordered splatting).
    hole_depth : value written into the warped ordering map at empty pixels.
    rule  : "zbuf" - a target pixel takes the samples of the NEAREST source layer only
                     (proper occlusion handling);
            "avg"  - every sample is accumulated and weight-normalised, i.e. the
                     behaviour of the provided warping.scatter_image, which mixes
                     foreground and background colours at depth edges.

    Returns
    -------
    warped_color : (H, W, C) float32
    warped_z     : (H, W) float32   ordering value of the winning sample, `hole_depth`
                                    where empty
    hole_mask    : (H, W) bool      True where no source sample landed
    weight       : (H, W) float32   summed (normalised) weight, >0 exactly on valid pixels
    """
    color = np.asarray(color, dtype=np.float32)
    dx = np.asarray(dx, dtype=np.float32)
    h, w = dx.shape
    if color.shape[:2] != (h, w):
        raise ValueError(f"shape mismatch: color {color.shape} vs displacement {(h, w)}")
    c = color.shape[2] if color.ndim == 3 else 1
    src = color.reshape(h, w, c)
    if z is None:
        z = np.ones((h, w), np.float32)
    else:
        z = np.asarray(z, dtype=np.float32)

    if dy is None:
        contributions = _splat_1d(dx)
    else:
        contributions = _splat_2d(dx, np.asarray(dy, dtype=np.float32))
    if splat != "sub":
        # integer single-tap splat: one target pixel per source pixel, no interpolation.
        # This is what an integer disparity map + nearest-neighbour splat produces, and
        # it is where the classical 1-2 px "crack" artifact comes from.
        contributions = _splat_single(dx, dy, splat)

    # ---- keep only samples that carry weight and land inside the image ----
    packed = []
    for tx, ty, wt, sx, sy in contributions:
        sel = (wt > 0) & (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
        if sel.any():
            packed.append((tx[sel], ty[sel], wt[sel], z[sel], sx[sel], sy[sel]))

    if rule == "zbuf":
        # nearest source layer per target pixel; only real (weight>0) samples compete
        zbuf = np.full((h, w), -np.inf, np.float32)
        for tx, ty, _wt, zz, _sx, _sy in packed:
            np.maximum.at(zbuf, (ty, tx), zz)
        used = []
        for tx, ty, wt, zz, sx, sy in packed:
            keep = zz >= zbuf[ty, tx]
            used.append((tx[keep], ty[keep], wt[keep], zz[keep], sx[keep], sy[keep]))
    else:
        used = packed

    wacc = np.zeros((h, w), np.float32)
    cacc = np.zeros((h, w, c), np.float32)
    zacc = np.zeros((h, w), np.float32)
    for tx, ty, wt, zz, sx, sy in used:
        if tx.size == 0:
            continue
        np.add.at(cacc, (ty, tx), src[sy, sx] * wt[:, None])
        np.add.at(wacc, (ty, tx), wt)
        np.add.at(zacc, (ty, tx), zz * wt)

    valid = wacc > 0
    warped_color = np.zeros_like(cacc)
    warped_color[valid] = cacc[valid] / wacc[valid][:, None]
    warped_z = np.full((h, w), hole_depth, np.float32)
    warped_z[valid] = zacc[valid] / wacc[valid]
    if c == 1:
        warped_color = warped_color[:, :, 0]
    return warped_color, warped_z, ~valid, wacc


def _splat_1d(dx):
    """Two contributions (floor, floor+1) for every source pixel.

    Yields (target_x, target_y, weight, source_x, source_y).
    """
    h, w = dx.shape
    y, x = np.mgrid[0:h, 0:w]
    t = x + dx
    x0 = np.floor(t).astype(np.int32)
    wx = (t - x0).astype(np.float32)
    y = y.astype(np.int32)
    x = x.astype(np.int32)
    return [(x0, y, 1.0 - wx, x, y), (x0 + 1, y, wx, x, y)]


def _splat_single(dx, dy, mode):
    """One contribution per source pixel (nearest target pixel), weight 1."""
    h, w = dx.shape
    y, x = np.mgrid[0:h, 0:w]
    tx = x + dx
    ty = y + (np.zeros_like(dx) if dy is None else np.asarray(dy, np.float32))
    f = np.floor if mode == "floor" else np.round
    return [(f(tx).astype(np.int32), f(ty).astype(np.int32),
             np.ones((h, w), np.float32), x.astype(np.int32), y.astype(np.int32))]


def _splat_2d(dx, dy):
    """Four contributions (2x2) for every source pixel.

    Yields (target_x, target_y, weight, source_x, source_y).
    """
    h, w = dx.shape
    y, x = np.mgrid[0:h, 0:w]
    tx = x + dx
    ty = y + dy
    x0 = np.floor(tx).astype(np.int32)
    y0 = np.floor(ty).astype(np.int32)
    wx = (tx - x0).astype(np.float32)
    wy = (ty - y0).astype(np.float32)
    y = y.astype(np.int32)
    x = x.astype(np.int32)
    out = []
    for ox, oy, ww in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)),
                       (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        out.append((x0 + ox, y0 + oy, ww, x, y))
    return out




