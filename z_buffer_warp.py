"""Proper Z-buffer splatting for the forward 3D warp, plus an objective comparison
against the two collision behaviours of `warping.scatter_image`.

Why this module exists
----------------------
`warping.scatter_image` accumulates every source sample that lands on a target pixel:

    target[x'] += src[x] * w        (w = sub-pixel split of the disparity)

Its two collision behaviours are both decided by traversal order rather than depth:

    inverse_ordering=False  the later-written source keeps accumulating on top, so a
                            far/background sample can be added over a near/foreground one
    inverse_ordering=True   the FAR sample wins the collision outright, which visibly
                            replaces the foreground silhouette with background texture
                            ("lost foreground")

Correct rule (Z-buffer): among all source samples landing on the same target pixel,
keep the NEAREST one (largest inverse depth), and split the sub-pixel weight only
between the two positions of the SAME source sample.

Measured on this dataset (see `compare` below), pixels deviating by >40 gray levels
from the nearest-sample colour:

    scatter_image inverse_ordering=True    14 141 px
    scatter_image inverse_ordering=False      665 px
    z_buffer_splat (this module)              103 px
"""

import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warp_and_visualize import imread_u, imwrite_u, label  # noqa: E402
from warping import scatter_image  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(ROOT, "input")
OUT = os.path.join(ROOT, "output")
TMP = os.path.join(ROOT, ".tmp")


def z_buffer_splat(input_frame, inverse_depth, direction, scale_factor,
                   reproject_depth=True, return_inverse_depth=True):
    """Forward warp keeping the nearest source sample per target pixel.

    Two passes, so the depth test is order-independent and the accumulation can use
    vectorised unbuffered adds:

      pass 1  best[target] = max inverse depth over every source sample landing there
      pass 2  accumulate colour/depth of the samples whose inverse depth equals best,
              weighted by their sub-pixel weight, then renormalise by the total weight

    Returns (reproj_img, hole_mask, warped_depth_map).

    WARNING about the third return value: `warping.scatter_image` returns
    `1 / (inverse_depth + 1e-6)` there, i.e. **real depth**, not the inverse depth it
    was given. That asymmetry silently breaks any `[0, 1]` visualisation or downstream
    use of the warped map. This implementation therefore returns the **warped inverse
    depth** by default (same units as the input, 1.0 = nearest, 0 = hole), and only
    converts to real depth when `return_inverse_depth=False` (for exact parity with
    `scatter_image`).
    """
    h, w = input_frame.shape[:2]
    disparity = (inverse_depth.astype(np.float32) * scale_factor).astype(np.float32)
    disparity_int = disparity.astype(np.int32)
    weight_plus1 = disparity - disparity_int.astype(np.float32)
    disparity_int_plus1 = (disparity + 1.0).astype(np.int32)

    x_coords, _ = np.meshgrid(np.arange(w), np.arange(h))
    contributions = (
        (x_coords + disparity_int * direction, 1.0 - weight_plus1),
        (x_coords + disparity_int_plus1 * direction, weight_plus1),
    )

    # ---- pass 1: nearest source depth per target pixel -------------------------
    best = np.full((h, w), -np.inf, dtype=np.float32)
    for tx, weight in contributions:
        valid = (tx >= 0) & (tx < w) & (weight > 0)
        ys, xs = np.where(valid)
        np.maximum.at(best, (ys, tx[ys, xs]), inverse_depth[ys, xs])
    filled = np.isfinite(best)

    # ---- pass 2: accumulate only the winning (nearest) samples ------------------
    frame_f = input_frame.astype(np.float32)
    acc = np.zeros((h, w, input_frame.shape[2]), dtype=np.float32)
    acc_invdepth = np.zeros((h, w), dtype=np.float32)
    acc_w = np.zeros((h, w), dtype=np.float32)
    for tx, weight in contributions:
        valid = (tx >= 0) & (tx < w) & (weight > 0)
        ys, xs = np.where(valid)
        txv = tx[ys, xs]
        win = inverse_depth[ys, xs] >= best[ys, txv]
        wy, wx = ys[win], txv[win]
        ww = weight[ys, xs][win]
        wsrc = xs[win]
        np.add.at(acc, (wy, wx), frame_f[wy, wsrc] * ww[:, None])
        np.add.at(acc_invdepth, (wy, wx), inverse_depth[wy, wsrc] * ww)
        np.add.at(acc_w, (wy, wx), ww)

    img = np.zeros_like(frame_f)
    warped_invdepth = np.zeros((h, w), dtype=np.float32)
    nz = filled & (acc_w > 0)
    img[nz] = acc[nz] / acc_w[nz][:, None]
    warped_invdepth[nz] = acc_invdepth[nz] / acc_w[nz]

    hole_mask = np.where(filled, 0, 255).astype(np.uint8)
    if not reproject_depth:
        out_depth = np.zeros((h, w), dtype=np.float32)
    elif return_inverse_depth:
        out_depth = warped_invdepth
    else:
        out_depth = np.where(filled, 1.0 / (warped_invdepth + 1e-6), 0.0).astype(np.float32)
    return img.astype(input_frame.dtype), hole_mask, out_depth


def nearest_sample_reference(color, inverse_depth, direction, scale_factor):
    """Build the reference answer using the INPUTS only.

    For every target pixel: which source samples land on it, what is the nearest one,
    and what colour does that nearest sample have.  Independent of any warp code under
    test, so it can be used as ground truth for the collision rule.
    """
    h, w = inverse_depth.shape
    disparity = inverse_depth * scale_factor
    disparity_int = disparity.astype(np.int32)
    weight_plus1 = disparity - disparity_int
    x_coords, _ = np.meshgrid(np.arange(w), np.arange(h))
    contributions = (
        (x_coords + disparity_int * direction, 1.0 - weight_plus1),
        (x_coords + (disparity + 1).astype(np.int32) * direction, weight_plus1),
    )

    best = np.full((h, w), -np.inf, np.float32)
    ncoll = np.zeros((h, w), np.int32)
    for tx, weight in contributions:
        valid = (tx >= 0) & (tx < w) & (weight > 0)
        ys, xs = np.where(valid)
        np.maximum.at(best, (ys, tx[ys, xs]), inverse_depth[ys, xs])
        np.add.at(ncoll, (ys, tx[ys, xs]), 1)

    ref_color = np.zeros((h, w, 3), np.float32)
    for tx, weight in contributions:
        valid = (tx >= 0) & (tx < w) & (weight > 0)
        ys, xs = np.where(valid)
        txv = tx[ys, xs]
        win = inverse_depth[ys, xs] >= best[ys, txv]
        ref_color[ys[win], txv[win]] = color[ys[win], xs[win]]

    filled = np.isfinite(best)
    return ref_color, filled, ncoll


def compare(frame_index="f000", scale_factor=44.8, direction=-1):
    """Quantify each collision rule against the nearest-sample reference, and write
    `output/step2_1_zbuffer_compare.png`."""
    color = imread_u(os.path.join(IN, f"color-cam6-{frame_index}.jpg"), cv2.IMREAD_COLOR)
    d = imread_u(os.path.join(IN, f"depth-cam6-{frame_index}.png"), cv2.IMREAD_UNCHANGED)
    inv = d[:, :, 0].astype(np.float32) / 255.0

    alt, hm_alt, _ = scatter_image(color, inv, direction, scale_factor,
                                   inverse_ordering=False, reproject_depth=False)
    cur, hm_cur, _ = scatter_image(color, inv, direction, scale_factor,
                                   inverse_ordering=True, reproject_depth=False)
    zb, hm_zb, _ = z_buffer_splat(color, inv, direction, scale_factor)
    alt = np.clip(alt, 0, 255).astype(np.uint8)
    cur = np.clip(cur, 0, 255).astype(np.uint8)
    zb = np.clip(zb, 0, 255).astype(np.uint8)

    print(f"hole ratio: ordering=False {(hm_alt > 0).mean()*100:.3f}%  "
          f"ordering=True {(hm_cur > 0).mean()*100:.3f}%  z_buffer {(hm_zb > 0).mean()*100:.3f}%")

    ref, filled, ncoll = nearest_sample_reference(color, inv, direction, scale_factor)
    collided = filled & (ncoll > 1)
    print(f"collided pixels: {collided.sum()} ({collided.mean()*100:.1f}% of frame)")

    def errvis(img):
        e = np.abs(img.astype(np.float32) - ref).max(axis=2)
        e[~filled] = 0
        vis = np.zeros((*filled.shape, 3), np.uint8)
        vis[..., 2] = np.clip(e * 3, 0, 255).astype(np.uint8)   # red = deviation
        return vis

    for name, img in (("scatter_image inverse_ordering=False", alt),
                      ("scatter_image inverse_ordering=True ", cur),
                      ("z_buffer_splat (proposed)          ", zb)):
        diff = np.abs(img.astype(np.float32) - ref).max(axis=2)
        mean_err = np.abs(img.astype(np.float32) - ref)[collided].mean()
        big = int(((diff > 40) & filled).sum())
        print(f"  {name}  mean err on collided {mean_err:5.2f}   px off>40 (whole frame): {big:>6d}")

    row = np.hstack([
        label(alt, "scatter_image ordering=False"),
        label(errvis(alt), "err vs nearest sample (red)"),
        label(cur, "scatter_image ordering=True (was used)"),
        label(errvis(cur), "err (red)"),
        label(zb, "z_buffer_splat (adopted)"),
        label(errvis(zb), "err (red)"),
    ])
    path = os.path.join(OUT, "step2_1_zbuffer_compare.png")
    imwrite_u(path, row)
    print("wrote", path)
    return alt, cur, zb


def main():
    compare()


if __name__ == "__main__":
    main()
