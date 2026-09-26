"""Drop-in replacement for the provided forward warp, with a proper Z-buffer.

The supplied `warping.scatter_image` decides collisions by traversal order:

    inverse_ordering=False  later samples keep accumulating on top
    inverse_ordering=True   the FAR sample wins outright  -> foreground texture is
                            replaced by background (the subject looks cut up)

This module keeps the exact same call signature and return triple, but resolves
collisions by depth (nearest source sample wins, sub-pixel weights are normalised inside
that sample), so existing code only has to change the import:

    from warping import scatter_image                 # broken collision rule
    from viewfill.compat import scatter_image_safe    # same signature, Z-buffer

    img, mask, depth = scatter_image_safe(frame, inverse_depth,
                                         direction=-1, scale_factor=44.8)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dibr import warp as _warp  # noqa: E402


def scatter_image_safe(input_frame, inverse_depth, direction, scale_factor,
                       inverse_ordering=None, reproject_depth=False,
                       return_inverse_depth=False, splat="sub"):
    """Z-buffer replacement for `warping.scatter_image`.

    Parameters are the ones of the original function:
      input_frame      : HxWxC image
      inverse_depth    : HxW, larger = nearer (0..1 normalised or 0..255)
      direction        : +1 / -1, sign of the horizontal shift
      scale_factor     : disparity in pixels per unit of inverse_depth
      inverse_ordering : accepted for compatibility only; it is what the original used to
                         pick a collision winner, which is exactly the bug this replaces
      reproject_depth  : also warp the depth map (returned in REAL depth, like the
                         original: 1 / warped_inverse_depth)
      return_inverse_depth : return the warped INVERSE depth instead (same units as the
                         input, holes = 0), which is what downstream filling expects
      splat            : "sub" (sub-pixel 2-tap) | "floor" | "round" (integer single tap)

    Returns (warped_image, hole_mask_uint8, warped_depth) exactly like the original.
    """
    frame = np.asarray(input_frame, np.float32)
    inv = np.asarray(inverse_depth, np.float32)
    dx = (float(direction) * float(scale_factor) * inv).astype(np.float32)
    img, warped_inv, hole, _ = _warp.forward_warp(frame, dx, None, z=inv,
                                                  hole_depth=-1.0, rule="zbuf",
                                                  splat=splat)
    if not reproject_depth:
        depth = np.zeros_like(inv, np.float32)
    elif return_inverse_depth:
        depth = np.where(hole, 0.0, np.maximum(warped_inv, 0.0)).astype(np.float32)
    else:
        depth = np.where(hole, 0.0,
                         1.0 / (np.maximum(warped_inv, 0.0) + 1e-6)).astype(np.float32)
    return img.astype(input_frame.dtype), (hole * 255).astype(np.uint8), depth
