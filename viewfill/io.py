"""File IO for viewfill (non-ASCII-path safe, reusing the validated helpers)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dibr import io_utils  # noqa: E402


def load_image(path):
    """Read a colour image as RGB uint8."""
    return io_utils.imread(path, gray=False)


def load_depth(path, mode="auto"):
    """Read a depth / inverse-depth map as float32.

    mode: "auto"  -> if the file is 8-bit, scale to 0..1; if it is float, keep as is
          "01"    -> force the 0..1 convention (8-bit input divided by 255)
          "255"   -> keep 8-bit values (0..255)
    """
    d = io_utils.imread(path, gray=True)
    a = np.asarray(d)
    if mode == "255":
        return a.astype(np.float32)
    if a.dtype != np.uint8 and float(a.max()) <= 1.0 + 1e-6:
        return a.astype(np.float32)
    if mode == "01":
        return a.astype(np.float32) / 255.0
    return (a.astype(np.float32) / 255.0) if a.dtype == np.uint8 else a.astype(np.float32)


def load_pair(image_path, depth_path, depth_mode="auto"):
    """Load a colour image and its inverse depth, checking that they are aligned."""
    rgb = load_image(image_path)
    inv = load_depth(depth_path, depth_mode)
    if rgb.shape[:2] != inv.shape[:2]:
        raise ValueError(f"size mismatch: image {rgb.shape[:2]} vs depth {inv.shape[:2]}")
    return rgb, inv


def save_image(path, rgb):
    io_utils.imwrite(path, np.clip(rgb, 0, 255).astype(np.uint8))


def save_gray(path, field, valid_mask=None, vmin=None, vmax=None):
    from dibr import viz
    viz.imwrite_gray(path, field, valid_mask=valid_mask, vmin=vmin, vmax=vmax)
