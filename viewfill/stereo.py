"""viewfill.stereo - center view + inverse depth -> left/right eye pair (2D video -> 3D).

Disparity model (from 逆深度转视差.md):

    total disparity   Delta = (far_pct - total_pct * inv) * W          [+ = 入屏]
    right eye shift   d_R   = +Delta / 2
    left  eye shift   d_L   = -Delta / 2
    x_R = x_C + d_R ,  x_L = x_C + d_L

with `inv` in [0, 1] (near = 1), `total_pct = 0.03` (total span) and `far_pct = 0.02`
(= total_pct - near_pct, the into-screen maximum).  At inv = 1 (nearest) the right eye
moves -0.5 % W and the left eye +0.5 % W (out of screen); at inv = 0 (farthest) the right
eye moves +1 % W and the left eye -1 % W (into screen); inv = 2/3 sits on the screen plane.

The paper's ghost step (II-B) is DISABLED by default here: it is the only stage that
rewrites *valid* foreground pixels (measured 917 px with changes up to 144 grey levels on
one frame, 0 px once disabled), and it was already measured as ~noise in this project, so
leaving it on only injects artifacts into the middle of the foreground.

Both eyes are produced from the center image itself, so the centre image is also the only
patch source used to fill the holes the warp creates.
"""
import time

import numpy as np

from dibr import warp as _warp
from .config import FillConfig
from .pipeline import depth_to_scale255, fill_warped

__all__ = ["disparity_fields", "stereo_pair", "make_sbs"]


def disparity_fields(inv01, width=None, total_pct=0.03, near_pct=0.01):
    """Return (dx_left, dx_right, delta) in pixels for an inverse-depth map in [0, 1].

    total_pct : total disparity span as a fraction of the image width (0.03 = 3 %)
    near_pct  : maximum OUT-of-screen disparity (0.01 = 1 %); the remaining
                total_pct - near_pct is the maximum into-screen disparity
    """
    inv = np.asarray(inv01, np.float32)
    if inv.max(initial=0.0) > 1.0 + 1e-6:              # accept 0..255 maps as well
        inv = inv / 255.0
    W = float(width if width is not None else inv.shape[1])
    far_pct = float(total_pct) - float(near_pct)
    delta = (far_pct - float(total_pct) * inv) * W      # + = into screen
    return -0.5 * delta, 0.5 * delta, delta


def stereo_pair(rgb, inv_depth, cfg=None, total_pct=0.03, near_pct=0.01, log=None,
                translucent="keep", skip_ghosts=True):
    """Warp + fill one frame into a left/right pair.

    Returns a dict with "left"/"right" (HxWx3 uint8), the displacement fields, and the
    per-eye statistics (holes before/after, iterations, back-projection consistency).
    """
    cfg = cfg or FillConfig()
    rgb = np.asarray(rgb, np.float32)
    P = depth_to_scale255(inv_depth)                    # 0..255 internal depth scale
    inv01 = P / 255.0
    H, W = P.shape
    dx_l, dx_r, delta = disparity_fields(inv01, W, total_pct, near_pct)
    out = dict(delta_px=(float(delta.min()), float(delta.max())),
               d_right_px=(float(dx_r.min()), float(dx_r.max())),
               d_left_px=(float(dx_l.min()), float(dx_l.max())), stats={})
    for name, dx in (("left", dx_l), ("right", dx_r)):
        t = time.time()
        dx = np.ascontiguousarray(dx, np.float32)
        I_w, D_w, hole, _ = _warp.forward_warp(rgb, dx, None, z=P, hole_depth=-1.0,
                                               rule="zbuf", splat=cfg.splat)
        # an explicit displacement field: fill_warped takes it as-is (no re-warp check)
        eye_cfg = FillConfig(**{**cfg.as_dict(), "repair_warp": "never",
                                "crack_translucent": translucent,
                                "skip_ghosts": bool(skip_ghosts)})
        res = fill_warped(I_w, hole, D_w, rgb, P, disp=(dx, None), cfg=eye_cfg, log=log)
        out[name] = res["filled"]
        s = res["stats"]
        out["stats"][name] = dict(
            hole_px=int(hole.sum()), hole_pct=float(hole.mean() * 100),
            residual=int(res["remaining"].sum()), iterations=int(s.get("iterations", 0)),
            back_proj_before=s.get("back_proj_psnr_before"),
            back_proj_after=s.get("back_proj_psnr_after"), seconds=time.time() - t)
    return out


def make_sbs(left, right, layout="full-sbs"):
    """Compose the two eyes: "full-sbs" = 2W x H, "half-sbs" = W x H (each squeezed)."""
    import cv2
    l, r = np.asarray(left), np.asarray(right)
    if layout.lower() in ("full", "full-sbs", "sbs", "side-by-side"):
        return np.hstack([l, r])
    if layout.lower() in ("half", "half-sbs"):
        h, w = l.shape[:2]
        return np.hstack([cv2.resize(l, (w // 2, h), interpolation=cv2.INTER_AREA),
                          cv2.resize(r, (w // 2, h), interpolation=cv2.INTER_AREA)])
    if layout.lower() in ("tab", "top-bottom", "over-under"):
        return np.vstack([l, r])
    raise ValueError("unknown layout %r" % layout)
