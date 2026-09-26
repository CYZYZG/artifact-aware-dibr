"""Step 0 - camera calibration handling and the reference->virtual displacement field.

Official data facts (MSR Ballet, README.txt of the distribution):
    depth PNG stores inverse depth:  z = 1 / ((P/255)*(1/MinZ - 1/MaxZ) + 1/MaxZ)
    MinZ = 42.0, MaxZ = 130.0          (same length unit as the translations in the
                                        calibration file; cameras are ~3.3-3.9 apart)
    calibParams-ballet.txt             per camera: K (3x3, with skew), distortion, [R|t]
    "(0,0) coordinate of an image is located in the bottom left corner"
"""
import os

import numpy as np

from . import io_utils

MINZ, MAXZ = 42.0, 130.0
IMG_H, IMG_W = 768, 1024


# --------------------------------------------------------------------------- #
# calibration file
# --------------------------------------------------------------------------- #
def parse_calib(path):
    """Parse calibParams-ballet.txt -> {cam_index: dict(K, R, t, C)}."""
    lines = open(path, "r", encoding="utf-8", errors="ignore").read().split("\n")
    cams, i = {}, 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        idx = int(lines[i].strip())
        K = np.array([[float(x) for x in lines[i + 1].split()],
                      [float(x) for x in lines[i + 2].split()],
                      [float(x) for x in lines[i + 3].split()]])
        Rt = np.array([[float(x) for x in lines[i + 5].split()],
                       [float(x) for x in lines[i + 6].split()],
                       [float(x) for x in lines[i + 7].split()]])
        R, t = Rt[:, :3], Rt[:, 3]
        cams[idx] = {"K": K, "R": R, "t": t, "C": -R.T @ t}
        i += 8
    return cams


def load_calib(dataset_root=io_utils.DATASET_ROOT_DEFAULT):
    path = os.path.join(dataset_root, "calibParams-ballet.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"calibration not found: {path}")
    return parse_calib(path)


# --------------------------------------------------------------------------- #
# official inverse depth model
# --------------------------------------------------------------------------- #
def depth_from_P(P):
    """8-bit inverse-depth intensity -> z (same unit as the calibration translations)."""
    P = np.asarray(P, dtype=np.float64)
    return 1.0 / ((P / 255.0) * (1.0 / MINZ - 1.0 / MAXZ) + 1.0 / MAXZ)


def invdepth_from_z(z):
    """z -> 8-bit intensity P (inverse of depth_from_P)."""
    z = np.asarray(z, dtype=np.float64)
    return ((1.0 / z - 1.0 / MAXZ) / (1.0 / MINZ - 1.0 / MAXZ) * 255.0)


def baseline(cams, a, b, axis=0):
    """Signed distance between two camera centres along a world axis."""
    return float(cams[b]["C"][axis] - cams[a]["C"][axis])


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
def _unproject(u, v, z, K, y_origin):
    """Image pixel (top-down u,v) + depth -> camera-frame 3D point."""
    cy = (IMG_H - 1) - K[1, 2] if y_origin == "bottom" else K[1, 2]
    sy = -1.0 if y_origin == "bottom" else 1.0
    yc = sy * (v - cy) / K[1, 1] * z
    xc = (u - K[0, 2] - K[0, 1] * yc / K[1, 1] / z) / K[0, 0] * z
    return xc, yc, z


def _project(xc, yc, zc, K, y_origin):
    cy = (IMG_H - 1) - K[1, 2] if y_origin == "bottom" else K[1, 2]
    sy = -1.0 if y_origin == "bottom" else 1.0
    u = K[0, 0] * xc / zc + K[0, 1] * yc / zc + K[0, 2]
    v = cy + sy * K[1, 1] * yc / zc
    return u, v


def displacement_field(cams, src, dst, P, y_origin="bottom", step=1):
    """Exact per-pixel (dx, dy) taking every source pixel to the dst camera image.

    dx/dy are in pixels of the top-down image, i.e.  target_xy = source_xy + (dx, dy).
    Returns (dx, dy, u_t, v_t, valid) with float64 arrays of shape (H/step, W/step).
    """
    Ks, Rs, ts = cams[src]["K"], cams[src]["R"], cams[src]["t"]
    Kd, Rd, td = cams[dst]["K"], cams[dst]["R"], cams[dst]["t"]
    z = depth_from_P(P[::step, ::step])
    v, u = np.mgrid[0:IMG_H:step, 0:IMG_W:step]
    u = u.astype(np.float64)
    v = v.astype(np.float64)
    xc, yc, zc = _unproject(u, v, z, Ks, y_origin)
    # camera src -> world -> camera dst
    p = np.stack([xc, yc, zc], -1) - ts
    p = p @ Rs                                    # R_src^T (X - t_src)
    p = p @ Rd.T + td                             # R_dst X + t_dst
    u_t, v_t = _project(p[..., 0], p[..., 1], p[..., 2], Kd, y_origin)
    valid = p[..., 2] > 1e-6
    return u_t - u, v_t - v, u_t, v_t, valid


def disparity_of_direction(cams, src, dst, axis=0):
    """Sign convention for a 1D disparity warp: +1 if content moves towards +x."""
    return 1.0 if baseline(cams, src, dst, axis) < 0 else -1.0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def describe(cams, src, dst, P, dataset_root=io_utils.DATASET_ROOT_DEFAULT, out_dir=None):
    """Print a Step-0 calibration report and (optionally) write artefacts."""
    dx, dy, u_t, v_t, valid = displacement_field(cams, src, dst, P)
    d = P / 255.0
    m = valid
    A = np.stack([d[m], np.ones(m.sum())], 1)
    coef, *_ = np.linalg.lstsq(A, dx[m], rcond=None)
    pred = A @ coef
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"--- Step 0 calibration report: cam{src} -> cam{dst} ---")
    log(f"camera centres : cam{src} = {np.round(cams[src]['C'], 4)}")
    log(f"                 cam{dst} = {np.round(cams[dst]['C'], 4)}")
    log(f"baseline       : |dC| = {np.linalg.norm(cams[dst]['C'] - cams[src]['C']):.4f} "
        f"(dX = {baseline(cams, src, dst, 0):+.4f})")
    log(f"focal (px)     : fx = {cams[src]['K'][0, 0]:.2f}, fy = {cams[src]['K'][1, 1]:.2f}")
    log(f"z range        : {depth_from_P(P).min():.2f} .. {depth_from_P(P).max():.2f} "
        f"(mean {depth_from_P(P).mean():.2f})")
    log(f"dx  (px)       : min {dx[m].min():+.2f}  max {dx[m].max():+.2f}  "
        f"mean {dx[m].mean():+.2f}  median {np.median(dx[m]):+.2f}  std {dx[m].std():.2f}")
    log(f"dy  (px)       : min {dy[m].min():+.2f}  max {dy[m].max():+.2f}  "
        f"mean {dy[m].mean():+.2f}  median {np.median(dy[m]):+.2f}  std {dy[m].std():.2f}")
    log(f"1D linear fit  : dx ~= {coef[0]:+.4f}*(P/255) {coef[1]:+.4f}   "
        f"residual median {np.median(np.abs(dx[m] - pred)):.2f} px, "
        f"p95 {np.percentile(np.abs(dx[m] - pred), 95):.2f} px")
    log(f"pure-horizontal model error: median |dy| = {np.median(np.abs(dy[m])):.2f} px, "
        f"p95 = {np.percentile(np.abs(dy[m]), 95):.2f} px")
    log(f"divergence of dx over the image: spread = {dx[m].max() - dx[m].min():.2f} px "
        f"(a pure translating rig would have a spread set only by depth)")

    if out_dir:
        import cv2
        from . import viz
        os.makedirs(out_dir, exist_ok=True)
        # normalise dx/dy for display
        viz.imwrite_gray(os.path.join(out_dir, f"dx_cam{src}_to_cam{dst}.png"), dx,
                         valid_mask=valid)
        viz.imwrite_gray(os.path.join(out_dir, f"dy_cam{src}_to_cam{dst}.png"), dy,
                         valid_mask=valid)
        np.save(os.path.join(out_dir, f"field_cam{src}_cam{dst}.npy"),
                np.stack([dx, dy, u_t, v_t]).astype(np.float32))
        with open(os.path.join(out_dir, f"report_cam{src}_to_cam{dst}.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return dict(dx=dx, dy=dy, u_t=u_t, v_t=v_t, valid=valid, fit=coef, report=lines)
