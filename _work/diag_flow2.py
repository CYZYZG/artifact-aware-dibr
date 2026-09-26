"""Diagnostic 2: (a) unit-test forward_warp with a constant shift,
(b) residual flow between the warped result and the real target image."""
import cv2
import numpy as np

from dibr import calib, io_utils, warp

ROOT = io_utils.DATASET_ROOT_DEFAULT
v6 = io_utils.load_view(ROOT, 6, "f000")
v7 = io_utils.load_view(ROOT, 7, "f000")
cams = calib.load_calib(ROOT)

# ---------------- (a) constant shift sanity ----------------
rng = np.random.default_rng(0)
img = rng.integers(0, 255, (64, 64, 3)).astype(np.float32)
for dx0 in (-7.0, -7.4, 5.0, 5.6):
    out, z, hole, w = warp.forward_warp(img, np.full((64, 64), dx0, np.float32))
    i0 = int(np.floor(dx0))
    ref = np.roll(img, i0, axis=1)
    if i0 < 0:
        ref[:, i0:] = 0
    else:
        ref[:, :i0] = 0
    ok = np.abs(out - ref).max()
    print(f"constant dx={dx0:+.1f}: max|forward_warp - np.roll| = {ok:.4g}, holes={hole.sum()}")

# ---------------- (b) residual flow after warping ----------------
g6 = cv2.cvtColor(v6["color"], cv2.COLOR_RGB2GRAY)
g7 = cv2.cvtColor(v7["color"], cv2.COLOR_RGB2GRAY)


def flow(a, b):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    return dis.calc(a, b, None)


f_raw = flow(g6, g7)
dxp, dyp, _, _, _ = calib.displacement_field(cams, 6, 7, v6["depth"], y_origin="bottom")
w_cal, _, hole, _ = warp.forward_warp(v6["color"].astype(np.float32), dxp, dyp,
                                      z=v6["depth"].astype(np.float32))
w_hor, _, hole_h, _ = warp.forward_warp(v6["color"].astype(np.float32), dxp,
                                        z=v6["depth"].astype(np.float32))
gw_cal = cv2.cvtColor(np.clip(w_cal, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
gw_hor = cv2.cvtColor(np.clip(w_hor, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
f_cal = flow(gw_cal, g7)
f_hor = flow(gw_hor, g7)


def report(name, f, mask=None):
    m = np.ones(f.shape[:2], bool) if mask is None else mask
    mag = np.hypot(f[..., 0], f[..., 1])[m]
    print(f"{name:34s} median|flow| {np.median(mag):6.2f} px   p90 {np.percentile(mag, 90):6.2f}   "
          f"median dx {np.median(f[..., 0][m]):+7.2f}   median dy {np.median(f[..., 1][m]):+6.2f}")


print("\n=== residual flow to the real cam7 (smaller = better aligned) ===")
report("raw cam6 -> cam7 (no warp)", f_raw)
report("after calibrated 2D warp", f_cal, ~hole)
report("after calibrated dx-only warp", f_hor, ~hole_h)

print("\n=== per-block residual flow magnitude after the 2D warp ===")
for by in range(0, 768, 192):
    row = []
    for bx in range(0, 1024, 256):
        sl = (slice(by, by + 192), slice(bx, bx + 256))
        m = ~hole[sl]
        if m.sum() < 100:
            row.append("   n/a")
            continue
        mag = np.hypot(f_cal[..., 0][sl], f_cal[..., 1][sl])[m]
        row.append(f"{np.median(mag):5.1f}")
    print(f"  y={by:3d}: " + " ".join(row))
print("raw-flow medians per block (reference):")
for by in range(0, 768, 192):
    row = []
    for bx in range(0, 1024, 256):
        sl = (slice(by, by + 192), slice(bx, bx + 256))
        mag = np.hypot(f_raw[..., 0][sl], f_raw[..., 1][sl])
        row.append(f"{np.median(mag):5.1f}")
    print(f"  y={by:3d}: " + " ".join(row))

# ---------------- (c) brightness / colour compatibility ----------------
print("\n=== colour statistics (mean RGB) ===")
print("  cam6", v6["color"].reshape(-1, 3).mean(0).round(2),
      " cam7", v7["color"].reshape(-1, 3).mean(0).round(2))
d = v6["color"].astype(np.float32) - v7["color"].astype(np.float32)
print("  cam6-cam7 mean", d.reshape(-1, 3).mean(0).round(2),
      " std", d.reshape(-1, 3).std(0).round(2))
