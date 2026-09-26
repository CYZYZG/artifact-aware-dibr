"""Diagnostic: measured (optical flow) vs predicted (calibration) displacement cam6->cam7."""
import cv2
import numpy as np
from PIL import Image

from dibr import calib, io_utils

ROOT = io_utils.DATASET_ROOT_DEFAULT
FRAME = "f000"


def _psnr(gt, pred, mask=None):
    d = ((gt.mean(2) - pred.mean(2)) ** 2)
    if mask is not None:
        d = d[mask]
    mse = float(d.mean())
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0


def gray(path):
    return io_utils.imread(path, gray=True)


cams = calib.load_calib(ROOT)
v6 = io_utils.load_view(ROOT, 6, FRAME)
v7 = io_utils.load_view(ROOT, 7, FRAME)
g6, g7 = v6["color"].mean(2).astype(np.uint8), v7["color"].mean(2).astype(np.uint8)

# ---- predicted field ----
for yo in ("bottom", "top"):
    dx, dy, _, _, valid = calib.displacement_field(cams, 6, 7, v6["depth"], y_origin=yo)
    print(f"predicted ({yo:6s}): dx [{dx.min():+7.2f},{dx.max():+7.2f}] med {np.median(dx):+7.2f} | "
          f"dy [{dy.min():+6.2f},{dy.max():+6.2f}] med {np.median(dy):+6.2f} "
          f"med|dy| {np.median(np.abs(dy)):5.2f}")

# ---- measured field: dense inverse search flow ----
for preset_name, preset in (("MEDIUM", cv2.DISOPTICAL_FLOW_PRESET_MEDIUM),
                            ("FAST", cv2.DISOPTICAL_FLOW_PRESET_FAST)):
    dis = cv2.DISOpticalFlow_create(preset)
    dis.setUseSpatialPropagation(True)
    flow = dis.calc(g6, g7, None)
    print(f"flow {preset_name}: dx [{flow[...,0].min():+7.2f},{flow[...,0].max():+7.2f}] "
          f"med {np.median(flow[...,0]):+7.2f} | dy med {np.median(flow[...,1]):+6.2f} "
          f"med|dy| {np.median(np.abs(flow[...,1])):5.2f}")

fx, fy = flow[..., 0], flow[..., 1]
np.save(r"D:\项目\空洞填补\_work\flow_6to7.npy", flow)

print("\n=== predicted vs measured, by image block (median dx) ===")
dxp, dyp, _, _, _ = calib.displacement_field(cams, 6, 7, v6["depth"], y_origin="bottom")
print(f"{'block':>6s} {'Pmed':>5s} {'pred dx':>8s} {'meas dx':>8s} {'pred dy':>8s} {'meas dy':>8s}")
for by in range(0, 768, 192):
    for bx in range(0, 1024, 256):
        sl = (slice(by, by + 192), slice(bx, bx + 256))
        print(f"{bx//256},{by//192:<3d} {np.median(v6['depth'][sl]):5.0f} "
              f"{np.median(dxp[sl]):+8.2f} {np.median(fx[sl]):+8.2f} "
              f"{np.median(dyp[sl]):+8.2f} {np.median(fy[sl]):+8.2f}")

# ---- where does the warp actually help? regional PSNR ----
from dibr import warp
w_id, _, hole_id, _ = warp.forward_warp(v6["color"].astype(np.float32), np.zeros_like(dxp),
                                        z=v6["depth"].astype(np.float32))
w_cal, _, hole_cal, _ = warp.forward_warp(v6["color"].astype(np.float32), dxp, dyp,
                                          z=v6["depth"].astype(np.float32))
print("\n=== regional PSNR vs real cam7: identity | calibrated warp (valid px only) ===")
gt = v7["color"].astype(np.float32)
for by in range(0, 768, 192):
    row = []
    for bx in range(0, 1024, 256):
        sl = (slice(by, by + 192), slice(bx, bx + 256))
        m = ~hole_cal[sl]
        row.append(f"{bx//256},{by//192}: {_psnr(gt[sl], v6['color'][sl].astype(np.float32)):5.2f}"
                   f" | {_psnr(gt[sl], w_cal[sl], m):5.2f}")
    print("   ".join(row))
