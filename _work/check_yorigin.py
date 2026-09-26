"""Check which y-origin convention reproduces the real target camera better."""
import cv2
import numpy as np

from dibr import calib, io_utils, viz, warp

ROOT = io_utils.DATASET_ROOT_DEFAULT
cams = calib.load_calib(ROOT)
v6 = io_utils.load_view(ROOT, 6, "f000")
v7 = io_utils.load_view(ROOT, 7, "f000")


def g(im):
    return cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


for yo in ("bottom", "top"):
    dx, dy, _, _, _ = calib.displacement_field(cams, 6, 7, v6["depth"], y_origin=yo)
    w, _, hole, _ = warp.forward_warp(v6["color"].astype(np.float32), dx, dy,
                                      z=v6["depth"].astype(np.float32))
    m = ~hole
    d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    d.setUseSpatialPropagation(True)
    f = d.calc(g(w), g(v7["color"]), None)
    mag = np.hypot(f[..., 0], f[..., 1])[m]
    print(f"y_origin={yo:6s}: holes {hole.mean()*100:5.2f}%  "
          f"median|residual flow| {np.median(mag):5.2f} px  p90 {np.percentile(mag, 90):6.2f} px  "
          f"PSNR(valid) {viz.psnr(v7['color'], w, mask=m):5.2f} dB  "
          f"SSIM(valid) {viz.ssim(v7['color'], w, mask=m):.4f}  "
          f"median|dy| {np.median(np.abs(dy)):.2f} px")
