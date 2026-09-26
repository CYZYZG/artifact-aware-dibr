"""Step 0 probe #2: full-projection prediction from calibration vs measured regional shift."""
import numpy as np
from PIL import Image

ROOT = r"D:\项目\3DVideos-distrib\MSR3DVideo-Ballet"
MINZ, MAXZ = 42.0, 130.0
H, W = 768, 1024


def load_calib():
    lines = open(rf"{ROOT}\calibParams-ballet.txt").read().split("\n")
    cams, i = {}, 0
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        idx = int(lines[i].strip())
        K = np.array([[float(x) for x in lines[i + 1].split()],
                      [float(x) for x in lines[i + 2].split()],
                      [float(x) for x in lines[i + 3].split()]])
        Rt = np.array([[float(x) for x in lines[i + 5].split()],
                       [float(x) for x in lines[i + 6].split()],
                       [float(x) for x in lines[i + 7].split()]])
        cams[idx] = dict(K=K, R=Rt[:, :3], t=Rt[:, 3])
        i += 8
    return cams


def gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float32)


def depth_from_P(P):
    return 1.0 / ((P / 255.0) * (1.0 / MINZ - 1.0 / MAXZ) + 1.0 / MAXZ)


def predict(cams, src, dst, us, vs, z, y_origin="bottom"):
    """Project src-camera pixels (top-down coords us,vs) with depth z into camera dst.
    Returns (u_dst, v_dst) in top-down pixel coordinates."""
    Ks, Rs, ts = cams[src]["K"], cams[src]["R"], cams[src]["t"]
    Kd, Rd, td = cams[dst]["K"], cams[dst]["R"], cams[dst]["t"]
    cy = (H - 1) - Ks[1, 2] if y_origin == "bottom" else Ks[1, 2]
    sy = -1.0 if y_origin == "bottom" else 1.0
    us = np.asarray(us, np.float64)
    vs = np.asarray(vs, np.float64)
    zz = np.broadcast_to(np.asarray(z, np.float64), us.shape)
    Xc = np.stack([(us - Ks[0, 2] - Ks[0, 1] * ((sy * (vs - cy)) / Ks[1, 1])) / Ks[0, 0] * zz,
                   sy * (vs - cy) / Ks[1, 1] * zz,
                   zz], axis=-1)
    Xw = (Xc - ts) @ Rs            # R^T (Xc - t)
    Xd = Xw @ Rd.T + td            # Rd Xw + td
    u = Kd[0, 0] * Xd[..., 0] / Xd[..., 2] + Kd[0, 1] * Xd[..., 1] / Xd[..., 2] + Kd[0, 2]
    cyd = (H - 1) - Kd[1, 2] if y_origin == "bottom" else Kd[1, 2]
    v = cyd + sy * Kd[1, 1] * Xd[..., 1] / Xd[..., 2]
    return u, v


cams = load_calib()
f = "f000"
P = gray(rf"{ROOT}\cam6\depth-cam6-{f}.png")
z = depth_from_P(P)

regions = {
    "left wall poster (far)": (slice(180, 330), slice(0, 120)),
    "right dancer body (near)": (slice(150, 600), slice(700, 850)),
    "left dancer body (near)": (slice(200, 600), slice(230, 330)),
    "floor foreground": (slice(620, 740), slice(200, 800)),
    "curtain (far)": (slice(60, 300), slice(600, 900)),
}

print("region                       Pmed   z_med | predicted dU (cam6->cam5 / cam6->cam7) [y_origin=bottom]"
      " | top")
for name, sl in regions.items():
    yy, xx = np.mgrid[sl[0], sl[1]]
    yy, xx = yy.ravel()[::37], xx.ravel()[::37]
    zz = z[yy, xx]
    Pmed = float(np.median(P[yy, xx]))
    zmed = float(np.median(zz))
    out = []
    for dst in (5, 7):
        for yo in ("bottom", "top"):
            u, v = predict(cams, 6, dst, xx.astype(np.float64), yy.astype(np.float64), zmed, yo)
            out.append((dst, yo, float(np.median(u - xx)), float(np.median(v - yy))))
    line = f"{name:26s} {Pmed:5.0f} {zmed:7.2f} |"
    for dst, yo, du, dv in out:
        line += f"  c{dst}({yo[0]}): dU={du:+7.2f} dV={dv:+6.2f} |"
    print(line)

print("\n=== measured best integer (dx,dy) per region, cam6 -> camK ===")
ref = gray(rf"{ROOT}\cam6\color-cam6-{f}.jpg")
refg = np.abs(np.diff(ref, axis=1, prepend=ref[:, :1]))


def best_shift2d(a, b, reg, drange, srange):
    ys, xs = reg
    ya, xa = ys.start, xs.start
    best = None
    for dv in srange:
        for du in drange:
            y0a, y1a = ya + max(dv, 0), ya + (ys.stop - ys.start) + min(dv, 0)
            x0a, x1a = xa + max(du, 0), xa + (xs.stop - xs.start) + min(du, 0)
            pa = a[y0a:y1a, x0a:x1a]
            pb = b[y0a - dv:y0a - dv + pa.shape[0], x0a - du:x0a - du + pa.shape[1]]
            if pa.size == 0:
                continue
            d = np.abs(pa - pb).mean()
            if best is None or d < best[2]:
                best = (du, dv, float(d))
    return best


for k in (5, 7):
    oth = gray(rf"{ROOT}\cam{k}\color-cam{k}-{f}.jpg")
    othg = np.abs(np.diff(oth, axis=1, prepend=oth[:, :1]))
    print(f"-- cam6 -> cam{k} --")
    for name, sl in regions.items():
        b = best_shift2d(refg, othg, sl, range(-220, 221, 1), range(-12, 13, 1))
        print(f"   {name:26s} best (du,dv)=({b[0]:+4d},{b[1]:+3d}) sad={b[2]:7.3f}")
