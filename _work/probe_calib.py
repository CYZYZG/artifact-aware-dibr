"""Step 0 probe: measure the real horizontal disparity between cam6 and neighbouring cameras,
and check the linear inverse-depth model z = 1/((P/255)*(1/42 - 1/130) + 1/130)."""
import numpy as np
from PIL import Image

ROOT = r"D:\项目\3DVideos-distrib\MSR3DVideo-Ballet"
MINZ, MAXZ = 42.0, 130.0


def gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float32)


def best_shift_sad(a, b, rng, use_grad=True):
    """b shifted by dx relative to a: minimise SAD of a(x) vs b(x+dx) over the whole overlap."""
    if use_grad:
        gx = np.array([-1, 0, 1], np.float32)[None, :]
        a = np.abs(np.apply_along_axis(lambda r: np.convolve(r, gx[0], "same"), 1, a))
        b = np.abs(np.apply_along_axis(lambda r: np.convolve(r, gx[0], "same"), 1, b))
    best = None
    for dx in rng:
        if dx >= 0:
            d = np.abs(a[:, dx:] - b[:, : b.shape[1] - dx]).mean()
        else:
            d = np.abs(a[:, : b.shape[1] + dx] - b[:, -dx:]).mean()
        if best is None or d < best[1]:
            best = (dx, float(d))
    return best


f = "f000"
ref = gray(rf"{ROOT}\cam6\color-cam6-{f}.jpg")
pref = np.asarray(Image.open(rf"{ROOT}\cam6\depth-cam6-{f}.png").convert("L")).astype(np.float32)

print("=== camera geometry from calibParams-ballet.txt ===")
lines = open(rf"{ROOT}\calibParams-ballet.txt").read().split("\n")
cams = {}
i = 0
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
    R, t = Rt[:, :3], Rt[:, 3]
    C = -R.T @ t
    cams[idx] = dict(K=K, R=R, t=t, C=C)
    print(f"cam{idx}: center=({C[0]:8.4f},{C[1]:8.4f},{C[2]:8.4f})  "
          f"fx={K[0,0]:8.2f} fy={K[1,1]:8.2f} cx={K[0,2]:8.2f} cy={K[1,2]:8.2f} skew={K[0,1]:7.3f}")
    i += 8

print("\n=== measured integer horizontal shift vs cam6 (positive = cam6 content appears shifted"
      " to the right in the other camera) ===")
for k in [4, 5, 7]:
    oth = gray(rf"{ROOT}\cam{k}\color-cam{k}-{f}.jpg")
    dx, sad = best_shift_sad(ref, oth, range(-260, 261))
    dxr, sadr = best_shift_sad(oth, ref, range(-260, 261))
    print(f"cam6 vs cam{k}:  argmin a(x)=b(x+dx): dx={dx:+4d} (sad={sad:7.3f})   "
          f"reverse dx={dxr:+4d} -> cam6->cam{k} shift ~ {dx:+d}")

print("\n=== depth model ===")
z = 1.0 / ((pref / 255.0) * (1.0 / MINZ - 1.0 / MAXZ) + 1.0 / MAXZ)
print(f"z: min {z.min():.2f} max {z.max():.2f} mean {z.mean():.2f}   (P min {pref.min():.0f} "
      f"max {pref.max():.0f} mean {pref.mean():.2f})")
for P in [0, 50, 100, 150, 200, 254]:
    zz = 1.0 / ((P / 255.0) * (1.0 / MINZ - 1.0 / MAXZ) + 1.0 / MAXZ)
    d = cams[6]["K"][0, 0] * abs(cams[7]["C"][0] - cams[6]["C"][0]) / zz
    print(f"  P={P:3d} -> z={zz:7.2f}  predicted |cam6-cam7| disparity = {d:7.2f} px")
