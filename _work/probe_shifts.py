"""Step 0 probe #3: per-patch local shift between cam6 and cam5/cam7 via template matching."""
import cv2
import numpy as np
from PIL import Image

ROOT = r"D:\项目\3DVideos-distrib\MSR3DVideo-Ballet"
f = "f000"


def gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float32)


def grad(a):
    return np.abs(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3))


ref = gray(rf"{ROOT}\cam6\color-cam6-{f}.jpg")
gref = grad(ref)
P = gray(rf"{ROOT}\cam6\depth-cam6-{f}.png")

# patches centred on informative spots (cx, cy, half-size)
patches = {
    "poster left wall (far)": (60, 250, 45),
    "curtain upper right (far)": (750, 150, 45),
    "barre mid-right (far)": (900, 430, 45),
    "right dancer torso (near)": (760, 300, 45),
    "right dancer legs (near)": (790, 560, 45),
    "left dancer torso (near)": (280, 330, 45),
    "left dancer arm (near)": (430, 340, 45),
    "floor centre (mid)": (520, 680, 45),
    "floor left (mid)": (180, 650, 45),
}

for k in (5, 7):
    oth = gray(rf"{ROOT}\cam{k}\color-cam{k}-{f}.jpg")
    goth = grad(oth)
    print(f"=== cam6 -> cam{k} (patch template from cam6 searched in cam{k}) ===")
    print(f"{'patch':28s} {'Pmed':>5s} {'best(dx,dy)':>13s} {'score':>8s}  {'top5 candidates':>30s}")
    for name, (cx, cy, hs) in patches.items():
        tpl = gref[cy - hs:cy + hs, cx - hs:cx + hs]
        # search window
        pad = 260
        y0, y1 = max(0, cy - hs - 20), min(oth.shape[0], cy + hs + 20)
        x0, x1 = max(0, cx - hs - pad), min(oth.shape[1], cx + hs + pad)
        win = goth[y0:y1, x0:x1]
        if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
            continue
        res = cv2.matchTemplate(win, tpl, cv2.TM_SQDIFF)
        flat = res.ravel()
        order = np.argsort(flat)[:5]
        cands = []
        for o in order:
            yy, xx = np.unravel_index(o, res.shape)
            cands.append((int(x0 + xx - (cx - hs)), int(y0 + yy - (cy - hs)), float(flat[o])))
        Pmed = float(np.median(P[cy - hs:cy + hs, cx - hs:cx + hs]))
        print(f"{name:28s} {Pmed:5.0f} {str((cands[0][0], cands[0][1])):>13s} {cands[0][2]:8.2f}  "
              f"{[(c[0], c[1]) for c in cands[1:4]]}")
