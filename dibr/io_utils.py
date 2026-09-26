"""Safe image IO and data loading (Windows non-ASCII path aware)."""
import os

import cv2
import numpy as np

DATASET_ROOT_DEFAULT = os.environ.get("BALLET_DATA_ROOT", r"D:\项目\3DVideos-distrib\MSR3DVideo-Ballet")


def imread(path, gray=False):
    """Read an image from a path that may contain non-ASCII characters."""
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        raise IOError(f"empty file or missing: {path}")
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(buf, flag)
    if img is None:
        raise IOError(f"imdecode failed: {path}")
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imwrite(path, img_bgr_or_rgb):
    """Write with imencode+tofile (encoding-safe).

    The pipeline keeps every image in RGB order, while cv2.imencode/imwrite assume
    BGR for 3-channel input, so convert on the way out.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = img_bgr_or_rgb
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"imencode failed: {path}")
    buf.tofile(path)


def resolve_paths(root, cam, frame):
    """Locate (color_path, depth_path) for a camera/frame.

    Supports two layouts:
      * flat      : <root>/color-cam6-f000.jpg      (the user's ./input folder)
      * per-camera: <root>/cam6/color-cam6-f000.jpg (the MSR distribution)
    """
    names = (f"color-cam{cam}-{frame}.jpg", f"depth-cam{cam}-{frame}.png")
    for layout in (os.path.join(root, f"cam{cam}"), root):
        cand = [os.path.join(layout, n) for n in names]
        if all(os.path.isfile(p) for p in cand):
            return tuple(cand)
    raise FileNotFoundError(f"no color/depth pair for cam{cam} {frame} under {root}")


def load_view(root, cam, frame):
    """Return dict(color=RGB uint8 HxWx3, invdepth=P uint8 HxW, paths=...)."""
    cpath, dpath = resolve_paths(root, cam, frame)
    color = imread(cpath, gray=False)
    depth = imread(dpath, gray=True)
    if color.shape[:2] != depth.shape[:2]:
        raise ValueError(f"size mismatch {color.shape} vs {depth.shape}")
    return {"cam": cam, "frame": frame, "color": color, "depth": depth,
            "color_path": cpath, "depth_path": dpath}
