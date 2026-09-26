"""Visualisation and metrics helpers for the reproduction checks."""
import cv2
import numpy as np

PALETTE = {"hole": (255, 0, 0), "crack": (0, 0, 255), "ghost": (0, 255, 255),
           "oofa": (255, 0, 0), "disocc": (0, 255, 0)}


def to_u8(img):
    return np.clip(np.asarray(img, np.float32), 0, 255).astype(np.uint8)


def colorize(field, valid_mask=None, invalid_color=40, vmin=None, vmax=None, cmap=True):
    """Robust-normalised TURBO colour map of a scalar field (RGB out)."""
    f = np.asarray(field, np.float32)
    lo = np.nanmin(f) if vmin is None else vmin
    hi = np.nanmax(f) if vmax is None else vmax
    n = np.zeros_like(f) if hi <= lo else np.clip((f - lo) / (hi - lo), 0, 1)
    u8 = (n * 255).astype(np.uint8)
    vis = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO) if cmap else np.repeat(u8[:, :, None], 3, 2)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    if valid_mask is not None:
        vis[~valid_mask] = (invalid_color,) * 3
    return vis


def imwrite_gray(path, field, valid_mask=None, vmin=None, vmax=None):
    """Save a scalar field as a TURBO PNG (invalid pixels grey)."""
    from . import io_utils
    io_utils.imwrite(path, colorize(field, valid_mask=valid_mask, vmin=vmin, vmax=vmax))


def overlay_mask(img_rgb, mask, color, alpha=0.75):
    vis = to_u8(img_rgb).copy()
    m = mask.astype(bool)
    vis[m] = (1 - alpha) * vis[m] + alpha * np.array(color, np.float32)
    return vis.astype(np.uint8)


def label(img, text, scale=0.5):
    bar = np.full((24, img.shape[1], 3), 28, np.uint8)
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def grid(tiles, cols):
    rows = int(np.ceil(len(tiles) / cols))
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    canvas = np.full((rows * h + (rows - 1) * 8, cols * w + (cols - 1) * 8, 3), 12, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * (h + 8):r * (h + 8) + t.shape[0], c * (w + 8):c * (w + 8) + t.shape[1]] = t
    return canvas


def text_card(lines, width=760):
    card = np.full((len(lines) * 19 + 16, width, 3), 20, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(card, ln, (10, 22 + i * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return card


def dilate(mask, k=1):
    """Binary dilation by a (2k+1)^2 box, for making sparse masks visible."""
    if k <= 0:
        return mask.astype(bool)
    ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), ker).astype(bool)


def zoom(img, cx, cy, half=110):
    h, w = img.shape[:2]
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    return img[y0:y1, x0:x1].copy()


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def psnr(a, b, mask=None, maxval=255.0):
    """PSNR on grayscale; `mask` selects the compared pixels (True = compare)."""
    a = _gray(a).astype(np.float64)
    b = _gray(b).astype(np.float64)
    d = (a - b) ** 2
    if mask is not None:
        d = d[mask.astype(bool)]
    mse = float(d.mean()) if d.size else float("nan")
    return float("inf") if mse <= 0 else 10.0 * np.log10(maxval ** 2 / mse)


def _gray(img):
    img = np.asarray(img)
    if img.ndim == 3:
        return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    return img.astype(np.float64)


def ssim(a, b, mask=None, sigma=1.5):
    """Grayscale SSIM with the usual 11x11 Gaussian window (Brkmann defaults)."""
    from scipy.ndimage import gaussian_filter
    x = _gray(a).astype(np.float64)
    y = _gray(b).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_x = gaussian_filter(x, sigma)
    mu_y = gaussian_filter(y, sigma)
    sxx = gaussian_filter(x * x, sigma) - mu_x ** 2
    syy = gaussian_filter(y * y, sigma) - mu_y ** 2
    sxy = gaussian_filter(x * y, sigma) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2)
    s = num / den
    if mask is not None:
        s = s[mask.astype(bool)]
    return float(s.mean())
