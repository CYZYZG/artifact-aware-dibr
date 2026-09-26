"""Step 2 - crack detection and filling  (paper II-A).

Paper recipe
------------
1. `D_w` has holes set to an artificial negative value (-1), so empty cracks behave like
   translucent ones (a crack pixel whose disparity is lower than its neighbourhood).
2. Filter with a line-shaped structuring element `H = [1 1 1 1]^T` (vertical line, for
   vertical cracks produced by a horizontal projection).  The paper calls it a closing,
   but describes exactly a max-filter: "replace the disparities in D_w with the highest
   value within the operation area of H" -> grayscale dilation.
3. A pixel is a crack where  D_hat_w(u,v) - D_w(u,v) >= lambda   (lambda = 5).
4. Crack pixels are removed from I_w and D_w.
5. `D_w` is filled by copying the already-estimated values from `D_hat_w`.
6. `I_w` is filled with Hierarchical Hole Filling (HHF, [16]): a Gaussian pyramid whose
   filters only ever use VALID pixels, so empty data never pollutes the estimate.
   Only cracks (<= 2 px wide) are reconstructed this way; disocclusions/OOFAs are left
   for the exemplar-based stage.
"""
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def line_se(length=4, orientation="v", anchor="matlab"):
    """Line-shaped structuring element; 'v' = vertical line (paper's H).

    The paper writes H = [1 1 1 1]^T but does not define the SE anchor.  For an even
    length the anchor is not the true centre, so the choice matters: with the MATLAB
    `imdilate` convention (origin at index floor((L+1)/2), 1-based) a 4-tap SE covers
    offsets {-1, 0, +1, +2}, whereas OpenCV's default anchor covers {-2, -1, 0, +1}
    -- i.e. the detection window is shifted by one row/column.  We follow MATLAB.
    """
    if orientation == "v":
        se = np.ones((length, 1), np.uint8)
    elif orientation == "h":
        se = np.ones((1, length), np.uint8)
    else:
        raise ValueError(orientation)
    if anchor == "matlab":
        a = max(0, length // 2 - 1)          # offsets {-1,0,1,2} for length 4
        return se, ((0, a) if orientation == "v" else (a, 0))
    return se, (-1, -1)                      # OpenCV default


def detect_cracks(D_w, lam=5.0, se_len=4, orientation="v", anchor="matlab"):
    """Return (crack_mask, D_hat, diff).

    D_w : (H, W) float32, holes = negative sentinel (-1).
    """
    D = np.asarray(D_w, np.float32)
    se, anc = line_se(se_len, orientation, anchor)
    # grayscale dilation = max inside the SE; outside pixels never contribute
    D_hat = cv2.dilate(D, se, anchor=anc, borderType=cv2.BORDER_CONSTANT,
                       borderValue=-np.inf if D.dtype.kind == "f" else 0)
    D_hat = np.where(np.isfinite(D_hat), D_hat, D)
    diff = D_hat - D
    return diff >= lam, D_hat, diff


def detect_cracks_both(D_w, lam=5.0, se_len=4):
    """Union of the vertical- and horizontal-SE detections (when a vertical projection
    also exists).  Reported separately because the horizontal SE reacts to the boundary
    column of large vertical holes (e.g. the OOFA strip), which is not a crack."""
    c_v, hat_v, d_v = detect_cracks(D_w, lam, se_len, "v")
    c_h, hat_h, d_h = detect_cracks(D_w, lam, se_len, "h")
    return c_v | c_h, np.maximum(hat_v, hat_h), np.maximum(d_v, d_h), c_v, c_h


# --------------------------------------------------------------------------- #
# hierarchical hole filling  (HHF, Solh & AlRegib [16])
# --------------------------------------------------------------------------- #
def _masked_blur(img, valid, sigma, ksize):
    """Gaussian blur that only ever sees valid pixels."""
    v = valid.astype(np.float32)
    num = cv2.GaussianBlur(img * v[..., None] if img.ndim == 3 else img * v,
                           (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)
    den = cv2.GaussianBlur(v, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)
    eps = 1e-8
    out = num / np.maximum(den, eps)[..., None] if img.ndim == 3 else num / np.maximum(den, eps)
    return out, den > 1e-6


def _down(img, valid, sigma, ksize):
    blur, v = _masked_blur(img, valid, sigma, ksize)
    return blur[::2, ::2], v[::2, ::2]


def hhf_fill(img, valid, sigma=1.0, ksize=5, min_size=8, coarse_iters=16,
             refine_iters=1):
    """Fill the invalid pixels of `img` using a Gaussian pyramid of valid data only.

    Parameters
    ----------
    img   : (H, W) or (H, W, C) float32
    valid : (H, W) bool, True where `img` holds real data
    sigma, ksize : Gaussian filter applied at every level (valid pixels only)
    min_size     : stop building the pyramid below this side length
    coarse_iters : masked-blur iterations used to complete the coarsest level
    refine_iters : masked-blur iterations per level during reconstruction

    Returns the filled image (float32, same shape).  Values at originally valid pixels
    are returned unchanged, and no invalid pixel value ever influences the result.
    """
    a = np.asarray(img, np.float32)
    squeeze = a.ndim == 2
    if squeeze:
        a = a[..., None]
    v = np.asarray(valid, bool)
    if v.shape != a.shape[:2]:
        raise ValueError("valid mask shape mismatch")
    if v.all():
        return a[..., 0] if squeeze else a

    pyr = []
    cur_i, cur_v = a, v
    while min(cur_i.shape[:2]) > min_size and not cur_v.all():
        pyr.append((cur_i, cur_v))
        cur_i, cur_v = _down(cur_i, cur_v, sigma, ksize)

    # ---- complete the coarsest level (only valid data is used) ----
    est, ev = cur_i.copy(), cur_v.copy()
    for _ in range(coarse_iters):
        if ev.all():
            break
        blur, vv = _masked_blur(est, ev, sigma, ksize)
        est = np.where(ev[..., None], est, blur)
        ev = ev | vv
    if not ev.all():                      # degenerate: no valid data anywhere near
        if ev.any():
            est = np.where(ev[..., None], est, est[ev].mean(axis=0))
        else:
            est = np.zeros_like(est)

    # ---- reconstruct upwards, never touching valid pixels ----
    out = est
    for img_k, v_k in reversed(pyr):
        up = cv2.resize(out, (img_k.shape[1], img_k.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
        if up.ndim == 2 and img_k.ndim == 3:
            up = up[..., None]
        out = np.where(v_k[..., None], img_k, up)
        for _ in range(refine_iters):
            blur, _ = _masked_blur(out, v_k, sigma, ksize)
            out = np.where(v_k[..., None], out, blur)
    return out[..., 0] if squeeze else out


# --------------------------------------------------------------------------- #
# joint step
# --------------------------------------------------------------------------- #
def fill_cracks(I_w, D_w, lam=5.0, se_len=4, orientation="v", hhf_sigma=1.0,
                hhf_ksize=5, shape_filter="none", max_thickness=3.0):
    """Detect cracks in D_w, refill D_w from D_hat, and refill I_w with HHF.

    shape_filter
        "none"  paper-faithful: every pixel whose disparity drops by >= lambda relative
                to the max inside the line SE is a crack.  On this data that also flags
                the top/bottom caps of large disocclusion holes (they have valid content
                within the SE), which the paper's crack model does not describe.
        "thin"  empty cracks are additionally restricted to hole components whose
                thickness is <= max_thickness (i.e. real 1-2 px slivers); translucent
                cracks (not holes) are kept as detected.

    Returns a dict with crack mask, D_hat, diff, filled colour/depth and the remaining
    (non-crack) hole mask.
    """
    hole = np.asarray(D_w) < 0
    if orientation == "both":
        crack, D_hat, diff, c_v, c_h = detect_cracks_both(D_w, lam, se_len)
        extra = {"crack_vertical": c_v, "crack_horizontal": c_h}
    else:
        crack, D_hat, diff = detect_cracks(D_w, lam, se_len, orientation)
        extra = {}
    crack_raw = crack.copy()
    if shape_filter == "thin":
        _, lab = component_stats(hole)
        thin = {r["label"] for r in component_stats(hole)[0]
                if r["thickness"] <= max_thickness}
        keep = np.isin(lab, list(thin)) if thin else np.zeros_like(hole)
        crack = (crack & hole & keep) | (crack & ~hole)
    # NOTE: a crack is NOT necessarily an empty pixel.  Empty cracks carry the -1
    # sentinel, while *translucent* cracks carry real (but too small) disparity and a
    # wrong texture; the paper asks for both to be detected.  Both are invalidated.
    valid = ~(hole | crack)
    D_filled = np.asarray(D_w, np.float32).copy()
    D_filled[crack] = D_hat[crack]
    I_hhf = hhf_fill(np.asarray(I_w, np.float32), valid, sigma=hhf_sigma, ksize=hhf_ksize)
    I_filled = np.where(crack[..., None], I_hhf, I_w) if I_w.ndim == 3 else \
        np.where(crack, I_hhf, I_w)
    res = dict(crack=crack, crack_raw=crack_raw, D_hat=D_hat, diff=diff, D_filled=D_filled,
               I_hhf=I_hhf, I_filled=I_filled, hole=hole,
               remaining_holes=hole & ~crack,
               empty_crack=crack & hole, translucent_crack=crack & ~hole)
    res.update(extra)
    return res


# --------------------------------------------------------------------------- #
# shape statistics of the hole components (how crack-like is the data?)
# --------------------------------------------------------------------------- #
def component_stats(mask):
    """Per-component area / bounding box / thickness / elongation of a binary mask."""
    m = mask.astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    edt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    rows = []
    for k in range(1, n):
        ys, xs = np.where(lab == k)
        area = int(stats[k, cv2.CC_STAT_AREA])
        bw = int(stats[k, cv2.CC_STAT_WIDTH])
        bh = int(stats[k, cv2.CC_STAT_HEIGHT])
        rows.append(dict(label=k, area=area, bw=bw, bh=bh,
                         thickness=float(2.0 * edt[ys, xs].max()),
                         elongation=float(max(bw, bh) / max(1, min(bw, bh))),
                         cx=float(cent[k][0]), cy=float(cent[k][1]),
                         touches_border=bool(stats[k, cv2.CC_STAT_LEFT] <= 1 or
                                             stats[k, cv2.CC_STAT_TOP] <= 1 or
                                             stats[k, cv2.CC_STAT_LEFT] + bw >= mask.shape[1] - 1 or
                                             stats[k, cv2.CC_STAT_TOP] + bh >= mask.shape[0] - 1)))
    return rows, lab


def synthetic_cracks(P, hole, n=60, width=2, height=15, uniform_tol=5.0, rng=None):
    """Place thin vertical slivers on real content in nearly-uniform-disparity areas,
    mimicking the paper's description of cracks.  Returns (mask, clean_values)."""
    rng = rng or np.random.default_rng(0)
    h, w = P.shape
    mask = np.zeros((h, w), bool)
    tries = 0
    made = 0
    while made < n and tries < n * 200:
        tries += 1
        x = int(rng.integers(4, w - 4))
        y = int(rng.integers(4, h - 4 - height))
        sl = (slice(y, y + height), slice(max(0, x - 3), min(w, x + 4)))
        if hole[sl].any():
            continue
        if float(P[sl].max() - P[sl].min()) > uniform_tol:
            continue
        mask[y:y + height, x:x + width] = ~hole[y:y + height, x:x + width]
        made += 1
    return mask, P.copy()


def validate_hhf(I_w, D_w, hole, n=60, width=2, height=15, uniform_tol=5.0,
                 max_pairs=40, seed=0):
    """Leave-one-out test of the crack filler.

    Thin slivers are carved out of *valid* content in nearly-uniform-disparity areas,
    filled with HHF, and compared with the true content.  A simple vertical linear
    interpolation is used as the baseline, so the numbers show whether the hierarchical
    Gaussian estimate is actually better than naive 1D interpolation.
    """
    I_w = np.asarray(I_w, np.float32)
    mask, _ = synthetic_cracks(D_w, hole, n=n, width=width, height=height,
                               uniform_tol=uniform_tol,
                               rng=np.random.default_rng(seed))
    if mask.sum() == 0:
        return None
    valid = ~(hole | mask)
    filled = hhf_fill(I_w, valid)
    # baseline: vertical linear interpolation between the nearest valid rows
    idx = np.arange(I_w.shape[0])[:, None] * np.ones((1, I_w.shape[1]), int)
    up = np.where(valid, idx, -10 ** 6)
    up = np.maximum.accumulate(up, axis=0)
    dn = np.where(valid, idx, 10 ** 6)
    dn = np.minimum.accumulate(dn[::-1], axis=0)[::-1]
    y0 = np.clip(up, 0, I_w.shape[0] - 1)
    y1 = np.clip(dn, 0, I_w.shape[0] - 1)
    c0 = I_w[y0, np.arange(I_w.shape[1])[None, :].repeat(I_w.shape[0], 0)]
    c1 = I_w[y1, np.arange(I_w.shape[1])[None, :].repeat(I_w.shape[0], 0)]
    denom = np.maximum(y1 - y0, 1)[..., None]
    w1 = ((idx - y0) / (y1 - y0 + 1e-9))[..., None]
    lin = np.where(valid[..., None], I_w, c0 * (1 - w1) + c1 * w1)

    def err(a):
        d = np.abs(a - I_w)
        return float(d[mask].mean()), float(np.sqrt((d[mask] ** 2).mean()))

    e_hhf = err(filled)
    e_lin = err(lin)
    return dict(n_slivers=n, sliver_px=int(mask.sum()),
                hhf_mae=e_hhf[0], hhf_rmse=e_hhf[1],
                lin_mae=e_lin[0], lin_rmse=e_lin[1],
                hhf_psnr=float(20 * np.log10(255.0 / max(e_hhf[1], 1e-6))),
                lin_psnr=float(20 * np.log10(255.0 / max(e_lin[1], 1e-6))),
                mask=mask, filled=filled, lin=lin)
