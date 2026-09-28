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
             refine_iters=1, return_support=False):
    """Fill the invalid pixels of `img` using a Gaussian pyramid of valid data only.

    Parameters
    ----------
    img   : (H, W) or (H, W, C) float32
    valid : (H, W) bool, True where `img` holds real data
    sigma, ksize : Gaussian filter applied at every level (valid pixels only)
    min_size     : stop building the pyramid below this side length
    coarse_iters : masked-blur iterations used to complete the coarsest level
    refine_iters : masked-blur iterations per level during reconstruction

    return_support : also return the (H, W) mask of pixels within the kernel's reach of
        real data; outside it the coarse estimate is kept instead of ~0 (which used to
        leave BLACK pixels 3-4 px away from content).

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
        # take the blurred value only where valid data actually supports it: where the
        # kernel reaches nothing, `blur` is num/max(den, eps) ~ 0 and would paint black
        est = np.where(ev[..., None], est, np.where(vv[..., None], blur, est))
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
            blur, sup_k = _masked_blur(out, v_k, sigma, ksize)
            # unsupported pixels keep the upsampled coarser estimate (a real colour from
            # the neighbourhood) rather than the ~0 produced by an empty kernel
            out = np.where(v_k[..., None], out,
                           np.where(sup_k[..., None], blur, out))
    res = out[..., 0] if squeeze else out
    if return_support:
        _, sup = _masked_blur(res, v, sigma, ksize)
        return res, sup
    return res


# --------------------------------------------------------------------------- #
# joint step
# --------------------------------------------------------------------------- #
def _side_neighbours(I_w, D_w, valid, max_reach=8):
    """Nearest valid pixel to the left / right along each row, with its distance."""
    H, W = valid.shape
    C = I_w.shape[2] if I_w.ndim == 3 else 1
    src = np.asarray(I_w, np.float32).reshape(H, W, C)
    D = np.asarray(D_w, np.float32)
    ys = np.arange(H)[:, None].repeat(W, 1)
    xs = np.arange(W)[None, :].repeat(H, 0)
    out = {}
    for side in ("l", "r"):
        col = np.zeros((H, W, C), np.float32)
        dd = np.full((H, W), np.nan, np.float32)
        dist = np.full((H, W), np.inf, np.float32)
        found = np.zeros((H, W), bool)
        for k in range(1, max_reach + 1):
            xk = xs - k if side == "l" else xs + k
            inb = (xk >= 0) & (xk < W) & ~found
            if not inb.any():
                continue
            xc = np.clip(xk, 0, W - 1)
            ok = inb & valid[ys, xc]
            if ok.any():
                col[ok] = src[ys[ok], xc[ok]]
                dd[ok] = D[ys[ok], xc[ok]]
                dist[ok] = k
                found[ok] = True
        out[side] = (col, dd, dist, found)
    return out


def interp_across(I_w, D_w, valid, crack, max_reach=6):
    """Fill a thin crack by LINEAR interpolation between its two sides.

    A 1-2 px crack is a missing sliver of a continuous image; interpolating across the gap
    (perpendicular to the crack) continues the local texture, whereas copying from one side
    shifts it by the copy distance (very visible on curtain stripes) and isotropic HHF
    pulls the dark foreground into the sliver.  Returns (colour, mask).
    """
    H, W = crack.shape
    C = I_w.shape[2] if I_w.ndim == 3 else 1
    nb = _side_neighbours(I_w, D_w, valid, max_reach)
    lcol, _ld, ldist, lfound = nb["l"]
    rcol, _rd, rdist, rfound = nb["r"]
    both = crack & lfound & rfound
    w = (ldist / np.maximum(ldist + rdist, 1e-6))[:, :, None]
    col = lcol * (1.0 - w) + rcol * w
    if C == 1:
        col = col[..., 0]
    return col, both


def bg_side_fill(I_w, D_w, crack, valid, lam=5.0, max_reach=8):
    """Fill crack pixels from the BACKGROUND side of the depth discontinuity.

    A crack that sits on a depth edge (|disparity on the two sides| >= lam) is a sliver of
    revealed background.  HHF interpolates isotropically and therefore pulls the (usually
    much darker) foreground edge into the sliver, which leaves a dark rim along the
    silhouette.  Here such a crack pixel copies the nearest valid pixel of the background
    side instead.  Cracks *inside* a uniform region (both sides the same disparity) are
    left untouched so that HHF, as in the paper, fills them.

    Returns (colour, mask) where mask marks the pixels handled here.
    """
    H, W = crack.shape
    img = np.asarray(I_w, np.float32)
    D = np.asarray(D_w, np.float32)
    val = np.asarray(valid, bool)
    C = img.shape[2] if img.ndim == 3 else 1
    src = img.reshape(H, W, C)
    out = np.zeros((H, W, C), np.float32)
    done = np.zeros((H, W), bool)

    # nearest valid pixel to the left / right along the row, plus its disparity
    lcol = np.zeros((H, W, C), np.float32)
    ld = np.full((H, W), np.nan, np.float32)
    rcol = np.zeros((H, W, C), np.float32)
    rd = np.full((H, W), np.nan, np.float32)
    lfound = np.zeros((H, W), bool)
    rfound = np.zeros((H, W), bool)
    xs = np.arange(W)[None, :].repeat(H, 0)
    ys = np.arange(H)[:, None].repeat(W, 1)
    for k in range(1, max_reach + 1):
        for side, xk, found, col, dd in ((0, xs - k, lfound, lcol, ld),
                                        (1, xs + k, rfound, rcol, rd)):
            inb = (xk >= 0) & (xk < W) & ~found
            if not inb.any():
                continue
            ok = inb & val[ys, np.clip(xk, 0, W - 1)]
            if ok.any():
                col[ok] = src[ys[ok], np.clip(xk[ok], 0, W - 1)]
                dd[ok] = D[ys[ok], np.clip(xk[ok], 0, W - 1)]
                found[ok] = True

    edge = np.isfinite(ld) & np.isfinite(rd) & (np.abs(ld - rd) >= lam)
    sel_l = crack & edge & (ld <= rd) & lfound          # left side is the background
    sel_r = crack & edge & (rd < ld) & rfound           # right side is the background
    out[sel_l] = lcol[sel_l]
    out[sel_r] = rcol[sel_r]
    done = sel_l | sel_r
    if C == 1:
        out = out[..., 0]
    return out, done


def fill_cracks(I_w, D_w, lam=5.0, se_len=4, orientation="v", hhf_sigma=1.0,
                hhf_ksize=5, shape_filter="none", max_thickness=3.0,
                fill_mode="hhf", slit_only=False):
    """Detect cracks in D_w, refill D_w from D_hat, and refill I_w with HHF.

    shape_filter
        "none"  paper-faithful: every pixel whose disparity drops by >= lambda relative
                to the max inside the line SE is a crack.  On this data that also flags
                the top/bottom caps of large disocclusion holes (they have valid content
                within the SE), which the paper's crack model does not describe.
        "thin"  empty cracks are additionally restricted to hole components whose
                thickness is <= max_thickness (i.e. real 1-2 px slivers); translucent
                cracks (not holes) are kept as detected.
    slit_only
        Per-pixel version of that idea and the reliable one: a crack is a THIN slit, so the
        hole's local half-width (distance transform) must stay <= max_thickness / 2 in a
        small neighbourhood.  Without it the raw rule also flags the caps/edges of LARGE
        disocclusion holes (they have valid content within the line SE and D_w carries the
        -1 sentinel there), and those pixels are then "fixed" by HHF instead of by the
        exemplar filling stage - the cause of the non-black/half-filled holes seen in
        practice.  Rejected pixels stay in `remaining_holes` and are filled normally.

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
    big_hole_px = 0
    if slit_only:
        dt = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
        rad = int(np.ceil(max_thickness)) + 1
        local = cv2.dilate(dt, np.ones((2 * rad + 1, 2 * rad + 1), np.uint8))
        thin = local <= max(1.0, max_thickness / 2.0)
        big_hole_px = int((crack & hole & ~thin).sum())
        crack = (crack & ~hole) | (crack & hole & thin)
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
    I_hhf, sup_hhf = hhf_fill(np.asarray(I_w, np.float32), valid, sigma=hhf_sigma,
                              ksize=hhf_ksize, return_support=True)
    # colour and depth are written together, so every pixel the depth map declares filled
    # also carries a real colour (never the ~0 of an empty kernel)
    I_filled = np.where(crack[..., None], I_hhf, I_w) if I_w.ndim == 3 else \
        np.where(crack, I_hhf, I_w)
    bg_done = None
    lin_done = None
    if fill_mode in ("auto", "linear", "bg"):
        if fill_mode in ("auto", "linear"):
            lin_col, lin_done = interp_across(I_w, D_w, valid, crack)
            use = lin_done[..., None] if I_w.ndim == 3 else lin_done
            I_filled = np.where(use, lin_col, I_filled)
        if fill_mode in ("auto", "bg"):
            # cracks still without a two-sided support (e.g. at an image border, or a
            # crack on a depth step wider than the reach) take the background side
            rest = crack & ~(lin_done if lin_done is not None else False)
            bg_col, bg_done = bg_side_fill(I_w, D_w, rest, valid, lam=lam)
            use = bg_done[..., None] if I_w.ndim == 3 else bg_done
            I_filled = np.where(use, bg_col, I_filled)
    res = dict(crack=crack, crack_raw=crack_raw, D_hat=D_hat, diff=diff, D_filled=D_filled,
               bg_side_px=int(0 if bg_done is None else bg_done.sum()),
               big_hole_crack_px=int(big_hole_px),
               hhf_unsupported_px=int((crack & ~sup_hhf).sum()),
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
