"""Step 3 - ghost detection and correction  (paper II-B).

Paper recipe
------------
1. `G` = all holes; OOFAs are cleared in `G` (they sit at one image edge, identified from
   the warping direction: a horizontal projection opens the slit at the trailing edge).
2. `G_hat = dilate(G, diamond(size 2))`, candidates `G_dO = G_hat \\ G` (a band just
   outside the disocclusions).
3. FG points are removed from the candidates with a per-hole threshold `T_O`, the
   trimmed mean (alpha = 10 %) of the disparity along the hole boundary.
4. For every surviving candidate `p`: `mu_BG` = median of the 9x9 patch at `p` using only
   valid (non-candidate) pixels; `p_FG` = its position in the foreground, obtained by
   warping with the *extended* foreground disparity
   `D_FG = dilate(dilate(D, H), H^T)`; `mu_FG` = median of the 9x9 patch at `p_FG`.
   `d_BG = |I(p) - mu_BG|`, `d_FG = |I(p) - mu_FG|`.
   `d_BG >= d_FG` or `d_BG > alpha_sim (11)`  =>  ghost.
5. The ghost is fixed by warping its content to the correct place `p_FG`, "without
   replacing or deleting the synthetic view content".
"""
import cv2
import numpy as np

from . import warp as _warp


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #
def diamond(radius=2):
    """MATLAB strel('diamond', radius)."""
    r = radius
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return ((np.abs(y) + np.abs(x)) <= r).astype(np.uint8)


def find_oofa(hole, direction):
    """Leading run of holes scanned from the image edge the projection vacates.

    direction = +1 (content moves right, reference is to the right of the virtual view)
    -> OOFA on the left edge; direction = -1 -> OOFA on the right edge.
    """
    h, w = hole.shape
    oofa = np.zeros_like(hole)
    if direction > 0:
        for y in range(h):
            x = 0
            while x < w and hole[y, x]:
                oofa[y, x] = True
                x += 1
    else:
        for y in range(h):
            x = w - 1
            while x >= 0 and hole[y, x]:
                oofa[y, x] = True
                x -= 1
    return oofa


def candidate_band(hole, radius=2, oofa=None):
    """(candidates, G) with G = holes minus OOFA and candidates = dilate(G) \\ G.

    Unlike the literal set expression of the paper, pixels that are still holes (which is
    what an OOFA pixel is after `G` is cleared) are excluded: a ghost is a *content*
    pixel that was warped to the wrong place, so a hole cannot be a ghost candidate.
    """
    G = hole.copy()
    if oofa is not None:
        G &= ~oofa
    Ghat = cv2.dilate(G.astype(np.uint8), diamond(radius)).astype(bool)
    return (Ghat & ~G) & ~hole, G


def hole_labels(hole):
    """Label image of the hole components (float32 so cv2 morphology can dilate it)."""
    n, lab = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    return n, lab.astype(np.float32)


def label_of_nearest_hole(hole, radius=1):
    """For every pixel, the label of the nearest hole component (max-label dilation,
    adequate for the thin band around each hole).  0 = no hole nearby."""
    n, lab = hole_labels(hole)
    lab_dil = cv2.dilate(lab, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    return np.rint(lab_dil).astype(np.int32)


# --------------------------------------------------------------------------- #
# local FG/BG extractor
# --------------------------------------------------------------------------- #
def trimmed_mean(vals, alpha=0.10):
    v = np.sort(np.asarray(vals, np.float64))
    k = int(np.floor(len(v) * alpha))
    if len(v) - 2 * k < 1:
        return float(np.mean(v)) if len(v) else 0.0
    return float(v[k:len(v) - k].mean())


def boundary_thresholds(D_w, hole, alpha=0.10, radius=1):
    """T_O per hole component = trimmed mean of the disparity along its boundary.

    Returns (T, lab_near) with T indexed by component label (0 unused).
    """
    lab_near = label_of_nearest_hole(hole, radius)
    _, lab = hole_labels(hole)
    n = int(lab.max()) + 1
    boundary = (lab_near > 0) & ~hole
    if not boundary.any():
        return np.zeros(n, np.float64), lab_near
    labels = lab_near[boundary]
    vals = np.asarray(D_w, np.float64)[boundary]
    order = np.argsort(labels, kind="stable")
    labels, vals = labels[order], vals[order]
    T = np.zeros(n, np.float64)
    edges = np.searchsorted(labels, np.arange(n + 1))
    for k in range(1, n):
        a, b = edges[k], edges[k + 1]
        if b > a:
            T[k] = trimmed_mean(vals[a:b], alpha)
    return T, lab_near


def background_mask(D_w, T, lab_near):
    """M_BG = 1 where the disparity says background.  Disparity larger = nearer = FG
    (paper II-A), so background is  disparity <= T_O."""
    Tmap = T[np.clip(lab_near, 0, len(T) - 1)]
    return (D_w <= Tmap) & (lab_near > 0)


def extended_fg(P_ref, se_len=4, orientation="h"):
    """D_FG = dilate(dilate(D, H), H^T): the foreground disparity grown in both axes."""
    from . import cracks
    se_v, anc_v = cracks.line_se(se_len, "v")
    se_h, anc_h = cracks.line_se(se_len, "h")
    d1 = cv2.dilate(np.asarray(P_ref, np.float32), se_v, anchor=anc_v)
    return cv2.dilate(d1, se_h, anchor=anc_h)


# --------------------------------------------------------------------------- #
# masked median (exact, over valid pixels only)
# --------------------------------------------------------------------------- #
def masked_median(img, valid, ksize=9, strip=96):
    """Median over the valid pixels of a ksize x ksize window (0 where none valid).

    Exact: gathers the window, pushes invalid entries to +inf, sorts, and picks the
    (m//2)-th element where m = number of valid pixels in the window.  Processed in
    horizontal strips so the 81-view stack stays small.
    """
    a = np.asarray(img, np.float32)
    v = np.asarray(valid, bool)
    r = ksize // 2
    h, w = a.shape
    big = np.float32(np.inf)
    P = np.full((h + 2 * r, w + 2 * r), big, np.float32)
    P[r:r + h, r:r + w] = np.where(v, a, big)
    cnt = cv2.boxFilter(v.astype(np.float32), -1, (ksize, ksize), normalize=False,
                        borderType=cv2.BORDER_CONSTANT)
    out = np.zeros((h, w), np.float32)
    for y0 in range(0, h, strip):
        y1 = min(h, y0 + strip)
        n = y1 - y0
        views = [P[y0 + dy: y0 + dy + n, dx: dx + w]
                 for dy in range(ksize) for dx in range(ksize)]
        stack = np.stack(views, 0)
        stack.sort(axis=0)
        m = np.clip(cnt[y0:y1].astype(np.int32), 0, ksize * ksize)
        # standard median: average the two central order statistics when m is even
        k1 = np.clip((m - 1) // 2, 0, ksize * ksize - 1)
        k2 = np.clip(m // 2, 0, ksize * ksize - 1)
        s1 = np.take_along_axis(stack, k1[None], axis=0)[0]
        s2 = np.take_along_axis(stack, k2[None], axis=0)[0]
        med = 0.5 * (s1 + s2)
        out[y0:y1] = np.where(m > 0, med, 0.0)
    return out


# --------------------------------------------------------------------------- #
# backward map (which source pixel landed on a given target pixel)
# --------------------------------------------------------------------------- #
def backward_index(shape, dx, dy=None, z=None, splat="sub"):
    """int32 array with the flat index of the winning source pixel per target pixel."""
    h, w = shape
    if splat == "sub":
        contribs = _warp._splat_1d(np.asarray(dx, np.float32)) if dy is None else \
            _warp._splat_2d(np.asarray(dx, np.float32), np.asarray(dy, np.float32))
    else:
        contribs = _warp._splat_single(np.asarray(dx, np.float32), dy, splat)
    zz = np.ones((h, w), np.float32) if z is None else np.asarray(z, np.float32)
    best = np.full((h, w), -np.inf, np.float32)
    src = np.full((h, w), -1, np.int32)
    packed = []
    for tx, ty, wt, sx, sy in contribs:
        sel = (wt > 0) & (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
        if sel.any():
            packed.append((tx[sel], ty[sel], wt[sel], zz[sel],
                           (sy[sel].astype(np.int64) * w + sx[sel]).astype(np.int64)))
    for tx, ty, wt, zv, idx in packed:
        np.maximum.at(best, (ty, tx), zv)
    for tx, ty, wt, zv, idx in packed:
        win = zv >= best[ty, tx]
        src[ty[win], tx[win]] = idx[win].astype(np.int32)
    return src


def sample_map(field, ys, xs):
    """Bilinear-free nearest lookup with clipping."""
    h, w = field.shape
    return field[np.clip(np.rint(ys).astype(np.int32), 0, h - 1),
                 np.clip(np.rint(xs).astype(np.int32), 0, w - 1)]


# --------------------------------------------------------------------------- #
# full step
# --------------------------------------------------------------------------- #
def detect_and_fix(I_w, D_w, hole, disp, disp_fg, direction=None, oofa=None,
                   band_radius=2, alpha_trim=0.10, alpha_sim=11.0, fg_side="gt",
                   ksize=9, fix_mode="copy", gray_weights=(0.299, 0.587, 0.114)):
    """Detect ghosts in the synthetic view and relocate their content to `p_FG`.

    disp / disp_fg : (dx, dy) displacement fields in the SOURCE domain
                     (disp from the nominal depth, disp_fg from D_FG)
    fix_mode       : "copy"  write the ghost content at p_FG, keep p as it is
                     "move"  same, and additionally turn p into a hole
                     "bg"    same, and replace p by the local background median
    """
    I_w = np.asarray(I_w, np.float32)
    D_w = np.asarray(D_w, np.float32)
    hole = np.asarray(hole, bool)
    dx, dy = np.asarray(disp[0], np.float32), np.asarray(disp[1], np.float32)
    fx, fy = np.asarray(disp_fg[0], np.float32), np.asarray(disp_fg[1], np.float32)
    h, w = hole.shape
    if direction is None:
        direction = 1 if float(np.median(dx)) > 0 else -1
    if oofa is None:
        oofa = find_oofa(hole, direction)

    cand, G = candidate_band(hole, band_radius, oofa)
    T, lab_near = boundary_thresholds(D_w, G, alpha_trim)
    Tmap = T[np.clip(lab_near, 0, len(T) - 1)]
    has_T = lab_near > 0
    # remove foreground points from the candidates (larger disparity = nearer = FG)
    if fg_side == "gt":
        fg_pts = cand & has_T & (D_w > Tmap)
    else:
        fg_pts = cand & has_T & (D_w < Tmap)
    cand = cand & ~fg_pts

    gray = (I_w[..., 0] * gray_weights[0] + I_w[..., 1] * gray_weights[1]
            + I_w[..., 2] * gray_weights[2]).astype(np.float32)
    valid = ~(hole | cand)
    med_field = masked_median(gray, valid, ksize)

    # backward map: which source pixel produced each target pixel
    src_idx = backward_index((h, w), dx, dy, z=D_w if D_w.max() > 0 else None)
    ys, xs = np.nonzero(cand)
    flat = src_idx[ys, xs]
    ok = flat >= 0
    ys, xs, flat = ys[ok], xs[ok], flat[ok]
    sy, sx = np.divmod(flat, w)
    # the same content warped with the EXTENDED foreground disparity
    px = sx.astype(np.float32) + fx[sy, sx]
    py = sy.astype(np.float32) + fy[sy, sx]

    mu_bg = med_field[ys, xs]
    mu_fg = sample_map(med_field, py, px)
    ip = gray[ys, xs]
    d_bg = np.abs(ip - mu_bg)
    d_fg = np.abs(ip - mu_fg)
    is_ghost = (d_bg >= d_fg) | (d_bg > alpha_sim)

    ghost = np.zeros_like(cand)
    ghost[ys[is_ghost], xs[is_ghost]] = True

    # ---- relocate the ghost content to p_FG ----
    I_out = I_w.copy()
    D_out = D_w.copy()
    gys_all, gxs_all = ys[is_ghost], xs[is_ghost]
    gpx, gpy = px[is_ghost], py[is_ghost]
    tx = np.clip(np.rint(gpx).astype(np.int64), 0, w - 1)
    ty = np.clip(np.rint(gpy).astype(np.int64), 0, h - 1)
    inside = (gpx >= 0) & (gpx < w) & (gpy >= 0) & (gpy < h)
    wys, wxs, wtx, wty = gys_all[inside], gxs_all[inside], tx[inside], ty[inside]
    # nearest content wins collisions: write in ascending depth order
    order = np.argsort(D_w[wys, wxs], kind="stable")
    dest = (wty[order] * w + wtx[order])
    I_out.reshape(-1, 3)[dest] = I_w[wys[order], wxs[order]]
    D_out.reshape(-1)[dest] = D_w[wys[order], wxs[order]]

    hole_out = hole.copy()
    if fix_mode == "move":
        hole_out |= ghost
        D_out[ghost] = -1.0
    elif fix_mode == "bg":
        mb = med_field[wys, wxs]
        for c in range(3):
            I_out[..., c][wys, wxs] = mb
        D_out[wys, wxs] = Tmap[wys, wxs]

    return dict(ghost=ghost, candidates=cand, G=G, oofa=oofa, fg_points=fg_pts,
                T=T, lab_near=lab_near, med_field=med_field,
                cand_yx=(ys, xs), cand_mu_bg=mu_bg, cand_mu_fg=mu_fg,
                d_bg=d_bg, d_fg=d_fg, classified=is_ghost,
                ghost_yx=(gys_all, gxs_all), write_yx=(wys, wxs),
                p_fg_xy=(gpx, gpy), write_xy=(wtx, wty),
                I_fixed=I_out, D_fixed=D_out, hole_out=hole_out,
                direction=direction, disp=disp, disp_fg=disp_fg)

