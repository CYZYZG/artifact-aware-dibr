"""Steps 5-7 - exemplar-based filling of disocclusions and OOFAs (paper II-C).

5) local FG/BG extractor + priority
       disocclusion : P(p) = B(p) * E(p)
       OOFA         : P(p) = C(p) * D(p)
   B = fraction of background pixels inside the valid part of Psi_p,
   E = depth term from [9] (favours background; the formula is not given in the paper),
   C, D = Criminisi's confidence and data terms.
6) patch matching in the REFERENCE image I, inside an N x N window centred on the
   backward warp p' of the patch centre; disocclusions search background only
   (M_BG eroded by a diamond of size 7); adaptive patch size 9x9 -> 3x3 (step -2)
   while min s > beta.
7) iterate until the hole is empty, updating C and B as content is filled in.
"""
import cv2
import numpy as np

DIAMOND7 = (((np.abs(np.mgrid[-7:8, -7:8][0]) + np.abs(np.mgrid[-7:8, -7:8][1])) <= 7)
            .astype(np.uint8))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def gray(img):
    a = np.asarray(img, np.float32)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)


def square_offsets(k):
    r = k // 2
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]


def gather(img, ys, xs, offs, valid, shape):
    """Gather K-neighbourhoods.  Returns (values (n,K,C), inbounds_and_valid (n,K))."""
    H, W = shape
    n, K = len(ys), len(offs)
    C = img.shape[2] if img.ndim == 3 else 1
    vals = np.zeros((n, K, C), np.float32)
    ok = np.zeros((n, K), bool)
    for i, (dy, dx) in enumerate(offs):
        yy, xx = ys + dy, xs + dx
        inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        yy2, xx2 = np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)
        vals[:, i] = img[yy2, xx2].reshape(n, C)
        ok[:, i] = inb & valid[yy2, xx2]
    return vals, ok


def masked_ssd(region, templ, mask):
    """SSD between `templ` and every position of `region`, counting only mask==1 px.

    region : (R, R, C) float32      templ : (k, k, C) float32     mask : (k, k) float32
    Exact and fully vectorised (sum of squared differences expanded into correlations).
    """
    mt = templ * mask[..., None]
    shape = (region.shape[0] - templ.shape[0] + 1,
             region.shape[1] - templ.shape[1] + 1)
    corr = np.zeros(shape, np.float32)
    for c in range(region.shape[2]):
        corr += cv2.matchTemplate(region[..., c], mt[..., c], cv2.TM_CCORR)
    r2 = (region ** 2).sum(-1).astype(np.float32)
    sum_r2 = cv2.matchTemplate(r2, mask, cv2.TM_CCORR)
    const = float(((templ ** 2).sum(-1))[mask > 0].sum())
    return const + sum_r2 - 2.0 * corr


# --------------------------------------------------------------------------- #
# priority terms
# --------------------------------------------------------------------------- #
def depth_term(D_w, valid):
    """E(p): larger for background.  Normalised complement of the disparity, so E=1 at
    the farthest valid pixel and E=0 at the nearest (the paper cites [9] without giving
    the formula; validated by monotonicity and by the ablation in step5_inpaint.py)."""
    v = D_w[valid]
    if v.size == 0:
        return np.ones_like(D_w, np.float32), (0.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    E = np.clip((hi - D_w) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    return E, (lo, hi)


def criminisi_terms(I_w, valid, rem_local, ys, xs, ys_l, xs_l, offs, conf, grads):
    """Confidence C(p) and data term D(p) of Criminisi [17] for the frontier pixels."""
    H, W = valid.shape
    cvals, cok = gather(conf[..., None], ys, xs, offs, valid, (H, W))
    C = (cvals[..., 0] * cok).sum(1) / max(1, len(offs))
    gx, gy = grads
    m = rem_local.astype(np.float32)
    nx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)[ys_l, xs_l]
    ny = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)[ys_l, xs_l]
    nrm = np.hypot(nx, ny) + 1e-8
    nx, ny = nx / nrm, ny / nrm
    iso_x, iso_y = -gy[ys, xs], gx[ys, xs]        # isophote = gradient rotated 90 deg
    D = np.abs(iso_x * nx + iso_y * ny) / 255.0
    return C.astype(np.float32), D.astype(np.float32)


def image_gradients(I_w):
    g = gray(I_w)
    return (cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def background_term(D_w, valid, ys, xs, offs, T):
    """B(p) = (# background pixels in the valid part of Psi_p) / (# valid pixels).
    Background = disparity <= T (paper II-A convention: larger disparity = nearer)."""
    H, W = valid.shape
    vals, ok = gather(D_w[..., None], ys, xs, offs, valid, (H, W))
    num = ((vals[..., 0] <= T) & ok).sum(1)
    den = np.maximum(ok.sum(1), 1)
    return (num / den).astype(np.float32)


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
SEARCH_DIAG = {"nv0": 0, "region": 0, "nonfinite": 0, "calls": 0, "ok_empty": 0}


def search_patch(I_w, valid, ref_color, ref_depth, py, px, wy, wx, T, n_window=69,
                 sizes=(9, 7, 5, 3), beta=35.0, bg_only=True, beta_mode="mean",
                 require_full_valid=False, D_w=None, bg_template=False,
                 require_full_bg=False):
    """Best source patch in the reference image.  Returns (qy, qx, k, cost, n_cand, used_bg).

    (py, px)  centre of the patch to be filled, in the SYNTHETIC view (template source)
    (wy, wx)  its backward warp p', in the REFERENCE view  (search-window centre)

    For disocclusions the candidate centres are restricted to `M_BG` eroded by a diamond
    of size 7 (paper II-C-2).  The erosion is computed on a crop with a 7 px margin so it
    is exact over the whole search window.

    `beta_mode` decides the scale of the adaptive-size test:
      "mean" (default) cost = SSD / (3 * number of valid pixels), i.e. a mean squared
             per-channel error in gray-level units.  The paper's beta = 35 is used on this
             scale, which is the only reading under which the threshold is reachable on
             8-bit data (with the raw SSD sum every patch bottoms out at 3x3).
      "sum"  the literal formulation of the paper; reported for reference.
    """
    H, W = valid.shape
    R = n_window // 2
    y0 = int(np.clip(wy - R, 0, max(0, H - n_window)))
    x0 = int(np.clip(wx - R, 0, max(0, W - n_window)))
    y1, x1 = min(H, y0 + n_window), min(W, x0 + n_window)
    region = ref_color[y0:y1, x0:x1].astype(np.float32)
    m = 7
    my0, mx0 = max(0, y0 - m), max(0, x0 - m)
    my1, mx1 = min(H, y1 + m), min(W, x1 + m)
    conn = (ref_depth[my0:my1, mx0:mx1] <= T).astype(np.uint8)
    ero = cv2.erode(conn, DIAMOND7).astype(bool)
    ok_crop = ero[y0 - my0: y1 - my0, x0 - mx0: x1 - mx0]
    # a wider margin (and a float copy) is needed when the WHOLE source patch has to be
    # background, so that no foreground pixel of the source can be copied into the hole
    mm = max(m, max(sizes)) if require_full_bg else m
    fy0, fx0 = max(0, y0 - mm), max(0, x0 - mm)
    fy1, fx1 = min(H, y1 + mm), min(W, x1 + mm)
    mbg_f = (ref_depth[fy0:fy1, fx0:fx1] <= T).astype(np.float32) if require_full_bg else None
    used_bg = False
    SEARCH_DIAG["calls"] += 1
    for k in sizes:
        r = k // 2
        ys = np.arange(py - r, py + r + 1)
        xs = np.arange(px - r, px + r + 1)
        inb = (ys[:, None] >= 0) & (ys[:, None] < H) & (xs[None, :] >= 0) & (xs[None, :] < W)
        yc, xc = np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)
        templ = I_w[yc][:, xc].astype(np.float32)
        tmask = (inb & valid[yc][:, xc]).astype(np.float32)
        if bg_only and bg_template and D_w is not None:
            # the template contains the (dark) foreground edge of the silhouette; if it is
            # kept in the SSD the matcher is driven to reproduce that edge inside the hole,
            # which shows up as a dark rim.  Mask those pixels out.
            bt = inb & (np.asarray(D_w)[yc][:, xc] <= T)
            keep = tmask * bt
            if keep.sum() >= max(9.0, 0.25 * tmask.sum()):
                tmask = keep
        nv = float(tmask.sum())
        if nv == 0:
            SEARCH_DIAG["nv0"] += 1
            continue
        if region.shape[0] < k or region.shape[1] < k:
            SEARCH_DIAG["region"] += 1
            continue
        ssd = masked_ssd(region, templ, tmask)
        cand = np.ones_like(ssd, bool)
        if require_full_valid:
            # the whole source patch must be real content (otherwise the trivial
            # "match the template against itself" solution wins with cost 0)
            full = cv2.boxFilter(valid[my0:my1, mx0:mx1].astype(np.float32), -1,
                                 (k, k), normalize=False,
                                 borderType=cv2.BORDER_CONSTANT)
            fc = (full >= k * k - 1e-6)[y0 - my0: y1 - my0, x0 - mx0: x1 - mx0]
            cand &= fc[r: r + ssd.shape[0], r: r + ssd.shape[1]]
        if bg_only:
            ok = ok_crop[r: r + ssd.shape[0], r: r + ssd.shape[1]]
            if require_full_bg and mbg_f is not None:
                fb = cv2.boxFilter(mbg_f, -1, (k, k), normalize=False,
                                   borderType=cv2.BORDER_CONSTANT)
                fbc = (fb >= k * k - 1e-6)[y0 - fy0: y1 - fy0, x0 - fx0: x1 - fx0]
                ok = ok & fbc[r: r + ssd.shape[0], r: r + ssd.shape[1]]
            if ok.any():
                cand = ok
                used_bg = True
            else:
                SEARCH_DIAG["ok_empty"] += 1
        ssd_m = np.where(cand, ssd, np.inf)
        flat = int(np.argmin(ssd_m))
        raw = float(ssd_m.ravel()[flat])
        if not np.isfinite(raw):
            SEARCH_DIAG["nonfinite"] += 1
            continue
        cost = raw / (3.0 * nv) if beta_mode == "mean" else raw
        oy, ox = np.unravel_index(flat, ssd.shape)
        qy, qx = y0 + r + oy, x0 + r + ox
        if cost <= beta or k == sizes[-1]:
            return qy, qx, k, cost, int(cand.sum()), used_bg
    return None


# --------------------------------------------------------------------------- #
# fill one component
# --------------------------------------------------------------------------- #
def fill_component(I_w, D_w, valid, lab, k, bbox, src_of, ref_color, ref_depth, T,
                   comp_type, conf, E, grads, params, ablate="none"):
    """Fill one hole component in place (I_w, D_w, valid, src_of, conf are modified).

    `valid` is the mutable "pixel has content" mask: newly filled pixels are marked valid
    immediately, which is what lets the frontier advance (the paper's iteration fills the
    hole from its boundary inwards).
    """
    H, W = valid.shape
    y0, y1, x0, x1 = bbox
    # pad by 1 so that the frontier (dilate(rem) minus rem) is representable even for a
    # single-pixel component, whose valid neighbours lie just outside its own bbox
    y0, y1 = max(0, y0 - 1), min(H, y1 + 1)
    x0, x1 = max(0, x0 - 1), min(W, x1 + 1)
    rem = (lab[y0:y1, x0:x1] == k)
    valid_l = valid[y0:y1, x0:x1]
    n_window, sizes, beta = params["n_window"], params["sizes"], params["beta"]
    max_iter = params["max_iter"]
    offs9 = square_offsets(9)
    bg_only = (comp_type != "oofa")
    log = dict(iterations=0, sizes={}, costs=[], failed=0, bg_search=0)
    ker3 = np.ones((3, 3), np.uint8)
    while rem.any() and log["iterations"] < max_iter:
        d = cv2.dilate(rem.astype(np.uint8), ker3).astype(bool) & ~rem & valid_l
        ys_l, xs_l = np.nonzero(d)
        if ys_l.size == 0:
            log["failed"] += 1
            log["fail_reason"] = "no frontier"
            break
        ys, xs = ys_l + y0, xs_l + x0
        C, Dt = criminisi_terms(I_w, valid, rem, ys, xs, ys_l, xs_l, offs9, conf, grads)
        if ablate == "nocd":
            P = C * Dt
        elif comp_type == "oofa":
            P = C * Dt
        else:
            B = background_term(D_w, valid, ys, xs, offs9, T)
            if ablate == "nob":          # drop the background term
                P = E[ys, xs]
            elif ablate == "nodepth":    # drop the depth term
                P = B
            elif ablate == "noconf":     # add Criminisi's terms back
                P = B * E[ys, xs] * Dt
            else:
                P = B * E[ys, xs]
        i = int(np.argmax(P))
        py, px = int(ys[i]), int(xs[i])
        src = int(src_of[py, px])
        qy0, qx0 = divmod(src, W) if src >= 0 else (py, px)
        if ablate == "search_iw":
            # ablation of paper contribution (ii): search inside the SYNTHETIC view
            # (classic Criminisi) instead of the artefact-free reference image.
            # The candidate patch must be completely real content, otherwise the trivial
            # self-match has zero cost and the hole would be copied onto itself.
            src_color, src_depth, wy, wx = I_w, D_w, py, px
            full_valid = True
        else:
            src_color, src_depth, wy, wx = ref_color, ref_depth, qy0, qx0
            full_valid = False
        if ablate == "nobgsearch":
            bg_only = False
        found = search_patch(I_w, valid, src_color, src_depth, py, px, wy, wx, T,
                             n_window, sizes, beta, bg_only,
                             params.get("beta_mode", "mean"),
                             require_full_valid=full_valid, D_w=D_w,
                             bg_template=params.get("bg_template", False),
                             require_full_bg=params.get("require_full_bg", False))
        if found is None:
            log["failed"] += 1
            log["fail_reason"] = "no patch"
            break
        qy, qx, ksz, cost, n_cand, used_bg = found
        log["sizes"][ksz] = log["sizes"].get(ksz, 0) + 1
        log["costs"].append(cost)
        if used_bg:
            log["bg_search"] += 1
        for dy, dx in square_offsets(ksz):
            ty, tx = py + dy, px + dx
            ly, lx = ty - y0, tx - x0
            if not (0 <= ly < rem.shape[0] and 0 <= lx < rem.shape[1]):
                continue                     # patch reaches outside this component
            if not rem[ly, lx]:
                continue
            sy = int(np.clip(qy + dy, 0, H - 1))
            sx = int(np.clip(qx + dx, 0, W - 1))
            I_w[ty, tx] = src_color[sy, sx]
            D_w[ty, tx] = src_depth[sy, sx]
            src_of[ty, tx] = sy * W + sx
            conf[ty, tx] = conf[py, px]
            rem[ly, lx] = False
            valid[ty, tx] = True
        log["iterations"] += 1
    log["rem"] = rem
    log["rem_left"] = int(rem.sum())
    return log


def bboxes(lab, n):
    """Per-component bounding boxes."""
    import scipy.ndimage as ndi
    sl = ndi.find_objects(lab)
    out = {}
    for k in range(1, n):
        s = sl[k - 1]
        if s is None:
            continue
        out[k] = (max(0, s[0].start), min(lab.shape[0], s[0].stop),
                  max(0, s[1].start), min(lab.shape[1], s[1].stop))
    return out


def hole_threshold(D_w, lab, k, alpha=0.10):
    """T_O = trimmed mean (alpha) of the valid disparity along hole component k's boundary."""
    import scipy.ndimage as ndi

    from .ghosts import trimmed_mean
    m = lab == k
    b = ndi.binary_dilation(m, np.ones((3, 3), bool)) & ~m
    v = np.asarray(D_w)[b]
    v = v[v >= 0]
    return trimmed_mean(v, alpha) if v.size else 0.0


def repair_src_of(src_of, valid):
    """Fill the -1 entries of a backward map by nearest-neighbour propagation.

    Pixels that were not produced by the warp itself (e.g. crack-filled by HHF) have no
    backward mapping; their neighbours do, so copying the nearest valid entry plus the
    pixel offset is a good approximation of where that content came from.
    """
    import scipy.ndimage as ndi
    bad = src_of < 0
    if not bad.any() or not (~bad).any():
        return src_of
    h, w = src_of.shape
    _, idx = ndi.distance_transform_edt(bad, return_indices=True)
    near = src_of[idx[0], idx[1]]
    off = (np.arange(h)[:, None] - idx[0]) * w + (np.arange(w)[None, :] - idx[1])
    return np.where(bad, near + off, src_of).astype(np.int32)


def fill_all(I_w, D_w, hole, ref_color, ref_depth, disp, src_of,
             direction=-1, params=None, ablate="none", oofa_frac=0.5, progress=None):
    """Fill every hole component of a warped view.  This is the Step 4-7 entry point.

    Parameters
    ----------
    I_w, D_w : target view; D_w is a nearer-is-larger scalar map with holes at -1
    hole     : (H, W) bool, True where the target view has no content
    ref_color, ref_depth : the reference image and its depth (the patch source)
    disp     : (dx, dy) displacement field in the SOURCE domain that produced the warp
    src_of   : backward map (flat index into the reference) or None to derive it
    params   : dict(n_window, sizes, beta, beta_mode, max_iter)

    Returns a dict with the filled arrays, the classification, and per-component logs.
    """
    from . import holes as _holes
    defaults = dict(n_window=69, sizes=(9, 7, 5, 3), beta=150.0, beta_mode="mean",
                    max_iter=400000)
    defaults.update(params or {})
    params = defaults
    H, W = hole.shape
    valid = ~np.asarray(hole, bool)
    oofa, disocc, lab, comp_type, info = _holes.classify(hole, direction, oofa_frac)
    n_comp = int(lab.max()) + 1
    areas = np.bincount(lab.ravel(), minlength=n_comp)
    order = [int(k) for k in np.argsort(-areas[1:]) + 1 if areas[k] > 0]

    conf = valid.astype(np.float32)
    E, E_range = depth_term(D_w, valid)
    grads = image_gradients(I_w)
    if src_of is None:
        from .ghosts import backward_index
        src_of = backward_index((H, W), np.asarray(disp[0], np.float32),
                                None if disp[1] is None else np.asarray(disp[1], np.float32),
                                z=np.where(valid, np.maximum(D_w, 0), 0.0),
                                splat=params.get("splat", "sub"))
    src_of = repair_src_of(np.asarray(src_of, np.int32), valid)

    bb = bboxes(lab, n_comp)
    logs = {}
    for i, k in enumerate(order):
        T = hole_threshold(D_w, lab, k)
        logs[k] = fill_component(I_w, D_w, valid, lab, k, bb[k], src_of, ref_color,
                                 ref_depth, T, comp_type[k], conf, E, grads, params,
                                 ablate=ablate)
        if progress is not None and (i + 1) % 50 == 0:
            progress(i + 1, len(order), logs)
    remaining = ~valid
    sizes_used = {}
    for l in logs.values():
        for kk, v in l["sizes"].items():
            sizes_used[kk] = sizes_used.get(kk, 0) + v
    costs = np.array([c for l in logs.values() for c in l["costs"]], np.float64)
    stats = dict(
        holes_before=int(hole.sum()), holes_after=int(remaining.sum()),
        filled_px=int(hole.sum()) - int(remaining.sum()),
        oofa_px=info["oofa_px"], disocc_px=info["disocc_px"],
        oofa_components=info["oofa_components"], disocc_components=info["disocc_components"],
        components=len(order), iterations=int(sum(l["iterations"] for l in logs.values())),
        failed_components=int(sum(1 for l in logs.values() if l["failed"])),
        E_lo=E_range[0], E_hi=E_range[1],
        patch_sizes=sizes_used,
        cost_median=float(np.median(costs)) if costs.size else float("nan"),
        cost_frac_below_beta=float((costs <= params["beta"]).mean()) if costs.size else 0.0,
    )
    return dict(I_filled=I_w, D_filled=D_w, remaining=remaining, oofa=oofa, disocc=disocc,
                lab=lab, comp_type=comp_type, logs=logs, stats=stats, valid=valid,
                src_of=src_of)
