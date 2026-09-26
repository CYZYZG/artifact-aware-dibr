"""Edge-domain blending: Poisson seamless cloning of the filled region into the background.

    min_f  sum_{p in Omega} sum_{q in N4(p)} ( (f(p)-f(q)) - (g(p)-g(q)) )^2

keeps the gradients of the filled content g and lets the level be set by the untouched
pixels on the boundary of Omega (Dirichlet).

MEASURED RESULT (this project, MSR Ballet cam6 with scale -44.8, see 复现方案.md 9.9):
with the guidance g taken from the filled image itself and the Dirichlet values taken from
the very same image, the exact solution is f = g on Omega: the blend is an IDENTITY, so it
cannot remove the seam (numerically verified: identical seam 10.22 / GT 20.39 / SSIM 0.7130
with and without).  Seamless cloning only re-bases the level when the content inside Omega
comes from a DIFFERENT source; here the seam is an edge/gradient discontinuity (the fill
reproduces a silhouette edge it copied), not a level step, so there is nothing to re-base.

The module is kept because it is the honest, reproducible record of that negative result and
because `anchor` documents the trap: anchoring on every boundary pixel also drags the band
towards the dark foreground silhouette that borders the hole on the other side.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["poisson_blend"]


def _neighbour_rows(H, W, hole, idx):
    """Return (rows, cols, counts, dirs) for the 4-neighbourhood of every hole pixel."""
    ys, xs = np.nonzero(hole)
    rows = idx[ys, xs]
    nbr = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        y2, x2 = ys + dy, xs + dx
        inb = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        yy = np.clip(y2, 0, H - 1)
        xx = np.clip(x2, 0, W - 1)
        nbr.append((inb, yy, xx))
    return ys, xs, rows, nbr


def poisson_blend(I_filled, hole, anchor=None, fallback_iters=0):
    """Blend the filled pixels into the untouched background (per channel).

    I_filled : HxWxC float image (already filled)
    hole     : HxW bool, the pixels that were filled (the region Omega)
    anchor   : HxW bool, which untouched pixels may act as the Dirichlet condition.  The
               default (None) uses every untouched neighbour, which drags the filled band
               towards the dark foreground silhouette that borders the other side of the
               hole; passing only the BACKGROUND side is the meaningful choice.
    Returns  : HxWxC float, with Omega re-based; pixels outside Omega are unchanged.
    """
    img = np.asarray(I_filled, np.float32)
    hole = np.asarray(hole, bool)
    if img.ndim == 2:
        img = img[:, :, None]
        squeeze = True
    else:
        squeeze = False
    H, W, C = img.shape
    n = int(hole.sum())
    if n == 0:
        return img[..., 0] if squeeze else img
    idx = np.full((H, W), -1, np.int64)
    idx[hole] = np.arange(n)

    rows_all, cols_all, vals_all, b_extra = [], [], [], np.zeros((n, C), np.float32)
    ys, xs, rows, nbr = _neighbour_rows(H, W, hole, idx)
    g = img
    diag = np.zeros(n, np.float32)
    for inb, yy, xx in nbr:
        is_hole = inb & hole[yy, xx]
        dirich = inb & ~hole[yy, xx]
        if anchor is not None:
            dirich = dirich & anchor[yy, xx]
        # guidance gradient, summed over the neighbours that are constrained (in Omega or
        # anchored); a neighbour that is neither is free (homogeneous Neumann) and drops out
        use = is_hole | dirich
        b_extra += np.where(use[..., None], g[ys, xs] - g[yy, xx], 0.0)
        rows_all.append(rows[is_hole])
        cols_all.append(idx[yy[is_hole], xx[is_hole]])
        vals_all.append(-np.ones(int(is_hole.sum()), np.float32))
        diag += use
        if dirich.any():
            b_extra[dirich] += g[yy[dirich], xx[dirich]]
    rows_all.append(rows)
    cols_all.append(rows)
    vals_all.append(diag)
    A = sp.coo_matrix((np.concatenate(vals_all),
                       (np.concatenate(rows_all), np.concatenate(cols_all))),
                      shape=(n, n)).tocsr()
    A = A + sp.identity(n, format="csr") * 1e-6      # keep it non-singular
    out = img.copy()
    solve = spla.splu(A.tocsc()) if n < 200000 else None
    for c in range(C):
        b = b_extra[:, c].astype(np.float64)
        if solve is not None:
            f = solve.solve(b)
        else:                                        # pragma: no cover - large fallback
            f = spla.cg(A, b, rtol=1e-6, maxiter=1000)[0]
        out[ys, xs, c] = f
    return out[..., 0] if squeeze else out
