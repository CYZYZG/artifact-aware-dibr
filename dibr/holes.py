"""Step 4 - split the remaining holes into OOFAs and disocclusions (paper II-C).

OOFA: out-of-field areas at the image edge the projection vacated; found by scanning each
row from that edge until the first non-empty pixel (paper II-B).
Everything else is a disocclusion.
"""
import cv2
import numpy as np

from . import ghosts


def classify(hole, direction, oofa_frac=0.5):
    """Return (oofa, disocc, lab, comp_type, info).

    comp_type[k] is "oofa" or "disocc" for hole component k; a component belongs to the
    OOFA class when at least `oofa_frac` of its pixels fall in the scanned edge strip.
    """
    oofa = ghosts.find_oofa(hole, direction)
    n, lab = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    comp_type = {0: "disocc"}
    fracs = {}
    for k in range(1, n):
        m = lab == k
        a = int(m.sum())
        o = int((m & oofa).sum())
        fracs[k] = o / max(1, a)
        comp_type[k] = "oofa" if fracs[k] >= oofa_frac else "disocc"
    disocc = hole & ~oofa
    touched = int((disocc & _edge_ring(hole.shape)).sum())
    info = dict(components=int(n - 1),
                oofa_px=int(oofa.sum()), disocc_px=int(disocc.sum()),
                oofa_components=int(sum(1 for k in comp_type if comp_type[k] == "oofa")),
                disocc_components=int(sum(1 for k in comp_type if comp_type[k] == "disocc")),
                mixed_components=int(sum(1 for k, f in fracs.items() if 0 < f < 1)),
                disocc_px_touching_image_edge=touched,
                direction=int(direction), fracs=fracs)
    return oofa, disocc, lab, comp_type, info


def _edge_ring(shape, width=1):
    m = np.zeros(shape, bool)
    m[:width], m[-width:] = True, True
    m[:, :width], m[:, -width:] = True, True
    return m
