"""Why does the fill stop early?  Inspect the per-component logs."""
import sys

import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补")
from dibr import calib, ghosts, holes, inpaint, io_utils  # noqa: E402

ROOT = io_utils.DATASET_ROOT_DEFAULT
S3 = r"D:\项目\空洞填补\output\step3_ghosts\cam6_to_cam7_f000\band2_gt_a11_copy"
I_w = np.load(S3 + r"\I_fixed.npy").astype(np.float32)
D_w = np.load(S3 + r"\D_fixed.npy").astype(np.float32)
hole = np.load(S3 + r"\hole_out.npy").astype(bool)
h, w = hole.shape
cams = calib.load_calib(ROOT)
ref = io_utils.load_view(ROOT, 6, "f000")
ref_color = ref["color"].astype(np.float32)
ref_depth = ref["depth"].astype(np.float32)
dx, dy, _, _, _ = calib.displacement_field(cams, 6, 7, ref_depth)
direction = 1 if float(np.median(dx)) > 0 else -1
oofa, disocc, lab, comp_type, info = holes.classify(hole, direction)
n_comp = int(lab.max()) + 1
areas = np.bincount(lab.ravel(), minlength=n_comp)
order = [k for k in np.argsort(-areas[1:]) + 1 if areas[k] > 0]
bb = inpaint.bboxes(lab, n_comp)
valid = ~hole
conf = valid.astype(np.float32)
E, _ = inpaint.depth_term(D_w, valid)
src_of = ghosts.backward_index((h, w), dx, dy, z=np.where(valid, D_w, 0.0))
bad = src_of < 0
import scipy.ndimage as ndi
if bad.any() and (~bad).any():
    _, idx = ndi.distance_transform_edt(bad, return_indices=True)
    near = src_of[idx[0], idx[1]]
    off = (np.arange(h)[:, None] - idx[0]) * w + (np.arange(w)[None, :] - idx[1])
    src_of = np.where(bad, near + off, src_of).astype(np.int32)

params = dict(n_window=69, sizes=(9, 7, 5, 3), beta=35.0, max_iter=400000)
for k in order[:5]:
    m = lab == k
    v = D_w[ndi.binary_dilation(m, np.ones((3, 3), bool)) & ~m]
    v = v[v >= 0]
    T = float(np.mean(np.sort(v)[max(0, len(v) // 10):len(v) - len(v) // 10])) if v.size else 0
    y0, y1, x0, x1 = bb[k]
    # how many of this component's pixels have a valid 4-neighbour?
    sub = m[y0:y1, x0:x1]
    val = ~hole[y0:y1, x0:x1]
    fr = ndi.binary_dilation(sub, np.ones((3, 3), bool)) & ~sub & val
    log = inpaint.fill_component(I_w, D_w, hole, lab, k, bb[k], src_of, ref_color,
                                 ref_depth, T, comp_type[k], conf, E, params)
    print(f"comp {k}: area {int(m.sum())} bbox {y1-y0}x{x1-x0} type={comp_type[k]} "
          f"frontier_px {int(fr.sum())} T={T:.1f} -> iters {log['iterations']} "
          f"rem_left {log['rem_left']} failed {log['failed']} "
          f"reason {log.get('fail_reason')} sizes {log['sizes']}")
print("\ntotal hole px", int(hole.sum()), "components", n_comp)
print("components with no valid 4/8-neighbour:")
cnt = 0
for k in order:
    y0, y1, x0, x1 = bb[k]
    sub = (lab[y0:y1, x0:x1] == k)
    val = ~hole[y0:y1, x0:x1]
    if not (ndi.binary_dilation(sub, np.ones((3, 3), bool)) & ~sub & val).any():
        cnt += 1
print("  ", cnt, "of", len(order))
