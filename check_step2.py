"""Automated checks for Step 2 (crack detection + HHF filling).  Run: python check_step2.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dibr import cracks, io_utils, viz  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "cam6_to_cam7_f000"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    src = os.path.join(HERE, "output", "step1_warp", TAG)
    D_w = np.load(os.path.join(src, "warped_P.npy")).astype(np.float32)
    I_w = np.load(os.path.join(src, "warped_color.npy")).astype(np.float32)
    hole = D_w < 0

    # ---- 1. detector semantics on synthetic cases ------------------------------
    # a 2 px wide vertical strip of uniform disparity -> the SE must be horizontal
    # (perpendicular to the slit) to bridge it; SE anchor = MATLAB convention.
    base = np.full((40, 40), 100.0, np.float32)
    crack_strip = np.zeros((40, 40), bool)
    crack_strip[10:30, 20:22] = True
    D = base.copy()
    D[crack_strip] = 90.0                      # 10 units below the neighbourhood
    det, _, _ = cracks.detect_cracks(D, lam=5, se_len=4, orientation="h")
    check("detector finds a 2 px wide uniform-region dip>=lambda",
          bool((det == crack_strip).all()),
          f"detected {int(det.sum())} px, expected {int(crack_strip.sum())}")

    ramp = np.tile(np.arange(40, dtype=np.float32)[:, None], (1, 40))
    det1, _, _ = cracks.detect_cracks(ramp * 1.0, lam=5, se_len=4, orientation="v")
    det2, _, _ = cracks.detect_cracks(ramp * 3.0, lam=5, se_len=4, orientation="v")
    check("smooth ramp with 2*slope < lambda is NOT flagged", det1.sum() == 0,
          f"slope 1/row -> {int(det1.sum())} px")
    check("steeper ramp (2*slope >= lambda) IS flagged (paper's own caveat)",
          det2.sum() > 0, f"slope 3/row -> {int(det2.sum())} px")

    # the SE must be PERPENDICULAR to the slit to bridge it: a long vertical slit is
    # bridged by the horizontal SE, while a vertical SE only reaches its two caps.
    # This is why the paper's crack model (vertical slits from a horizontal projection)
    # needs H^T, even though the paper's sentence names H first.
    slit = np.zeros((40, 40), bool)
    slit[8:32, 20] = True
    Ds = np.full((40, 40), 100.0, np.float32)
    Ds[slit] = 80.0
    det_h, _, _ = cracks.detect_cracks(Ds, lam=5, se_len=4, orientation="h")
    det_v, _, _ = cracks.detect_cracks(Ds, lam=5, se_len=4, orientation="v")
    check("horizontal SE bridges a 1 px vertical slit completely",
          int((det_h & slit).sum()) == int(slit.sum()),
          f"h detects {int((det_h & slit).sum())}/{int(slit.sum())} slit px")
    check("vertical SE only reaches the slit caps (documented property)",
          0 < int((det_v & slit).sum()) < int(slit.sum()),
          f"v detects {int((det_v & slit).sum())}/{int(slit.sum())} slit px")

    # ---- 2. detection bookkeeping on the real D_w ------------------------------
    crack_raw, D_hat, diff = cracks.detect_cracks(D_w, lam=5, se_len=4, orientation="v")
    check("dilation is a max-filter: D_hat >= D_w everywhere", bool((D_hat >= D_w - 1e-6).all()))
    check("crack mask == (D_hat - D_w >= lambda)",
          bool((crack_raw == (diff >= 5)).all()))
    check("the empty-crack sentinel is -1", bool((D_w[hole] == -1).all()),
          f"hole values unique: {np.unique(D_w[hole])[:3]}")

    # ---- 3. fill bookkeeping ---------------------------------------------------
    res = cracks.fill_cracks(I_w, D_w, lam=5, se_len=4, orientation="v")
    crack = res["crack"]
    check("shape filter only removes detections (none == raw here)",
          bool((crack == crack_raw).all()))
    check("D_filled == D_hat at cracks, >= 0 there",
          bool((res["D_filled"][crack] == D_hat[crack]).all()) and
          bool((res["D_filled"][crack] >= 0).all()),
          f"min at cracks {res['D_filled'][crack].min():.1f}")
    check("D_filled unchanged outside the cracks",
          bool((res["D_filled"][~crack] == D_w[~crack]).all()))
    check("remaining holes == holes minus detected cracks",
          bool((res["remaining_holes"] == (hole & ~crack)).all()),
          f"{int(hole.sum())} - {int((crack & hole).sum())} = "
          f"{int(res['remaining_holes'].sum())}")
    check("crack filling actually repaired empty holes",
          int(hole.sum()) - int(res["remaining_holes"].sum()) > 0,
          f"repaired {int(hole.sum()) - int(res['remaining_holes'].sum())} px")

    # ---- 4. HHF invariants ----------------------------------------------------
    keep = ~(hole | crack)          # pixels that HHF must NOT touch
    check("HHF leaves untouched every pixel that was valid and not a crack",
          float(np.abs(res["I_hhf"][keep] - I_w[keep]).max()) == 0.0,
          f"max change {np.abs(res['I_hhf'][keep] - I_w[keep]).max():g}")
    check("HHF output is finite everywhere", bool(np.isfinite(res["I_hhf"]).all()))

    rng = np.random.default_rng(0)
    corrupted = I_w.copy()
    corrupted[hole] = rng.uniform(0, 255, (int(hole.sum()), 3))
    res2 = cracks.fill_cracks(corrupted, D_w, lam=5, se_len=4, orientation="v")
    same = float(np.abs(res2["I_hhf"][hole] - res["I_hhf"][hole]).max())
    check("HHF never uses empty data (garbage in holes changes nothing)", same == 0.0,
          f"max difference {same:g}")

    # ---- 5. HHF accuracy on a known signal ------------------------------------
    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.float32)[None, :], (128, 1))
    ramp3 = np.stack([ramp, ramp * 0.5, 255 - ramp], -1)
    v = np.ones((128, 256), bool)
    v[:, 100:102] = False
    filled = cracks.hhf_fill(ramp3, v)
    mae = float(np.abs(filled[:, 100:102] - ramp3[:, 100:102]).mean())
    check("HHF reconstructs a horizontal ramp through a 2 px gap (MAE < 1)", mae < 1.0,
          f"MAE {mae:.3f} gray levels")

    # ---- 6. HHF beats naive vertical interpolation on crack-like slivers ------
    val = cracks.validate_hhf(I_w, D_w, hole, n=60)
    check("HHF beats linear interpolation on synthetic slivers",
          val is not None and val["hhf_psnr"] > val["lin_psnr"],
          f"HHF PSNR {val['hhf_psnr']:.2f} dB vs linear {val['lin_psnr']:.2f} dB "
          f"({val['sliver_px']} px)")

    # ---- 7. ground truth: does the crack treatment help? ----------------------
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 7, "f000")["color"]
    for name, m in (("all cracks", crack), ("empty cracks", res["empty_crack"]),
                    ("translucent cracks", res["translucent_crack"])):
        b = viz.psnr(gt, I_w, mask=m)
        a = viz.psnr(gt, res["I_filled"], mask=m)
        check(f"filling helps on {name} (vs real cam7)", a > b + 0.5,
              f"{b:.2f} -> {a:.2f} dB over {int(m.sum())} px")
    mvalid = ~res["remaining_holes"]
    check("whole-frame PSNR improves after the crack step",
          viz.psnr(gt, res["I_filled"], mask=mvalid) > viz.psnr(gt, I_w, mask=mvalid) + 0.5,
          f"{viz.psnr(gt, I_w, mask=mvalid):.2f} -> "
          f"{viz.psnr(gt, res['I_filled'], mask=mvalid):.2f} dB")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
