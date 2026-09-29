"""Configuration for the view-filling pipeline (defaults are the validated settings)."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FillConfig:
    # ---- warping -----------------------------------------------------------
    scale: float = -44.8
    splat: str = "sub"        # sub (sub-pixel 2-tap, few cracks) | floor | round (integer)
    rule: str = "zbuf"        # zbuf (nearest source wins) | avg (weight average)
    # A warp produced elsewhere may have a broken collision rule (e.g. the provided
    # warping.scatter_image with inverse_ordering=True lets the FAR sample win, which
    # replaces foreground texture with background and looks like the person is cut).
    # "auto" = detect it against a correct Z-buffer warp and repair when it matters.
    repair_warp: str = "auto"     # auto | always | never
    repair_threshold_pct: float = 1.0   # deviation (% of valid px) that triggers a repair
    dev_threshold_gray: float = 40.0    # per-pixel difference considered damage

    # ---- cracks (paper II-A) ----------------------------------------------
    lam: float = 5.0          # crack threshold on the 0..255 depth scale
    se_len: int = 4           # line structuring element length (paper: 4)
    se_orientation: str = "auto"   # auto | v | h | both  (SE must cross the slit)
    bg_template: bool = True     # mask the template's FG pixels out of the SSD
    crack_shape: str = "none"      # none (paper) | thin (only thin hole components)
    max_thickness: float = 3.0
    hhf_sigma: float = 1.0
    hhf_ksize: int = 5

    # ---- ghosts (paper II-B) ----------------------------------------------
    skip_ghosts: bool = False
    band_radius: int = 2
    alpha_trim: float = 0.10
    alpha_sim: float = 11.0
    fg_side: str = "gt"       # gt: remove the nearer (foreground) candidates (paper II-A)
    fix_mode: str = "copy"    # copy | move | bg
    ksize: int = 9

    # ---- exemplar filling (paper II-C) ------------------------------------
    n_window: int = 69        # search window N (paper: 69)
    sizes: Tuple[int, ...] = (9, 7, 5, 3)   # adaptive patch sizes (paper: 9 -> 3, step 2)
    beta: float = 150.0       # acceptance threshold on the normalised patch cost
    beta_mode: str = "mean"   # mean (per-pixel MSE; paper's 35 is unreachable on 8-bit) | sum
    max_iter: int = 400000
    # Crack detection is a rule about THIN slits.  The raw paper rule (D_hat - D >= lam)
    slit_only: bool = True
    # Widen the splat footprint before warping so that a fast disparity ramp at a


    depth_dilate: object = "auto7"
    # RECTIFIED INPUT: the disparity has no vertical component, so a hole pixel may only be
    # filled from the SAME row.  None = off (search the full 2D window, needed when the
    # displacement field has a y component); 0 = same row only; n = allow +-n rows.
    # Without it the matcher can slide a source patch vertically, which shifts horizontal
    # structures (rails/barres) up or down inside the filled band.
    epipolar: object = None
    # STRUCTURE-AWARE CROSS-ROW PENALTY: a source patch may be borrowed from another
    struct_pen: float = 8.0
    # Low-texture protection: where the surroundings of a hole are smooth (plain wall), the
    # only informative content nearby is the dark baseboard / contact shadow, and the matcher
    # copies it -> a dark smeared band along the silhouette.  edge_pen charges
    # edge_pen * w_low * mean|grad|(source patch), w_low = 1 for a smooth neighbourhood and 0
    # where texture is rich (curtain), so only the problematic case is affected.
    edge_pen: float = 15.0
    edge_ref: float = 6.0
    # MEASURED VERDICT - constraining this does NOT help here.  Error decomposition over


    # ---- misc --------------------------------------------------------------
    oofa_frac: float = 0.5
    ablate: str = "none"      # none | nob | nodepth | nocd | noconf
    seed: int = 0

    def patch_params(self) -> dict:
        return dict(n_window=self.n_window, sizes=tuple(self.sizes), beta=self.beta,
                    beta_mode=self.beta_mode, max_iter=self.max_iter, splat=self.splat,
                    bg_template=self.bg_template, epipolar=self.epipolar,
                    struct_pen=self.struct_pen,
                    edge_pen=self.edge_pen, edge_ref=self.edge_ref)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["sizes"] = list(self.sizes)
        return d
