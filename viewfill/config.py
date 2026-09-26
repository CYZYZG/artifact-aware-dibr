"""Configuration for the view-filling pipeline (defaults are the validated settings)."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FillConfig:
    # ---- warping -----------------------------------------------------------
    # disparity in pixels = (depth normalised to 0..1) * scale.
    # The 2D->3D convention this package was written for: scale = -44.8, i.e. content
    # shifts LEFT by up to 44.8 px and the virtual view is to the RIGHT of the reference.
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
    # Dark rim along the filled-region boundary: the template patch contains the (dark)
    # foreground edge of the silhouette, so the matcher is driven to reproduce that edge
    # inside the hole.  bg_template masks those pixels out of the SSD; require_full_bg
    # additionally demands that no pixel of the source patch is foreground.
    bg_template: bool = True
    require_full_bg: bool = False
    # Cracks lying on a depth step are disocclusion slivers: "linear" interpolates across
    # them, "bg" copies the background side, "hhf" is the paper-faithful isotropic fill.
    # On the seam metric (jump at the filled/background junction) none of them beats hhf,
    # so hhf stays the default; the real lever is depth_dilate below (see 复现方案.md 9.7).
    crack_fill: str = "hhf"
    # Widen the splat footprint before warping so that a fast disparity ramp at a
    # silhouette does not leave a 1-2 px crack network (the main source of the visible seam
    # at the filled/background junction).  int n = n iterations of a 3x3 max filter
    # everywhere; "auto"/"auto5"/"auto7" instead widen only where the disparity gradient
    # demands it, with the demand map smoothed by a w x w max filter first.
    # PAIRED test over 8 frames of cam6 -> cam7 with the CALIBRATED displacement field
    # (seam mean +- sd / per-sample best / p90 / GT PSNR):
    #   3      5.989 +- 0.756 (best 6/8)  15.796  28.231
    #   auto7  6.080 +- 0.769 (best 2/8)  16.521  28.233   <- default
    #   auto5  6.673 +- 0.884 (best 0/8)  17.560  28.062   (3 and auto7 both beat it)
    # 3 vs auto7 is NOT significant on any metric (p = 0.38..0.84); auto7 is taken as the
    # default for its (marginally) best mean GT PSNR, the smaller p90 spread (+-2.27 vs
    # +-2.91) and its clearly better result on curtain-stripe content (e.g. frame f004,
    # seam 5.68 vs 7.10) - see 复现方案.md 9.11 and _work/visual_cmp.py for the side-by-side.


    depth_dilate: object = "auto7"
    # RECTIFIED INPUT: the disparity has no vertical component, so a hole pixel may only be
    # filled from the SAME row.  None = off (search the full 2D window, needed when the
    # displacement field has a y component); 0 = same row only; n = allow +-n rows.
    # Without it the matcher can slide a source patch vertically, which shifts horizontal
    # structures (rails/barres) up or down inside the filled band.
    epipolar: object = None
    # MEASURED VERDICT - constraining this does NOT help here.  Error decomposition over
    # 2 frames (all / filled-band / barre-row PSNR): None 28.40/24.66/24.64 (default,
    # best), 2 = 28.33/24.30/24.29, 0 = 28.12/23.40/19.92 (much worse, even on the
    # barre rows).  The row offsets are real (58.8 % of patches, p90 22.7 px) but mostly
    # BENEFICIAL: in a disocclusion the correct background is occluded in the reference
    # at that very row, so a same-row candidate pool cannot contain it - borrowing the
    # same background surface from another row is usually right.  Use 0/1/2 only when
    # the background behind your objects has strong horizontal structure (rails, fences)
    # that must not shift vertically.

    # Reject source patches that straddle a depth edge (patch inverse-depth std above this
    # tolerance, on the 0..255 depth scale).  Such a patch carries the dark silhouette edge
    # or its shadow into the hole, which is what shows up as a ghost contour along the seam.
    # 0 = off.  See 复现方案.md 9.10 for the measured effect.
    src_depth_tol: float = 0.0

    # ---- misc --------------------------------------------------------------
    oofa_frac: float = 0.5
    ablate: str = "none"      # none | nob | nodepth | nocd | noconf
    seed: int = 0

    def patch_params(self) -> dict:
        return dict(n_window=self.n_window, sizes=tuple(self.sizes), beta=self.beta,
                    beta_mode=self.beta_mode, max_iter=self.max_iter, splat=self.splat,
                    bg_template=self.bg_template, require_full_bg=self.require_full_bg,
                    src_depth_tol=self.src_depth_tol, epipolar=self.epipolar)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["sizes"] = list(self.sizes)
        return d
