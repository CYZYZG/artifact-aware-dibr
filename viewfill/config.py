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
    # everywhere; "auto" = widen only where |dx gradient| demands it (no unnecessary
    # foreground inflation).  Measured (scale -44.8): seam 10.27 -> 7.58, p90 39.1 -> 26.0,
    # Measured (scale -44.8): 0 -> seam 10.27/p90 39.1/cracks 12682; 2 (default) ->
    # 7.90/22.3/9526; 3 -> 4.49/11.2/9294 but the foreground is inflated by 3 px; "auto"
    # (demand-driven, best GT 20.46 dB) -> 7.50/21.9/10815 but leaves visible thin lines
    # along the silhouette.  Uniform 2 is the best visual/geometric compromise.
    # See 复现方案.md 9.10.
    depth_dilate: object = 2
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
                    src_depth_tol=self.src_depth_tol)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["sizes"] = list(self.sizes)
        return d
