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

    # ---- misc --------------------------------------------------------------
    oofa_frac: float = 0.5
    ablate: str = "none"      # none | nob | nodepth | nocd | noconf
    seed: int = 0

    def patch_params(self) -> dict:
        return dict(n_window=self.n_window, sizes=tuple(self.sizes), beta=self.beta,
                    beta_mode=self.beta_mode, max_iter=self.max_iter, splat=self.splat)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["sizes"] = list(self.sizes)
        return d
