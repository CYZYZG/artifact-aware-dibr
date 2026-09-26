"""viewfill - generic hole filling for warped views (2D->3D / DIBR).

Give it a colour image plus its inverse depth; it warps to a virtual viewpoint
(disparity = inverse_depth * scale, the convention used by the 2D->3D work this was
written for) and fills every hole with the artifact-type aware pipeline of

    A. Q. de Oliveira, M. Walter, C. R. Jung,
    "An Artifact-type Aware DIBR Method for View Synthesis", IEEE SPL 2018.

or feed it an already warped image with holes and it fills those.

    from viewfill import FillConfig, warp_and_fill, fill_warped

    res = warp_and_fill(rgb, inv_depth, FillConfig(scale=-44.8))
    res["filled"]        # HxWx3 uint8, no holes
    res["stats"]         # per-stage numbers
"""
from .api import fill_holes                          # noqa: F401
from .config import FillConfig                        # noqa: F401
from .io import load_image, load_depth, load_pair     # noqa: F401
from .pipeline import fill_warped, warp_and_fill      # noqa: F401

__all__ = ["FillConfig", "warp_and_fill", "fill_warped", "fill_holes",
           "load_image", "load_depth", "load_pair"]
__version__ = "1.1.0"
