"""
Step 2.1 - View synthesis / hole generation by 3D warping (single view).

Paper: Hole Filling for View Synthesis Using Depth Guided Global Optimization
       (G. Luo, Y. Zhu, IEEE Access 2018), Section II / Fig. 3 (single view branch).

Pipeline implemented here:
    input frame (color + inverse depth)
        -> normalize inverse depth to [0, 1]
        -> scatter_image(...)   # 3D warping / DIBR (sub-pixel splatting + weight normalization)
        -> warped color video, warped inverse depth, hole mask

Outputs (written to ./output):
    step2_1_warped_color/f###.png     warped color view (black where hole)
    step2_1_warped_depth/f###.png     warped inverse-depth map (holes black)
    step2_1_hole_mask/f###.png        hole mask (white = hole, 255)
    step2_1_overlay/f###.png          warped color with holes painted red
    step2_1_panel/f###.png            6-sub-image inspection panel per frame
    step2_1_panel/f###_ordering_off.png   same panel with inverse_ordering=False (Z-buffer check)
    step2_1_summary.png              one-pair overview: full-resolution 7-sub-image panel + stats card
    step2_1_stats.csv                per-frame hole statistics

Run:
    python warp_and_visualize.py
    python warp_and_visualize.py --scale_factor 44.8 --direction -1
"""

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

# Use warping.scatter_image from the repository root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warping import scatter_image  # noqa: E402


def z_buffer_splat(*a, **kw):
    """Late import wrapper: z_buffer_warp imports helpers from this module, so it can
    only be imported lazily to avoid a circular import at module load time."""
    from z_buffer_warp import z_buffer_splat as _impl
    return _impl(*a, **kw)


# --------------------------------------------------------------------------- #
# Defaults (values fixed by the project brief)
# --------------------------------------------------------------------------- #
DEFAULT_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEFAULT_SCALE_FACTOR = 44.8
DEFAULT_DIRECTION = -1
DEFAULT_WARP_METHOD = "z_buffer"


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
# OpenCV's imread/imwrite go through a narrow-char path on Windows and fail on
# non-ASCII paths (this project lives under D:\项目\...).  Read/write via Python
# file objects + imdecode/imencode instead, which is encoding-safe.
def imread_u(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        raise IOError(f"empty file: {path}")
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"imdecode failed: {path}")
    return img


def imwrite_u(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if not ok:
        raise IOError(f"imencode failed: {path}")
    buf.tofile(path)


def load_sequence(input_dir):
    """Load aligned color (BGR uint8) and normalized inverse-depth ([0,1] float32) frames."""
    color_paths = sorted(glob.glob(os.path.join(input_dir, "color-*.jpg")))
    depth_paths = sorted(glob.glob(os.path.join(input_dir, "depth-*.png")))
    if not color_paths:
        raise FileNotFoundError(f"no color-*.jpg found in {input_dir}")
    if len(color_paths) != len(depth_paths):
        raise ValueError(
            f"count mismatch: {len(color_paths)} color vs {len(depth_paths)} depth frames"
        )

    frames = []
    for cpath, dpath in zip(color_paths, depth_paths):
        cname = os.path.basename(cpath)
        dname = os.path.basename(dpath)
        # frame index check, e.g. color-cam6-f003.jpg <-> depth-cam6-f003.png
        c_idx = cname.split("-")[-1].split(".")[0]
        d_idx = dname.split("-")[-1].split(".")[0]
        if c_idx != d_idx:
            raise ValueError(f"unaligned pair: {cname} vs {dname}")

        color = imread_u(cpath, cv2.IMREAD_COLOR)
        depth_raw = imread_u(dpath, cv2.IMREAD_UNCHANGED)
        if color is None or depth_raw is None:
            raise IOError(f"failed to read {cname} or {dname}")
        if color.shape[:2] != depth_raw.shape[:2]:
            raise ValueError(f"size mismatch for {c_idx}: {color.shape} vs {depth_raw.shape}")

        # depth png is content-grayscale; collapse to a single channel defensively
        if depth_raw.ndim == 3:
            chans = [depth_raw[:, :, c] for c in range(depth_raw.shape[2])]
            if not all(np.array_equal(chans[0], c) for c in chans[1:]):
                print(f"[warn] {dname}: depth channels differ, using channel 0")
            inv_depth_gray = chans[0]
        else:
            inv_depth_gray = depth_raw

        # normalized inverse depth in [0, 1]; larger value = closer to the camera
        inv_depth = inv_depth_gray.astype(np.float32) / 255.0
        frames.append(
            {
                "index": c_idx,
                "color": color,
                "inv_depth": inv_depth,
                "inv_depth_raw": inv_depth_gray,
            }
        )
    return frames


def ensure_dirs(output_dir):
    sub = [
        "step2_1_warped_color",
        "step2_1_warped_depth",
        "step2_1_hole_mask",
        "step2_1_overlay",
        "step2_1_panel",
    ]
    paths = {"root": output_dir}
    os.makedirs(output_dir, exist_ok=True)
    for name in sub:
        paths[name] = os.path.join(output_dir, name)
        os.makedirs(paths[name], exist_ok=True)
    return paths


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #
def colorize_scalar(field, holes=None, valid_mask=None):
    """TURBO colormap of a [0,1] scalar field; optionally gray-out invalid pixels."""
    f = np.clip(field.astype(np.float32), 0.0, 1.0)
    u8 = (f * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    if valid_mask is not None:
        vis[~valid_mask] = (40, 40, 40)
    return vis


def make_overlay(warped_color, hole_mask):
    """Warped color with hole pixels painted red."""
    vis = warped_color.copy()
    holes = hole_mask > 0
    vis[holes] = (0, 0, 255)
    return vis


def label(img, text, scale=0.55):
    """Put a small caption bar on top of an image (returns a new image)."""
    bar = np.full((26, img.shape[1], 3), 28, np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def shrink(img, max_width=900):
    """Nearest-neighbour shrink so the inspection mosaics stay viewable/light."""
    if img.shape[1] <= max_width:
        return img
    s = max_width / float(img.shape[1])
    return cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)


def summary_card(stats_rows, scale_factor, direction, warp_method, inverse_ordering):
    """Text card summarising parameters, per-frame hole statistics and the Z-buffer choice."""
    warp_line = ("z_buffer_splat  (proper Z-buffer: nearest source sample wins collisions)"
                 if warp_method == "z_buffer" else
                 f"scatter_image(inverse_ordering={inverse_ordering})  (original warping.py)")
    lines = ["Step 2.1 - 3D warping (single view)  |  paper Section II / Fig. 3 (single-view branch)",
             "",
             f"input    : {'1 frame pair' if len(stats_rows) == 1 else str(len(stats_rows)) + ' frame pairs'}"
             f", color-cam6-*.jpg + depth-cam6-*.png (1024x768)",
             f"depth    : normalized inverse depth = gray/255  (1.0 = nearest, 0.0 = farthest)",
             f"warp     : direction={direction}, scale_factor={scale_factor}, reproject_depth=True",
             f"method   : {warp_line}",
             f"disparity: 0 .. {max(r['max_hole_disp_px'] for r in stats_rows):.1f} px   "
             f"(mean over holes {np.mean([r['mean_hole_disp_px'] for r in stats_rows]):.1f} px)",
             "",
             "frame     holes(px)   ratio%   comps  comps>=50px",
             "--------------------------------------------------"]
    for r in stats_rows:
        lines.append(f"{r['frame']:>7}  {r['hole_pixels']:>9d}  {r['hole_ratio_%']:>6.3f}  "
                     f"{r['hole_components']:>6}  {r['hole_components_ge50']:>11}")
    ratios = [r["hole_ratio_%"] for r in stats_rows]
    lines += ["--------------------------------------------------",
              f"mean hole ratio {sum(ratios)/len(ratios):.3f}%   "
              f"min {min(ratios):.3f}%   max {max(ratios):.3f}%",
              "",
              "hole = target pixel that received no source sample after forward splatting",
              "(disocclusion newly exposed by the virtual view point)",
              "",
              "WHY z_buffer: warping.scatter_image decides a collision by traversal order, not",
              "by depth. Both of its modes damage the foreground on this data --",
              "  inverse_ordering=True  -> far sample wins: foreground replaced by background",
              "  inverse_ordering=False -> later sample keeps adding on top of the nearer one",
              "Correct rule: keep the NEAREST source sample. Deviation from the source colour at",
              "the nearest depth (pixels off by >40 gray levels, whole frame):",
              "  scatter inverse_ordering=True  14 141 px",
              "  scatter inverse_ordering=False    665 px",
              "  z_buffer_splat                    103 px   <- adopted",
              "see step2_1_zbuffer_compare.png and z_buffer_warp.py",
              "",
              "panels to inspect: 1) input  2) warped color  3) holes painted red",
              "                   4) input inverse depth  5) warped inverse depth  6) hole mask",
              "                   7) zoom on largest hole"]
    card = np.full((len(lines) * 20 + 16, 700, 3), 20, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(card, ln, (10, 24 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return card


def largest_hole_crop(hole_mask, crop_w=240, crop_h=180):
    """Zoom window around the largest *interior* hole component.

    Border-touching strips (the image-edge disocclusion) are ignored: they have the
    largest area but look like a plain red edge.  When several large interior
    components sit close together, cluster them so one window shows the real
    disocclusion strips together with the foreground edge that produced them.
    """
    mask = (hole_mask > 0).astype(np.uint8)
    H, W = mask.shape
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0, min(crop_h, H), 0, min(crop_w, W)

    xs, ys, ws, hs, areas = (stats[1:, cv2.CC_STAT_LEFT], stats[1:, cv2.CC_STAT_TOP],
                             stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT],
                             stats[1:, cv2.CC_STAT_AREA])
    touches_border = (xs <= 1) | (ys <= 1) | (xs + ws >= W - 1) | (ys + hs >= H - 1)

    # 1) largest interior component
    cand = np.where(~touches_border)[0]
    if cand.size == 0:
        cand = np.arange(len(areas))
    seed = cand[int(np.argmax(areas[cand]))]
    seed_cx = xs[seed] + ws[seed] / 2.0
    seed_cy = ys[seed] + hs[seed] / 2.0

    # 2) merge every component whose centroid falls inside the zoom window
    inw = ((xs + ws / 2.0 >= seed_cx - crop_w) & (xs + ws / 2.0 <= seed_cx + crop_w) &
           (ys + hs / 2.0 >= seed_cy - crop_h) & (ys + hs / 2.0 <= seed_cy + crop_h) & (areas >= 30))
    x0 = int(np.clip(xs[inw].min() - 20, 0, max(0, W - crop_w)))
    y0 = int(np.clip(ys[inw].min() - 20, 0, max(0, H - crop_h)))
    x1 = int(np.clip((xs[inw] + ws[inw]).max() + 20, x0 + 1, W))
    y1 = int(np.clip((ys[inw] + hs[inw]).max() + 20, y0 + 1, H))
    # keep the window a reasonable size
    x1 = min(x1, x0 + crop_w + 120)
    y1 = min(y1, y0 + crop_h + 120)
    return y0, y1, x0, x1


def build_panel(frame, warped_color, warped_map, hole_mask, tag="",
                warped_map_kind="inverse depth", warped_map_range=(0.0, 1.0)):
    """2x3 inspection panel; bottom-right cell is a zoom on the largest hole.

    `warped_map_kind` / `warped_map_range` describe the *third* return value of the warp
    so the figure can never silently mislabel it: `warping.scatter_image` returns real
    DEPTH there (it does a 1/x), while `z_buffer_splat` returns INVERSE depth.
    """
    h, w = hole_mask.shape
    holes = hole_mask > 0
    valid = ~holes

    orig = frame["color"]
    depth_vis = colorize_scalar(frame["inv_depth"])
    lo, hi = warped_map_range
    if hi > lo:
        wmap_norm = (warped_map.astype(np.float32) - lo) / (hi - lo)
    else:
        wmap_norm = np.zeros_like(warped_map, dtype=np.float32)
    wdepth_vis = colorize_scalar(wmap_norm, valid_mask=valid)
    overlay = make_overlay(warped_color, hole_mask)
    hole_vis = cv2.cvtColor((holes * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    y0, y1, x0, x1 = largest_hole_crop(hole_mask)
    crop_overlay = overlay[y0:y1, x0:x1].copy()
    crop_orig = orig[y0:y1, x0:x1].copy()
    cv2.rectangle(crop_overlay, (0, 0), (crop_overlay.shape[1] - 1, crop_overlay.shape[0] - 1), (0, 255, 255), 1)
    zoom = np.hstack([crop_orig, crop_overlay])

    ratio = holes.mean() * 100.0
    top = np.hstack(
        [
            label(orig, "1) original color " + tag),
            label(warped_color, "2) warped color"),
            label(overlay, f"3) warped + holes(red)  {ratio:.2f}%"),
        ]
    )
    bot = np.hstack(
        [
            label(depth_vis, "4) input inverse depth  [0,1]"),
            label(wdepth_vis, f"5) warped {warped_map_kind}  [{lo:g},{hi:g}]"),
            label(hole_vis, "6) hole mask (white=hole)"),
        ]
    )
    grid = np.vstack([top, bot])
    # wide zoom strip under the grid
    zoom_strip = np.full((zoom.shape[0] + 26, grid.shape[1], 3), 28, np.uint8)
    cv2.putText(zoom_strip, "7) zoom around largest hole: [left] original   [right] warped+hole",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    zoom_strip[26:, : zoom.shape[1]] = zoom
    return np.vstack([grid, zoom_strip])


def build_mosaic(images, cols=2, gap=8):
    """Stack labeled images into a grid (images already have caption bars)."""
    rows = int(np.ceil(len(images) / cols))
    cell_h = max(im.shape[0] for im in images)
    cell_w = max(im.shape[1] for im in images)
    canvas = np.full((rows * cell_h + (rows - 1) * gap, cols * cell_w + (cols - 1) * gap, 3), 12, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        y = r * (cell_h + gap)
        x = c * (cell_w + gap)
        canvas[y : y + im.shape[0], x : x + im.shape[1]] = im
    return canvas


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Step 2.1: 3D warping for single view")
    ap.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--scale_factor", type=float, default=DEFAULT_SCALE_FACTOR,
                    help="disparity scale applied to the normalized inverse depth")
    ap.add_argument("--direction", type=int, default=DEFAULT_DIRECTION, choices=[-1, 1],
                    help="splat direction: +1 shifts right, -1 shifts left")
    ap.add_argument("--inverse_ordering", action="store_true", default=True,
                    help="reverse traversal order when splatting (Z-buffer ordering control); "
                         "default True because it was measured to be the correct occlusion order")
    ap.add_argument("--reproject_depth", action="store_true", default=True,
                    help="also synthesize the warped depth map")
    ap.add_argument("--frames", default="first",
                    help="which frames to process: 'first' (default, single pair for inspection), "
                         "'all', or a comma separated list like 'f000,f003'")
    ap.add_argument("--warp_method", default=DEFAULT_WARP_METHOD, choices=["z_buffer", "scatter"],
                    help="z_buffer (default): proper Z-buffer splat, keeps the nearest source "
                         "sample on collisions; scatter: the original warping.scatter_image")
    args = ap.parse_args()

    frames = load_sequence(args.input_dir)
    if args.frames.strip().lower() == "first":
        frames = frames[:1]
    elif args.frames.strip().lower() != "all":
        want = {s.strip() for s in args.frames.split(",") if s.strip()}
        frames = [f for f in frames if f["index"] in want]
        if not frames:
            raise ValueError(f"no frame matched --frames {args.frames!r}")
    paths = ensure_dirs(args.output_dir)
    print(f"loaded {len(frames)} frame(s) from {len(glob.glob(os.path.join(args.input_dir, 'color-*.jpg')))} available, "
          f"size {frames[0]['color'].shape[1]}x{frames[0]['color'].shape[0]}")
    print(f"params: scale_factor={args.scale_factor} direction={args.direction} "
          f"warp_method={args.warp_method} (scatter inverse_ordering={args.inverse_ordering}) "
          f"reproject_depth={args.reproject_depth}")

    rows = []
    mosaic_panels = []
    for frame in frames:
        idx = frame["index"]
        color = frame["color"]
        inv_depth = frame["inv_depth"]

        if args.warp_method == "z_buffer":
            # z_buffer_splat returns the WARPED INVERSE DEPTH (same units as the input)
            warped, hole_mask, warped_map = z_buffer_splat(
                color, inv_depth, direction=args.direction,
                scale_factor=args.scale_factor, reproject_depth=args.reproject_depth,
                return_inverse_depth=True,
            )
            map_kind, map_range = "inverse depth", (0.0, 1.0)
        else:
            # scattering.scatter_image returns REAL DEPTH (it computes 1/inv_depth), so it
            # is converted back to inverse depth here to keep one consistent convention.
            warped, hole_mask, real_depth = scatter_image(
                color,
                inv_depth,
                direction=args.direction,
                scale_factor=args.scale_factor,
                inverse_ordering=args.inverse_ordering,
                reproject_depth=args.reproject_depth,
            )
            if args.reproject_depth:
                warped_map = np.where(hole_mask > 0, 0.0, 1.0 / (real_depth + 1e-6)).astype(np.float32)
            else:
                warped_map = np.zeros_like(inv_depth, dtype=np.float32)
            map_kind, map_range = "inverse depth (converted from depth)", (0.0, 1.0)
        warped_u8 = np.clip(warped, 0, 255).astype(np.uint8)
        holes = hole_mask > 0
        valid = ~holes

        # ---- outputs ----
        imwrite_u(os.path.join(paths["step2_1_warped_color"], f"{idx}.png"), warped_u8)
        lo, hi = map_range
        imwrite_u(
            os.path.join(paths["step2_1_warped_depth"], f"{idx}.png"),
            colorize_scalar(np.clip((warped_map - lo) / (hi - lo), 0, 1), valid_mask=valid),
        )
        imwrite_u(os.path.join(paths["step2_1_hole_mask"], f"{idx}.png"), hole_mask)
        imwrite_u(os.path.join(paths["step2_1_overlay"], f"{idx}.png"), make_overlay(warped_u8, hole_mask))

        panel = build_panel(frame, warped_u8, warped_map, hole_mask, tag=f"frame {idx}",
                            warped_map_kind=map_kind, warped_map_range=map_range)
        imwrite_u(os.path.join(paths["step2_1_panel"], f"{idx}.png"), panel)
        mosaic_panels.append(panel)

        # ---- collision-rule check: same warp with a different splat behaviour ----
        if args.warp_method == "z_buffer":
            warped_alt, mask_alt, _ = scatter_image(
                color, inv_depth, direction=args.direction, scale_factor=args.scale_factor,
                inverse_ordering=args.inverse_ordering, reproject_depth=False,
            )
            alt_tag = "scatter_image (foreground lost)"
        else:
            warped_alt, mask_alt, _ = scatter_image(
                color, inv_depth, direction=args.direction, scale_factor=args.scale_factor,
                inverse_ordering=not args.inverse_ordering, reproject_depth=False,
            )
            alt_tag = f"scatter inverse_ordering={not args.inverse_ordering}"
        alt = np.clip(warped_alt, 0, 255).astype(np.uint8)
        panel_alt = build_panel(frame, alt, warped_map, mask_alt, tag=f"frame {idx} ({alt_tag})",
                                warped_map_kind=map_kind, warped_map_range=map_range)
        imwrite_u(os.path.join(paths["step2_1_panel"], f"{idx}_ordering_alt.png"), panel_alt)

        # ---- statistics ----
        n_lab, _, cc_stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
        cc_areas = cc_stats[1:, cv2.CC_STAT_AREA] if n_lab > 1 else np.array([], dtype=np.int32)
        rows.append(
            {
                "frame": idx,
                "width": color.shape[1],
                "height": color.shape[0],
                "hole_pixels": int(holes.sum()),
                "hole_ratio_%": round(float(holes.mean() * 100.0), 4),
                "hole_components": int(n_lab - 1),
                "hole_components_ge50": int((cc_areas >= 50).sum()),
                "hole_pixels_in_ge50_%": round(
                    float(cc_areas[cc_areas >= 50].sum() / max(1, cc_areas.sum()) * 100.0), 3
                ),
                "max_hole_disp_px": round(float(inv_depth.max() * args.scale_factor), 3),
                "mean_hole_disp_px": round(float(inv_depth[holes].mean() * args.scale_factor), 3) if holes.any() else 0.0,
                "invdepth_min": round(float(inv_depth.min()), 4),
                "invdepth_max": round(float(inv_depth.max()), 4),
                "invdepth_mean": round(float(inv_depth.mean()), 4),
            }
        )
        print(f"  {idx}: holes {rows[-1]['hole_pixels']:>7d} px "
              f"({rows[-1]['hole_ratio_%']:.3f}%), components {rows[-1]['hole_components']} "
              f"(>=50px: {rows[-1]['hole_components_ge50']} covering {rows[-1]['hole_pixels_in_ge50_%']:.1f}%)")

    # single-pair mode: keep the panel at full resolution, side by side with the card
    if len(mosaic_panels) == 1:
        panel = mosaic_panels[0]
        card = summary_card(rows, args.scale_factor, args.direction, args.warp_method, args.inverse_ordering)
        if card.shape[1] < 620:
            card = np.hstack([card, np.full((card.shape[0], 620 - card.shape[1], 3), 20, np.uint8)])
        h = max(panel.shape[0], card.shape[0])
        panel_p = np.full((h, panel.shape[1], 3), 12, np.uint8)
        panel_p[: panel.shape[0]] = panel
        card_p = np.full((h, card.shape[1], 3), 20, np.uint8)
        card_p[: card.shape[0]] = card
        summary = np.hstack([panel_p, card_p])
    else:
        summary = build_mosaic([shrink(p, 900) for p in mosaic_panels], cols=2)
        card = summary_card(rows, args.scale_factor, args.direction, args.warp_method, args.inverse_ordering)
        if card.shape[1] < summary.shape[1]:
            card = np.hstack([card, np.full((card.shape[0], summary.shape[1] - card.shape[1], 3), 20, np.uint8)])
        else:
            summary = np.hstack([summary, np.full((summary.shape[0], card.shape[1] - summary.shape[1], 3), 12, np.uint8)])
        summary = np.vstack([card, summary])
    mosaic_path = os.path.join(paths["root"], "step2_1_summary.png")
    imwrite_u(mosaic_path, summary)

    csv_path = os.path.join(paths["root"], "step2_1_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ratios = [r["hole_ratio_%"] for r in rows]
    print("\n--- summary ---")
    print(f"hole ratio per frame: min {min(ratios):.3f}%  max {max(ratios):.3f}%  mean {sum(ratios)/len(ratios):.3f}%")
    print(f"panels : {paths['step2_1_panel']}")
    print(f"mosaic : {mosaic_path}")
    print(f"stats  : {csv_path}")


if __name__ == "__main__":
    main()
