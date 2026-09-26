"""Step 0 - calibration report for a reference -> target camera pair.

    python step0_calibrate.py --ref_cam 6 --dst_cam 7 --frame f000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dibr import calib, io_utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Step 0: camera calibration / displacement field")
    ap.add_argument("--dataset_root", default=io_utils.DATASET_ROOT_DEFAULT)
    ap.add_argument("--input_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "input"),
                    help="folder with the depth map of the reference camera (fallback)")
    ap.add_argument("--ref_cam", type=int, default=6)
    ap.add_argument("--dst_cam", type=int, default=7)
    ap.add_argument("--frame", default="f000")
    ap.add_argument("--y_origin", default="bottom", choices=["bottom", "top"])
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "output", "step0_calib"))
    args = ap.parse_args()

    cams = calib.load_calib(args.dataset_root)
    try:
        view = io_utils.load_view(args.dataset_root, args.ref_cam, args.frame)
        src = args.dataset_root
    except FileNotFoundError:
        view = io_utils.load_view(args.input_dir, args.ref_cam, args.frame)
        src = args.input_dir
    print(f"depth map: {view['depth_path']}")
    res = calib.describe(cams, args.ref_cam, args.dst_cam, view["depth"], out_dir=args.out_dir)
    print(f"\nartefacts written to {args.out_dir}")
    return 0 if res["valid"].all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
