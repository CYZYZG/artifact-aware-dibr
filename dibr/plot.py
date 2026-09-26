"""Minimal chart rendering with OpenCV (matplotlib is not available in this runtime)."""
import cv2
import numpy as np

# BGR colours
C = {"red": (60, 60, 230), "green": (80, 200, 80), "blue": (230, 140, 40),
     "orange": (40, 160, 255), "violet": (200, 80, 200), "grey": (150, 150, 150),
     "yellow": (60, 220, 220)}


def _canvas(w, h, bg=24):
    return np.full((h, w, 3), bg, np.uint8)


def line_chart(series, title, ylabel, size=(980, 430), ylim=None, ref_lines=(),
               xlabels=None):
    """series: list of (name, values, colour-name).  ref_lines: list of (value, label)."""
    w, h = size
    img = _canvas(w, h)
    ml, mr, mt, mb = 78, 190, 46, 46
    x0, x1 = ml, w - mr
    y0, y1 = mt, h - mb
    allv = [v for _, vals, _ in series for v in vals if np.isfinite(v)]
    allv += [v for v, _ in ref_lines]
    lo = min(allv) if allv else 0.0
    hi = max(allv) if allv else 1.0
    if ylim:
        lo, hi = ylim
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    n = max(len(vals) for _, vals, _ in series)

    def xy(i, v):
        xx = x0 + (x1 - x0) * (i / max(1, n - 1))
        yy = y1 - (y1 - y0) * ((v - lo) / (hi - lo))
        return int(round(xx)), int(round(yy))

    # grid + y ticks
    for t in range(6):
        v = lo + (hi - lo) * t / 5
        _, yy = xy(0, v)
        cv2.line(img, (x0, yy), (x1, yy), (52, 52, 52), 1)
        cv2.putText(img, f"{v:.4g}", (8, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (170, 170, 170), 1, cv2.LINE_AA)
    for v, lab in ref_lines:
        _, yy = xy(0, v)
        cv2.line(img, (x0, yy), (x1, yy), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(img, lab, (x1 - 178, yy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (150, 150, 150), 1, cv2.LINE_AA)
    # series
    for si, (name, vals, col) in enumerate(series):
        col = C.get(col, (255, 255, 255))
        pts = [xy(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 3, col, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1 + 14, y0 + 8 + si * 22), (x1 + 28, y0 + 18 + si * 22),
                      col, -1)
        cv2.putText(img, name, (x1 + 34, y0 + 18 + si * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (220, 220, 220), 1, cv2.LINE_AA)
    if xlabels:
        for i, lab in enumerate(xlabels):
            if i % max(1, len(xlabels) // 10) == 0:
                xx, _ = xy(i, lo)
                cv2.putText(img, lab, (xx - 16, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                            (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(img, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1,
                cv2.LINE_AA)
    cv2.putText(img, ylabel, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
                cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    return img


def bar_chart(groups, title, ylabel, size=(980, 380), ylim=None):
    """groups: list of (label, [(series_name, value, colour), ...])"""
    w, h = size
    img = _canvas(w, h)
    ml, mr, mt, mb = 78, 200, 46, 54
    x0, x1 = ml, w - mr
    y0, y1 = mt, h - mb
    vals = [v for _, bars in groups for _, v, _ in bars]
    lo, hi = (0.0, max(vals) * 1.15) if not ylim else ylim
    bw = (x1 - x0) / max(1, len(groups))
    for t in range(6):
        v = lo + (hi - lo) * t / 5
        yy = int(y1 - (y1 - y0) * (v - lo) / (hi - lo))
        cv2.line(img, (x0, yy), (x1, yy), (52, 52, 52), 1)
        cv2.putText(img, f"{v:.4g}", (8, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (170, 170, 170), 1, cv2.LINE_AA)
    for gi, (lab, bars) in enumerate(groups):
        nb = len(bars)
        for bi, (name, v, col) in enumerate(bars):
            col = C.get(col, (255, 255, 255))
            bx = int(x0 + bw * gi + bw * 0.18 + (bw * 0.64 / nb) * bi)
            bwid = int(bw * 0.64 / nb) - 4
            by = int(y1 - (y1 - y0) * (v - lo) / (hi - lo))
            cv2.rectangle(img, (bx, by), (bx + bwid, y1), col, -1)
            cv2.putText(img, f"{v:.1f}", (bx - 2, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, lab, (int(x0 + bw * gi + bw * 0.1), y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    names = [n for n, _, _ in groups[0][1]] if groups else []
    for si, n in enumerate(names):
        cv2.rectangle(img, (x1 + 14, y0 + 8 + si * 22), (x1 + 28, y0 + 18 + si * 22),
                      C.get(groups[0][1][si][2], (255, 255, 255)), -1)
        cv2.putText(img, n, (x1 + 34, y0 + 18 + si * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1,
                cv2.LINE_AA)
    cv2.putText(img, ylabel, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
                cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    return img
