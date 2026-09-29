"""viewfill.video - write a stereo video with Python libraries only (no system ffmpeg).

Encoder preference:
  1. imageio + imageio-ffmpeg -> H.264 (libx264) using the static ffmpeg shipped inside the
     wheel (Lib/site-packages/imageio_ffmpeg/binaries), so neither a system install nor the
     local D:\\ffmpeg copy is ever used;
  2. OpenCV's bundled encoder (mp4v) as a fallback when imageio is unavailable.

    with VideoWriter("sbs.mp4", fps=30) as vw:
        vw.append(rgb_uint8)          # HxWx3 RGB
"""
import os

__all__ = ["VideoWriter", "write_video", "video_info"]


class VideoWriter:
    def __init__(self, path, fps=30, crf=18, codec="libx264", quiet=True, ffmpeg_exe=None):
        self.path = os.path.abspath(path)
        self.fps = float(fps)
        self.crf = int(crf)
        self.codec = codec
        self.quiet = quiet
        self.ffmpeg_exe = ffmpeg_exe
        self.n = 0
        self.kind = None
        self.size = None
        self.imageio_error = None
        self._w = None
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    # ------------------------------------------------------------------ openers
    def _open_imageio(self, frame):
        import imageio
        h, w = frame.shape[:2]
        base = dict(fps=self.fps, codec=self.codec, quality=None, macro_block_size=1,
                    ffmpeg_params=["-crf", str(self.crf), "-movflags", "+faststart"])
        if self.ffmpeg_exe:
            base["ffmpeg_exe"] = self.ffmpeg_exe
        attempts = ([dict(base, ffmpeg_log="error")] if self.quiet else []) + [base]
        last = None
        for kw in attempts:                       # `ffmpeg_log` is not accepted everywhere
            try:
                self._w = imageio.get_writer(self.path, **kw)
                self.kind = "imageio/libx264"
                self.size = (w, h)
                return
            except Exception as e:                # noqa: BLE001 - fall through to OpenCV
                last = e
        raise RuntimeError("imageio writer failed: %s" % last)

    def _open_cv2(self, frame):
        import cv2
        h, w = frame.shape[:2]
        self._w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not self._w.isOpened():
            raise RuntimeError("no usable encoder: %s" % (self.imageio_error,))
        self.kind = "opencv/mp4v"
        self.size = (w, h)

    # ------------------------------------------------------------------ API
    def append(self, frame_rgb):
        import numpy as np
        f = np.ascontiguousarray(frame_rgb)
        if f.ndim == 2:
            f = np.repeat(f[:, :, None], 3, axis=2)
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        if self._w is None:
            try:
                self._open_imageio(f)
            except Exception as e:                # noqa: BLE001
                self.imageio_error = str(e)
                self._open_cv2(f)
        if self.kind == "opencv/mp4v":
            import cv2
            self._w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        else:
            self._w.append_data(f)
        self.n += 1

    def close(self):
        if self._w is not None:
            close = getattr(self._w, "close", None) or getattr(self._w, "release", None)
            if close:
                close()
            self._w = None
        return dict(path=self.path, frames=self.n, encoder=self.kind, size=self.size,
                    bytes=os.path.getsize(self.path) if os.path.exists(self.path) else 0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_video(frames, path, fps=30, crf=18, quiet=True, ffmpeg_exe=None):
    vw = VideoWriter(path, fps=fps, crf=crf, quiet=quiet, ffmpeg_exe=ffmpeg_exe)
    try:
        for f in frames:
            vw.append(f)
    finally:
        info = vw.close()
    return info


def video_info(path):
    """(frames, width, height, fps) read back from the written file."""
    import imageio
    try:
        rd = imageio.get_reader(path)
        meta = rd.get_meta_data()
        n = rd.count_frames()
        rd.close()
        w, h = meta.get("size", (0, 0))
        return int(n), int(w), int(h), float(meta.get("fps", 0.0))
    except Exception:                             # noqa: BLE001
        import cv2
        cap = cv2.VideoCapture(path)
        out = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
               int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
               float(cap.get(cv2.CAP_PROP_FPS)))
        cap.release()
        return out
