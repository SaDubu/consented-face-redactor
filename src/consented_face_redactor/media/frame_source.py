"""OpenCV-based FrameSource and FrameWriter implementations."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional

import numpy as np


class OpenCvFrameSource:
    """Read frames from video/image files using OpenCV cv2.VideoCapture."""

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"}

    def __init__(self, path: str | Path) -> None:
        self._path = str(Path(path).resolve())
        self._cap = None  # type: ignore[attr-defined] (cv2.VideoCapture handle is dynamic)
        self._opened = False
        self._fps: float = -1.0
        self._frame_count: int = -1
        self._width: int = 0
        self._height: int = 0
        self._current_index: int = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def current_frame_index(self) -> int:
        return self._current_index

    @property
    def path(self) -> str:
        return self._path

    def open(self) -> None:
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "OpenCvFrameSource requires opencv-python. Install via: pip install opencv-python"
            ) from exc

        if not hasattr(cv2, "VideoCapture"):
            raise ImportError("OpenCV version missing VideoCapture module")

        self._cap = cv2.VideoCapture(str(self._path))

        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open file: {self._path}")

        ext = Path(self._path).suffix.lower()
        if ext in self._IMAGE_EXTENSIONS:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._cap.release()
                raise ValueError(f"Failed to read image from {self._path}")
            self._height, self._width = frame.shape[:2]
            self._fps = -1.0
            self._frame_count = 1
        else:
            # Video file
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or -1.0
            self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._opened = True
        self._current_index = 0

    def close(self) -> None:
        if self._cap is not None and hasattr(self._cap, "release"):
            self._cap.release()
        self._opened = False
        self._current_index = 0

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._opened or self._cap is None or not hasattr(self._cap, "read"):
            raise RuntimeError("OpenCvFrameSource not opened — call open() first")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return (False, None)

        self._current_index += 1
        bgr_frame = np.asarray(frame, dtype=np.uint8)
        return (True, bgr_frame)

    def seek(self, frame_index: int) -> bool:
        if not self._opened or self._cap is None or not hasattr(self._cap, "set"):
            raise RuntimeError("OpenCvFrameSource not opened — call open() first")

        if frame_index < 0:
            return False

        success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        if success:
            self._current_index = frame_index
            ok, _ = self.read()
            return ok and (self.current_frame_index == frame_index + 1)
        return False


class FrameWriter(abc.ABC):
    """Abstract base class for writing frames to video/image files."""

    @property
    @abc.abstractmethod
    def is_open(self) -> bool: ...

    @abc.abstractmethod
    def open(self, path: str | Path, fps: float, codec: str = ".avc1") -> None: ...

    @abc.abstractmethod
    def write(self, frame: np.ndarray) -> bool: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class OpenCvFrameWriter(FrameWriter):
    """Write frames to video files using OpenCV VideoWriter."""

    _SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, path: str | Path, fps: float = 30.0, codec: str = "avc1") -> None:
        self._path = str(Path(path).resolve())
        self._fps = fps
        self._codec = codec
        self._writer = None  # type: ignore[attr-defined] (cv2.VideoWriter handle)
        self._is_open = False
        self._width: int = 0
        self._height: int = 0

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def path(self) -> str:
        return self._path

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def open(self, path=None, fps=None, codec=None):
        """Open the output file for writing."""
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "OpenCvFrameWriter requires opencv-python. Install via: pip install opencv-python"
            ) from exc

        out_path = str(Path(path or self._path).resolve())
        out_fps = fps or self._fps
        out_codec = codec or self._codec

        if Path(out_path).suffix.lower() not in self._SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported output format: {Path(out_path).suffix}")

        fourcc = cv2.VideoWriter.fourcc(*out_codec[:4]) if len(out_codec) >= 4 else -1
        size_to_set = (320, 240)  # Default before first frame is written
        self._writer = cv2.VideoWriter(out_path, fourcc, out_fps, size_to_set)

        if not self._writer.isOpened():
            raise OSError(f"VideoWriter could not open file: {out_path}")

        self._is_open = True

    def _ensure_first_frame(self, frame):
        """Set video dimensions from first frame during writer construction."""
        h, w = frame.shape[:2]
        if self._height == 0 or self._width == 0:
            size_to_update = (w, h)
            # Re-open with corrected size
            fourcc = cv2.VideoWriter.fourcc(*self._codec[:4]) if len(self._codec) >= 4 else -1
            new_writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, size_to_update)
            if not new_writer.isOpened():
                raise OSError(f"VideoWriter could not open file: {self._path}")
            self._writer.release()  # Release the incorrectly-sized writer
            self._writer = new_writer
            self._width = w
            self._height = h

    def write(self, frame):
        """Write a single frame to the file."""
        if not self._is_open or self._writer is None:
            raise RuntimeError("Writer not opened — call open() first")

        out_frame = np.asarray(frame, dtype=np.uint8)

        if out_frame.ndim != 3 or out_frame.shape[2] not in (1, 3):
            raise ValueError("Frame must be a 3D array with 1 or 3 channels")

        h, w = out_frame.shape[:2]
        # Resize only if dimensions differ and writer is already opened with a different size
        if self._height > 0 and self._width > 0 and (h != self._height or w != self._width):
            import cv2  # type: ignore[import-untyped]

            out_frame = cv2.resize(out_frame, (self._width, self._height))

        ok = self._writer.write(out_frame) if hasattr(self._writer, "write") else False
        return bool(ok) if ok is not None else False

    def close(self):
        """Release the writer resources."""
        if self._writer is not None and hasattr(self._writer, "release"):
            self._writer.release()
        self._is_open = False
        self._width = 0
        self._height = 0
