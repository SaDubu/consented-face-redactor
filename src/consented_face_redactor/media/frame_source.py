"""OpenCV frame reader and writer implementations."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import FrameSource


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "OpenCV media support requires the 'inference' project extra"
        ) from exc
    return cv2


class OpenCvFrameSource(FrameSource):
    """Read BGR uint8 frames from an image or video file."""

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"}

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve(strict=False)
        self._cap: Any = None
        self._image_frame: np.ndarray | None = None
        self._opened = False
        self._fps = -1.0
        self._frame_count = -1
        self._width = 0
        self._height = 0
        self._current_index = 0

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
        return str(self._path)

    def open(self) -> None:
        self.close()
        if not self._path.is_file():
            raise FileNotFoundError(f"Input media is unavailable: {self._path.name}")

        cv2 = _load_cv2()
        if self._path.suffix.lower() in self._IMAGE_EXTENSIONS:
            frame = cv2.imread(str(self._path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Input image could not be decoded: {self._path.name}")
            image = np.asarray(frame)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Decoded image must be a uint8 BGR frame")
            self._image_frame = image.copy()
            self._height, self._width = image.shape[:2]
            self._fps = -1.0
            self._frame_count = 1
        else:
            cap = cv2.VideoCapture(str(self._path))
            if not cap.isOpened():
                cap.release()
                raise OSError(f"Input media could not be opened: {self._path.name}")
            self._cap = cap
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            self._fps = fps if np.isfinite(fps) and fps > 0 else -1.0
            count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._frame_count = int(count) if np.isfinite(count) and count >= 0 else -1
            self._width = int(width) if np.isfinite(width) and width >= 1 else 0
            self._height = int(height) if np.isfinite(height) and height >= 1 else 0
            if self._width < 1 or self._height < 1:
                self.close()
                raise ValueError("Input video reports invalid frame dimensions")

        self._opened = True
        self._current_index = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._image_frame = None
        self._opened = False
        self._current_index = 0
        self._fps = -1.0
        self._frame_count = -1
        self._width = 0
        self._height = 0

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            raise RuntimeError("OpenCvFrameSource is not open")

        if self._image_frame is not None:
            if self._current_index > 0:
                return False, None
            self._current_index = 1
            return True, self._image_frame.copy()

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        result = np.asarray(frame)
        if result.dtype != np.uint8 or result.ndim != 3 or result.shape[2] != 3:
            raise ValueError("Decoded video frame must be a uint8 BGR array")
        self._current_index += 1
        return True, result

    def seek(self, frame_index: int) -> bool:
        if not self._opened:
            raise RuntimeError("OpenCvFrameSource is not open")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            return False

        if self._image_frame is not None:
            if frame_index not in (0, 1):
                return False
            self._current_index = frame_index
            return True

        if self._frame_count > 0 and frame_index > self._frame_count:
            return False
        cv2 = _load_cv2()
        if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
            return False
        self._current_index = frame_index
        return True


class FrameWriter(abc.ABC):
    """Abstract frame writer."""

    @property
    @abc.abstractmethod
    def is_open(self) -> bool: ...

    @abc.abstractmethod
    def open(
        self,
        path: str | Path | None = None,
        fps: float | None = None,
        codec: str | None = None,
    ) -> None: ...

    @abc.abstractmethod
    def write(self, frame: np.ndarray) -> bool: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class OpenCvFrameWriter(FrameWriter):
    """Write fixed-size BGR frames without silently resizing them."""

    _SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        path: str | Path,
        fps: float = 30.0,
        codec: str = "avc1",
        *,
        overwrite: bool = False,
    ) -> None:
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise TypeError("fps must be numeric")
        if not isinstance(codec, str):
            raise TypeError("codec must be a string")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        self._path = Path(path).resolve(strict=False)
        self._fps = float(fps)
        self._codec = codec
        self._overwrite = overwrite
        self._writer: Any = None
        self._is_open = False
        self._width = 0
        self._height = 0

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def open(
        self,
        path: str | Path | None = None,
        fps: float | None = None,
        codec: str | None = None,
    ) -> None:
        self.close()
        if path is not None:
            self._path = Path(path).resolve(strict=False)
        if fps is not None:
            if isinstance(fps, bool) or not isinstance(fps, (int, float)):
                raise TypeError("fps must be numeric")
            self._fps = float(fps)
        if codec is not None:
            if not isinstance(codec, str):
                raise TypeError("codec must be a string")
            self._codec = codec

        if self._path.suffix.lower() not in self._SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported output format: {self._path.suffix.lower()}")
        if not np.isfinite(self._fps) or self._fps <= 0:
            raise ValueError("fps must be finite and positive")
        if len(self._codec) != 4 or not self._codec.isascii():
            raise ValueError("codec must contain exactly four ASCII characters")
        if self._path.exists() and not self._overwrite:
            raise FileExistsError(f"Output already exists: {self._path.name}")
        if not self._path.parent.is_dir():
            raise FileNotFoundError("Output directory is unavailable")
        _load_cv2()
        self._is_open = True

    def _initialize_writer(self, width: int, height: int) -> None:
        if self._path.exists() and not self._overwrite:
            self._is_open = False
            raise FileExistsError(f"Output already exists: {self._path.name}")
        cv2 = _load_cv2()
        fourcc_factory = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc_factory is None:
            fourcc_factory = cv2.VideoWriter.fourcc
        fourcc = fourcc_factory(*self._codec)
        try:
            writer = cv2.VideoWriter(
                str(self._path),
                fourcc,
                self._fps,
                (width, height),
            )
        except Exception:
            self._is_open = False
            raise
        if not writer.isOpened():
            writer.release()
            self._is_open = False
            raise OSError(f"Output media could not be opened: {self._path.name}")
        self._writer = writer
        self._width = width
        self._height = height

    def write(self, frame: np.ndarray) -> bool:
        if not self._is_open:
            raise RuntimeError("OpenCvFrameWriter is not open")
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy array")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a uint8 BGR array with shape (H, W, 3)")

        height, width = frame.shape[:2]
        if height < 1 or width < 1:
            raise ValueError("frame dimensions must be positive")
        if self._writer is None:
            self._initialize_writer(width, height)
        elif (width, height) != (self._width, self._height):
            raise ValueError("all output frames must have identical dimensions")

        self._writer.write(np.ascontiguousarray(frame))
        return True

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
        self._writer = None
        self._is_open = False
        self._width = 0
        self._height = 0
