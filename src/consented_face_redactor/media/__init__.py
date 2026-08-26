"""Abstract frame source — read frames from video/image file paths."""

from __future__ import annotations

import abc
from typing import Iterator, Optional, Sequence

import numpy as np


class FrameSource(abc.ABC):
    """Abstract base class for reading frames from a file source."""

    @property
    @abc.abstractmethod
    def fps(self) -> float:
        """Frames per second of the source (-1 if unknown/dictation)."""

    @property
    @abc.abstractmethod
    def frame_count(self) -> int:
        """Total number of frames (or -1 for unknown/streaming)."""

    @property
    @abc.abstractmethod
    def width(self) -> int:
        """Frame width in pixels."""

    @property
    @abc.abstractmethod
    def height(self) -> int:
        """Frame height in pixels."""

    @abc.abstractmethod
    def open(self) -> None:
        """Open the file and prepare for frame reading. Must be called before any read operations."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources. After this, no frames can be read until `open()` is called again."""

    @abc.abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the next frame.

        Returns
        -------
        success : bool
            True if a frame was successfully read.
        frame : np.ndarray or None
            Frame as uint8 numpy array (H, W, C). None when read() returns success=False.
        """

    @abc.abstractmethod
    def seek(self, frame_index: int) -> bool:
        """Seek to a specific frame index.

        Returns
        -------
        success : bool
            True if seek was successful.
        """

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[type-arg]
        self.close()


class FakeFrameReader(FrameSource):
    """Fake frame source for testing — yields deterministic all-zero frames."""

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        n_frames: int = 60,
        fps: float = 30.0,
        ch: int = 3,
    ) -> None:
        self._width = width
        self._height = height
        self._n_frames = n_frames
        self._fps = fps
        self._ch = ch
        self._opened = False
        self._closed = False
        self._current_index = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._n_frames

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def current_frame_index(self) -> int:
        return self._current_index

    def open(self) -> None:
        if self._closed:
            self._closed = False
        self._opened = True
        self._current_index = 0

    def close(self) -> None:
        self._closed = True
        self._opened = False
        self._current_index = 0

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._opened or self._closed:
            return (False, None)
        if self._current_index >= self._n_frames:
            return (False, None)
        frame = np.zeros((self._height, self._width, self._ch), dtype=np.uint8)
        # Add deterministic gradient based on frame index so frames are distinguishable
        frame[:, :, 0] = (self._current_index * 5) % 256  # R channel increments
        self._current_index += 1
        return (True, frame)

    def seek(self, frame_index: int) -> bool:
        if not self._opened or self._closed:
            return False
        if frame_index < 0 or frame_index > self._n_frames:
            return False
        self._current_index = frame_index
        return True
