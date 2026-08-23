"""Detector interface contract — adapter for YuNet (or interchangeable detectors).

All detector adapters must implement this unified contract so the pipeline
depends only on internal interfaces, never raw vendor output.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Half-open face bounding box in pixel coordinates: ``[x1, x2)``."""
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("bounding-box coordinates must be integers")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive width and height")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class FaceDetection:
    """Output of face detection with five-point landmarks."""
    bbox: BoundingBox  # top-level bounding box in image coords
    landmarks: np.ndarray  # shape (5,2), float32, aligned to bbox
    confidence: float  # detector confidence score, range [0,1]

    def __post_init__(self) -> None:
        if not isinstance(self.bbox, BoundingBox):
            raise TypeError("bbox must be a BoundingBox")
        landmarks = np.asarray(self.landmarks, dtype=np.float32)
        if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
            raise ValueError("landmarks must be a finite float32 array with shape (5, 2)")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float, np.floating)
        ):
            raise TypeError("confidence must be numeric")
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        landmarks = landmarks.copy()
        landmarks.setflags(write=False)
        object.__setattr__(self, "landmarks", landmarks)
        object.__setattr__(self, "confidence", confidence)

    def as_yunet_row(self) -> np.ndarray:
        """Return the 15-value row expected by OpenCV ``alignCrop``."""
        row = np.empty(15, dtype=np.float32)
        row[:4] = (self.bbox.x1, self.bbox.y1, self.bbox.width, self.bbox.height)
        row[4:14] = self.landmarks.reshape(-1)
        row[14] = self.confidence
        return row


class DetectorAdapter(abc.ABC):
    """Abstract base for face detection adapters."""

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        pass

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> Sequence[FaceDetection]:
        """Detect all faces in image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR channel order; dtype uint8. Never mutated.

        Returns
        -------
        Sequence[FaceDetection]
            Zero or more detections. Empty sequence when no faces present.
            Does not filter by quality — caller does validation gating.
        """


class EmbedderAdapter(abc.ABC):
    """Abstract base for face embedding adapters."""

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        pass

    @abc.abstractmethod
    def embed(
        self,
        image: np.ndarray,
        detection: FaceDetection,
    ) -> tuple[np.ndarray, int]:
        """Align a detected face and extract a normalized embedding.

        Parameters
        ----------
        image : np.ndarray
            Original BGR uint8 image. Never mutated.
        detection : FaceDetection
            Bounding box, five landmarks, and detector confidence.

        Returns
        -------
        embedding : np.ndarray
            L2-normalized float32 vector, shape (D,).
        model_version : int
            Embedder version number for provenance tracking.

        Raises
        ------
        ValueError on invalid crop size, non-finite output, etc.
        """
