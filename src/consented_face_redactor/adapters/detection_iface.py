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
    """Face bounding box in pixel coordinates (x1, y1, x2, y2)."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        return self.y2 - self.y1 + 1

    @property
    def area(self) -> int:
        w = max(0, self.width)
        h = max(0, self.height)
        return w * h


@dataclass(frozen=True)
class FaceDetection:
    """Output of face detection with five-point landmarks."""
    bbox: BoundingBox  # top-level bounding box in image coords
    landmarks: np.ndarray  # shape (5,2), float32, aligned to bbox
    confidence: float  # detector confidence score, range [0,1]


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
            Input image — may be BGR or RGB; dtype uint8. Never mutated.

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
        face_crop: np.ndarray,
        landmarks: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Extract normalized embedding from a aligned face crop.

        Parameters
        ----------
        face_crop : np.ndarray
            Aligned face image (e.g. 112x112 RGB, uint8). Never mutated.
        landmarks : np.ndarray
            Five-point landmarks used for alignment (5,2).

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
