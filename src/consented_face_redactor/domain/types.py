"""Domain types for face redaction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FaceBox:
    """Bounding box of a detected face in pixel coordinates."""

    x1: int  # top-left x
    y1: int  # top-left y
    x2: int  # bottom-right x
    y2: int  # bottom-right y

    @property
    def width(self) -> int:
        return max(self.x2 - self.x1, 1)

    @property
    def height(self) -> int:
        return max(self.y2 - self.y1, 1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> int:
        return max(self.width, 1) * max(self.height, 1)


@dataclass(frozen=True)
class FiveLandmarks:
    """Five-eye/face landmarks used for alignment and rotation."""

    # Left eye (2 points), right eye (2 points), nose tip (1 point)
    left_eye_x: float
    left_eye_y: float
    right_eye_x: float
    right_eye_y: float
    nose_x: float
    nose_y: float

    @property
    def eye_angle(self) -> float:
        """Return the eye-line angle in radians (for rotation)."""
        dx = self.right_eye_x - self.left_eye_x
        dy = self.right_eye_y - self.left_eye_y
        return float(__import__("math").atan2(dy, dx))


@dataclass(frozen=True)
class MosaicConfig:
    """Configuration for the mosaic effect."""

    grid_cells: int = 12
    padding_ratio: float = 0.18
    block_side_px: int = 8  # legacy compatibility; force_block_size remains authoritative
    force_block_size: Optional[int] = None  # override auto-sizing logic
    min_block_px: int = 10
    shape: str = "rectangle"
    ellipse_horizontal_scale: float = 1.40
    ellipse_vertical_scale: float = 1.50


@dataclass(frozen=True)
class BlurConfig:

    """Configuration for Gaussian blur effect."""

    sigma_px: float = 4.0
    min_radius_px: int = 3
