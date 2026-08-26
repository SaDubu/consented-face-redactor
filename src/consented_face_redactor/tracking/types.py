"""Immutable value types for temporal face tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _readonly_float_array(value: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}D array")
    array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class PointTrackResult:
    """One tracker's non-authorizing point/visibility observation."""

    frame_index: int
    points_xy: np.ndarray
    visibility: np.ndarray
    model_revision: str

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        points = _readonly_float_array(self.points_xy, name="points_xy", ndim=2)
        visibility = _readonly_float_array(self.visibility, name="visibility", ndim=1)
        if points.shape[1:] != (2,) or len(points) == 0:
            raise ValueError("points_xy must have shape (N, 2) with N > 0")
        if len(visibility) != len(points):
            raise ValueError("visibility length must match points")
        if np.any((visibility < 0.0) | (visibility > 1.0)):
            raise ValueError("visibility values must be in [0, 1]")
        if not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise ValueError("model_revision must be non-empty")
        object.__setattr__(self, "points_xy", points)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "model_revision", self.model_revision.strip())


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    """Robust 2D scale/rotation/translation estimate."""

    scale: float
    rotation_radians: float
    translation_x: float
    translation_y: float
    inlier_count: int

    def __post_init__(self) -> None:
        values = (
            self.scale,
            self.rotation_radians,
            self.translation_x,
            self.translation_y,
        )
        if not np.isfinite(values).all() or self.scale <= 0:
            raise ValueError("similarity transform values must be finite with positive scale")
        if isinstance(self.inlier_count, bool) or not isinstance(self.inlier_count, int) or self.inlier_count < 2:
            raise ValueError("inlier_count must be an integer >= 2")


@dataclass(frozen=True, slots=True)
class TrackedFaceBox:
    """One predicted face box and its localization evidence."""

    bbox: tuple[float, float, float, float]
    visible_point_ratio: float
    inlier_point_count: int
    source: Literal["tracker", "detection_fused", "gallery_anchor", "bidirectional"]

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        if not np.isfinite(self.bbox).all() or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must contain finite ordered coordinates")
        if not np.isfinite(self.visible_point_ratio) or not 0.0 <= self.visible_point_ratio <= 1.0:
            raise ValueError("visible_point_ratio must be in [0, 1]")
        if (
            isinstance(self.inlier_point_count, bool)
            or not isinstance(self.inlier_point_count, int)
            or self.inlier_point_count < 0
        ):
            raise ValueError("inlier_point_count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class TrackAuthorization:
    """Gallery-derived identity lease attached to one continuous track."""

    track_id: str
    profile_id: str
    gallery_revision: str
    origin_frame_index: int
    last_gallery_approval_frame: int

    def __post_init__(self) -> None:
        for name in ("track_id", "profile_id", "gallery_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("origin_frame_index", "last_gallery_approval_frame"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.last_gallery_approval_frame < self.origin_frame_index:
            raise ValueError("last approval cannot precede track origin")


@dataclass(frozen=True, slots=True)
class TrackFrameDecision:
    """Final temporal decision for one profile on one frame."""

    frame_index: int
    track_id: str | None
    profile_id: str | None
    bbox: tuple[float, float, float, float] | None
    authorized: bool
    bbox_source: str
    visible_point_ratio: float | None
    review_required: bool
    reason_code: str

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if not isinstance(self.authorized, bool) or not isinstance(self.review_required, bool):
            raise TypeError("authorized and review_required must be booleans")
        if not isinstance(self.bbox_source, str) or not self.bbox_source:
            raise ValueError("bbox_source must be non-empty")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("reason_code must be non-empty")
        if self.authorized and (not self.track_id or not self.profile_id or self.bbox is None):
            raise ValueError("authorized decisions require track, profile, and bbox")
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            if not np.isfinite(self.bbox).all() or x2 <= x1 or y2 <= y1:
                raise ValueError("bbox must contain finite ordered coordinates")
        if self.visible_point_ratio is not None and (
            not np.isfinite(self.visible_point_ratio)
            or not 0.0 <= self.visible_point_ratio <= 1.0
        ):
            raise ValueError("visible_point_ratio must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class BboxValidation:
    valid: bool
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be boolean")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("reason_code must be non-empty")
