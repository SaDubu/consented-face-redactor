"""Robust geometry helpers for face point tracks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .types import BboxValidation, SimilarityTransform


def seed_face_points(
    bbox: tuple[float, float, float, float],
    landmarks: np.ndarray,
    *,
    grid_side: int = 4,
) -> np.ndarray:
    """Return five landmarks plus an inset regular grid inside a face box."""
    x1, y1, x2, y2 = map(float, bbox)
    if not np.isfinite((x1, y1, x2, y2)).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must contain finite ordered coordinates")
    if isinstance(grid_side, bool) or not isinstance(grid_side, int) or not 2 <= grid_side <= 16:
        raise ValueError("grid_side must be an integer in [2, 16]")
    landmark_array = np.asarray(landmarks, dtype=np.float32)
    if landmark_array.shape != (5, 2) or not np.isfinite(landmark_array).all():
        raise ValueError("landmarks must be a finite array with shape (5, 2)")
    xs = np.linspace(x1 + 0.15 * (x2 - x1), x2 - 0.15 * (x2 - x1), grid_side)
    ys = np.linspace(y1 + 0.15 * (y2 - y1), y2 - 0.15 * (y2 - y1), grid_side)
    grid = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32)
    points = np.concatenate((landmark_array, grid), axis=0)
    points.setflags(write=False)
    return points


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    denominator = float(np.sum(centered_source * centered_source))
    if denominator <= np.finfo(np.float32).eps:
        raise ValueError("source points do not span a usable shape")
    covariance = centered_source.T @ centered_target
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    scale = float(np.sum(singular_values) / denominator)
    translation = target_center - scale * (rotation @ source_center)
    return rotation, scale, translation


def estimate_similarity_transform(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    visibility: np.ndarray,
    *,
    minimum_visibility: float = 0.5,
) -> SimilarityTransform:
    """Estimate a similarity transform and remove large residual outliers once."""
    previous = np.asarray(previous_points, dtype=np.float64)
    current = np.asarray(current_points, dtype=np.float64)
    visible = np.asarray(visibility, dtype=np.float64)
    if previous.ndim != 2 or previous.shape[1:] != (2,) or current.shape != previous.shape:
        raise ValueError("point arrays must have matching shape (N, 2)")
    if visible.shape != (len(previous),):
        raise ValueError("visibility must have shape (N,)")
    if not np.isfinite(previous).all() or not np.isfinite(current).all() or not np.isfinite(visible).all():
        raise ValueError("point inputs must be finite")
    mask = visible >= minimum_visibility
    if int(mask.sum()) < 2:
        raise ValueError("at least two visible points are required")

    source = previous[mask]
    target = current[mask]
    rotation, scale, translation = _fit_similarity(source, target)
    predicted = scale * (source @ rotation.T) + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))
    cutoff = max(1.0, median_residual + 3.0 * max(mad, 0.25))
    inliers = residuals <= cutoff
    if int(inliers.sum()) >= 2 and not np.all(inliers):
        source = source[inliers]
        target = target[inliers]
        rotation, scale, translation = _fit_similarity(source, target)
    angle = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return SimilarityTransform(
        scale=scale,
        rotation_radians=angle,
        translation_x=float(translation[0]),
        translation_y=float(translation[1]),
        inlier_count=len(source),
    )


def transform_points(points: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (2,) or not np.isfinite(array).all():
        raise ValueError("points must be a finite array with shape (N, 2)")
    cosine = math.cos(transform.rotation_radians)
    sine = math.sin(transform.rotation_radians)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    result = transform.scale * (array @ rotation.T)
    result += np.asarray((transform.translation_x, transform.translation_y))
    return result.astype(np.float32)


def transform_bbox(
    bbox: tuple[float, float, float, float],
    transform: SimilarityTransform,
) -> tuple[float, float, float, float]:
    """Transform all four corners and return their axis-aligned enclosure."""
    x1, y1, x2, y2 = map(float, bbox)
    if not np.isfinite((x1, y1, x2, y2)).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must contain finite ordered coordinates")
    corners = np.asarray(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=np.float32)
    moved = transform_points(corners, transform)
    return (
        float(np.min(moved[:, 0])),
        float(np.min(moved[:, 1])),
        float(np.max(moved[:, 0])),
        float(np.max(moved[:, 1])),
    )


def validate_tracked_bbox(
    previous_bbox: Sequence[float],
    current_bbox: Sequence[float],
    *,
    frame_shape: tuple[int, ...],
    visible_point_ratio: float,
    minimum_visible_point_ratio: float = 0.6,
    maximum_scale_ratio: float = 1.25,
    maximum_center_shift_ratio: float = 0.35,
) -> BboxValidation:
    """Reject impossible or weak face-box propagation."""
    if len(previous_bbox) != 4 or len(current_bbox) != 4:
        return BboxValidation(False, "malformed_bbox")
    previous = tuple(map(float, previous_bbox))
    current = tuple(map(float, current_bbox))
    if not np.isfinite(previous + current).all():
        return BboxValidation(False, "nonfinite_bbox")
    if previous[2] <= previous[0] or previous[3] <= previous[1] or current[2] <= current[0] or current[3] <= current[1]:
        return BboxValidation(False, "degenerate_bbox")
    if not np.isfinite(visible_point_ratio) or visible_point_ratio < minimum_visible_point_ratio:
        return BboxValidation(False, "tracker_visibility_insufficient")
    previous_area = (previous[2] - previous[0]) * (previous[3] - previous[1])
    current_area = (current[2] - current[0]) * (current[3] - current[1])
    scale_ratio = math.sqrt(current_area / previous_area)
    if not 1.0 / maximum_scale_ratio <= scale_ratio <= maximum_scale_ratio:
        return BboxValidation(False, "track_scale_change_exceeded")
    if len(frame_shape) < 2 or frame_shape[0] < 1 or frame_shape[1] < 1:
        return BboxValidation(False, "invalid_frame_shape")
    previous_center = np.asarray(((previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2))
    current_center = np.asarray(((current[0] + current[2]) / 2, (current[1] + current[3]) / 2))
    diagonal = math.hypot(float(frame_shape[1]), float(frame_shape[0]))
    if float(np.linalg.norm(current_center - previous_center)) / diagonal > maximum_center_shift_ratio:
        return BboxValidation(False, "track_center_shift_exceeded")
    if current[2] <= 0 or current[3] <= 0 or current[0] >= frame_shape[1] or current[1] >= frame_shape[0]:
        return BboxValidation(False, "track_outside_frame")
    return BboxValidation(True, "track_geometry_valid")
