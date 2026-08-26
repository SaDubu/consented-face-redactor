"""Geometry tests for point-seeded face tracking."""

from __future__ import annotations

import math

import numpy as np

from consented_face_redactor.tracking.geometry import (
    estimate_similarity_transform,
    seed_face_points,
    transform_bbox,
    validate_tracked_bbox,
)


def test_seed_face_points_combines_landmarks_and_grid() -> None:
    landmarks = np.asarray(((20, 20), (40, 20), (30, 30), (22, 40), (38, 40)), dtype=np.float32)
    points = seed_face_points((10, 10, 50, 50), landmarks, grid_side=4)

    assert points.shape == (21, 2)
    assert np.array_equal(points[:5], landmarks)
    assert points.flags.writeable is False


def test_similarity_transform_recovers_translation_scale_and_rotation() -> None:
    source = np.asarray(((0, 0), (10, 0), (10, 10), (0, 10), (5, 5)), dtype=np.float32)
    angle = math.radians(15)
    rotation = np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    target = 1.1 * (source @ rotation.T) + (12, -4)
    target[-1] += (40, 40)  # one large outlier
    transform = estimate_similarity_transform(source, target, np.ones(len(source)))

    assert abs(transform.scale - 1.1) < 0.05
    assert abs(transform.rotation_radians - angle) < 0.05
    assert transform.inlier_count == 4


def test_transform_bbox_returns_rotated_axis_aligned_enclosure() -> None:
    source = np.asarray(((0, 0), (10, 0), (10, 10), (0, 10)), dtype=np.float32)
    target = source + (5, 7)
    transform = estimate_similarity_transform(source, target, np.ones(4))

    assert np.allclose(transform_bbox((0, 0, 10, 10), transform), (5, 7, 15, 17))


def test_bbox_validation_rejects_visibility_and_scale_jumps() -> None:
    assert validate_tracked_bbox(
        (10, 10, 50, 50),
        (12, 11, 52, 51),
        frame_shape=(100, 100, 3),
        visible_point_ratio=0.9,
    ).valid
    assert validate_tracked_bbox(
        (10, 10, 50, 50),
        (12, 11, 52, 51),
        frame_shape=(100, 100, 3),
        visible_point_ratio=0.2,
    ).reason_code == "tracker_visibility_insufficient"
    assert validate_tracked_bbox(
        (10, 10, 50, 50),
        (10, 10, 90, 90),
        frame_shape=(100, 100, 3),
        visible_point_ratio=0.9,
    ).reason_code == "track_scale_change_exceeded"
