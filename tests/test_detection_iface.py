"""Tests for the vendor-neutral face detection contract."""

from __future__ import annotations

import numpy as np
import pytest

from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection


class TestBoundingBox:
    def test_uses_half_open_coordinates(self):
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=15)
        assert bbox.width == 10
        assert bbox.height == 15
        assert bbox.area == 150

    @pytest.mark.parametrize(
        "coordinates",
        [
            (5, 5, 5, 9),
            (5, 5, 9, 5),
            (5, 5, 0, 9),
            (5, 5, 9, 0),
        ],
    )
    def test_rejects_degenerate_or_reversed_box(self, coordinates):
        with pytest.raises(ValueError, match="positive width"):
            BoundingBox(*coordinates)

    @pytest.mark.parametrize("value", [True, 1.5, "1"])
    def test_rejects_non_integer_coordinates(self, value):
        with pytest.raises(TypeError, match="coordinates"):
            BoundingBox(value, 0, 10, 10)


class TestFaceDetection:
    def test_copies_and_freezes_landmarks(self):
        landmarks = np.arange(10, dtype=np.float32).reshape(5, 2)
        detection = FaceDetection(
            bbox=BoundingBox(0, 0, 10, 15),
            landmarks=landmarks,
            confidence=0.95,
        )

        landmarks[0, 0] = 99
        assert detection.landmarks[0, 0] == 0
        assert detection.landmarks.flags.writeable is False
        with pytest.raises(ValueError):
            detection.landmarks[0, 0] = 2

    def test_reconstructs_yunet_row_for_alignment(self):
        landmarks = np.arange(10, dtype=np.float32).reshape(5, 2)
        detection = FaceDetection(BoundingBox(2, 3, 12, 18), landmarks, 0.75)

        row = detection.as_yunet_row()

        np.testing.assert_array_equal(row[:4], [2, 3, 10, 15])
        np.testing.assert_array_equal(row[4:14], np.arange(10, dtype=np.float32))
        assert row[14] == pytest.approx(0.75)

    @pytest.mark.parametrize(
        "landmarks",
        [
            np.zeros((4, 2), dtype=np.float32),
            np.full((5, 2), np.nan, dtype=np.float32),
        ],
    )
    def test_rejects_invalid_landmarks(self, landmarks):
        with pytest.raises(ValueError, match="landmarks"):
            FaceDetection(BoundingBox(0, 0, 10, 10), landmarks, 0.9)

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
    def test_rejects_invalid_confidence(self, confidence):
        with pytest.raises(ValueError, match="confidence"):
            FaceDetection(
                BoundingBox(0, 0, 10, 10),
                np.zeros((5, 2), dtype=np.float32),
                confidence,
            )

    def test_rejects_non_bbox_object(self):
        with pytest.raises(TypeError, match="BoundingBox"):
            FaceDetection(object(), np.zeros((5, 2), dtype=np.float32), 0.9)
