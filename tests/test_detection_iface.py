"""Tests for detection_iface module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing src without install (conftest also patches sys.path)
_HERE = Path(__file__).resolve()
_src = _HERE.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


class TestBoundingBox:
    def test_bbox_properties(self):
        from consented_face_redactor.adapters.detection_iface import BoundingBox

        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=15)
        assert bbox.width == 11
        assert bbox.height == 16
        assert bbox.area == 176

    def test_bbox_can_have_zero_area(self):
        from consented_face_redactor.adapters.detection_iface import BoundingBox

        # x2 < x1 or y2 < y1 can technically happen in edge cases
        bbox = BoundingBox(x1=5, y1=5, x2=0, y2=0)
        assert bbox.width == -4  # negative but valid object
        assert bbox.area == 0    # width clamped to max(0,w)


class TestFaceDetection:
    def test_face_detection_dataclass(self):
        from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection
        import numpy as np

        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=15)
        lm = np.zeros((5, 2), dtype=np.float32)
        fd = FaceDetection(bbox=bbox, landmarks=lm, confidence=0.95)

        assert fd.confidence == 0.95
        assert isinstance(fd.landmarks, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
