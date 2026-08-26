"""Regression tests preventing one approval from redacting every face."""

from __future__ import annotations

import numpy as np

from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection
from consented_face_redactor.config import Config
from consented_face_redactor.gallery_approval import GalleryApproval
from consented_face_redactor.pipeline import RedactionPipeline, TrackState


def _face(x1: int, x2: int) -> FaceDetection:
    return FaceDetection(
        BoundingBox(x1, 10, x2, 40),
        np.array([[x1 + 5, 18], [x2 - 5, 18], [x1 + 15, 25], [x1 + 8, 34], [x2 - 8, 34]], dtype=np.float32),
        0.99,
    )


class _Detector:
    model_id = "yunet"

    def detect(self, frame):
        return [_face(5, 35), _face(55, 85)]


class _FaceByFaceGallery:
    def evaluate(self, frame, detection):
        if detection.bbox.x1 == 5:
            return GalleryApproval(True, "prof-00000000", 0.99, "test_consent", "v1")
        return GalleryApproval.denied("profile_not_approved", profile_id="prof-00000001", similarity=0.99, gallery_revision="v1")


def test_only_explicitly_approved_face_roi_is_redacted():
    frame = np.random.default_rng(13).integers(0, 255, (60, 100, 3), dtype=np.uint8)
    original = frame.copy()
    pipeline = RedactionPipeline(
        Config(effect_mode="mosaic"), detector=_Detector(), gallery=_FaceByFaceGallery()
    )

    result = pipeline.process_frame(frame, 0, 0.0, None)

    assert result.is_redacted is True
    assert result.track_state is TrackState.CONFIRMED
    assert result.review_required is True
    assert len(pipeline.last_frame_approvals) == 2
    assert len(pipeline.last_frame_decisions) == 2
    assert pipeline.last_frame_decisions[0].bbox == (5.0, 10.0, 35.0, 40.0)
    assert pipeline.last_frame_approvals[0].approved is True
    assert pipeline.last_frame_approvals[1].approved is False
    np.testing.assert_array_equal(result.result_frame[10:40, 55:85], original[10:40, 55:85])
    assert not np.array_equal(result.result_frame[10:40, 5:35], original[10:40, 5:35])
