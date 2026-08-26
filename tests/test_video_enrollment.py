"""Tests for target-only video enrollment and reference coverage."""

from __future__ import annotations

import numpy as np
import pytest

from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection
from consented_face_redactor.gallery import LocalGallery
from consented_face_redactor.media import FakeFrameReader
from consented_face_redactor.video_enrollment import (
    EnrollmentCandidate,
    EnrollmentSkip,
    VideoEnrollmentOptions,
    VideoEnrollmentService,
    extract_enrollment_candidate,
    select_diverse_references,
)


def _face(width: int = 40, height: int = 40) -> FaceDetection:
    return FaceDetection(
        BoundingBox(5, 5, 5 + width, 5 + height),
        np.zeros((5, 2), dtype=np.float32), 0.95,
    )


def _candidate(index: int, values: list[float]) -> EnrollmentCandidate:
    return EnrollmentCandidate(index, float(index), np.asarray(values, dtype=np.float32), 0.95)


class _Detector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame):
        return self._detections


class _Embedder:
    def __init__(self, vector: np.ndarray, raises: bool = False):
        self._vector = vector
        self._raises = raises

    def embed(self, frame, detection):
        if self._raises:
            raise ValueError("embedding failure")
        return self._vector, 1


@pytest.mark.parametrize(
    ("detections", "reason"), [([], "no_face"), ([_face(), _face()], "multiple_faces"), ([_face(10, 10)], "face_too_small")],
)
def test_candidate_reports_non_enrollment_reasons(detections, reason):
    result = extract_enrollment_candidate(
        np.zeros((80, 80, 3), dtype=np.uint8), frame_index=3, timestamp_s=0.1,
        detector=_Detector(detections), embedder=_Embedder(np.array([1.0, 0.0], dtype=np.float32)),
        options=VideoEnrollmentOptions(min_face_width=32, min_face_height=32),
    )
    assert isinstance(result, EnrollmentSkip)
    assert result.reason_code == reason


def test_candidate_returns_detached_normalized_embedding():
    result = extract_enrollment_candidate(
        np.zeros((80, 80, 3), dtype=np.uint8), frame_index=3, timestamp_s=0.1,
        detector=_Detector([_face()]), embedder=_Embedder(np.array([3.0, 4.0], dtype=np.float32)),
        options=VideoEnrollmentOptions(),
    )
    assert isinstance(result, EnrollmentCandidate)
    np.testing.assert_allclose(result.embedding, [0.6, 0.8])
    assert result.embedding.flags.writeable is False


def test_diversity_selection_deduplicates_and_marks_extreme_outlier_for_review():
    selected, review, duplicates = select_diverse_references(
        [_candidate(0, [1.0, 0.0]), _candidate(1, [0.999, 0.001]), _candidate(2, [0.0, 1.0]), _candidate(3, [-1.0, -1.0])],
        options=VideoEnrollmentOptions(duplicate_similarity=0.99, max_references=8),
    )
    assert [item.frame_index for item in selected] == [0, 2]
    assert [item.frame_index for item in review] == [3]
    assert duplicates == 1


def test_service_collects_without_retaining_frames_and_enrolls_many_references():
    source = FakeFrameReader(n_frames=5, width=80, height=80, fps=10)
    source.open()
    vectors = iter([
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.7, 0.7], dtype=np.float32),
    ])

    class _SequencedEmbedder:
        def embed(self, frame, detection):
            return next(vectors), 1

    service = VideoEnrollmentService(
        detector=_Detector([_face()]), embedder=_SequencedEmbedder(),
        options=VideoEnrollmentOptions(sample_every_n_frames=2, max_references=3, duplicate_similarity=0.999),
    )
    candidates, report = service.collect(source)
    source.close()
    selected, report = service.select(candidates, report)
    gallery = LocalGallery()
    profile_id = gallery.enroll_many([item.embedding.copy() for item in selected])

    assert report.sampled_frame_count == 3
    assert report.selected_reference_count == 2
    assert gallery.to_dict()["profiles"][profile_id]["v_count"] == 2


def test_selection_quarantines_smaller_disconnected_embedding_island():
    main = [
        _candidate(0, [1.0, 0.0, 0.0]),
        _candidate(1, [0.9, 0.4, 0.0]),
        _candidate(2, [0.7, 0.7, 0.0]),
        _candidate(3, [0.4, 0.9, 0.0]),
    ]
    false_island = [
        _candidate(10, [0.0, 0.0, 1.0]),
        _candidate(11, [0.0, 0.1, 0.995]),
    ]

    selected, review, _ = select_diverse_references(
        main + false_island,
        options=VideoEnrollmentOptions(
            duplicate_similarity=0.9999,
            minimum_cluster_similarity=0.45,
        ),
    )

    assert [item.frame_index for item in selected] == [0, 1, 2, 3]
    assert [item.frame_index for item in review] == [10, 11]
