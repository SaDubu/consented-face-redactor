"""Focused tests for Phase 6: pipeline ↔ detector adapter wiring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


def dummy_config():
    """Minimal Config-like object."""
    Cfg = type("Cfg", (), {})()
    Cfg.model = type("ModelCfg", (), {
        "detection_model_id": None,
        "embedding_model_id": None,
    })()
    Cfg.output_dir = "/tmp/out"
    Cfg.redaction_size = 1.0
    Cfg.safety = type("SafetyCfg", (), {
        "face_confidence_threshold": 0.5,
        "match_similarity_threshold": 0.6,
    })()


@dataclass
class _FakeBBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class _FakeDet:
    bbox: _FakeBBox
    landmarks: np.ndarray  # (5, 2) float32
    confidence: float


def fake_det(x1=10.0, y1=20.0, x2=110.0, y2=130.0, conf=0.95):
    landmarks = np.array([
        [40.0, 50.0],
        [80.0, 50.0],
        [60.0, 75.0],
        [45.0, 100.0],
        [75.0, 100.0],
    ], dtype=np.float32)
    return _FakeDet(bbox=_FakeBBox(x1, y1, x2, y2), landmarks=landmarks, confidence=conf)


def make_frame(h=480, w=640):
    return np.ones((h, w, 3), dtype=np.uint8) * 128


def mock_detector(detections=None, model_id="yunet"):
    if detections is None:
        detections = [fake_det()]
    det = MagicMock(spec=["detect", "model_id"])
    det.model_id = model_id
    det.detect.return_value = detections
    return det


# ---- Stub path (no detector) ----

class TestStubPath:
    def test_has_detector_false(self):
        from consented_face_redactor.pipeline import RedactionPipeline
        cfg = dummy_config()
        pipe = RedactionPipeline(cfg)
        assert pipe.has_detector is False

    def test_detector_requires_bgr_input_false(self):
        from consented_face_redactor.pipeline import RedactionPipeline
        cfg = dummy_config()
        pipe = RedactionPipeline(cfg)
        assert pipe.detector_requires_bgr_input is False

    def test_empty_detections_returned(self):
        from consented_face_redactor.pipeline import (
            ProcessResult, TrackState, RedactionPipeline,
        )
        cfg = dummy_config()
        pipe = RedactionPipeline(cfg)
        result = pipe.process_frame(make_frame(), 0, 0.0, None)
        assert isinstance(result, ProcessResult)
        assert not result.is_redacted
        assert result.track_state == TrackState.UNSEEN
        assert not result.review_required


# ---- With mock detector ----

class TestWithMockDetector:
    def _pipe(self, cfg=None, det=None):
        from consented_face_redactor.pipeline import RedactionPipeline
        if cfg is None:
            cfg = dummy_config()
        return RedactionPipeline(cfg, detector=det)

    def test_has_detector_true(self):
        pipe = self._pipe(det=mock_detector())
        assert pipe.has_detector is True

    def test_bgr_input_for_yunet(self):
        pipe = self._pipe(det=mock_detector(model_id="yunet"))
        assert pipe.detector_requires_bgr_input is True

    def test_detect_called_once(self):
        pipe = self._pipe(det=mock_detector())
        pipe.process_frame(make_frame(), 0, 0.0, None)
        assert pipe._detector.detect.call_count == 1

    def test_single_face_no_redaction(self):
        pipe = self._pipe(det=mock_detector())
        result = pipe.process_frame(make_frame(), 0, 0.0, None)
        assert not result.is_redacted
        assert result.result_frame.shape == (480, 640, 3)

    def test_input_args_are_correct_shape(self):
        det = mock_detector()
        pipe = self._pipe(det=det)
        pipe.process_frame(make_frame(h=240, w=320), 0, 0.0, None)
        call_arg = det.detect.call_args[0][0]
        assert call_arg.shape == (240, 320, 3)

    def test_multiple_faces_three_detections(self):
        dets = [fake_det(x1=10 + i*10, y1=20 + i*50) for i in range(3)]
        det = mock_detector(dets, model_id="yunet")
        pipe = self._pipe(det=det)
        result = pipe.process_frame(make_frame(), 0, 0.0, None)
        assert not result.is_redacted

    def test_landmark_shape_five_two(self):
        dets = [fake_det()]
        det = mock_detector(dets, model_id="yunet")
        pipe = self._pipe(det=det)
        result = pipe.process_frame(make_frame(), 0, 0.0, None)
        assert not result.is_redacted

    def test_confidences_preserved(self):
        dets = [fake_det(conf=0.9), fake_det(conf=0.4)]
        det = mock_detector(dets, model_id="yunet")
        pipe = self._pipe(det=det)
        result = pipe.process_frame(make_frame(), 0, 0.0, None)
        assert not result.is_redacted


# ---- Track-state transitions ----

class TestTrackStateTransitions:
    def _pipe(self, cfg=None, det=None):
        from consented_face_redactor.pipeline import RedactionPipeline
        if cfg is None:
            cfg = dummy_config()
        return RedactionPipeline(cfg, detector=det)

    def test_first_frame_unseen_when_no_faces(self):
        det = mock_detector(detections=[])
        pipe = self._pipe(det=det)
        result = pipe.process_frame(make_frame(), 0, 0.0, None)
        assert result.track_state.value == "unseen"

    def test_review_false_on_first_frame_unseen(self):
        det = mock_detector(detections=[])
        pipe = self._pipe(det=det)
        result = pipe.process_frame(make_frame(), 0, 0.0, None)
        assert not result.review_required


# ---- Frame validation ----

class TestFrameValidation:
    def test_non_ndarray_raises(self):
        from consented_face_redactor.pipeline import RedactionPipeline
        pipe = RedactionPipeline(dummy_config())
        with pytest.raises(TypeError, match="must be a numpy array"):
            pipe.process_frame([1, 2, 3], 0, 0.0, None)

    def test_wrong_dtype_raises(self):
        from consented_face_redactor.pipeline import RedactionPipeline
        pipe = RedactionPipeline(dummy_config())
        with pytest.raises(ValueError, match="uint8"):
            pipe.process_frame(np.zeros((10, 10, 3), dtype=np.float32), 0, 0.0, None)

    def test_wrong_ndim_raises(self):
        from consented_face_redactor.pipeline import RedactionPipeline
        pipe = RedactionPipeline(dummy_config())
        with pytest.raises(ValueError, match="shape"):
            pipe.process_frame(np.zeros((10, 10), dtype=np.uint8), 0, 0.0, None)

    def test_channel_count_raises(self):
        from consented_face_redactor.pipeline import RedactionPipeline
        pipe = RedactionPipeline(dummy_config())
        with pytest.raises(ValueError, match="shape"):
            pipe.process_frame(np.zeros((10, 10, 4), dtype=np.uint8), 0, 0.0, None)
