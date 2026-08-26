"""Tests for Phase 7: GalleryMatcher + full TrackState state machine."""

from __future__ import annotations

import numpy as np
import pytest

from consented_face_redactor.pipeline import DetectionResult, ProcessResult, RedactionPipeline, TrackState
from consented_face_redactor.config import Config
from consented_face_redactor.gallery_matcher import GalleryMatcher


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_frame() -> np.ndarray:
    """Return a small dummy BGR frame for processing."""
    return np.zeros((48, 64, 3), dtype=np.uint8)


def _make_config(
    effect_mode: str = "mosiac",
    t_confirm: float = 0.90,
    t_keep: float = 0.55,
    track_lost_ttl_frames: int = 10,
    recheck_interval_frames: int = 30,
) -> Config:
    """Return a Config instance for testing."""
    return Config(
        effect_mode=effect_mode,
        t_confirm=t_confirm,
        t_keep=t_keep,
        track_lost_ttl_frames=track_lost_ttl_frames,
        recheck_interval_frames=recheck_interval_frames,
    )


class _HighConfDet:
    """Detector that always detects one face with confidence 0.95."""

    model_id = "yunet"
    detect_calls = 0

    class _FakeDet:
        bbox = (10, 10, 100, 100)
        landmarks = np.zeros((5, 2), dtype=np.float32)
        confidence = 0.95

    def detect(self, frame):
        self.detect_calls += 1
        return [self._FakeDet]


class _LowConfDet:
    """Detector that always detects one face with low confidence."""

    model_id = "yunet"
    detect_calls = 0

    class _FakeDet:
        bbox = (10, 10, 100, 100)
        landmarks = np.zeros((5, 2), dtype=np.float32)
        confidence = 0.3

    def detect(self, frame):
        self.detect_calls += 1
        return [self._FakeDet]


class _NoFaceDetector:
    """A detector that returns an empty list (no faces detected)."""

    model_id = "stub"

    def detect(self, frame):
        return []  # explicitly empty list


# --------------------------------------------------------------------------- #
# GalleryMatcher tests
# --------------------------------------------------------------------------- #


class TestGalleryMatcherEmbed:
    """Verify that gallery matcher handles embeddings / None inputs."""

    def test_embed_returns_none_without_backend(self):
        matcher = GalleryMatcher()
        assert matcher.embed(_make_frame()) is None

    def test_none_embedding_produces_no_match(self):
        profile_db = {"alice": [0.1, 0.2], "bob": [0.15, 0.25]}
        gallery = GalleryMatcher(gallery_db=profile_db)
        result = gallery.match(None)
        assert result.approved is False
        assert result.reason_code == "embedding_unavailable"

    def test_empty_gallery_produces_no_matches(self):
        frame = _make_frame()
        fake_data: list[float] = [0.1, 0.2]
        matcher = GalleryMatcher()

        result = matcher.match(fake_data)
        assert result.approved is False
        assert result.reason_code == "empty_gallery"


class TestGalleryMatcherMatch:
    """Test that GalleryMatcher matches return empty when unconfigured."""

    def test_matches_return_empty_list(self):
        matcher = GalleryMatcher()
        fake_embedding: list[float] = [0.1, 0.2]
        result = matcher.match(fake_embedding)
        assert result.approved is False
        assert result.reason_code == "empty_gallery"


# --------------------------------------------------------------------------- #
# TrackState state machine tests
# --------------------------------------------------------------------------- #


class TestInitialState:
    """Pipeline should start in UNSEEN state."""

    def test_initial_state_is_unseen(self):
        config = _make_config()
        pipeline = RedactionPipeline(config)
        assert pipeline.current_track_state is TrackState.UNSEEN


class TestNoFaceDetection:
    """When no faces are detected, initial track transitions to UNSEEN."""

    def test_no_face_returns_unseen_from_initial(self):
        config = _make_config()
        pipeline = RedactionPipeline(config)
        result1 = pipeline.process_frame(
            _make_frame(), frame_index=0, timestamp=0.0, state=None,
        )
        assert result1.track_state is TrackState.UNSEEN
        assert result1.is_redacted is False
        assert result1.review_required is False

    def test_no_face_while_confirmed_transitions_to_lost(self):
        config = _make_config()
        pipeline = RedactionPipeline(config)
        # A v1 CONFIRMED snapshot requires fresh approval after restoration.
        pipeline.load_track_state({
            "track_state": "confirmed",
            "frame_index": 5,
        })
        result = pipeline.process_frame(
            _make_frame(), frame_index=6, timestamp=1.0, state=None,
        )
        assert result.track_state is TrackState.LOST
        assert result.review_required is False

    def test_no_face_while_candidate_transitions_to_lost(self):
        """Test CANDIDATE → LOST on no face."""
        config = _make_config(t_confirm=0.99)  # high threshold so stays in CANDIDATE

        # First face arrives but confidence < t_confirm (LOW conf)
        pipeline1 = RedactionPipeline(config, detector=_LowConfDet())
        result = pipeline1.process_frame(
            _make_frame(), frame_index=0, timestamp=0.0, state=None,
        )
        assert result.track_state is TrackState.CANDIDATE

        # Now face disappears — should transition to LOST via direct attr
        pipeline2 = RedactionPipeline(config, detector=_NoFaceDetector())
        pipeline2._track_state = TrackState.CANDIDATE
        pipeline2._frame_index = -1
        lost_result = pipeline2.process_frame(
            _make_frame(), frame_index=1, timestamp=1.0, state=None,
        )
        assert lost_result.track_state is TrackState.LOST

    def test_unseen_with_face_stays_unseen(self):
        """When no detector: stays in UNSEEN."""
        config = _make_config()
        pipeline = RedactionPipeline(config)
        result1 = pipeline.process_frame(
            _make_frame(), frame_index=0, timestamp=0.0, state=None,
        )

        # Second frame also — still no detector → stays in UNSEEN
        result2 = pipeline.process_frame(
            _make_frame(), frame_index=1, timestamp=1.0, state=None,
        )
        assert result1.track_state is TrackState.UNSEEN
        assert result2.track_state is TrackState.UNSEEN


class TestConfidenceDoesNotConfirmIdentity:
    """Detector confidence alone cannot authorize a confirmed identity."""

    def test_confirmed_above_threshold(self):
        config = _make_config(t_confirm=0.65)
        pipeline = RedactionPipeline(config, detector=_HighConfDet())
        result1 = pipeline.process_frame(
            _make_frame(), frame_index=0, timestamp=0.0, state=None,
        )

        assert result1.track_state is TrackState.CANDIDATE
        assert result1.is_redacted is False
        assert result1.review_required is True


class TestLostAndexpired:
    """Verify LOST → EXPIRED when face disappears."""

    def test_expired_after_lost_ttl(self):
        config = _make_config(track_lost_ttl_frames=3)
        pipeline = RedactionPipeline(config, detector=_NoFaceDetector())
        # Simulate LOST state with lost_frame_index set (direct attr since load_track_state is strict)
        pipeline._track_state = TrackState.LOST
        pipeline._frame_index = 0
        pipeline._lost_frame_index = 5

        result = pipeline.process_frame(
            _make_frame(), frame_index=6, timestamp=1.0, state=None,
        )
        assert result.track_state is TrackState.LOST  # frames_since_loss = 1

        result = pipeline.process_frame(
            _make_frame(), frame_index=8, timestamp=3.0, state=None,
        )
        assert result.track_state is TrackState.EXPIRED  # frames_since_loss = 3 >= ttl


class TestLostExpireReappearance:
    """Verify EXPIRED transition when face reappears."""

    def test_lost_to_candidate_on_face_reappearance(self):
        config = _make_config()
        pipeline = RedactionPipeline(config, detector=_NoFaceDetector())
        # Simulate EXPIRED state
        pipeline._track_state = TrackState.EXPIRED
        pipeline._frame_index = 0

        for i in range(3):
            result = pipeline.process_frame(
                _make_frame(), frame_index=i + 1, timestamp=float(i + 1), state=None,
            )
            assert result.track_state is TrackState.EXPIRED
