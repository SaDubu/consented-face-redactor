"""Tests for the bounded pipeline skeleton and track-state persistence."""

from __future__ import annotations

import numpy as np
import pytest

from consented_face_redactor.pipeline import ProcessResult, RedactionPipeline, TrackState


def _pipeline() -> RedactionPipeline:
    return RedactionPipeline(
        {"effect_mode": "mosaic", "t_confirm": 0.65, "t_keep": 0.55}
    )


def _frame() -> np.ndarray:
    return np.zeros((48, 64, 3), dtype=np.uint8)


class TestProcessFrame:
    def test_no_face_returns_unseen_without_mutating_input(self):
        pipeline = _pipeline()
        frame = np.random.default_rng(7).integers(
            0, 255, (48, 64, 3), dtype=np.uint8
        )
        original = frame.copy()

        result = pipeline.process_frame(frame, frame_index=0, timestamp=0.0, state=None)

        assert isinstance(result, ProcessResult)
        assert result.is_redacted is False
        assert result.review_required is False
        assert result.track_state is TrackState.UNSEEN
        np.testing.assert_array_equal(frame, original)
        assert result.result_frame is not frame

    def test_confirmed_track_without_face_transitions_to_lost(self):
        pipeline = _pipeline()
        pipeline.load_track_state({"track_state": "confirmed", "frame_index": 3})

        result = pipeline.process_frame(_frame(), 4, 0.2, state=None)

        assert result.track_state is TrackState.LOST
        assert result.review_required is True
        assert pipeline.current_track_state is TrackState.LOST

    def test_frame_index_reversal_expires_track(self):
        pipeline = _pipeline()
        pipeline.process_frame(_frame(), 5, 0.2, state=None)

        result = pipeline.process_frame(_frame(), 4, 0.3, state=None)

        assert result.track_state is TrackState.EXPIRED
        assert result.review_required is True

    @pytest.mark.parametrize(
        "frame",
        [
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.float32),
            np.zeros((0, 4, 3), dtype=np.uint8),
        ],
    )
    def test_rejects_invalid_frame(self, frame):
        with pytest.raises(ValueError):
            _pipeline().process_frame(frame, 0, 0.0, None)

    @pytest.mark.parametrize("frame_index", [-1, True, 1.5])
    def test_rejects_invalid_frame_index(self, frame_index):
        with pytest.raises(ValueError):
            _pipeline().process_frame(_frame(), frame_index, 0.0, None)

    @pytest.mark.parametrize("timestamp", [-0.1, float("nan"), float("inf"), True])
    def test_rejects_invalid_timestamp(self, timestamp):
        with pytest.raises((TypeError, ValueError)):
            _pipeline().process_frame(_frame(), 0, timestamp, None)


class TestTrackStatePersistence:
    def test_save_and_load_round_trip_uses_controlled_string(self):
        pipeline = _pipeline()
        pipeline.load_track_state({"track_state": "candidate", "frame_index": 42})

        snapshot = pipeline.save_track_state()

        assert snapshot == {"track_state": "candidate", "frame_index": 42}
        assert pipeline.current_track_state is TrackState.CANDIDATE

    @pytest.mark.parametrize(
        "snapshot",
        [
            None,
            {},
            {"track_state": "unknown", "frame_index": 0},
            {"track_state": "unseen", "frame_index": True},
            {"track_state": "unseen", "frame_index": -2},
            {"track_state": "unseen", "frame_index": 0, "extra": 1},
        ],
    )
    def test_rejects_invalid_snapshot_without_partial_update(self, snapshot):
        pipeline = _pipeline()
        pipeline.load_track_state({"track_state": "candidate", "frame_index": 7})
        with pytest.raises(ValueError):
            pipeline.load_track_state(snapshot)
        assert pipeline.save_track_state() == {
            "track_state": "candidate",
            "frame_index": 7,
        }


def test_stub_output_is_deterministic():
    frame = np.ones((48, 64, 3), dtype=np.uint8) * 128
    first = _pipeline().process_frame(frame, 0, 0.0, None)
    second = _pipeline().process_frame(frame, 0, 0.0, None)
    np.testing.assert_array_equal(first.result_frame, second.result_frame)
