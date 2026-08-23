"""Tests for pipeline module — skeleton with fake adapters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# ------------------------------------------------------------------ #
# Test suite
# ------------------------------------------------------------------ #


class TestPipelineInit:
    def test_creates_without_side_effects(self, tmp_path: Path):
        from consented_face_redactor.pipeline import RedactionPipeline

        before = set(str(p) for p in Path(".").rglob("*"))  # type: ignore[assignment] (string path)
        cfg = {"effect_mode": "mosaic", "t_confirm": 0.65, "t_keep": 0.55}  # mock config
        RedactionPipeline(cfg)
        after = set(str(p) for p in Path(".").rglob("*"))  # type: ignore[assignment]

        assert before == after


class TestProcessFrame:
    def test_no_faces_returns_unseen(self):
        from consented_face_redactor.pipeline import ProcessResult, TrackState, RedactionPipeline

        cfg = {"effect_mode": "mosaic", "t_confirm": 0.65, "t_keep": 0.55}
        pipe = RedactionPipeline(cfg)  # type: ignore[arg-type] (mock config stub)

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        result = pipe.process_frame(frame, frame_index=0, timestamp=0.0, state=None)

        assert isinstance(result, ProcessResult)
        assert result.is_redacted is False
        assert result.review_required is False
        assert result.track_state == TrackState.UNSEEN

    def test_input_not_mutated(self):
        from consented_face_redactor.pipeline import RedactionPipeline

        cfg = {"effect_mode": "mosaic", "t_confirm": 0.65, "t_keep": 0.55}
        pipe = RedactionPipeline(cfg)  # type: ignore[arg-type]

        original = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        frame_copy = original.copy()

        result = pipe.process_frame(original, frame_index=0, timestamp=0.0, state=None)

        # Input should be identical after processing
        np.testing.assert_array_equal(frame_copy, original)


class TestTrackStatePersistence:
    def test_save_load_roundtrip(self):
        from consented_face_redactor.pipeline import RedactionPipeline

        cfg = {"effect_mode": "mosaic", "t_confirm": 0.65, "t_keep": 0.55}
        pipe = RedactionPipeline(cfg)  # type: ignore[arg-type]

        # Simulate some state changes (normally done internally)
        pipe._track_state = "candidate"  # type: ignore[attr-defined]
        pipe._frame_index = 42

        snapshot = pipe.save_track_state()
        pipe.load_track_state(snapshot)

        assert pipe.current_track_state == "candidate"  # type: ignore[comparison-overlap] (for testing)
        assert pipe._frame_index == 42  # type: ignore[attr-defined]


class TestDeterminism:
    def test_deterministic_output_for_stub(self):
        """For the same stub input, result_frame must be byte-identical."""
        from consented_face_redactor.pipeline import RedactionPipeline

        cfg = {"effect_mode": "mosaic", "t_confirm": 0.65, "t_keep": 0.55}
        pipe1 = RedactionPipeline(cfg)  # type: ignore[arg-type]
        pipe2 = RedactionPipeline(cfg)  # type: ignore[arg-type]

        frame = np.ones((48, 64, 3), dtype=np.uint8) * 128

        r1 = pipe1.process_frame(frame, frame_index=0, timestamp=0.0, state=None)
        r2 = pipe2.process_frame(frame, frame_index=0, timestamp=0.0, state=None)

        np.testing.assert_array_equal(r1.result_frame, r2.result_frame)
