"""Contract tests for model-independent point tracking types."""

from __future__ import annotations

import numpy as np
import pytest

from consented_face_redactor.tracking import PointTrackResult, PointTracker


class _FakePointTracker:
    model_id = "fake-point-tracker"

    def __init__(self) -> None:
        self.points: np.ndarray | None = None

    def reset(self) -> None:
        self.points = None

    def initialize(self, frame, *, frame_index, query_points):
        self.points = np.asarray(query_points, dtype=np.float32).copy()
        return PointTrackResult(frame_index, self.points, np.ones(len(self.points)), "fake-v1")

    def update(self, frame, *, frame_index):
        assert self.points is not None
        self.points = self.points + (1.0, 2.0)
        return PointTrackResult(frame_index, self.points, np.ones(len(self.points)), "fake-v1")


def test_fake_tracker_satisfies_runtime_protocol() -> None:
    assert isinstance(_FakePointTracker(), PointTracker)


def test_point_result_copies_and_freezes_arrays() -> None:
    points = np.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=np.float32)
    result = PointTrackResult(0, points, np.ones(2), "fake-v1")
    points[:] = 99

    assert result.points_xy[0, 0] == 1.0
    assert result.points_xy.flags.writeable is False
    with pytest.raises(ValueError):
        result.points_xy[0, 0] = 5


def test_point_result_rejects_malformed_output() -> None:
    with pytest.raises(ValueError, match="visibility length"):
        PointTrackResult(0, np.zeros((2, 2)), np.ones(1), "fake-v1")
    with pytest.raises(ValueError, match="finite"):
        PointTrackResult(0, np.asarray(((np.nan, 1.0),)), np.ones(1), "fake-v1")
