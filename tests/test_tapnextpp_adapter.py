from __future__ import annotations

import hashlib

import numpy as np
import pytest

from consented_face_redactor.tracking import PointTracker, TapNextPlusPlusAdapter


class _Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def track_frame(self, frame, query_points_xy=None, state=None):
        if query_points_xy is not None:
            points = np.asarray(query_points_xy, dtype=np.float32)
            next_state = {"points": points + 1.0}
            self.calls.append(("initialize", query_points_xy))
            return points, np.ones(len(points), dtype=bool), next_state
        points = np.asarray(state["points"], dtype=np.float32)
        self.calls.append(("update", state))
        return points, np.array([True, False, True]), {"points": points + 1.0}


def _adapter(tmp_path, *, backend=None, digest=None):
    checkpoint = tmp_path / "tapnextpp.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    entry = {
        "model_id": "tapnextpp-512",
        "role": "tracker",
        "source": "google-deepmind/tapnet",
        "filename": checkpoint.name,
        "sha256": digest or hashlib.sha256(b"checkpoint").hexdigest(),
        "license": "Apache-2.0",
        "input_shape": [3, 512, 512],
        "preprocessing_revision": 1,
        "provider": "PyTorch",
    }
    implementation = backend or _Backend()

    def factory(*args, **kwargs):
        assert args[0] == checkpoint
        assert kwargs["input_resolution"] == 512
        return implementation

    return TapNextPlusPlusAdapter(
        checkpoint_path=checkpoint,
        vendor_source_dir=tmp_path / "unused",
        manifest_entry=entry,
        backend_factory=factory,
    ), implementation


def test_adapter_satisfies_protocol_and_tracks_consecutive_frames(tmp_path):
    adapter, backend = _adapter(tmp_path)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    queries = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)

    first = adapter.initialize(frame, frame_index=4, query_points=queries)
    second = adapter.update(frame, frame_index=5)

    assert isinstance(adapter, PointTracker)
    assert adapter.model_id == "tapnextpp-512"
    np.testing.assert_array_equal(first.points_xy, queries)
    np.testing.assert_array_equal(second.visibility, [1.0, 0.0, 1.0])
    assert [call[0] for call in backend.calls] == ["initialize", "update"]


def test_adapter_reset_revokes_recurrent_state(tmp_path):
    adapter, _ = _adapter(tmp_path)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    adapter.initialize(frame, frame_index=0, query_points=np.array([[1, 1], [2, 2]]))
    adapter.reset()
    with pytest.raises(RuntimeError, match="initialized"):
        adapter.update(frame, frame_index=1)


def test_adapter_rejects_nonconsecutive_frame(tmp_path):
    adapter, _ = _adapter(tmp_path)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    adapter.initialize(frame, frame_index=0, query_points=np.array([[1, 1], [2, 2]]))
    with pytest.raises(ValueError, match="consecutive"):
        adapter.update(frame, frame_index=2)


@pytest.mark.parametrize(
    "frame",
    [np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.float32)],
)
def test_adapter_rejects_invalid_frames(tmp_path, frame):
    adapter, _ = _adapter(tmp_path)
    with pytest.raises(ValueError, match="frame"):
        adapter.initialize(frame, frame_index=0, query_points=np.array([[1, 1], [2, 2]]))


def test_adapter_verifies_checksum_before_backend_load(tmp_path):
    with pytest.raises(Exception, match="checksum mismatch"):
        _adapter(tmp_path, digest="0" * 64)
