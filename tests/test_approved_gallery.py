"""Tests for the explicit-approval adapter used by real-model CLI paths."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection
from consented_face_redactor.approval_store import ApprovalRecord, ApprovalStore
from consented_face_redactor.approved_gallery import ApprovedLocalGalleryAdapter
from consented_face_redactor.gallery import LocalGallery


def _unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _detection() -> FaceDetection:
    return FaceDetection(
        BoundingBox(1, 1, 10, 10), np.zeros((5, 2), dtype=np.float32), 0.95
    )


class _Embedder:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector

    def embed(self, frame: np.ndarray, detection: FaceDetection):
        assert frame.dtype == np.uint8
        assert isinstance(detection, FaceDetection)
        return self.vector, 1


def _adapter(record: ApprovalRecord | None):
    gallery = LocalGallery()
    profile_id = gallery.enroll(_unit([1.0, 0.0, 0.0]))
    records = {} if record is None else {profile_id: record}
    approvals = ApprovalStore(records, gallery_revision="gallery-test-v1")
    return ApprovedLocalGalleryAdapter(
        embedder=_Embedder(_unit([1.0, 0.0, 0.0])),
        gallery=gallery,
        approvals=approvals,
    ), profile_id


def test_adapter_requires_explicit_approval_record():
    adapter, profile_id = _adapter(None)

    result = adapter.evaluate(np.zeros((12, 12, 3), dtype=np.uint8), _detection())

    assert result.approved is False
    assert result.profile_id == profile_id
    assert result.reason_code == "profile_not_explicitly_approved"


def test_adapter_approves_only_current_explicit_record():
    adapter, profile_id = _adapter(ApprovalRecord(True, "test_subject_consent"))

    result = adapter.evaluate(np.zeros((12, 12, 3), dtype=np.uint8), _detection())

    assert result.approved is True
    assert result.profile_id == profile_id
    assert result.reason_code == "test_subject_consent"
    assert result.gallery_revision == "gallery-test-v1"


def test_adapter_rejects_expired_approval():
    expired = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    adapter, _ = _adapter(ApprovalRecord(True, "test_subject_consent", expired))

    result = adapter.evaluate(np.zeros((12, 12, 3), dtype=np.uint8), _detection())

    assert result.approved is False
    assert result.reason_code == "approval_expired"


def test_approval_store_round_trip(tmp_path):
    store = ApprovalStore(
        {"prof-00000000": ApprovalRecord(True, "test_subject_consent")},
        gallery_revision="gallery-test-v1",
    )
    path = tmp_path / "approvals.json"

    store.save(path)
    restored = ApprovalStore.load(path)

    assert restored.to_dict() == store.to_dict()
