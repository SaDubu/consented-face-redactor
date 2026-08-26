"""Contract tests for structured gallery authority decisions."""

import pytest

from consented_face_redactor.gallery_approval import GalleryApproval


def test_denial_has_no_profile_and_is_immutable() -> None:
    denial = GalleryApproval.denied("empty_gallery", gallery_revision="test-v1")

    assert denial.approved is False
    assert denial.profile_id is None
    assert denial.reason_code == "empty_gallery"
    with pytest.raises(AttributeError):
        denial.approved = True


def test_approval_preserves_audit_fields() -> None:
    approval = GalleryApproval(True, "profile-1", 0.93, "explicit_approval", "test-v1")

    assert approval.approved is True
    assert approval.profile_id == "profile-1"
    assert approval.similarity == 0.93
    assert approval.gallery_revision == "test-v1"
