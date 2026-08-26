import numpy as np
import pytest

from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection
from consented_face_redactor.gallery_approval import GalleryApproval
from consented_face_redactor.tracking.authorization import (
    ContinuityPolicy,
    TrackObservation,
    create_authorized_track,
    may_propagate_authorization,
)
from consented_face_redactor.tracking.types import BboxValidation, TrackedFaceBox


def _detection():
    return FaceDetection(BoundingBox(10, 10, 30, 30), np.zeros((5, 2), np.float32), 0.9)


def _approval(profile="prof-a"):
    return GalleryApproval(True, profile, 0.8, "approved", "gallery-r1")


def _observation(frame, *, matched=True, approval=None, valid=True, ambiguous=False):
    return TrackObservation(
        frame_index=frame,
        tracked_box=TrackedFaceBox((11, 10, 31, 30), 0.9, 12, "tracker"),
        bbox_validation=BboxValidation(valid, "track_geometry_valid" if valid else "tracker_visibility_insufficient"),
        detection_matched=matched,
        ambiguous=ambiguous,
        gallery_approval=approval,
    )


def test_tracker_only_cannot_create_authorization():
    with pytest.raises(ValueError, match="approved=True"):
        create_authorized_track(GalleryApproval.denied("no_match"), _detection(), frame_index=0, track_id="t1")


def test_valid_continuity_propagates_existing_profile_only():
    track = create_authorized_track(_approval(), _detection(), frame_index=0, track_id="t1")
    decision = may_propagate_authorization(track, _observation(1))
    assert decision.authorized
    assert decision.track.authorization.profile_id == "prof-a"
    assert decision.reason_code == "tracked_from_explicit_approval"


def test_conflicting_explicit_profile_revokes_lease():
    track = create_authorized_track(_approval(), _detection(), frame_index=0, track_id="t1")
    decision = may_propagate_authorization(track, _observation(1, approval=_approval("prof-b")))
    assert not decision.authorized
    assert not decision.track.active
    assert decision.reason_code == "conflicting_profile"


def test_tracker_only_limit_is_fail_closed():
    track = create_authorized_track(_approval(), _detection(), frame_index=0, track_id="t1")
    policy = ContinuityPolicy(tracker_only_max_frames=1)
    first = may_propagate_authorization(track, _observation(1, matched=False), policy=policy)
    second = may_propagate_authorization(first.track, _observation(2, matched=False), policy=policy)
    assert first.authorized
    assert not second.authorized
    assert second.reason_code == "tracker_only_limit_exceeded"


def test_ambiguity_immediately_revokes():
    track = create_authorized_track(_approval(), _detection(), frame_index=0, track_id="t1")
    decision = may_propagate_authorization(track, _observation(1, ambiguous=True))
    assert not decision.authorized
    assert decision.review_required
