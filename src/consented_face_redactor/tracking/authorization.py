"""Anchor-derived authorization leases for temporal localization."""

from __future__ import annotations

from dataclasses import dataclass, replace

from consented_face_redactor.adapters.detection_iface import FaceDetection
from consented_face_redactor.gallery_approval import GalleryApproval

from .types import BboxValidation, TrackAuthorization, TrackedFaceBox


@dataclass(frozen=True, slots=True)
class ContinuityPolicy:
    tracker_only_max_frames: int = 12
    identity_refresh_max_frames: int = 90
    minimum_visible_point_ratio: float = 0.60


@dataclass(frozen=True, slots=True)
class AuthorizedTrack:
    authorization: TrackAuthorization
    bbox: tuple[float, float, float, float]
    last_frame_index: int
    last_detection_frame: int
    active: bool = True
    reason_code: str = "explicit_gallery_anchor"
    revoked_frame_index: int | None = None


@dataclass(frozen=True, slots=True)
class TrackObservation:
    frame_index: int
    tracked_box: TrackedFaceBox | None
    bbox_validation: BboxValidation
    detection_matched: bool
    ambiguous: bool = False
    model_output_valid: bool = True
    gallery_approval: GalleryApproval | None = None


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    authorized: bool
    review_required: bool
    reason_code: str
    track: AuthorizedTrack


def _detection_bbox(detection: FaceDetection) -> tuple[float, float, float, float]:
    box = detection.bbox
    return float(box.x1), float(box.y1), float(box.x2), float(box.y2)


def create_authorized_track(
    approval: GalleryApproval,
    detection: FaceDetection,
    *,
    frame_index: int,
    track_id: str,
) -> AuthorizedTrack:
    """Create a lease only from a current, explicit gallery approval."""
    if approval.approved is not True:
        raise ValueError("an authorized track requires approved=True")
    if not approval.profile_id or not approval.gallery_revision:
        raise ValueError("approved gallery result requires profile_id and gallery_revision")
    authorization = TrackAuthorization(
        track_id=track_id,
        profile_id=approval.profile_id,
        gallery_revision=approval.gallery_revision,
        origin_frame_index=frame_index,
        last_gallery_approval_frame=frame_index,
    )
    return AuthorizedTrack(
        authorization=authorization,
        bbox=_detection_bbox(detection),
        last_frame_index=frame_index,
        last_detection_frame=frame_index,
    )


def refresh_authorized_track(
    track: AuthorizedTrack,
    approval: GalleryApproval,
    *,
    frame_index: int,
    bbox: tuple[float, float, float, float] | None = None,
) -> AuthorizedTrack:
    """Refresh an existing lease with a same-profile explicit approval."""
    if not track.active:
        raise ValueError("revoked authorization cannot be refreshed")
    if approval.approved is not True or approval.profile_id != track.authorization.profile_id:
        raise ValueError("refresh approval must explicitly approve the existing profile")
    if not approval.gallery_revision:
        raise ValueError("refresh approval requires gallery_revision")
    authorization = replace(
        track.authorization,
        gallery_revision=approval.gallery_revision,
        last_gallery_approval_frame=frame_index,
    )
    return replace(
        track,
        authorization=authorization,
        bbox=bbox or track.bbox,
        last_frame_index=frame_index,
        last_detection_frame=frame_index,
        reason_code="explicit_gallery_anchor",
    )


def revoke_track_authorization(
    track: AuthorizedTrack,
    *,
    frame_index: int,
    reason_code: str,
) -> AuthorizedTrack:
    """Irreversibly end one lease; reacquisition requires a new anchor/track."""
    return replace(
        track,
        active=False,
        last_frame_index=frame_index,
        reason_code=reason_code,
        revoked_frame_index=frame_index,
    )


def may_propagate_authorization(
    track: AuthorizedTrack,
    observation: TrackObservation,
    *,
    policy: ContinuityPolicy = ContinuityPolicy(),
) -> AuthorizationDecision:
    """Propagate an existing identity lease; never invent a new identity."""
    reason: str | None = None
    if not track.active:
        reason = "authorization_already_revoked"
    elif observation.frame_index != track.last_frame_index + 1:
        reason = "tracker_frame_sequence_invalid"
    elif not observation.model_output_valid or observation.tracked_box is None:
        reason = "tracker_output_malformed"
    elif observation.ambiguous:
        reason = "track_detection_ambiguous"
    elif not observation.bbox_validation.valid:
        reason = observation.bbox_validation.reason_code
    elif observation.gallery_approval is not None and observation.gallery_approval.approved is True:
        approval = observation.gallery_approval
        if approval.profile_id != track.authorization.profile_id:
            reason = "conflicting_profile"
        else:
            refreshed = refresh_authorized_track(
                track,
                approval,
                frame_index=observation.frame_index,
                bbox=observation.tracked_box.bbox,
            )
            return AuthorizationDecision(True, False, "explicit_gallery_anchor", refreshed)
    elif observation.frame_index - track.authorization.last_gallery_approval_frame > policy.identity_refresh_max_frames:
        reason = "identity_refresh_required"
    elif not observation.detection_matched and observation.frame_index - track.last_detection_frame > policy.tracker_only_max_frames:
        reason = "tracker_only_limit_exceeded"

    if reason is not None:
        revoked = revoke_track_authorization(track, frame_index=observation.frame_index, reason_code=reason)
        return AuthorizationDecision(False, True, reason, revoked)

    assert observation.tracked_box is not None
    updated = replace(
        track,
        bbox=observation.tracked_box.bbox,
        last_frame_index=observation.frame_index,
        last_detection_frame=(
            observation.frame_index if observation.detection_matched else track.last_detection_frame
        ),
        reason_code="tracked_from_explicit_approval",
    )
    return AuthorizationDecision(True, False, "tracked_from_explicit_approval", updated)
