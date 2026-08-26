"""Temporal tracking contracts separated from identity authorization."""

from .protocol import PointTracker
from .tapnextpp import TapNextPlusPlusAdapter
from .authorization import (
    AuthorizationDecision,
    AuthorizedTrack,
    ContinuityPolicy,
    TrackObservation,
    create_authorized_track,
    may_propagate_authorization,
    refresh_authorized_track,
    revoke_track_authorization,
)
from .bidirectional import (
    AnalyzedFace,
    FrameAnalysis,
    IdentityAnchor,
    RedactionTrackPlan,
    ReconciliationPolicy,
    build_redaction_track_plan,
)
from .types import (
    BboxValidation,
    PointTrackResult,
    SimilarityTransform,
    TrackAuthorization,
    TrackedFaceBox,
    TrackFrameDecision,
)

__all__ = [
    "BboxValidation",
    "AuthorizationDecision",
    "AnalyzedFace",
    "AuthorizedTrack",
    "ContinuityPolicy",
    "FrameAnalysis",
    "IdentityAnchor",
    "PointTrackResult",
    "PointTracker",
    "RedactionTrackPlan",
    "ReconciliationPolicy",
    "TapNextPlusPlusAdapter",
    "SimilarityTransform",
    "TrackAuthorization",
    "TrackObservation",
    "TrackedFaceBox",
    "create_authorized_track",
    "build_redaction_track_plan",
    "may_propagate_authorization",
    "refresh_authorized_track",
    "revoke_track_authorization",
    "TrackFrameDecision",
]
