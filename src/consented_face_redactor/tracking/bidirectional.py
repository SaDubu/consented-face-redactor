"""Bidirectional reconciliation between explicit gallery anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from consented_face_redactor.adapters.detection_iface import FaceDetection
from consented_face_redactor.gallery_approval import GalleryApproval

from .association import bbox_iou
from .geometry import (
    estimate_similarity_transform,
    seed_face_points,
    transform_bbox,
    validate_tracked_bbox,
)
from .protocol import PointTracker
from .types import TrackFrameDecision


@dataclass(frozen=True, slots=True)
class AnalyzedFace:
    detection: FaceDetection
    approval: GalleryApproval


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    frame_index: int
    faces: tuple[AnalyzedFace, ...]
    reason_code: str = "frame_analyzed"


@dataclass(frozen=True, slots=True)
class IdentityAnchor:
    frame_index: int
    profile_id: str
    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    gallery_revision: str

    def __post_init__(self) -> None:
        landmarks = np.asarray(self.landmarks, dtype=np.float32)
        if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
            raise ValueError("anchor landmarks must have shape (5, 2)")
        landmarks = landmarks.copy()
        landmarks.setflags(write=False)
        object.__setattr__(self, "landmarks", landmarks)


@dataclass(frozen=True, slots=True)
class AnchorSegment:
    left: IdentityAnchor
    right: IdentityAnchor


@dataclass(frozen=True, slots=True)
class TrackPath:
    direction: str
    profile_id: str
    decisions: tuple[TrackFrameDecision, ...]


@dataclass(frozen=True, slots=True)
class FrameRange:
    first_frame: int
    last_frame: int
    reason_code: str


@dataclass(frozen=True, slots=True)
class RedactionTrackPlan:
    input_frame_count: int
    profile_ids: tuple[str, ...]
    decisions: tuple[TrackFrameDecision, ...]
    ambiguous_ranges: tuple[FrameRange, ...]


@dataclass(frozen=True, slots=True)
class ReconciliationPolicy:
    maximum_anchor_gap_frames: int = 90
    maximum_one_way_frames: int = 90
    tracker_only_max_frames: int = 12
    association_ambiguity_margin: float = 0.15
    minimum_path_iou: float = 0.30
    minimum_visible_point_ratio: float = 0.60
    maximum_scale_ratio: float = 1.25


def _bbox(detection: FaceDetection) -> tuple[float, float, float, float]:
    box = detection.bbox
    return float(box.x1), float(box.y1), float(box.x2), float(box.y2)


def collect_identity_anchors(frame_analysis: Sequence[FrameAnalysis]) -> tuple[IdentityAnchor, ...]:
    """Collect only direct, structurally complete gallery approvals."""
    anchors: list[IdentityAnchor] = []
    for frame in frame_analysis:
        for face in frame.faces:
            approval = face.approval
            if approval.approved is True and approval.profile_id and approval.gallery_revision:
                anchors.append(
                    IdentityAnchor(
                        frame.frame_index,
                        approval.profile_id,
                        _bbox(face.detection),
                        face.detection.landmarks,
                        approval.gallery_revision,
                    )
                )
    return tuple(sorted(anchors, key=lambda item: (item.frame_index, item.profile_id)))


def split_anchor_segments(
    anchors: Sequence[IdentityAnchor],
    *,
    max_gap_frames: int,
) -> tuple[AnchorSegment, ...]:
    """Return only weak gaps bounded by consecutive same-profile anchors."""
    by_profile: dict[str, list[IdentityAnchor]] = {}
    for anchor in anchors:
        by_profile.setdefault(anchor.profile_id, []).append(anchor)
    segments: list[AnchorSegment] = []
    for profile_id in sorted(by_profile):
        ordered = sorted(by_profile[profile_id], key=lambda item: item.frame_index)
        for left, right in zip(ordered, ordered[1:]):
            gap = right.frame_index - left.frame_index
            if 1 < gap <= max_gap_frames and left.gallery_revision == right.gallery_revision:
                segments.append(AnchorSegment(left, right))
    return tuple(sorted(segments, key=lambda item: (item.left.frame_index, item.left.profile_id)))


def _track_segment(
    frames: Mapping[int, np.ndarray],
    segment: AnchorSegment,
    tracker: PointTracker,
    *,
    reverse: bool,
    policy: ReconciliationPolicy,
) -> TrackPath:
    anchor = segment.right if reverse else segment.left
    stop = segment.left if reverse else segment.right
    actual_indices = (
        range(anchor.frame_index, stop.frame_index - 1, -1)
        if reverse
        else range(anchor.frame_index, stop.frame_index + 1)
    )
    tracker.reset()
    query_points = seed_face_points(anchor.bbox, anchor.landmarks)
    first_result = tracker.initialize(
        frames[anchor.frame_index], frame_index=0, query_points=query_points
    )
    previous_points = first_result.points_xy
    previous_bbox = anchor.bbox
    active = True
    decisions: list[TrackFrameDecision] = []
    for logical_index, actual_index in enumerate(actual_indices):
        if logical_index == 0:
            decisions.append(
                TrackFrameDecision(
                    actual_index,
                    f"{anchor.profile_id}:{segment.left.frame_index}-{segment.right.frame_index}",
                    anchor.profile_id,
                    anchor.bbox,
                    True,
                    "gallery_anchor",
                    1.0,
                    False,
                    "explicit_gallery_anchor",
                )
            )
            continue
        if not active:
            decisions.append(
                TrackFrameDecision(actual_index, None, None, None, False, "none", None, True, "track_path_revoked")
            )
            continue
        try:
            result = tracker.update(frames[actual_index], frame_index=logical_index)
            transform = estimate_similarity_transform(previous_points, result.points_xy, result.visibility)
            current_bbox = transform_bbox(previous_bbox, transform)
            visible_ratio = float(np.mean(result.visibility >= 0.5))
            validation = validate_tracked_bbox(
                previous_bbox,
                current_bbox,
                frame_shape=frames[actual_index].shape,
                visible_point_ratio=visible_ratio,
                minimum_visible_point_ratio=policy.minimum_visible_point_ratio,
                maximum_scale_ratio=policy.maximum_scale_ratio,
            )
        except (RuntimeError, TypeError, ValueError):
            validation = None
            current_bbox = None
            visible_ratio = None
        if validation is None or not validation.valid or current_bbox is None:
            active = False
            reason = validation.reason_code if validation is not None else "tracker_output_malformed"
            decisions.append(
                TrackFrameDecision(actual_index, None, None, None, False, "none", visible_ratio, True, reason)
            )
            continue
        decisions.append(
            TrackFrameDecision(
                actual_index,
                f"{anchor.profile_id}:{segment.left.frame_index}-{segment.right.frame_index}",
                anchor.profile_id,
                current_bbox,
                True,
                "tracker",
                visible_ratio,
                False,
                "tracked_from_explicit_approval",
            )
        )
        previous_points, previous_bbox = result.points_xy, current_bbox
    return TrackPath(
        "backward" if reverse else "forward",
        anchor.profile_id,
        tuple(sorted(decisions, key=lambda item: item.frame_index)),
    )


def track_segment_forward(
    frames: Mapping[int, np.ndarray], segment: AnchorSegment, tracker: PointTracker, *, policy: ReconciliationPolicy
) -> TrackPath:
    return _track_segment(frames, segment, tracker, reverse=False, policy=policy)


def track_segment_backward(
    frames: Mapping[int, np.ndarray], segment: AnchorSegment, tracker: PointTracker, *, policy: ReconciliationPolicy
) -> TrackPath:
    return _track_segment(frames, segment, tracker, reverse=True, policy=policy)


def reconcile_bidirectional_paths(
    forward: TrackPath,
    backward: TrackPath,
    *,
    policy: ReconciliationPolicy,
) -> tuple[TrackFrameDecision, ...]:
    """Authorize interior frames only where both same-profile paths agree."""
    if forward.profile_id != backward.profile_id:
        raise ValueError("bidirectional paths must refer to the same profile")
    left = {item.frame_index: item for item in forward.decisions}
    right = {item.frame_index: item for item in backward.decisions}
    decisions: list[TrackFrameDecision] = []
    for frame_index in sorted(set(left) & set(right)):
        fwd, bwd = left[frame_index], right[frame_index]
        if fwd.reason_code == "explicit_gallery_anchor":
            decisions.append(fwd)
            continue
        if bwd.reason_code == "explicit_gallery_anchor":
            decisions.append(bwd)
            continue
        if fwd.authorized and bwd.authorized and fwd.bbox and bwd.bbox:
            overlap = bbox_iou(fwd.bbox, bwd.bbox)
            if overlap >= policy.minimum_path_iou:
                fused = tuple((a + b) / 2.0 for a, b in zip(fwd.bbox, bwd.bbox))
                decisions.append(
                    TrackFrameDecision(
                        frame_index,
                        fwd.track_id,
                        forward.profile_id,
                        fused,  # type: ignore[arg-type]
                        True,
                        "bidirectional",
                        min(fwd.visible_point_ratio or 0.0, bwd.visible_point_ratio or 0.0),
                        False,
                        "bidirectional_anchor_consensus",
                    )
                )
                continue
        decisions.append(
            TrackFrameDecision(
                frame_index, None, None, None, False, "none", None, True, "bidirectional_path_disagreement"
            )
        )
    return tuple(decisions)


def build_redaction_track_plan(
    *,
    input_frame_count: int,
    analyses: Sequence[FrameAnalysis],
    gap_frames: Mapping[int, np.ndarray],
    forward_tracker: PointTracker,
    backward_tracker: PointTracker,
    policy: ReconciliationPolicy = ReconciliationPolicy(),
) -> RedactionTrackPlan:
    """Build deterministic direct-anchor and consensus decisions."""
    anchors = collect_identity_anchors(analyses)
    decisions: dict[tuple[int, str], TrackFrameDecision] = {}
    for anchor in anchors:
        decisions[(anchor.frame_index, anchor.profile_id)] = TrackFrameDecision(
            anchor.frame_index,
            f"{anchor.profile_id}:anchor:{anchor.frame_index}",
            anchor.profile_id,
            anchor.bbox,
            True,
            "gallery_anchor",
            1.0,
            False,
            "explicit_gallery_anchor",
        )
    ambiguous: list[FrameRange] = []
    for segment in split_anchor_segments(anchors, max_gap_frames=policy.maximum_anchor_gap_frames):
        required = range(segment.left.frame_index, segment.right.frame_index + 1)
        if any(index not in gap_frames for index in required):
            ambiguous.append(FrameRange(segment.left.frame_index, segment.right.frame_index, "missing_gap_frame"))
            continue
        forward = track_segment_forward(gap_frames, segment, forward_tracker, policy=policy)
        backward = track_segment_backward(gap_frames, segment, backward_tracker, policy=policy)
        reconciled = reconcile_bidirectional_paths(forward, backward, policy=policy)
        for decision in reconciled:
            decisions[(decision.frame_index, segment.left.profile_id)] = decision
        failed = [item.frame_index for item in reconciled if not item.authorized]
        if failed:
            ambiguous.append(FrameRange(min(failed), max(failed), "bidirectional_path_disagreement"))
    ordered = tuple(decisions[key] for key in sorted(decisions))
    return RedactionTrackPlan(
        input_frame_count=input_frame_count,
        profile_ids=tuple(sorted({item.profile_id for item in anchors})),
        decisions=ordered,
        ambiguous_ranges=tuple(ambiguous),
    )
