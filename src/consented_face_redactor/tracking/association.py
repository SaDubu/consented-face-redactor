"""Deterministic geometric association without identity authority."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from consented_face_redactor.adapters.detection_iface import FaceDetection

from .types import TrackedFaceBox


BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class AssociationPolicy:
    minimum_iou: float = 0.08
    maximum_center_distance_ratio: float = 0.12
    maximum_scale_ratio: float = 1.8
    maximum_cost: float = 1.0
    crossing_iou: float = 0.25


@dataclass(frozen=True, slots=True)
class AssociationCost:
    eligible: bool
    total: float
    iou: float
    center_distance_ratio: float
    scale_ratio: float
    reason_code: str


@dataclass(frozen=True, slots=True)
class TrackDetectionAssignment:
    track_index: int
    detection_index: int
    cost: AssociationCost


@dataclass(frozen=True, slots=True)
class AssociationResult:
    assignments: tuple[TrackDetectionAssignment, ...]
    unmatched_track_indices: tuple[int, ...]
    unmatched_detection_indices: tuple[int, ...]
    track_boxes: tuple[BBox, ...]
    detection_boxes: tuple[BBox, ...]


@dataclass(frozen=True, slots=True)
class Ambiguity:
    track_indices: tuple[int, int]
    reason_code: str


def _bbox(value: FaceDetection | TrackedFaceBox) -> BBox:
    box = value.bbox
    if hasattr(box, "x1"):
        return float(box.x1), float(box.y1), float(box.x2), float(box.y2)
    return tuple(map(float, box))  # type: ignore[return-value]


def bbox_iou(left: BBox, right: BBox) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def association_cost(
    predicted_track: TrackedFaceBox,
    detection: FaceDetection,
    *,
    frame_shape: tuple[int, ...],
    policy: AssociationPolicy = AssociationPolicy(),
) -> AssociationCost:
    """Score geometric continuity; the result cannot grant a profile identity."""
    track_box, detection_box = _bbox(predicted_track), _bbox(detection)
    diagonal = math.hypot(float(frame_shape[1]), float(frame_shape[0]))
    if diagonal <= 0:
        raise ValueError("frame_shape must contain positive height and width")
    tc = np.asarray(((track_box[0] + track_box[2]) / 2, (track_box[1] + track_box[3]) / 2))
    dc = np.asarray(((detection_box[0] + detection_box[2]) / 2, (detection_box[1] + detection_box[3]) / 2))
    distance = float(np.linalg.norm(tc - dc)) / diagonal
    track_area = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
    detection_area = (detection_box[2] - detection_box[0]) * (detection_box[3] - detection_box[1])
    scale = max(math.sqrt(detection_area / track_area), math.sqrt(track_area / detection_area))
    overlap = bbox_iou(track_box, detection_box)
    eligible = (
        overlap >= policy.minimum_iou
        and distance <= policy.maximum_center_distance_ratio
        and scale <= policy.maximum_scale_ratio
    )
    total = (1.0 - overlap) * 0.60 + distance * 2.5 + max(0.0, scale - 1.0) * 0.25
    eligible = eligible and total <= policy.maximum_cost
    return AssociationCost(
        eligible=eligible,
        total=total,
        iou=overlap,
        center_distance_ratio=distance,
        scale_ratio=scale,
        reason_code="association_gate_passed" if eligible else "association_gate_failed",
    )


def associate_tracks_to_detections(
    tracks: Sequence[TrackedFaceBox],
    detections: Sequence[FaceDetection],
    *,
    frame_shape: tuple[int, ...],
    policy: AssociationPolicy = AssociationPolicy(),
) -> AssociationResult:
    """Return a minimum-cost one-to-one assignment after hard gating."""
    costs = tuple(
        tuple(
            association_cost(track, detection, frame_shape=frame_shape, policy=policy)
            for detection in detections
        )
        for track in tracks
    )

    @lru_cache(maxsize=None)
    def solve(track_index: int, used_mask: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if track_index == len(tracks):
            return 0, 0.0, ()
        best = solve(track_index + 1, used_mask)
        for detection_index, cost in enumerate(costs[track_index]):
            if not cost.eligible or used_mask & (1 << detection_index):
                continue
            count, total, pairs = solve(track_index + 1, used_mask | (1 << detection_index))
            candidate = count + 1, total + cost.total, ((track_index, detection_index),) + pairs
            if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1:] < best[1:]):
                best = candidate
        return best

    if len(detections) > 20:
        raise ValueError("association supports at most 20 detections per frame")
    _, _, pairs = solve(0, 0)
    assignments = tuple(
        TrackDetectionAssignment(track_index, detection_index, costs[track_index][detection_index])
        for track_index, detection_index in sorted(pairs)
    )
    used_tracks = {item.track_index for item in assignments}
    used_detections = {item.detection_index for item in assignments}
    return AssociationResult(
        assignments=assignments,
        unmatched_track_indices=tuple(i for i in range(len(tracks)) if i not in used_tracks),
        unmatched_detection_indices=tuple(i for i in range(len(detections)) if i not in used_detections),
        track_boxes=tuple(_bbox(item) for item in tracks),
        detection_boxes=tuple(_bbox(item) for item in detections),
    )


def detect_crossing_ambiguity(
    assignments: AssociationResult,
    *,
    previous_assignments: AssociationResult | None,
    crossing_iou: float = 0.25,
) -> tuple[Ambiguity, ...]:
    """Flag overlapping tracks whose assigned left/right order flips."""
    if previous_assignments is None:
        return ()
    current = {item.track_index: item.detection_index for item in assignments.assignments}
    previous = {item.track_index: item.detection_index for item in previous_assignments.assignments}
    shared = sorted(set(current) & set(previous))
    ambiguities: list[Ambiguity] = []
    for offset, left_track in enumerate(shared):
        for right_track in shared[offset + 1 :]:
            if bbox_iou(assignments.track_boxes[left_track], assignments.track_boxes[right_track]) < crossing_iou:
                continue
            current_left_x = sum(assignments.detection_boxes[current[left_track]][::2]) / 2
            current_right_x = sum(assignments.detection_boxes[current[right_track]][::2]) / 2
            previous_left_x = sum(previous_assignments.detection_boxes[previous[left_track]][::2]) / 2
            previous_right_x = sum(previous_assignments.detection_boxes[previous[right_track]][::2]) / 2
            if (current_left_x - current_right_x) * (previous_left_x - previous_right_x) <= 0:
                ambiguities.append(Ambiguity((left_track, right_track), "track_detection_ambiguous"))
    return tuple(ambiguities)
