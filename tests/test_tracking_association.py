import numpy as np

from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection
from consented_face_redactor.tracking.association import (
    AssociationPolicy,
    associate_tracks_to_detections,
    association_cost,
)
from consented_face_redactor.tracking.types import TrackedFaceBox


def _track(box):
    return TrackedFaceBox(box, 1.0, 10, "tracker")


def _detection(box):
    return FaceDetection(BoundingBox(*box), np.zeros((5, 2), np.float32), 0.9)


def test_cost_accepts_nearby_overlap_and_rejects_far_face():
    near = association_cost(_track((10, 10, 30, 30)), _detection((12, 11, 32, 31)), frame_shape=(100, 100, 3))
    far = association_cost(_track((10, 10, 30, 30)), _detection((70, 70, 90, 90)), frame_shape=(100, 100, 3))
    assert near.eligible
    assert not far.eligible


def test_assignment_is_one_to_one_and_prefers_maximum_cardinality():
    tracks = [_track((10, 10, 30, 30)), _track((40, 10, 60, 30))]
    detections = [_detection((11, 10, 31, 30)), _detection((41, 10, 61, 30))]
    result = associate_tracks_to_detections(tracks, detections, frame_shape=(100, 100, 3))
    assert [(x.track_index, x.detection_index) for x in result.assignments] == [(0, 0), (1, 1)]
    assert result.unmatched_track_indices == ()
    assert result.unmatched_detection_indices == ()


def test_hard_gate_leaves_unrelated_items_unmatched():
    result = associate_tracks_to_detections(
        [_track((0, 0, 10, 10))],
        [_detection((80, 80, 90, 90))],
        frame_shape=(100, 100, 3),
        policy=AssociationPolicy(maximum_center_distance_ratio=0.05),
    )
    assert result.assignments == ()
    assert result.unmatched_track_indices == (0,)
    assert result.unmatched_detection_indices == (0,)
