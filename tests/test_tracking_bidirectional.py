import numpy as np

from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection
from consented_face_redactor.gallery_approval import GalleryApproval
from consented_face_redactor.tracking.bidirectional import (
    AnalyzedFace,
    FrameAnalysis,
    ReconciliationPolicy,
    build_redaction_track_plan,
    collect_identity_anchors,
)
from consented_face_redactor.tracking.types import PointTrackResult


class _TranslationTracker:
    model_id = "translation"

    def reset(self):
        self.points = None

    def initialize(self, frame, *, frame_index, query_points):
        self.points = np.asarray(query_points, np.float32)
        return PointTrackResult(frame_index, self.points, np.ones(len(self.points)), "fake-v1")

    def update(self, frame, *, frame_index):
        self.points = self.points + np.array([1.0, 0.0], np.float32)
        return PointTrackResult(frame_index, self.points, np.ones(len(self.points)), "fake-v1")


def _analysis(index, approved):
    detection = FaceDetection(
        BoundingBox(10 + index, 10, 30 + index, 30),
        np.array([[14, 15], [25, 15], [20, 20], [15, 25], [25, 25]], np.float32) + [index, 0],
        0.9,
    )
    approval = (
        GalleryApproval(True, "prof-a", 0.8, "approved", "r1")
        if approved
        else GalleryApproval.denied("no_match", gallery_revision="r1")
    )
    return FrameAnalysis(index, (AnalyzedFace(detection, approval),))


def test_collects_only_explicit_anchors():
    anchors = collect_identity_anchors([_analysis(0, True), _analysis(1, False)])
    assert len(anchors) == 1
    assert anchors[0].frame_index == 0


def test_bidirectional_consensus_fills_same_profile_gap():
    analyses = [_analysis(0, True), _analysis(1, False), _analysis(2, True)]
    frames = {index: np.zeros((64, 64, 3), np.uint8) for index in range(3)}
    plan = build_redaction_track_plan(
        input_frame_count=3,
        analyses=analyses,
        gap_frames=frames,
        forward_tracker=_TranslationTracker(),
        backward_tracker=_TranslationTracker(),
        policy=ReconciliationPolicy(minimum_path_iou=0.5),
    )
    by_frame = {item.frame_index: item for item in plan.decisions}
    assert by_frame[0].reason_code == "explicit_gallery_anchor"
    assert by_frame[1].authorized
    assert by_frame[1].reason_code == "bidirectional_anchor_consensus"
    assert by_frame[2].reason_code == "explicit_gallery_anchor"


def test_plan_does_not_fill_unbounded_gap():
    analyses = [_analysis(0, True), _analysis(3, True)]
    frames = {index: np.zeros((64, 64, 3), np.uint8) for index in range(4)}
    plan = build_redaction_track_plan(
        input_frame_count=4,
        analyses=analyses,
        gap_frames=frames,
        forward_tracker=_TranslationTracker(),
        backward_tracker=_TranslationTracker(),
        policy=ReconciliationPolicy(maximum_anchor_gap_frames=2),
    )
    assert [item.frame_index for item in plan.decisions] == [0, 3]
