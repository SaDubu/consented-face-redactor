from __future__ import annotations

import numpy as np

from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection
from consented_face_redactor.config import Config
from consented_face_redactor.gallery_approval import GalleryApproval
from consented_face_redactor.temporal_video_processor import TemporalVideoProcessor
from consented_face_redactor.tracking.types import PointTrackResult


class _Detector:
    def __init__(self):
        self.index = 0

    def detect(self, frame):
        index = self.index
        self.index += 1
        landmarks = np.array([[14, 15], [25, 15], [20, 20], [15, 25], [25, 25]], np.float32) + [index, 0]
        return [FaceDetection(BoundingBox(10 + index, 10, 30 + index, 30), landmarks, 0.9)]


class _Gallery:
    def __init__(self):
        self.index = 0

    def evaluate(self, frame, detection):
        approved = self.index in (0, 2)
        self.index += 1
        if approved:
            return GalleryApproval(True, "prof-a", 0.8, "consent", "r1")
        return GalleryApproval.denied("view_weak", gallery_revision="r1")


class _LastFrameGallery:
    def __init__(self):
        self.index = 0

    def evaluate(self, frame, detection):
        approved = self.index == 2
        self.index += 1
        if approved:
            return GalleryApproval(True, "prof-a", 0.8, "consent", "r1")
        return GalleryApproval.denied("view_weak", gallery_revision="r1")


class _Tracker:
    model_id = "fake"

    def reset(self):
        self.points = None

    def initialize(self, frame, *, frame_index, query_points):
        self.points = np.asarray(query_points, np.float32)
        return PointTrackResult(frame_index, self.points, np.ones(len(self.points)), "fake-v1")

    def update(self, frame, *, frame_index):
        self.points = self.points + [1.0, 0.0]
        return PointTrackResult(frame_index, self.points, np.ones(len(self.points)), "fake-v1")


def _video(path, frame_count=3):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    assert writer.isOpened()
    rng = np.random.default_rng(7)
    for _ in range(frame_count):
        writer.write(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    writer.release()


def test_two_pass_analysis_and_atomic_render(tmp_path):
    source = tmp_path / "input.mp4"
    destination = tmp_path / "output.mp4"
    _video(source)
    processor = TemporalVideoProcessor(
        config=Config(), detector=_Detector(), gallery=_Gallery(), tracker=_Tracker()
    )

    analysis = processor.analyze(source)
    evidence = processor.render(source, destination, analysis.plan)

    by_frame = {item.frame_index: item for item in analysis.plan.decisions}
    assert by_frame[1].reason_code == "bidirectional_anchor_consensus"
    assert evidence.input_frame_count == evidence.output_frame_count == 3
    assert evidence.redacted_frame_count == 3
    assert destination.is_file()
    assert source.is_file()


def test_render_rejects_overwrite_before_writing(tmp_path):
    source = tmp_path / "input.mp4"
    _video(source)
    processor = TemporalVideoProcessor(
        config=Config(), detector=_Detector(), gallery=_Gallery(), tracker=_Tracker()
    )
    analysis = processor.analyze(source)
    try:
        processor.render(source, source, analysis.plan)
    except ValueError as exc:
        assert "differ" in str(exc)
    else:
        raise AssertionError("same source/destination was accepted")


def test_last_anchor_extends_with_single_detection_continuity(tmp_path):
    source = tmp_path / "tail.mp4"
    _video(source, frame_count=4)
    processor = TemporalVideoProcessor(
        config=Config(), detector=_Detector(), gallery=_Gallery(), tracker=_Tracker()
    )

    analysis = processor.analyze(source)
    by_frame = {item.frame_index: item for item in analysis.plan.decisions}

    assert by_frame[3].authorized
    assert by_frame[3].reason_code == "tracked_from_explicit_approval"
    assert by_frame[3].bbox_source == "detection_fused"


def test_first_anchor_propagates_backward_with_single_detection_continuity(tmp_path):
    source = tmp_path / "leading.mp4"
    _video(source)
    processor = TemporalVideoProcessor(
        config=Config(),
        detector=_Detector(),
        gallery=_LastFrameGallery(),
        tracker=_Tracker(),
    )

    analysis = processor.analyze(source)
    by_frame = {item.frame_index: item for item in analysis.plan.decisions}

    assert set(by_frame) == {0, 1, 2}
    assert by_frame[0].reason_code == "tracked_from_explicit_approval"
    assert by_frame[1].reason_code == "tracked_from_explicit_approval"
    assert by_frame[2].reason_code == "explicit_gallery_anchor"
