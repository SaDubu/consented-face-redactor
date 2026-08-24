"""E2E tests for RedactionPipeline — verified against production exports."""

from __future__ import annotations
import tempfile
import numpy as np
import pytest
from io import BytesIO
from PIL import Image

# ------- Production imports (real classes) -------
from consented_face_redactor.pipeline import (
    RedactionPipeline,
    DetectionResult,
    EmbeddingResult,
    MatchDecision,
    ProcessResult,
    TrackState,
)
from consented_face_redactor.config import Config
from consented_face_redactor.domain.types import FaceBox


# ======= Shared helpers ======= #

class _MockFace:
    """Minimal mock for a single detector face-detection row."""
    __slots__ = ('confidence', 'bbox')

    def __init__(self, bbox=(20.0, 20.0, 40.0, 40.0), conf=1.0):
        self.confidence = conf
        self.bbox = list(bbox)  # must support iteration of 4 values


class _MockDetector:
    """A stub detector that returns faces on every frame."""

    def __init__(self, bboxes=None, confidences=None):
        if bboxes is None:
            bboxes = [(20.0, 20.0, 40.0, 40.0)]
        self.bboxes = [list(b) for b in bboxes]
        self.confidences = confidences or [1.0]

    def detect(self, frame):
        rows = []
        for bb, c in zip(self.bboxes, self.confidences):
            det = object.__new__(_MockFace)
            det.confidence = c
            det.bbox = bb
            rows.append(det)
        return rows


class _MockEmptyDetector:
    """A stub detector that returns no faces on every frame."""

    def detect(self, frame):
        return []


class _MockTwoShotDetector:
    """Returns 'face' on first call, empty detection thereafter (simulates face appearing then disappearing)."""

    def __init__(self, first_call_bboxes=None, first_call_confidences=None):
        self.count = 0
        if first_call_bboxes is None:
            first_call_bboxes = [(20.0, 20.0, 40.0, 40.0)]
        self.first_bboxes = [list(b) for b in first_call_bboxes]
        self.first_confs = first_call_confidences or [1.0]

    def detect(self, frame):
        if self.count == 0:
            rows = []
            for bb, c in zip(self.first_bboxes, self.first_confs):
                det = object.__new__(_MockFace)
                det.confidence = c
                det.bbox = bb
                rows.append(det)
            self.count += 1
            return rows
        else:
            self.count += 1
            return []


class TestConfigDefaults:
    def test_effect_mode_is_string_mosaic(self) -> None:
        """EffectMode does not exist; mode is a plain string."""
        cfg = Config(effect_mode="mosaic", t_confirm=0.5)
        assert cfg.effect_mode == "mosaic"

# ======= Pipeline construction ======= #

class TestPipelineConstruction:
    def test_pipeline_initializes_with_config(self) -> None:
        """RedactionPipeline accepts a Config; no default() method exists."""
        pipeline = RedactionPipeline(Config.default())
        assert isinstance(pipeline, RedactionPipeline)
        assert pipeline.current_track_state == TrackState.UNSEEN

# ======= Detection → Confidence gate (never authorizes redaction) ======= #

class TestDetectionConfidenceGate:
    def test_high_confidence_stays_candidate(self) -> None:
        """Even with confidence > t_confirm, transition is CANDIDATE only."""
        cfg = Config(t_confirm=0.65)
        pipeline = RedactionPipeline(cfg)
        # Synthetic frame: 64x64 black, detection over ROI [20,20,40,40]
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        det = DetectionResult(bboxes=[(20., 20., 40., 40.)], landmarks=[], confidences=[1.0])

        # Inject mock detector so the pipeline actually sees a face
        pipeline._detector = _MockDetector()
        
        # Frame 0: UNSEEN → CANDIDATE
        result = pipeline.process_frame(frame, frame_index=0, timestamp=0.0, state=None)
        assert pipeline.current_track_state == TrackState.CANDIDATE
        # Confidence alone NEVER triggers CONFIRMED or redaction
        assert not result.is_redacted

# ======= Gallery identity match → CONFIRMED ======= #

class TestGalleryMatchConfirmsRedaction:
    def test_gallery_match_transitions_to_confirmed(self) -> None:
        """Explicit gallery match triggers CONFIRMED; confidence alone does not."""
        # Create a fake gallery matcher (mocks the _gallery attribute).
        class FakeGallery:
            def embed(self, frame):
                return np.zeros(512, dtype=np.float32)

            def match(self, vec):
                return [("subject1", 0.95)]
        cfg = Config(effect_mode="mosaic", recheck_interval_frames=1, t_confirm=0.65)
        pipeline = RedactionPipeline(cfg)
        pipeline._gallery = FakeGallery()
        # Inject mock detector so the pipeline sees faces AND uses gallery for CONFIRMED
        pipeline._detector = _MockDetector(
            bboxes=[(20.0, 20.0, 40.0, 40.0)], confidences=[0.8]
        )

        frame: np.ndarray = np.zeros((64, 64, 3), dtype=np.uint8)

        # Frame 0 → CANDIDATE (no recheck yet, interval=1 so triggers next time)
        r0 = pipeline.process_frame(frame, frame_index=0, timestamp=0.0, state=None)
        assert pipeline.current_track_state == TrackState.CANDIDATE
        assert not r0.is_redacted

        # Frame 5 ≥ recheck_interval_frames (1) since last candidate frame → match triggers CONFIRMED
        det_with_face = DetectionResult(
            bboxes=[(20., 20., 40., 40.)], landmarks=[], confidences=[1.0]
        )
        # Frame with detection and gallery embedding triggers CONFIRMED
        result_confirmed = pipeline.process_frame(frame, frame_index=5, timestamp=2.5, state=None)
        assert pipeline.current_track_state == TrackState.CONFIRMED
# ======= Transition state transitions (detector only, no gallery) ======= #

class TestTransitionStates:
    def test_unseen_becomes_candidate_on_detection(self) -> None:
        """First detection on UNSEEN → CANDIDATE."""
        cfg = Config()
        pipeline = RedactionPipeline(cfg)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        assert pipeline.current_track_state == TrackState.UNSEEN
        det = DetectionResult(bboxes=[(20., 20., 40., 40.)], landmarks=[], confidences=[1.0])
        # Inject mock detector so the CANIDATE transition fires
        pipeline._detector = _MockDetector()
        pipeline.process_frame(frame, frame_index=0, timestamp=0.0, state=None)
        assert pipeline.current_track_state == TrackState.CANDIDATE

    def test_candidate_no_detection_briefly_stays_candidate_then_lost(self) -> None:
        """Face disappears briefly while CANDIDATE → LOST, not UNSEEN."""
        cfg = Config(t_confirm=0.65)
        pipeline = RedactionPipeline(cfg)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        # First: detect face → CANDIDATE (two-shot detector returns face then empty on subsequent calls)
        pipeline._detector = _MockTwoShotDetector(
            first_call_bboxes=[(20.0, 20.0, 40.0, 40.0)], first_call_confidences=[1.0]
        )
        r = pipeline.process_frame(frame, frame_index=0, timestamp=0.0, state=None)
        assert pipeline.current_track_state == TrackState.CANDIDATE
        # No detection on next frame → LOST (not back to UNSEEN)
        r = pipeline.process_frame(frame, frame_index=1, timestamp=0.5, state=None)
        assert pipeline.current_track_state == TrackState.LOST
# ======= Pipeline renders correct overlay shapes ======= #

class TestRenderOutput:
    def test_process_returns_correct_named_tuple(self) -> None:
        """ProcessResult fields are (result_frame, is_redacted, track_state, review_required)."""
        cfg = Config(effect_mode="mosaic")
        pipeline = RedactionPipeline(cfg)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        # First detection -> CANDIDATE (no gallery match, so no redaction)
        det_with_face = DetectionResult(
            bboxes=[(20., 20., 40., 40.)], landmarks=[], confidences=[1.0]
        )
        frame_with_face = np.zeros((64, 64, 3), dtype=np.uint8)

        # Frame 0: UNSEEN → CANDIDATE (no detections since detector is None)
        result = pipeline.process_frame(frame, frame_index=0, timestamp=0.0, state=None)
        assert isinstance(result, ProcessResult)
        assert result.result_frame.shape == (64, 64, 3)
        assert result.is_redacted is False
        assert isinstance(result.track_state, TrackState)
        assert isinstance(result.review_required, bool)

# ======= Raw asset / StickerEffect (no mode string in __init__) ======= #

class TestStickerInitWithBytes:
    def test_sticker_accepts_raw_png_bytes(self) -> None:
        """StickerEffect.__init__ must accept raw PNG bytes (no mode string)."""
        from consented_face_redactor.effects.sticker import StickerEffect

        # Minimal 1×1 white PNG
        img = Image.new("RGBA", (1, 1), (255, 255, 255, 255))
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        effect = StickerEffect(png_bytes)
        assert effect._sticker is not None
        assert effect._scale_factor == pytest.approx(1.0)


# ======= MatchDecision / EmbeddingResult contract ======= #

class TestDataClasses:
    def test_match_decision_fields(self) -> None:
        """MatchDecision fields are: is_target, confidence, reason_code, profile_id."""
        m = MatchDecision(is_target=True, confidence=0.98, reason_code="gallery_match", profile_id="subject1")
        assert m.is_target is True
        assert m.confidence == pytest.approx(0.98)
        assert m.profile_id == "subject1"

class TestDetectionResult:
    def test_detection_result_bboxes(self) -> None:
        """DetectionResult.bboxes must be set correctly at init."""
        dr = DetectionResult(bboxes=[(1.0, 2.0, 3.0, 4.0)], landmarks=[], confidences=[0.5])
        assert dr.bboxes[0] == (1.0, 2.0, 3.0, 4.0)
