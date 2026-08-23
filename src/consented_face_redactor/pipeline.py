"""Processing pipeline skeleton for consented-face-redactor.

This module provides the core frame processing API:
    process_frame(frame, frame_index, timestamp, state) -> (result_frame, new_state)

It depends only on internal interfaces -- never raw vendor output.
The input frame is never mutated by default; all operations return new objects.
"""

from __future__ import annotations

from contextlib import suppress
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np


# ------------------------------------------------------------------ #
# Types
# ------------------------------------------------------------------ #


class DetectionResult(NamedTuple):
    """Outputs of face detection + alignment."""
    bboxes: list[tuple[float, float, float, float]]  # (x1,y1,x2,y2) in pixels
    landmarks: list[np.ndarray]                       # shape (5,2), image coords
    confidences: list[float]


class EmbeddingResult(NamedTuple):
    """Outputs of face embedding."""
    vectors: list[np.ndarray]                          # L2-normalized float32 vectors
    model_revision: str                                 # version string


class MatchDecision(NamedTuple):
    """Identity match decision for a candidate frame."""
    is_target: bool
    confidence: float                                   # cosine similarity score
    reason_code: str                                     # controlled enum as string
    profile_id: str | None                            # opaque profile ID if matched


class TrackState(str, Enum):
    """Track state machine values."""
    UNSEEN = "unseen"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    LOST = "lost"
    EXPIRED = "expired"


class ProcessResult(NamedTuple):
    """Output of a single frame processing step."""
    result_frame: np.ndarray   # same shape as input, redaction applied if matched
    is_redacted: bool          # True only when identity is confirmed
    track_state: TrackState    # current track state after this frame
    review_required: bool      # True if frame range needs manual review


# ------------------------------------------------------------------ #
# Pipeline class (placeholder -- no real model integration yet)
# ------------------------------------------------------------------ #


if TYPE_CHECKING:
    from consented_face_redactor.adapters.detection_iface import DetectorAdapter  # noqa: F401


class RedactionPipeline:
    """Skeleton pipeline for face redaction processing.

    The pipeline accepts an optional detector which is invoked on every
    frame when present.  When no detector is supplied the stub path
    (empty detections) is taken so callers can test downstream logic
    without requiring inference dependencies.
    """

    def __init__(
        self,
        config: Any,  # Config class imported by caller
        *,
        detector: 'DetectorAdapter | None' = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._detects_bgr_input: bool = False
        if hasattr(self._detector, 'model_id') and (
            isinstance(self._detector.model_id, str)
            and self._detector.model_id == 'yunet'
        ):
            self._detects_bgr_input = True
        self._track_state: TrackState = TrackState.UNSEEN
        self._frame_index: int = -1

    @property
    def current_track_state(self) -> TrackState:
        return self._track_state

    @property
    def detector_requires_bgr_input(self) -> bool:
        """Return True when the attached detector expects BGR input."""
        return self._detects_bgr_input

    @property
    def has_detector(self) -> bool:
        """Return True when a detector was provided to the pipeline."""
        return self._detector is not None

    # -- public detection bridge -------------------------------------- #

    def _run_detector(
        self,
        frame: np.ndarray,
    ) -> list[Any]:
        """Call the detector on *frame*, handling BGR/RGB colour space.

        Returns a (possibly empty) list of whatever detection rows the
        detector produces  -- typically instances of
        ``detection_iface.FaceDetection``.
        """
        if self._detector is None:
            return []

        input_frame = frame
        if self._detects_bgr_input and frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3:
            # cvtColor to BGR -- this mirrors the existing adapter contract.
            # We use the minimal helper that only imports when needed.
            with suppress(ImportError):
                import cv2 as _cv2bgr  # type: ignore[import-not-found, unused-ignore]
                input_frame = _cv2bgr.cvtColor(frame, _cv2bgr.COLOR_BGR2RGB)

        return list(self._detector.detect(input_frame))

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
        state: Any,  # track state snapshot for serialization
    ) -> ProcessResult:
        """Process one frame without mutating the input.

        Returns
        -------
        ProcessResult
            result_frame : new array (copy when redaction applied)
            is_redacted : bool -- apply only on confirmed identity
            track_state : TrackState after evaluation
            review_required : bool -- frame ranges needing manual review
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy array")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a uint8 array with shape (H, W, 3)")
        if frame.shape[0] < 1 or frame.shape[1] < 1:
            raise ValueError("frame dimensions must be positive")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if isinstance(timestamp, bool) or not isinstance(
            timestamp, (int, float, np.integer, np.floating)
        ):
            raise TypeError("timestamp must be numeric")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if state is not None:
            self.load_track_state(state)

        if frame_index < self._frame_index:
            self._track_state = TrackState.EXPIRED
            self._frame_index = frame_index
            return ProcessResult(
                result_frame=frame.copy(),
                is_redacted=False,
                track_state=self._track_state,
                review_required=True,
            )

        # Phase 6: call detector when available; otherwise fall back to stub.
        raw_detections = self._run_detector(frame)

        # Map detector output (FaceDetection objects from detection_iface)
        # into pipeline's DetectionResult for downstream consumers.
        if raw_detections:
            bboxes: list[tuple[float, float, float, float]] = []
            landmarks: list[np.ndarray] = []
            confidences: list[float] = []
            for det in raw_detections:
                bbox = det.bbox
                bboxes.append((float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2)))
                feats = det.landmarks  # (5,2) float32 array
                landmarks.append(feats.copy())
                confidences.append(float(det.confidence))
            detections = DetectionResult(
                bboxes=bboxes,
                landmarks=landmarks,
                confidences=confidences,
            )
        else:
            detections = DetectionResult(
                bboxes=[],
                landmarks=[],
                confidences=[],
            )

        # No faces detected -- always review_required=False unless
        # previous track was CONFIRMED and now lost.
        if not detections.bboxes:
            if self._track_state == TrackState.CONFIRMED:
                self._track_state = TrackState.LOST
                self._frame_index = frame_index
                return ProcessResult(
                    result_frame=frame.copy(),
                    is_redacted=False,
                    track_state=self._track_state,
                    review_required=True,
                )
            if self._track_state in (TrackState.LOST, TrackState.EXPIRED):
                self._frame_index = frame_index
                return ProcessResult(
                    result_frame=frame.copy(),
                    is_redacted=False,
                    track_state=self._track_state,
                    review_required=True,
                )
            self._track_state = TrackState.UNSEEN
            self._frame_index = frame_index
            return ProcessResult(
                result_frame=frame.copy(),
                is_redacted=False,
                track_state=self._track_state,
                review_required=False,
            )

        # Face detected but no embedding yet (Phase 3 stub).
        # In Phase 4+ this calls gallery matcher.
        return ProcessResult(
            result_frame=frame.copy(),
            is_redacted=False,
            track_state=self._track_state,
            review_required=False,
        )

    def save_track_state(self) -> Any:
        """Serialize current track state for persistence."""
        return {
            "track_state": self._track_state.value,
            "frame_index": self._frame_index,
        }

    def load_track_state(self, snapshot: Any) -> None:
        """Restore track state from serialized snapshot."""
        if not isinstance(snapshot, dict) or set(snapshot) != {"track_state", "frame_index"}:
            raise ValueError("track state snapshot has an invalid schema")
        try:
            track_state = TrackState(snapshot["track_state"])
        except (TypeError, ValueError) as exc:
            raise ValueError("track state snapshot has an invalid state") from exc
        frame_index = snapshot["frame_index"]
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < -1
        ):
            raise ValueError("track state snapshot has an invalid frame index")
        self._track_state = track_state
        self._frame_index = frame_index
