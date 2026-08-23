"""Processing pipeline skeleton for consented-face-redactor.

This module provides the core frame processing API:
    process_frame(frame, frame_index, timestamp, state) -> (result_frame, new_state)

It depends only on internal interfaces — never raw vendor output.
The input frame is never mutated by default; all operations return new objects.
"""

from __future__ import annotations

from typing import Any, NamedTuple

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


class TrackState(str):
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
# Pipeline class (placeholder — no real model integration yet)
# ------------------------------------------------------------------ #


class RedactionPipeline:
    """Skeleton pipeline for face redaction processing.

    At this stage the pipeline uses fake detector/embedder stubs and
    a mock effect renderer. Real implementation will occur in Phase 6+.
    """

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def] (Config class imported by caller)
        self._config = config if hasattr(config, "effect_mode") else config
        self._track_state: TrackState = TrackState.UNSEEN
        self._frame_index: int = 0

    @property
    def current_track_state(self) -> TrackState:
        return self._track_state

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
            is_redacted : bool — apply only on confirmed identity
            track_state : TrackState after evaluation
            review_required : bool — frame ranges needing manual review
        """
        # Phase 3: stub detector returns no detections (empty list).
        # Real integration happens in Phase 6.
        detections = DetectionResult(
            bboxes=[],
            landmarks=[],
            confidences=[],
        )

        # No faces detected → always review_required=False unless
        # previous track was CONFIRMED and now lost.
        if not detections.bboxes:
            if self._track_state == TrackState.CONFIRMED:
                # Potential loss — would transition to LOST in full impl
                return ProcessResult(
                    result_frame=frame.copy(),
                    is_redacted=False,
                    track_state=self._track_state,
                    review_required=True,
                )
            return ProcessResult(
                result_frame=frame.copy(),
                is_redacted=False,
                track_state=TrackState.UNSEEN,
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
            "track_state": self._track_state,
            "frame_index": self._frame_index,
        }

    def load_track_state(self, snapshot: Any) -> None:
        """Restore track state from serialized snapshot."""
        if isinstance(snapshot, dict):
            self._track_state = TrackState(snapshot.get("track_state", "unseen"))
            self._frame_index = snapshot.get("frame_index", 0)
