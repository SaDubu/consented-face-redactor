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
# Effect / domain imports (Phase 8) — wired at import time for lazy access
# ------------------------------------------------------------------ #

def _import_effects():
    """Lazily fetch StickerEffect, MosaicConfig, FaceBox."""
    from consented_face_redactor.effects.sticker import StickerEffect
    from consented_face_redactor.domain.types import MosaicConfig, FaceBox
    return StickerEffect, MosaicConfig, FaceBox


def _build_sticker_effect(config, scale_factor, anchor):
    """Create a StickerEffect instance using config attr look-ups."""
    StickerEffect = _import_effects()[0]
    png_bytes = getattr(config, "sticker_png_bytes") or b""
    return StickerEffect(
        png_bytes, scale_factor=scale_factor, anchor=anchor, eye_rotation=True
    )


def _apply_effect_to_bbox(frame, bbox, mode, config, effect_proxy):
    """Apply mosaic / sticker / none effect to *frame* using the confirmed bbox.

    Parameters
    ----------
    frame : np.ndarray
        RGB uint8 (H,W,3) — never mutated; copy is returned.
    bbox : tuple[float,float,float,float]
        Face bounding box (x1,y1,x2,y2) in pixel coords.
    mode : str
        Effect mode: ``'mosaic'``, ``'sticker'``, or ``'none'``.
    config : Config
        Pipeline config for attribute look-ups (sticker_* etc.).
    effect_proxy : StickerEffect | None
        Pre-built sticker instance (used when proxy is available).

    Returns
    -------
    np.ndarray
        New frame with the selected redaction applied to the bbox ROI.
    """
    MosaicConfig, FaceBox = _import_effects()[1:]

    x1 = max(int(bbox[0]), 0)
    y1 = max(int(bbox[1]), 0)
    x2 = min(int(bbox[2]), frame.shape[1])
    y2 = min(int(bbox[3]), frame.shape[0])

    if x2 <= x1 or y2 <= y1:
        return frame.copy()

    face_roi = FaceBox(x1, y1, x2, y2)

    # --- MOSAIC --------------------------------------------------------- #
    if mode == "mosaic":
        mosaic_cfg = MosaicConfig(force_block_size=8)
        from consented_face_redactor.effects.mosaic import MosaicEffect
        mosaic_inst = MosaicEffect(mosaic_cfg)
        return mosaic_inst.render(frame, face_roi)

    # --- STICKER -------------------------------------------------------- #
    if mode == "sticker":
        StickerEffect = _import_effects()[0]
        five_landmarks = getattr(config, "effect_five_landmarks", None)
        if effect_proxy is not None:
            return effect_proxy.render(frame, face_roi, five_landmarks or type("LM", (), {"eye_angle": 0.0})())
        else:
            sf = float(getattr(config, "sticker_scale_factor", 1.0))
            anc = str(getattr(config, "sticker_anchor", "center"))
            sticker = _build_sticker_effect(config, sf, anc)
            return sticker.render(frame, face_roi, five_landmarks or type("LM", (), {"eye_angle": 0.0})())

    # --- NONE ------------------------------------------------------------ #
    if mode == "none":
        return frame.copy()

    # unknown mode — same as none (no effect applied)
    return frame.copy()


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
        config: Any,  # Config class or plain dict (for baseline compat)
        *,
        detector: 'DetectorAdapter | None' = None,
    ) -> None:
        from consented_face_redactor.config import Config as _Config

        if isinstance(config, dict):
            self._config = _Config.from_dict(config)  # type: ignore[unreachable]
        else:
            self._config = config
        self._detector = detector
        self._detects_bgr_input: bool = False
        if hasattr(self._detector, 'model_id') and (
            isinstance(self._detector.model_id, str)
            and self._detector.model_id == 'yunet'
        ):
            self._detects_bgr_input = True

        # Phase 7 config bindings
        self._track_state: TrackState = TrackState.UNSEEN
        self._frame_index: int = -1
        self._gallery: Any = None  # GalleryMatcher instance

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
                bbox_ref = getattr(det, 'bbox', None) or getattr(det, 'box', None)
                if isinstance(bbox_ref, (list, tuple)) and len(bbox_ref) >= 4:
                    bboxes.append((float(bbox_ref[0]), float(bbox_ref[1]), float(bbox_ref[2]), float(bbox_ref[3])))
                else:
                    bboxes.append(
                        (float(bbox_ref.x1), float(bbox_ref.y1), float(bbox_ref.x2), float(bbox_ref.y2))
                    )
                feats = getattr(det, 'landmarks', np.zeros((5, 2), dtype=np.float32))
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

        # No faces detected -- handle LOST/EXPIRED transitions.
        if not detections.bboxes:
            self._frame_index = frame_index
            if self._track_state == TrackState.CONFIRMED:
                self._track_state = TrackState.LOST
                self._lost_frame_index = frame_index
                review_required = True
                return ProcessResult(
                    result_frame=frame.copy(),
                    is_redacted=False,
                    track_state=self._track_state,
                    review_required=True,
                )
            if self._track_state == TrackState.LOST:
                # Check TTL: if no face seen for track_lost_ttl_frames, expire
                frames_since_loss = frame_index - self._lost_frame_index if (
                    hasattr(self, '_lost_frame_index') and self._lost_frame_index >= 0
                ) else 0
                if frames_since_loss >= self._config.track_lost_ttl_frames:
                    self._track_state = TrackState.EXPIRED
                    return ProcessResult(
                        result_frame=frame.copy(),
                        is_redacted=False,
                        track_state=self._track_state,
                        review_required=True,
                    )
                # Still in grace period
                return ProcessResult(
                    result_frame=frame.copy(),
                    is_redacted=False,
                    track_state=self._track_state,
                    review_required=True,
                )
            if self._track_state == TrackState.EXPIRED:
                return ProcessResult(
                    result_frame=frame.copy(),
                    is_redacted=False,
                    track_state=self._track_state,
                    review_required=True,
                )
            if self._track_state in (TrackState.UNSEEN, TrackState.CANDIDATE):
                # Face disappeared while candidate — transition to LOST
                if self._track_state == TrackState.CANDIDATE:
                    self._track_state = TrackState.LOST
                    self._lost_frame_index = frame_index
                return ProcessResult(
                    result_frame=frame.copy(),
                    is_redacted=False,
                    track_state=self._track_state,
                    review_required=False,
                )
            self._track_state = TrackState.UNSEEN
            return ProcessResult(
                result_frame=frame.copy(),
                is_redacted=False,
                track_state=self._track_state,
                review_required=False,
            )

        # -----------------------------------------------------------------------
        # Face detected -- Phase 7 state-machine wiring (UNSEEN/CANDIDATE/CONFIRMED)
        # -----------------------------------------------------------------------
        new_state: TrackState = self._track_state
        is_redacted: bool = False
        review_required: bool = True  # defaults to True for candidate states

        if self._track_state == TrackState.UNSEEN:
            # First detection — initialise tracking
            self._frame_index = frame_index
            self._candidate_score = 0.0

            # Compute mean confidence from detections
            if detections.confidences and len(detections.confidences) > 0:
                mean_confidence: float = sum(detections.confidences) / len(detections.confidences)
            else:
                mean_confidence = 0.0
            self._candidate_score = mean_confidence

            # Direct transition to CONFIRMED if confidence meets threshold
            if mean_confidence >= self._config.t_confirm and self._config.t_confirm < 1.0:
                new_state = TrackState.CONFIRMED
                is_redacted = True
                review_required = False
                # Wire Phase 8: apply effect to the detected face bbox on first CONFIRMED entry
                if detections.bboxes:
                    out_frame_accumulator = frame.copy()
                    for one_bbox in detections.bboxes:
                        out_frame_accumulator = _apply_effect_to_bbox(
                            out_frame_accumulator,
                            one_bbox,
                            self._config.effect_mode,
                            self._config,
                            None,  # no pre-built sticker proxy needed here
                        )
            else:
                new_state = TrackState.CANDIDATE

        elif self._track_state == TrackState.CANDIDATE:
            # Compute running confidence score
            if detections.confidences and len(detections.confidences) > 0:
                mean_confidence = sum(detections.confidences) / len(detections.confidences)
            else:
                mean_confidence = 0.0

            self._candidate_score = (
                mean_confidence
                if self._frame_index < 0
                else ((self._candidate_score * float(self._frame_index)) + mean_confidence)
                / (float(self._frame_index) + 1.0)
            )
            is_confirmed: bool = False

            # Check gallery matcher at the configured interval
            frame_count_since_candidate = (
                frame_index - self._frame_index if self._frame_index >= 0 else 0
            )
            if frame_count_since_candidate >= self._config.recheck_interval_frames:
                matches: list[tuple[str, float]] = []
                try:
                    embedding = getattr(self._gallery, 'embed', lambda f=None: None)(frame)
                except Exception:
                    embedding = None
                if embedding is not None and hasattr(self._gallery, 'match'):
                    match_result = self._gallery.match(embedding)
                    if isinstance(match_result, list) and len(match_result) > 0:
                        matches = match_result

            # CONFIRMED transition when score meets threshold
            if self._candidate_score >= self._config.t_confirm and self._config.t_confirm < 1.0:
                is_confirmed = True

            if is_confirmed:
                new_state = TrackState.CONFIRMED
                is_redacted = True
                review_required = False
                # Extract matched profile from the gallery result (if any)
                if matches and len(matches) > 0:
                    self._confirmed_profile_id = float(matches[0][1])  # score as profile proxy
                else:
                    self._confirmed_profile_id = None
                # Wire Phase 8: apply effect to detected faces on CANDIDATE→CONFIRMED transition
                if detections.bboxes:
                    out_frame_accumulator = frame.copy()
                    for one_bbox in detections.bboxes:
                        out_frame_accumulator = _apply_effect_to_bbox(
                            out_frame_accumulator,
                            one_bbox,
                            self._config.effect_mode,
                            self._config,
                            None,  # no pre-built sticker proxy needed here
                        )

        elif self._track_state == TrackState.CONFIRMED:
            # Face still present while confirmed — keep redacting
            if detections.bboxes:
                is_redacted = True
                review_required = False
                new_state = TrackState.CONFIRMED
                # Wire Phase 8: apply effect to each detected face bbox
                out_frame_accumulator = frame.copy()
                for one_bbox in detections.bboxes:
                    out_frame_accumulator = _apply_effect_to_bbox(
                        out_frame_accumulator,
                        one_bbox,
                        self._config.effect_mode,
                        self._config,
                        None,  # no pre-built sticker proxy needed here
                    )
            else:
                review_required = True

        elif self._track_state in (TrackState.LOST, TrackState.EXPIRED) and detections.bboxes:
            # Face reappeared while LOST — go back to CANDIDATE for re-confirmation
            review_required = True
            new_state = TrackState.CANDIDATE

        else:
            new_state = self._track_state

        self._frame_index = frame_index
        self._track_state = new_state

        # Phase 8: wire effect results back into the ProcessResult frame
        out_frame = None
        if self._track_state == TrackState.CONFIRMED and detections.bboxes:
            if 'out_frame_accumulator' in dir():
                out_frame = out_frame_accumulator
            else:
                out_frame = frame.copy()
        elif is_redacted and detections.bboxes:
            # First-time CONFIRMED (UNSEEN→CONFIRMED or CANDIDATE→CONFIRMED) — result accumulated above
            if 'out_frame_accumulator' in dir():
                out_frame = out_frame_accumulator
            else:
                out_frame = frame.copy()
        else:
            # no redaction, no detection — return unmodified frame
            out_frame = frame.copy()

        assert out_frame is not None, "Phase 8: out_frame must always be set"

        return ProcessResult(
            result_frame=out_frame,
            is_redacted=is_redacted,
            track_state=new_state,
            review_required=review_required,
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
