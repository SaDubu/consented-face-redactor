"""Processing pipeline skeleton for consented-face-redactor.

This module provides the core frame processing API:
    process_frame(frame, frame_index, timestamp, state) -> (result_frame, new_state)

It depends only on internal interfaces -- never raw vendor output.
The input frame is never mutated by default; all operations return new objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from consented_face_redactor.gallery_approval import GalleryApproval


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


@dataclass(frozen=True, slots=True)
class FaceDecision:
    """One detection's bbox and explicit approval result for diagnostics."""

    bbox: tuple[float, float, float, float]
    approval: GalleryApproval


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
        mosaic_cfg = MosaicConfig(
            grid_cells=int(getattr(config, "mosaic_grid_cells", 12)),
            padding_ratio=float(getattr(config, "mosaic_padding_ratio", 0.18)),
            min_block_px=int(getattr(config, "mosaic_min_block_px", 10)),
            shape=str(getattr(config, "mosaic_shape", "ellipse")),
            ellipse_horizontal_scale=float(
                getattr(config, "mosaic_ellipse_horizontal_scale", 1.40)
            ),
            ellipse_vertical_scale=float(
                getattr(config, "mosaic_ellipse_vertical_scale", 1.50)
            ),
        )
        from consented_face_redactor.effects.mosaic import MosaicEffect, expand_bbox

        if mosaic_cfg.shape == "rectangle":
            expanded = expand_bbox(
                (x1, y1, x2, y2),
                frame_shape=frame.shape,
                padding_ratio=mosaic_cfg.padding_ratio,
            )
            face_roi = FaceBox(*expanded)
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
        gallery: Any | None = None,
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
        self._gallery: Any = gallery  # GalleryMatcher instance
        self._lost_frame_index: int | None = None
        self._confirmed_profile_id: str | None = None
        self._candidate_confidences: list[float] = []
        self._gallery_recheck_count = 0
        self._last_gallery_approval = GalleryApproval.denied("not_checked")
        self._last_frame_approvals: tuple[GalleryApproval, ...] = ()
        self._last_frame_decisions: tuple[FaceDecision, ...] = ()

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

    @property
    def last_gallery_approval(self) -> GalleryApproval:
        """Return the latest structured gallery decision without granting authority."""
        return self._last_gallery_approval

    @property
    def last_frame_approvals(self) -> tuple[GalleryApproval, ...]:
        """Return the face-by-face decisions made for the latest frame."""
        return self._last_frame_approvals

    @property
    def last_frame_decisions(self) -> tuple[FaceDecision, ...]:
        """Return face-local decisions; callers must opt in before persisting bboxes."""
        return self._last_frame_decisions

    @property
    def telemetry_snapshot(self) -> dict[str, Any]:
        """Return non-authorizing observation metrics for benchmarks and diagnostics."""
        return {
            "candidate_confidences": tuple(self._candidate_confidences),
            "gallery_recheck_count": self._gallery_recheck_count,
            "approval_reason": self._last_gallery_approval.reason_code,
            "gallery_revision": self._last_gallery_approval.gallery_revision,
        }

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

        # FrameSource and OpenCvYuNetDetector both use BGR. Do not convert a
        # BGR frame to RGB here: that would silently degrade real inference.
        return list(self._detector.detect(frame))

    def _evaluate_detection(self, frame: np.ndarray, detection: Any) -> GalleryApproval:
        """Evaluate one face with an adapter exposing ``evaluate(frame, detection)``."""
        try:
            decision = self._gallery.evaluate(frame, detection)
        except Exception:
            return GalleryApproval.denied("gallery_evaluation_error")
        if not isinstance(decision, GalleryApproval):
            return GalleryApproval.denied("malformed_approval")
        return decision

    def _process_face_by_face_approvals(
        self,
        frame: np.ndarray,
        frame_index: int,
        raw_detections: list[Any],
        bboxes: list[tuple[float, float, float, float]],
        confidences: list[float],
    ) -> ProcessResult:
        """Fail closed per face, applying effects only to explicitly approved ROIs."""
        approvals = tuple(
            self._evaluate_detection(frame, detection) for detection in raw_detections
        )
        self._last_frame_approvals = approvals
        self._last_frame_decisions = tuple(
            FaceDecision(bbox=bbox, approval=approval)
            for bbox, approval in zip(bboxes, approvals)
        )
        self._gallery_recheck_count += len(approvals)
        self._candidate_confidences.extend(float(value) for value in confidences)
        self._last_gallery_approval = next(
            (approval for approval in approvals if approval.approved is True),
            approvals[0],
        )
        approved_bboxes = [
            bbox for bbox, approval in zip(bboxes, approvals) if approval.approved is True
        ]
        self._frame_index = frame_index
        if not approved_bboxes:
            self._track_state = TrackState.CANDIDATE
            self._confirmed_profile_id = None
            return ProcessResult(frame.copy(), False, self._track_state, True)

        output = frame.copy()
        for bbox in approved_bboxes:
            output = _apply_effect_to_bbox(
                output, bbox, self._config.effect_mode, self._config, None
            )
        self._track_state = TrackState.CONFIRMED
        self._confirmed_profile_id = self._last_gallery_approval.profile_id
        # An unapproved face in the same frame is never redacted and remains a
        # review item even though another face was explicitly approved.
        review_required = any(approval.approved is not True for approval in approvals)
        return ProcessResult(output, True, self._track_state, review_required)

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

        # Production gallery adapters evaluate every detected face with its
        # own landmarks. This path prevents one approved face from authorizing
        # redaction of other people in the same frame. Legacy test adapters
        # retain the historical embed()/match() path below.
        if raw_detections and callable(getattr(self._gallery, "evaluate", None)):
            return self._process_face_by_face_approvals(
                frame,
                frame_index,
                raw_detections,
                detections.bboxes,
                detections.confidences,
            )

        # No faces detected -- handle LOST/EXPIRED transitions.
        if not detections.bboxes:
            self._last_frame_approvals = ()
            self._last_frame_decisions = ()
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
            self._candidate_confidences.append(mean_confidence)

            # Safety gate: detector confidence alone NEVER authorizes redaction.
            # Unseen → CANDIDATE always; gallery match in CANDIDANT branch is what
            # actually authorizes CONFIRMED + redaction.
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
            self._candidate_confidences.append(mean_confidence)

            # Check gallery matcher at the configured interval
            frame_count_since_candidate = (
                frame_index - self._frame_index if self._frame_index >= 0 else 0
            )
            approval = GalleryApproval.denied("recheck_not_due")
            if frame_count_since_candidate >= self._config.recheck_interval_frames:
                self._gallery_recheck_count += 1
                if self._gallery is None:
                    approval = GalleryApproval.denied("gallery_unavailable")
                else:
                    try:
                        embedding = self._gallery.embed(frame)
                    except Exception:
                        approval = GalleryApproval.denied("embedding_error")
                    else:
                        if embedding is None:
                            approval = GalleryApproval.denied("embedding_unavailable")
                        else:
                            try:
                                candidate_approval = self._gallery.match(embedding)
                            except Exception:
                                approval = GalleryApproval.denied("gallery_match_error")
                            else:
                                if isinstance(candidate_approval, GalleryApproval):
                                    approval = candidate_approval
                                else:
                                    approval = GalleryApproval.denied("malformed_approval")
            self._last_gallery_approval = approval

            # Safety gate: CONFIRMED transition requires explicit gallery match.
            # confidence is a quality signal only — it never authorizes redaction.
            has_gallery_match = approval.approved is True

            if has_gallery_match:
                new_state = TrackState.CONFIRMED
                is_redacted = True
                review_required = False
                self._confirmed_profile_id = approval.profile_id
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
        """Serialize v2 track state, including temporal and approval context."""
        return {
            "schema_version": 2,
            "track_state": self._track_state.value,
            "frame_index": self._frame_index,
            "lost_frame_index": self._lost_frame_index,
            "confirmed_profile_id": self._confirmed_profile_id,
        }

    def load_track_state(self, snapshot: Any) -> None:
        """Restore legacy v1 or complete v2 track state without fail-open recovery."""
        if not isinstance(snapshot, dict):
            raise ValueError("track state snapshot has an invalid schema")
        legacy_keys = {"track_state", "frame_index"}
        v2_keys = legacy_keys | {"schema_version", "lost_frame_index", "confirmed_profile_id"}
        keys = set(snapshot)
        if keys == legacy_keys:
            version = 1
        elif keys == v2_keys and snapshot.get("schema_version") == 2:
            version = 2
        else:
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
        if version == 1 and track_state in (TrackState.CONFIRMED, TrackState.LOST):
            # Older snapshots cannot prove who was approved or when a loss began.
            # Degrade to a candidate requiring an explicit gallery approval.
            track_state = TrackState.CANDIDATE
        lost_frame_index = snapshot.get("lost_frame_index") if version == 2 else None
        if lost_frame_index is not None and (
            isinstance(lost_frame_index, bool) or not isinstance(lost_frame_index, int) or lost_frame_index < 0
        ):
            raise ValueError("track state snapshot has an invalid lost frame index")
        profile_id = snapshot.get("confirmed_profile_id") if version == 2 else None
        if profile_id is not None and not isinstance(profile_id, str):
            raise ValueError("track state snapshot has an invalid confirmed profile")
        self._track_state = track_state
        self._frame_index = frame_index
        self._lost_frame_index = lost_frame_index
        self._confirmed_profile_id = profile_id if track_state is TrackState.CONFIRMED else None
