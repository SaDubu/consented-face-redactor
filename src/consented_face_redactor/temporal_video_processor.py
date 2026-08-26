"""Two-pass temporal video analysis and atomic rendering."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from consented_face_redactor.gallery_approval import GalleryApproval
from consented_face_redactor.media.frame_source import OpenCvFrameSource, OpenCvFrameWriter
from consented_face_redactor.pipeline import _apply_effect_to_bbox
from consented_face_redactor.tracking.bidirectional import (
    AnalyzedFace,
    AnchorSegment,
    FrameAnalysis,
    FrameRange,
    IdentityAnchor,
    RedactionTrackPlan,
    ReconciliationPolicy,
    collect_identity_anchors,
    reconcile_bidirectional_paths,
    split_anchor_segments,
    track_segment_backward,
    track_segment_forward,
)
from consented_face_redactor.tracking.association import AssociationPolicy, association_cost
from consented_face_redactor.tracking.protocol import PointTracker
from consented_face_redactor.tracking.types import TrackFrameDecision, TrackedFaceBox


@dataclass(frozen=True, slots=True)
class TemporalAnalysisResult:
    plan: RedactionTrackPlan
    frame_evidence: tuple[dict[str, object], ...]
    fps: float
    frame_shape: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ProcessingEvidence:
    input_frame_count: int
    output_frame_count: int
    redacted_frame_count: int
    explicit_anchor_count: int
    propagated_frame_count: int
    ambiguous_ranges: tuple[FrameRange, ...]


class TemporalVideoProcessor:
    """Analyze identity first, then render the immutable authorization plan."""

    def __init__(
        self,
        *,
        config: Any,
        detector: Any,
        gallery: Any,
        tracker: PointTracker,
        policy: ReconciliationPolicy = ReconciliationPolicy(),
    ) -> None:
        if detector is None or gallery is None:
            raise ValueError("temporal processing requires detector and gallery runtimes")
        self._config = config
        self._detector = detector
        self._gallery = gallery
        self._tracker = tracker
        self._policy = policy

    @staticmethod
    def _read_segment(path: Path, first: int, last: int) -> dict[int, Any]:
        source = OpenCvFrameSource(path)
        source.open()
        frames: dict[int, Any] = {}
        try:
            if not source.seek(first):
                raise OSError("video segment seek failed")
            for frame_index in range(first, last + 1):
                ok, frame = source.read()
                if not ok or frame is None:
                    raise OSError("video ended inside an anchor segment")
                frames[frame_index] = frame
        finally:
            source.close()
        return frames

    def analyze(self, source_path: Path) -> TemporalAnalysisResult:
        """Run detector/gallery analysis and bidirectional weak-gap tracking."""
        path = Path(source_path).resolve(strict=False)
        source = OpenCvFrameSource(path)
        source.open()
        analyses: list[FrameAnalysis] = []
        evidence: list[dict[str, object]] = []
        frame_index = 0
        frame_shape: tuple[int, int, int] | None = None
        fps = source.fps if source.fps > 0 else 30.0
        try:
            while True:
                ok, frame = source.read()
                if not ok or frame is None:
                    break
                started = time.perf_counter()
                faces: list[AnalyzedFace] = []
                frame_reason = "frame_analyzed"
                try:
                    detections = tuple(self._detector.detect(frame))
                except Exception:
                    detections = ()
                    frame_reason = "detector_error"
                for detection in detections:
                    try:
                        approval = self._gallery.evaluate(frame, detection)
                    except Exception:
                        approval = GalleryApproval.denied("gallery_evaluation_error")
                    if not isinstance(approval, GalleryApproval):
                        approval = GalleryApproval.denied("malformed_approval")
                    faces.append(AnalyzedFace(detection, approval))
                duration_ms = (time.perf_counter() - started) * 1000.0
                analyses.append(FrameAnalysis(frame_index, tuple(faces), frame_reason))
                evidence.append(
                    {
                        "frame_index": frame_index,
                        "analysis_duration_ms": duration_ms,
                        "detected_face_count": len(faces),
                        "approved_face_count": sum(face.approval.approved is True for face in faces),
                        "approval_reason_codes": [face.approval.reason_code for face in faces],
                        "analysis_reason_code": frame_reason,
                    }
                )
                frame_shape = tuple(frame.shape)
                frame_index += 1
        finally:
            source.close()
        if frame_shape is None or frame_index == 0:
            raise ValueError("input video contains no decodable frames")

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
        segments = split_anchor_segments(
            anchors, max_gap_frames=self._policy.maximum_anchor_gap_frames
        )
        for segment in segments:
            try:
                frames = self._read_segment(
                    path, segment.left.frame_index, segment.right.frame_index
                )
                forward = track_segment_forward(
                    frames, segment, self._tracker, policy=self._policy
                )
                backward = track_segment_backward(
                    frames, segment, self._tracker, policy=self._policy
                )
                reconciled = reconcile_bidirectional_paths(
                    forward, backward, policy=self._policy
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                ambiguous.append(
                    FrameRange(
                        segment.left.frame_index,
                        segment.right.frame_index,
                        "tracker_segment_error",
                    )
                )
                continue
            for decision in reconciled:
                decisions[(decision.frame_index, segment.left.profile_id)] = decision
            failed = [item.frame_index for item in reconciled if not item.authorized]
            if failed:
                ambiguous.append(
                    FrameRange(min(failed), max(failed), "bidirectional_path_disagreement")
                )

        # A stored video can end after its final explicit anchor. Extend only
        # while the point path remains valid and is associated with exactly one
        # current face detection. Detection/gallery weakness may not create a
        # track; it can only support continuity of this already-authorized one.
        by_profile: dict[str, list[IdentityAnchor]] = {}
        for anchor in anchors:
            by_profile.setdefault(anchor.profile_id, []).append(anchor)
        analyses_by_index = {item.frame_index: item for item in analyses}
        association_policy = AssociationPolicy()
        for profile_id in sorted(by_profile):
            first_anchor = min(by_profile[profile_id], key=lambda item: item.frame_index)
            first_index = max(
                0,
                first_anchor.frame_index - self._policy.maximum_one_way_frames,
            )
            if first_index < first_anchor.frame_index:
                synthetic_left = IdentityAnchor(
                    first_index,
                    profile_id,
                    first_anchor.bbox,
                    first_anchor.landmarks,
                    first_anchor.gallery_revision,
                )
                try:
                    leading_frames = self._read_segment(
                        path, first_index, first_anchor.frame_index
                    )
                    leading_path = track_segment_backward(
                        leading_frames,
                        AnchorSegment(synthetic_left, first_anchor),
                        self._tracker,
                        policy=self._policy,
                    )
                except (OSError, RuntimeError, TypeError, ValueError):
                    ambiguous.append(
                        FrameRange(first_index, first_anchor.frame_index - 1, "tracker_edge_error")
                    )
                else:
                    tracker_only_streak = 0
                    revoked_at: int | None = None
                    for decision in sorted(
                        leading_path.decisions,
                        key=lambda item: item.frame_index,
                        reverse=True,
                    ):
                        if decision.frame_index == first_anchor.frame_index:
                            continue
                        if not decision.authorized or decision.bbox is None:
                            revoked_at = decision.frame_index
                            break
                        faces = analyses_by_index[decision.frame_index].faces
                        tracked = TrackedFaceBox(
                            decision.bbox,
                            decision.visible_point_ratio or 0.0,
                            0,
                            "tracker",
                        )
                        eligible = []
                        for face in faces:
                            cost = association_cost(
                                tracked,
                                face.detection,
                                frame_shape=frame_shape,
                                policy=association_policy,
                            )
                            if cost.eligible:
                                eligible.append((cost.total, face))
                        eligible.sort(key=lambda item: item[0])
                        if len(eligible) > 1 and eligible[1][0] - eligible[0][0] < self._policy.association_ambiguity_margin:
                            revoked_at = decision.frame_index
                            break
                        matched_face = eligible[0][1] if eligible else None
                        if matched_face is None:
                            tracker_only_streak += 1
                            if tracker_only_streak > self._policy.tracker_only_max_frames:
                                revoked_at = decision.frame_index
                                break
                        else:
                            tracker_only_streak = 0
                            approval = matched_face.approval
                            if approval.approved is True and approval.profile_id != profile_id:
                                revoked_at = decision.frame_index
                                break
                        decisions[(decision.frame_index, profile_id)] = TrackFrameDecision(
                            decision.frame_index,
                            decision.track_id,
                            profile_id,
                            decision.bbox,
                            True,
                            "detection_fused" if matched_face else "tracker",
                            decision.visible_point_ratio,
                            False,
                            "tracked_from_explicit_approval",
                        )
                    if revoked_at is not None:
                        ambiguous.append(
                            FrameRange(first_index, revoked_at, "one_way_continuity_revoked")
                        )

            last_anchor = max(by_profile[profile_id], key=lambda item: item.frame_index)
            final_index = min(
                frame_index - 1,
                last_anchor.frame_index + self._policy.maximum_one_way_frames,
            )
            if final_index <= last_anchor.frame_index:
                continue
            synthetic_right = IdentityAnchor(
                final_index,
                profile_id,
                last_anchor.bbox,
                last_anchor.landmarks,
                last_anchor.gallery_revision,
            )
            try:
                edge_frames = self._read_segment(path, last_anchor.frame_index, final_index)
                edge_path = track_segment_forward(
                    edge_frames,
                    AnchorSegment(last_anchor, synthetic_right),
                    self._tracker,
                    policy=self._policy,
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                ambiguous.append(
                    FrameRange(last_anchor.frame_index + 1, final_index, "tracker_edge_error")
                )
                continue
            tracker_only_streak = 0
            revoked_at: int | None = None
            for decision in edge_path.decisions:
                if decision.frame_index == last_anchor.frame_index:
                    continue
                if not decision.authorized or decision.bbox is None:
                    revoked_at = decision.frame_index
                    break
                faces = analyses_by_index[decision.frame_index].faces
                tracked = TrackedFaceBox(
                    decision.bbox,
                    decision.visible_point_ratio or 0.0,
                    0,
                    "tracker",
                )
                eligible: list[tuple[float, AnalyzedFace]] = []
                for face in faces:
                    cost = association_cost(
                        tracked,
                        face.detection,
                        frame_shape=frame_shape,
                        policy=association_policy,
                    )
                    if cost.eligible:
                        eligible.append((cost.total, face))
                eligible.sort(key=lambda item: item[0])
                if len(eligible) > 1 and eligible[1][0] - eligible[0][0] < self._policy.association_ambiguity_margin:
                    revoked_at = decision.frame_index
                    break
                if len(eligible) == 0:
                    tracker_only_streak += 1
                    if tracker_only_streak > self._policy.tracker_only_max_frames:
                        revoked_at = decision.frame_index
                        break
                else:
                    tracker_only_streak = 0
                    approval = eligible[0][1].approval
                    if approval.approved is True and approval.profile_id != profile_id:
                        revoked_at = decision.frame_index
                        break
                decisions[(decision.frame_index, profile_id)] = TrackFrameDecision(
                    decision.frame_index,
                    decision.track_id,
                    profile_id,
                    decision.bbox,
                    True,
                    "detection_fused" if eligible else "tracker",
                    decision.visible_point_ratio,
                    False,
                    "tracked_from_explicit_approval",
                )
            if revoked_at is not None:
                ambiguous.append(
                    FrameRange(revoked_at, final_index, "one_way_continuity_revoked")
                )

        ordered = tuple(decisions[key] for key in sorted(decisions))
        plan = RedactionTrackPlan(
            input_frame_count=frame_index,
            profile_ids=tuple(sorted({anchor.profile_id for anchor in anchors})),
            decisions=ordered,
            ambiguous_ranges=tuple(ambiguous),
        )
        decisions_by_frame: dict[int, list[TrackFrameDecision]] = {}
        for decision in plan.decisions:
            decisions_by_frame.setdefault(decision.frame_index, []).append(decision)
        for row in evidence:
            rows = decisions_by_frame.get(int(row["frame_index"]), [])
            row["redaction_decision_reason_codes"] = [item.reason_code for item in rows]
            row["temporally_authorized_face_count"] = sum(item.authorized for item in rows)
        return TemporalAnalysisResult(plan, tuple(evidence), fps, frame_shape)

    def render(
        self,
        source_path: Path,
        destination: Path,
        plan: RedactionTrackPlan,
    ) -> ProcessingEvidence:
        """Render a fixed plan to a sibling temporary file, then atomically publish."""
        source_path = Path(source_path).resolve(strict=False)
        destination = Path(destination).resolve(strict=False)
        if source_path == destination:
            raise ValueError("output path must differ from input path")
        if destination.exists():
            raise FileExistsError(f"output already exists: {destination.name}")
        if not destination.parent.is_dir():
            raise FileNotFoundError("output directory is unavailable")
        temporary = destination.with_name(
            f".{destination.stem}.temporal-{uuid.uuid4().hex}{destination.suffix}"
        )
        source = OpenCvFrameSource(source_path)
        source.open()
        fps = source.fps if source.fps > 0 else 30.0
        writer = OpenCvFrameWriter(temporary, fps=fps, codec="mp4v")
        writer.open()
        decisions_by_frame: dict[int, list[TrackFrameDecision]] = {}
        for decision in plan.decisions:
            if decision.authorized:
                decisions_by_frame.setdefault(decision.frame_index, []).append(decision)
        frame_index = 0
        redacted_frames = 0
        try:
            while True:
                ok, frame = source.read()
                if not ok or frame is None:
                    break
                output = frame
                frame_decisions = decisions_by_frame.get(frame_index, [])
                for decision in frame_decisions:
                    if decision.bbox is not None:
                        output = _apply_effect_to_bbox(
                            output,
                            decision.bbox,
                            self._config.effect_mode,
                            self._config,
                            None,
                        )
                if frame_decisions:
                    redacted_frames += 1
                writer.write(output)
                frame_index += 1
            if frame_index != plan.input_frame_count:
                raise ValueError("rendered frame count does not match the analysis plan")
            writer.close()
            source.close()
            os.replace(temporary, destination)
        except Exception:
            writer.close()
            source.close()
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return ProcessingEvidence(
            input_frame_count=plan.input_frame_count,
            output_frame_count=frame_index,
            redacted_frame_count=redacted_frames,
            explicit_anchor_count=sum(
                item.reason_code == "explicit_gallery_anchor" for item in plan.decisions
            ),
            propagated_frame_count=sum(
                item.reason_code == "bidirectional_anchor_consensus" for item in plan.decisions
            ),
            ambiguous_ranges=plan.ambiguous_ranges,
        )
