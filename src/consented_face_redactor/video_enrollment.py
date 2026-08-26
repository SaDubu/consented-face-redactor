"""Target-only video enrollment with diversity-based reference coverage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median
from typing import Any, Iterator, Sequence

import numpy as np

from consented_face_redactor.adapters.detection_iface import DetectorAdapter, EmbedderAdapter
from consented_face_redactor.gallery import LocalGallery
from consented_face_redactor.media import FrameSource


@dataclass(frozen=True, slots=True)
class VideoEnrollmentOptions:
    """Non-authorizing controls for sampling and reference coverage."""

    sample_every_n_frames: int = 6
    max_references: int = 64
    duplicate_similarity: float = 0.995
    minimum_cluster_similarity: float = 0.45
    min_face_width: int = 32
    min_face_height: int = 32
    max_review_candidates: int = 32

    def __post_init__(self) -> None:
        for name in (
            "sample_every_n_frames",
            "max_references",
            "min_face_width",
            "min_face_height",
            "max_review_candidates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("duplicate_similarity", "minimum_cluster_similarity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class EnrollmentCandidate:
    """A valid vector from a target-only registration video."""

    frame_index: int
    timestamp_s: float
    embedding: np.ndarray
    detector_confidence: float

    def __post_init__(self) -> None:
        vector = np.asarray(self.embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size < 1 or not np.isfinite(vector).all():
            raise ValueError("embedding must be a finite non-empty vector")
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float32).eps:
            raise ValueError("embedding must have non-zero norm")
        detached = vector.copy() / norm
        detached.setflags(write=False)
        object.__setattr__(self, "embedding", detached)
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if not np.isfinite(self.detector_confidence) or not 0.0 <= self.detector_confidence <= 1.0:
            raise ValueError("detector_confidence must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class EnrollmentSkip:
    """One sampled frame intentionally excluded from automatic enrollment."""

    frame_index: int
    timestamp_s: float
    reason_code: str


@dataclass(frozen=True, slots=True)
class EnrollmentReport:
    """Privacy-minimal enrollment evidence without pixels or vectors."""

    input_frame_count: int | None
    sampled_frame_count: int
    candidate_count: int
    selected_reference_count: int
    no_face_count: int
    multiple_face_count: int
    face_too_small_count: int
    embedding_error_count: int
    duplicate_count: int
    review_frame_indices: tuple[int, ...]
    nearest_similarity_min: float | None
    nearest_similarity_median: float | None
    nearest_similarity_p95: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "input_frame_count": self.input_frame_count,
            "sampled_frame_count": self.sampled_frame_count,
            "candidate_count": self.candidate_count,
            "selected_reference_count": self.selected_reference_count,
            "no_face_count": self.no_face_count,
            "multiple_face_count": self.multiple_face_count,
            "face_too_small_count": self.face_too_small_count,
            "embedding_error_count": self.embedding_error_count,
            "duplicate_count": self.duplicate_count,
            "review_frame_indices": list(self.review_frame_indices),
            "nearest_similarity_min": self.nearest_similarity_min,
            "nearest_similarity_median": self.nearest_similarity_median,
            "nearest_similarity_p95": self.nearest_similarity_p95,
        }


def iter_sampled_frames(
    source: FrameSource, *, sample_every_n_frames: int
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield sampled BGR frames without retaining previously read frames."""
    if isinstance(sample_every_n_frames, bool) or not isinstance(sample_every_n_frames, int) or sample_every_n_frames < 1:
        raise ValueError("sample_every_n_frames must be a positive integer")
    frame_index = 0
    fps = source.fps
    while True:
        success, frame = source.read()
        if not success or frame is None:
            return
        if frame_index % sample_every_n_frames == 0:
            timestamp = frame_index / fps if isinstance(fps, (int, float)) and fps > 0 else float(frame_index)
            yield frame_index, timestamp, frame
        frame_index += 1


def extract_enrollment_candidate(
    frame: np.ndarray,
    *,
    frame_index: int,
    timestamp_s: float,
    detector: DetectorAdapter,
    embedder: EmbedderAdapter,
    options: VideoEnrollmentOptions,
) -> EnrollmentCandidate | EnrollmentSkip:
    """Extract one valid target-only candidate or an auditable skip reason."""
    try:
        detections = list(detector.detect(frame))
    except Exception:
        return EnrollmentSkip(frame_index, timestamp_s, "detector_error")
    if not detections:
        return EnrollmentSkip(frame_index, timestamp_s, "no_face")
    if len(detections) != 1:
        return EnrollmentSkip(frame_index, timestamp_s, "multiple_faces")
    detection = detections[0]
    bbox = getattr(detection, "bbox", None)
    width = getattr(bbox, "width", 0)
    height = getattr(bbox, "height", 0)
    if width < options.min_face_width or height < options.min_face_height:
        return EnrollmentSkip(frame_index, timestamp_s, "face_too_small")
    try:
        vector, _revision = embedder.embed(frame, detection)
        return EnrollmentCandidate(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            embedding=vector,
            detector_confidence=float(detection.confidence),
        )
    except Exception:
        return EnrollmentSkip(frame_index, timestamp_s, "embedding_error")


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def nearest_reference_similarity(
    candidate: np.ndarray, references: Sequence[np.ndarray]
) -> float | None:
    """Return the closest selected reference similarity for coverage metrics."""
    if not references:
        return None
    vector = np.asarray(candidate, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size < 1 or norm <= np.finfo(np.float32).eps:
        raise ValueError("candidate must be a non-zero vector")
    vector = vector / norm
    scores = []
    for reference in references:
        item = np.asarray(reference, dtype=np.float32).reshape(-1)
        if item.shape != vector.shape:
            raise ValueError("reference dimensions must match")
        item_norm = float(np.linalg.norm(item))
        if item_norm <= np.finfo(np.float32).eps:
            raise ValueError("reference must be a non-zero vector")
        scores.append(_similarity(vector, item / item_norm))
    return max(scores)


def select_diverse_references(
    candidates: Sequence[EnrollmentCandidate],
    *,
    options: VideoEnrollmentOptions,
) -> tuple[list[EnrollmentCandidate], list[EnrollmentCandidate], int]:
    """Select coverage references; retain extreme vectors only for review.

    ``duplicate_similarity`` removes repeated adjacent appearances. Negative
    cosine similarity to every retained reference is treated as an outlier to
    review rather than being silently enrolled.
    """
    deduplicated: list[EnrollmentCandidate] = []
    review: list[EnrollmentCandidate] = []
    duplicate_count = 0
    for candidate in candidates:
        nearest = nearest_reference_similarity(
            candidate.embedding, [item.embedding for item in deduplicated]
        )
        if nearest is not None and nearest >= options.duplicate_similarity:
            duplicate_count += 1
            continue
        if len(deduplicated) >= 2:
            centroid = np.mean(
                np.stack([item.embedding for item in deduplicated]), axis=0
            )
            centroid /= float(np.linalg.norm(centroid))
            if _similarity(candidate.embedding, centroid) >= options.duplicate_similarity:
                duplicate_count += 1
                continue
        if nearest is not None and nearest <= -0.5:
            if len(review) < options.max_review_candidates:
                review.append(candidate)
            continue
        deduplicated.append(candidate)

    # A persistent false detector crop can form a small, internally coherent
    # embedding island. Preserve the largest connected view component and send
    # disconnected islands to review. Chained edges retain extreme poses when
    # intermediate views connect them to the dominant target trajectory.
    if len(deduplicated) >= 4:
        vectors = np.stack([item.embedding for item in deduplicated])
        similarities = vectors @ vectors.T
        seen: set[int] = set()
        components: list[list[int]] = []
        for start in range(len(deduplicated)):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            component: list[int] = []
            while stack:
                index = stack.pop()
                component.append(index)
                neighbors = np.where(
                    similarities[index] >= options.minimum_cluster_similarity
                )[0]
                for neighbor_value in neighbors:
                    neighbor = int(neighbor_value)
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            components.append(sorted(component))
        components.sort(key=lambda item: (-len(item), item[0]))
        dominant = set(components[0])
        disconnected = [
            item for index, item in enumerate(deduplicated) if index not in dominant
        ]
        available = max(0, options.max_review_candidates - len(review))
        review.extend(disconnected[:available])
        deduplicated = [
            item for index, item in enumerate(deduplicated) if index in dominant
        ]

    if len(deduplicated) <= options.max_references:
        return deduplicated, review, duplicate_count

    selected = [deduplicated[0]]
    remaining = deduplicated[1:]
    while remaining and len(selected) < options.max_references:
        choice_index = min(
            range(len(remaining)),
            key=lambda index: nearest_reference_similarity(
                remaining[index].embedding, [item.embedding for item in selected]
            ),
        )
        selected.append(remaining.pop(choice_index))
    return selected, review, duplicate_count


class VideoEnrollmentService:
    """Collect and enroll diverse references without any tracking component."""

    def __init__(
        self,
        *,
        detector: DetectorAdapter,
        embedder: EmbedderAdapter,
        options: VideoEnrollmentOptions,
    ) -> None:
        self._detector = detector
        self._embedder = embedder
        self._options = options

    def collect(self, source: FrameSource) -> tuple[list[EnrollmentCandidate], EnrollmentReport]:
        """Scan an already-open source once and return candidates plus skips."""
        candidates: list[EnrollmentCandidate] = []
        skipped: list[EnrollmentSkip] = []
        sampled = 0
        for frame_index, timestamp_s, frame in iter_sampled_frames(
            source, sample_every_n_frames=self._options.sample_every_n_frames
        ):
            sampled += 1
            result = extract_enrollment_candidate(
                frame,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                detector=self._detector,
                embedder=self._embedder,
                options=self._options,
            )
            if isinstance(result, EnrollmentCandidate):
                candidates.append(result)
            else:
                skipped.append(result)
        counts = {reason: sum(item.reason_code == reason for item in skipped) for reason in (
            "no_face", "multiple_faces", "face_too_small", "embedding_error"
        )}
        return candidates, EnrollmentReport(
            input_frame_count=source.frame_count if source.frame_count >= 0 else None,
            sampled_frame_count=sampled,
            candidate_count=len(candidates),
            selected_reference_count=0,
            no_face_count=counts["no_face"],
            multiple_face_count=counts["multiple_faces"],
            face_too_small_count=counts["face_too_small"],
            embedding_error_count=counts["embedding_error"],
            duplicate_count=0,
            review_frame_indices=(),
            nearest_similarity_min=None,
            nearest_similarity_median=None,
            nearest_similarity_p95=None,
        )

    def select(
        self,
        candidates: Sequence[EnrollmentCandidate],
        report: EnrollmentReport,
    ) -> tuple[list[EnrollmentCandidate], EnrollmentReport]:
        """Apply diversity selection and update only report-derived metrics."""
        selected, review, duplicate_count = select_diverse_references(
            candidates, options=self._options
        )
        similarities: list[float] = []
        for index, candidate in enumerate(selected):
            others = [item.embedding for other_index, item in enumerate(selected) if other_index != index]
            nearest = nearest_reference_similarity(candidate.embedding, others)
            if nearest is not None:
                similarities.append(nearest)
        p95 = float(np.percentile(similarities, 95)) if similarities else None
        return selected, replace(
            report,
            selected_reference_count=len(selected),
            duplicate_count=duplicate_count,
            review_frame_indices=tuple(item.frame_index for item in review),
            nearest_similarity_min=min(similarities) if similarities else None,
            nearest_similarity_median=median(similarities) if similarities else None,
            nearest_similarity_p95=p95,
        )

    def enroll(
        self,
        source: FrameSource,
        gallery: LocalGallery,
        *,
        profile_id: str | None = None,
    ) -> tuple[str, EnrollmentReport]:
        """Collect, select, and atomically add a target-only reference set."""
        candidates, report = self.collect(source)
        selected, report = self.select(candidates, report)
        if not selected:
            raise ValueError("No diverse enrollment references were selected")
        enrolled_profile_id = gallery.enroll_many(
            [item.embedding.copy() for item in selected], profile_id=profile_id
        )
        return enrolled_profile_id, report
