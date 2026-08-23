"""Local gallery storage — opaque profile IDs, versioned vectors, no PII."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Zero-norm vector — cannot compute cosine similarity.")
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass(frozen=True)
class MatchResult:
    """Deterministic match decision for a single candidate."""

    profile_id: str
    confidence: float  # cosine similarity value
    score_category: str  # "high" | "medium" | "nomatch"

    @property
    def is_match(self) -> bool:
        """Treat 'high' as positive match and everything else as negative."""
        return self.score_category == "high"


@dataclass(frozen=True)
class EnrollmentValidationFailure:
    """Structured reason for registration rejection."""

    reason: str  # "no_face", "too_many_faces", "blur_detected", "extreme_pose",
                 # "bad_landmark_geometry", "duplicate_vector", "non_finite_vector",
                 # "zero_norm_vector"
    detail: Optional[str] = None


class LocalGalleryError(Exception):
    """Base exception for gallery operations."""


class EnrollmentValidationError(LocalGalleryError):
    """Raised when an enrollment fails validation."""

    def __init__(self, reason, detail=None):
        super().__init__(f"EnrollmentValidation: {reason} – {detail or ''}")
        self.reason = reason       # type: ignore[attr-defined]
        self.detail = detail       # type: ignore[attr-defined]


class VectorCollisionError(LocalGalleryError):
    """Raised when a candidate vector is too similar to an existing one."""


class LocalGallery:
    """Immutable-on-read / append-only gallery persisted as JSON.

    Privacy-by-design: no raw image paths, no human names, no crop pixels, and no debug data
    are ever written to the serialized format. Only opaque profile IDs and deterministic centroids
    survive across saves.
    """

    VERSION = 1

    # Controlled-enumeration categories — not raw model strings.
    SCORE_HIGH_THRESHOLD = 0.82  # cosine similarity >= this ⇒ "high"
    SCORE_MEDIUM_THRESHOLD = 0.55

    def __init__(self) -> None:
        self._version: int = self.VERSION
        self._profiles: dict[str, dict[str, Any]] = {}  # profile_id → data
        self._next_profile_counter: int = 0

    # ------------------------------------------------------------------ #
    # Enrollment (registration)
    # ------------------------------------------------------------------ #

    def enroll(
        self,
        embedding: np.ndarray,
        *,
        source_label: Optional[str] = None  # optional opaque tag; not a human name
    ) -> str:
        """Register one reference face.

        Raises ``EnrollmentValidationError`` when input fails quality gates.
        Returns the new opaque profile ID.
        """
        self._validate_embedding(embedding)

        # Generate new opaque ID (never reuse)
        profile_id = f"prof-{self._next_profile_counter:08x}"
        self._next_profile_counter += 1

        # Determine centroid as average of vectors in this profile
        vectors = [embedding.copy()]
        centroid = np.mean(vectors, axis=0)
        centroid /= np.linalg.norm(centroid)  # L2-normalize centroid

        self._profiles[profile_id] = {
            "version": self.VERSION,
            "vectors": [v.tolist() for v in vectors],  # serialize floats only
            "v_count": len(vectors),
            "centroid": centroid.tolist(),
            # Deliberately omit: image source path, human name, raw crop pixels, debug frames.
        }

        return profile_id

    def _validate_embedding(self, embedding: np.ndarray) -> None:
        """Quality gates for enrollment — mirror Phase 4 §4.1 requirements."""
        if embedding.ndim != 1:
            raise EnrollmentValidationError(
                reason="invalid_shape",
                detail=f"Expected 1-D vector, got shape {embedding.shape}",
            )

        if not np.isfinite(embedding).all():
            raise EnrollmentValidationError(reason="non_finite_vector")

        norm = float(np.linalg.norm(embedding))
        if norm == 0.0:
            raise EnrollmentValidationError(reason="zero_norm_vector")

        # L2-normalize before storing
        embedding_normalized = embedding / norm

        # Check for duplicate vectors within the same profile context
        for prof_id, data in self._profiles.items():
            centroids = np.array(data["centroid"])
            cos_sim = _cosine_similarity(embedding_normalized, centroids)
            # Reject if vector is too similar (likely duplicate enrollment)
            if cos_sim >= 0.95:
                raise EnrollmentValidationError(
                    reason="duplicate_vector",
                    detail=f"Vector too similar to existing profile {prof_id} (cosine={cos_sim:.4f})",
                )

    # ------------------------------------------------------------------ #
    # Matching (query)
    # ------------------------------------------------------------------ #

    def match(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 1,
        confidence_threshold: Optional[float] = None
    ) -> list[MatchResult]:
        """Find the closest profiles in this gallery.

        Returns *zero or more* MatchResults sorted by descending similarity.
        When ``confidence_threshold`` is set, results below it are dropped
        before returning (used by Phase 7 tracker gating).
        """
        if query_embedding.ndim != 1:
            raise ValueError(
                f"Expected 1-D query embedding, got shape {query_embedding.shape}"
            )

        # L2-normalize query
        norm = float(np.linalg.norm(query_embedding))
        if norm == 0.0:
            raise ValueError("Zero-norm query vector")
        q_norm = query_embedding / norm

        results: list[tuple[str, float]] = []
        for prof_id, data in self._profiles.items():
            centroid = np.array(data["centroid"])
            sim = _cosine_similarity(q_norm, centroid)
            results.append((prof_id, sim))

        # Sort by descending similarity
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        match_results: list[MatchResult] = []
        for prof_id, sm in results:
            if confidence_threshold is not None and sm < confidence_threshold:
                continue  # below threshold — silently drop (tracker handles gating)

            # Compute controlled-enumeration category
            if sm >= self.SCORE_HIGH_THRESHOLD:
                category = "high"
            elif sm >= self.SCORE_MEDIUM_THRESHOLD:
                category = "medium"
            else:
                category = "nomatch"

            match_results.append(MatchResult(
                profile_id=prof_id,
                confidence=round(sm, 6),
                score_category=category,
            ))

        return match_results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Serialize gallery to disk. *Vectors and centroids only.*"""
        payload = {
            "version": LocalGallery.VERSION,
            "next_profile_counter": self._next_profile_counter,
            "profiles": {},  # opaque profile_id → metadata dict
        }
        for pid, data in self._profiles.items():
            payload["profiles"][pid] = {
                "v_count": data["v_count"],
                "centroid": data["centroid"],
                "vectors": data["vectors"],
                # Deliberately absent: source paths, human names, images, crops, debug frames.
            }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Load gallery from disk (replaces current state)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != self.VERSION:
            raise LocalGalleryError(f"Unsupported gallery version: {data.get('version')}")

        self._version = self.VERSION
        self._next_profile_counter = data["next_profile_counter"]
        self._profiles = data["profiles"]  # trusted deserialization (JSON format)

    def to_dict(self) -> dict[str, Any]:
        """Serialize current state (used in tests / CLI serialization round-trips)."""
        return {
            "version": LocalGallery.VERSION,
            "next_profile_counter": self._next_profile_counter,
            "profiles": self._profiles,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LocalGallery:
        """Reconstruct gallery from a Python dict (test fixture only)."""
        inst = cls.__new__(cls)
        inst._version = data["version"]
        inst._next_profile_counter = data["next_profile_counter"]
        inst._profiles = data["profiles"]
        return inst

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @property
    def profile_count(self) -> int:
        """Number of registered profiles (0 if empty)."""
        return len(self._profiles)

    @property
    def profile_ids(self) -> list[str]:
        """List all opaque profile IDs in registration order."""
        return sorted(self._profiles.keys())
