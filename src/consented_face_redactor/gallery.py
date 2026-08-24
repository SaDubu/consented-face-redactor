"""Validated local storage and matching for consented face embeddings."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


_PROFILE_ID_RE = re.compile(r"^prof-([0-9a-f]{8})$")
_ROOT_KEYS = {
    "version",
    "embedding_dimension",
    "thresholds",
    "next_profile_counter",
    "profiles",
}
_PROFILE_KEYS = {"version", "v_count", "centroid", "vectors"}
_THRESHOLD_KEYS = {"high", "medium", "profile_collision", "duplicate"}
_MAX_PROFILE_NUMBER = 0xFFFFFFFF


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return a bounded cosine similarity for two finite vectors."""
    if a.shape != b.shape:
        raise ValueError("Embedding dimensions do not match")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= np.finfo(np.float32).eps or norm_b <= np.finfo(np.float32).eps:
        raise ValueError("Cannot compare a zero-norm embedding")
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, similarity))


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LocalGalleryError("Gallery JSON contains a duplicate key")
        result[key] = value
    return result


@dataclass(frozen=True)
class MatchResult:
    """Deterministic match decision for one gallery profile."""

    profile_id: str
    confidence: float
    score_category: str

    @property
    def is_match(self) -> bool:
        """Only the calibrated high category is a positive identity match."""
        return self.score_category == "high"


@dataclass(frozen=True)
class EnrollmentValidationFailure:
    """Structured registration rejection reason."""

    reason: str
    detail: Optional[str] = None


class LocalGalleryError(Exception):
    """Base exception for gallery operations."""


class EnrollmentValidationError(LocalGalleryError):
    """Raised when an enrollment vector fails a controlled quality gate."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        message = f"EnrollmentValidation: {reason}"
        if detail:
            message = f"{message} - {detail}"
        super().__init__(message)
        self.reason = reason
        self.detail = detail


class VectorCollisionError(EnrollmentValidationError):
    """Raised when a reference could belong to another stored profile."""

    def __init__(self, profile_id: str, similarity: float) -> None:
        super().__init__(
            "duplicate_vector",
            f"Reference collides with {profile_id} (cosine={similarity:.4f})",
        )
        self.profile_id = profile_id
        self.similarity = similarity


class LocalGallery:
    """Append-only local gallery containing opaque IDs and normalized vectors."""

    VERSION = 2
    SCORE_HIGH_THRESHOLD = 0.82
    SCORE_MEDIUM_THRESHOLD = 0.55
    PROFILE_COLLISION_THRESHOLD = 0.95
    DUPLICATE_VECTOR_THRESHOLD = 0.9999

    def __init__(
        self,
        *,
        high_threshold: float = SCORE_HIGH_THRESHOLD,
        medium_threshold: float = SCORE_MEDIUM_THRESHOLD,
        profile_collision_threshold: float = PROFILE_COLLISION_THRESHOLD,
        duplicate_vector_threshold: float = DUPLICATE_VECTOR_THRESHOLD,
    ) -> None:
        high = self._threshold(high_threshold, "high_threshold")
        medium = self._threshold(medium_threshold, "medium_threshold")
        collision = self._threshold(
            profile_collision_threshold, "profile_collision_threshold"
        )
        duplicate = self._threshold(
            duplicate_vector_threshold, "duplicate_vector_threshold"
        )
        if medium >= high:
            raise ValueError("medium_threshold must be lower than high_threshold")
        if collision > duplicate:
            raise ValueError(
                "profile_collision_threshold cannot exceed duplicate_vector_threshold"
            )

        self._high_threshold = high
        self._medium_threshold = medium
        self._profile_collision_threshold = collision
        self._duplicate_vector_threshold = duplicate
        self._profiles: dict[str, dict[str, Any]] = {}
        self._next_profile_counter = 0
        self._embedding_dimension: int | None = None

    @staticmethod
    def _threshold(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be numeric")
        result = float(value)
        if not np.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return result

    @staticmethod
    def _normalize_enrollment(embedding: np.ndarray) -> np.ndarray:
        if not isinstance(embedding, np.ndarray):
            raise EnrollmentValidationError("invalid_type")
        if embedding.ndim != 1 or embedding.size < 1:
            raise EnrollmentValidationError(
                "invalid_shape", f"Expected a non-empty 1-D vector, got {embedding.shape}"
            )
        if embedding.dtype.kind not in "fiu":
            raise EnrollmentValidationError("invalid_dtype")
        vector = embedding.astype(np.float32, copy=True)
        if not np.isfinite(vector).all():
            raise EnrollmentValidationError("non_finite_vector")
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float32).eps:
            raise EnrollmentValidationError("zero_norm_vector")
        return vector / norm

    def _assert_dimension(self, vector: np.ndarray, *, enrollment: bool) -> None:
        if self._embedding_dimension is None:
            return
        if vector.size != self._embedding_dimension:
            message = f"Expected dimension {self._embedding_dimension}, got {vector.size}"
            if enrollment:
                raise EnrollmentValidationError("incompatible_embedding", message)
            raise ValueError(message)

    @staticmethod
    def _normalized_centroid(vectors: list[np.ndarray]) -> np.ndarray:
        centroid = np.mean(np.stack(vectors), axis=0, dtype=np.float64).astype(np.float32)
        norm = float(np.linalg.norm(centroid))
        if norm <= np.finfo(np.float32).eps:
            raise EnrollmentValidationError(
                "incompatible_embedding", "References produce a zero-norm centroid"
            )
        return centroid / norm

    @staticmethod
    def _profile_vectors(data: dict[str, Any]) -> list[np.ndarray]:
        return [np.asarray(vector, dtype=np.float32) for vector in data["vectors"]]

    def _profile_similarity(self, vector: np.ndarray, data: dict[str, Any]) -> float:
        candidates = self._profile_vectors(data)
        candidates.append(np.asarray(data["centroid"], dtype=np.float32))
        return max(_cosine_similarity(vector, candidate) for candidate in candidates)

    def enroll(
        self,
        embedding: np.ndarray,
        *,
        profile_id: str | None = None,
    ) -> str:
        """Create a profile or append a reference to an existing profile."""
        vector = self._normalize_enrollment(embedding)
        self._assert_dimension(vector, enrollment=True)

        if profile_id is not None:
            if not isinstance(profile_id, str) or _PROFILE_ID_RE.fullmatch(profile_id) is None:
                raise EnrollmentValidationError("invalid_profile_id")
            if profile_id not in self._profiles:
                raise EnrollmentValidationError("unknown_profile")

        for existing_id, data in self._profiles.items():
            similarity = self._profile_similarity(vector, data)
            if existing_id == profile_id:
                if similarity >= self._duplicate_vector_threshold:
                    raise EnrollmentValidationError(
                        "duplicate_vector", f"Reference already exists in {existing_id}"
                    )
            elif similarity >= self._profile_collision_threshold:
                raise VectorCollisionError(existing_id, similarity)

        if profile_id is None:
            if self._next_profile_counter > _MAX_PROFILE_NUMBER:
                raise LocalGalleryError("Gallery profile ID space is exhausted")
            profile_id = f"prof-{self._next_profile_counter:08x}"
            self._next_profile_counter += 1
            vectors = [vector]
        else:
            vectors = self._profile_vectors(self._profiles[profile_id])
            vectors.append(vector)

        centroid = self._normalized_centroid(vectors)
        if self._embedding_dimension is None:
            self._embedding_dimension = int(vector.size)
        self._profiles[profile_id] = {
            "version": self.VERSION,
            "vectors": [item.tolist() for item in vectors],
            "v_count": len(vectors),
            "centroid": centroid.tolist(),
        }
        return profile_id

    def add_reference(self, profile_id: str, embedding: np.ndarray) -> str:
        """Append a distinct reference vector to an existing opaque profile."""
        return self.enroll(embedding, profile_id=profile_id)

    def match(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 1,
        confidence_threshold: Optional[float] = None,
    ) -> list[MatchResult]:
        """Return deterministic profile matches ordered by descending similarity."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if confidence_threshold is not None:
            if isinstance(confidence_threshold, bool) or not isinstance(
                confidence_threshold, (int, float, np.integer, np.floating)
            ):
                raise TypeError("confidence_threshold must be numeric")
            confidence_threshold = float(confidence_threshold)
            if not np.isfinite(confidence_threshold) or not -1.0 <= confidence_threshold <= 1.0:
                raise ValueError("confidence_threshold must be finite and in [-1, 1]")

        try:
            query = self._normalize_enrollment(query_embedding)
        except EnrollmentValidationError as exc:
            raise ValueError(f"Invalid query embedding: {exc.reason}") from exc
        self._assert_dimension(query, enrollment=False)

        ranked = [
            (profile_id, self._profile_similarity(query, data))
            for profile_id, data in self._profiles.items()
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))

        results: list[MatchResult] = []
        for profile_id, similarity in ranked[:top_k]:
            if confidence_threshold is not None and similarity < confidence_threshold:
                continue
            if similarity >= self._high_threshold:
                category = "high"
            elif similarity >= self._medium_threshold:
                category = "medium"
            else:
                category = "nomatch"
            results.append(
                MatchResult(
                    profile_id=profile_id,
                    confidence=round(similarity, 6),
                    score_category=category,
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        """Atomically write a validated, privacy-minimal JSON gallery."""
        target = Path(path).expanduser().resolve(strict=False)
        if not target.parent.is_dir():
            raise LocalGalleryError("Gallery directory is unavailable")
        if target.exists() and not target.is_file():
            raise LocalGalleryError("Gallery destination is not a regular file")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(temporary_path, 0o600)
            except OSError:
                pass
            os.replace(temporary_path, target)
            temporary_path = None
        except OSError as exc:
            raise LocalGalleryError(f"Gallery could not be saved: {target.name}") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def load(self, path: str | Path) -> None:
        """Validate a gallery completely before replacing the current state."""
        source = Path(path).expanduser().resolve(strict=False)
        try:
            text = source.read_text(encoding="utf-8")
            raw = json.loads(text, object_pairs_hook=_json_object_without_duplicate_keys)
        except LocalGalleryError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LocalGalleryError(f"Gallery is unreadable or invalid: {source.name}") from exc

        candidate = self.from_dict(raw)
        self._high_threshold = candidate._high_threshold
        self._medium_threshold = candidate._medium_threshold
        self._profile_collision_threshold = candidate._profile_collision_threshold
        self._duplicate_vector_threshold = candidate._duplicate_vector_threshold
        self._profiles = candidate._profiles
        self._next_profile_counter = candidate._next_profile_counter
        self._embedding_dimension = candidate._embedding_dimension

    def to_dict(self) -> dict[str, Any]:
        """Return a detached serialization payload containing no PII fields."""
        profiles: dict[str, dict[str, Any]] = {}
        for profile_id in self.profile_ids:
            data = self._profiles[profile_id]
            profiles[profile_id] = {
                "version": data["version"],
                "v_count": data["v_count"],
                "centroid": list(data["centroid"]),
                "vectors": [list(vector) for vector in data["vectors"]],
            }
        return {
            "version": self.VERSION,
            "embedding_dimension": self._embedding_dimension,
            "thresholds": {
                "high": self._high_threshold,
                "medium": self._medium_threshold,
                "profile_collision": self._profile_collision_threshold,
                "duplicate": self._duplicate_vector_threshold,
            },
            "next_profile_counter": self._next_profile_counter,
            "profiles": profiles,
        }

    @staticmethod
    def _saved_vector(value: Any, dimension: int, name: str) -> np.ndarray:
        if not isinstance(value, list) or len(value) != dimension:
            raise LocalGalleryError(f"{name} has an incompatible dimension")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
        ):
            raise LocalGalleryError(f"{name} must contain only numbers")
        vector = np.asarray(value, dtype=np.float32)
        if not np.isfinite(vector).all():
            raise LocalGalleryError(f"{name} must be finite")
        norm = float(np.linalg.norm(vector))
        if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-6):
            raise LocalGalleryError(f"{name} must be L2-normalized")
        return vector / norm

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LocalGallery:
        """Construct a gallery from a strictly validated serialization payload."""
        if not isinstance(data, dict):
            raise LocalGalleryError("Gallery payload must be an object")
        if set(data) != _ROOT_KEYS:
            raise LocalGalleryError("Gallery payload has missing or unknown fields")
        if data["version"] != cls.VERSION:
            raise LocalGalleryError("Unsupported gallery version")

        thresholds = data["thresholds"]
        if not isinstance(thresholds, dict) or set(thresholds) != _THRESHOLD_KEYS:
            raise LocalGalleryError("Gallery thresholds are invalid")
        try:
            instance = cls(
                high_threshold=thresholds["high"],
                medium_threshold=thresholds["medium"],
                profile_collision_threshold=thresholds["profile_collision"],
                duplicate_vector_threshold=thresholds["duplicate"],
            )
        except (TypeError, ValueError) as exc:
            raise LocalGalleryError("Gallery thresholds are invalid") from exc

        profiles = data["profiles"]
        if not isinstance(profiles, dict):
            raise LocalGalleryError("Gallery profiles must be an object")
        if any(not isinstance(profile_id, str) for profile_id in profiles):
            raise LocalGalleryError("Gallery profile IDs must be strings")
        counter = data["next_profile_counter"]
        if (
            isinstance(counter, bool)
            or not isinstance(counter, int)
            or not 0 <= counter <= _MAX_PROFILE_NUMBER + 1
        ):
            raise LocalGalleryError("Gallery profile counter is invalid")
        dimension = data["embedding_dimension"]
        if profiles:
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
                raise LocalGalleryError("Gallery embedding dimension is invalid")
        elif dimension is not None:
            raise LocalGalleryError("An empty gallery must not declare a dimension")

        highest_profile_number = -1
        for profile_id in sorted(profiles):
            match = _PROFILE_ID_RE.fullmatch(profile_id)
            if match is None:
                raise LocalGalleryError("Gallery contains an invalid profile ID")
            highest_profile_number = max(highest_profile_number, int(match.group(1), 16))
            profile = profiles[profile_id]
            if not isinstance(profile, dict) or set(profile) != _PROFILE_KEYS:
                raise LocalGalleryError("Gallery profile has missing or unknown fields")
            if profile["version"] != cls.VERSION:
                raise LocalGalleryError("Gallery profile version is unsupported")
            vectors_raw = profile["vectors"]
            count = profile["v_count"]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
                or not isinstance(vectors_raw, list)
                or len(vectors_raw) != count
            ):
                raise LocalGalleryError("Gallery profile vector count is invalid")

            assert isinstance(dimension, int)
            vectors = [
                cls._saved_vector(item, dimension, "Gallery vector")
                for item in vectors_raw
            ]
            for index, vector in enumerate(vectors):
                for previous in vectors[:index]:
                    if (
                        _cosine_similarity(vector, previous)
                        >= instance._duplicate_vector_threshold
                    ):
                        raise LocalGalleryError(
                            "Gallery profile contains duplicate reference vectors"
                        )
            centroid = cls._saved_vector(profile["centroid"], dimension, "Gallery centroid")
            expected_centroid = cls._normalized_centroid(vectors)
            if not np.allclose(centroid, expected_centroid, rtol=1e-5, atol=1e-6):
                raise LocalGalleryError("Gallery centroid does not match its vectors")

            for existing_id, existing in instance._profiles.items():
                if any(
                    instance._profile_similarity(vector, existing)
                    >= instance._profile_collision_threshold
                    for vector in vectors
                ):
                    raise LocalGalleryError(
                        f"Gallery profiles {existing_id} and {profile_id} collide"
                    )
            instance._profiles[profile_id] = {
                "version": cls.VERSION,
                "v_count": count,
                "vectors": [vector.tolist() for vector in vectors],
                "centroid": centroid.tolist(),
            }

        if counter <= highest_profile_number:
            raise LocalGalleryError("Gallery profile counter would reuse an ID")
        instance._next_profile_counter = counter
        instance._embedding_dimension = dimension
        return instance

    @property
    def profile_count(self) -> int:
        return len(self._profiles)

    @property
    def profile_ids(self) -> list[str]:
        return sorted(self._profiles)

    @property
    def embedding_dimension(self) -> int | None:
        return self._embedding_dimension

    def save_to_json_file(self, path: str | Path) -> None:
        """Write the gallery to a JSON file at *path*."""
        import os
        import tempfile

        target = Path(path).expanduser().resolve(strict=False)
        if not target.parent.is_dir():
            raise LocalGalleryError("Gallery directory is unavailable")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.tmp_",
            suffix=".json",
            delete=False,
        ) as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        target.replace(Path(handle.name))

    @classmethod
    def from_json_file(cls, path: str | Path) -> "LocalGallery":
        """Load and validate a gallery JSON file and return an instance."""
        source = Path(path).expanduser().resolve(strict=False)
        text = source.read_text(encoding="utf-8")
        raw = json.loads(text)
        return cls.from_dict(raw)
