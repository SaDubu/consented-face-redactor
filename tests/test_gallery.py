"""Tests for LocalGallery — deterministic ordering, malformed input rejection, calibrated threshold fixture support."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow importing src without install
_HERE = Path(__file__).resolve().parent.parent / "src"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from consented_face_redactor.gallery import (
    EnrollmentValidationError,
    LocalGallery,
    MatchResult,
)


class TestEnrollmentValidation:
    """Phase 4 §4.1: quality gates on enrollment."""

    def test_enrolls_valid_embedding(self):
        """Valid normalized vector should create one profile and return opaque ID."""
        vec = np.array([0.6, 0.8, 0.0]) / np.linalg.norm([0.6, 0.8, 0.0])
        gallery = LocalGallery()
        pid = gallery.enroll(vec)
        assert pid.startswith("prof-")
        assert len(gallery.profile_ids) == 1

    def test_enrollment_rejects_2d_vector(self):
        """Non-1-D embedding → EnrollmentValidationError."""
        vec = np.array([[0.6, 0.8]])
        gallery = LocalGallery()
        with pytest.raises(EnrollmentValidationError, match="invalid_shape"):
            gallery.enroll(vec)

    def test_enrollment_rejects_non_finite(self):
        """Vector containing NaN/Inf → EnrollmentValidationError."""
        vec = np.array([np.nan, 0.8, 0.2])
        gallery = LocalGallery()
        with pytest.raises(EnrollmentValidationError, match="non.finite"):
            gallery.enroll(vec)

    def test_enrollment_rejects_zero_norm_vector(self):
        """Zero vector → EnrollmentValidationError."""
        vec = np.zeros(3)
        gallery = LocalGallery()
        with pytest.raises(EnrollmentValidationError, match="zero.norm"):
            gallery.enroll(vec)

    def test_enrollment_rejects_duplicate_vector_in_same_profile(self):
        """Highly similar vectors (>= 0.95 cosine to centroid) should fail."""
        vec = np.array([1.0, 0.0, 0.0])
        gallery = LocalGallery()
        id1 = gallery.enroll(vec)

        near_dup = vec + np.array([1e-4, 1e-4, 0.0])  # cosine ≈ 0.9999
        with pytest.raises(EnrollmentValidationError, match="duplicate_vector"):
            gallery.enroll(near_dup)


class TestMatch:
    """Cosine matcher — threshold calibration & result format."""

    def test_exact_self_match_returns_high(self):
        """Matching a vector against itself should yield 'high' score (cosine=1.0)."""
        vec = np.array([0.6, 0.8, 0.0]) / np.linalg.norm([0.6, 0.8, 0.0])
        gallery = LocalGallery()
        pid = gallery.enroll(vec)

        results = gallery.match(vec)
        assert len(results) == 1
        assert results[0].profile_id == pid
        assert results[0].confidence == pytest.approx(1.0, abs=1e-6)
        assert results[0].score_category == "high"
        assert results[0].is_match is True

    def test_unrelated_faces_below_medium_threshold(self):
        """Orthogonal vectors should receive 0 cosine and not appear."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        gallery = LocalGallery()
        gallery.enroll(v1)

        results = gallery.match(v2, confidence_threshold=0.5)
        assert len(results) == 0

    def test_match_filters_by_confidence_threshold(self):
        """results[confidence] < threshold should be excluded from return."""
        vec = np.array([1.0, 0.1, 0.0]) / np.linalg.norm([1.0, 0.1, 0.0])
        gallery = LocalGallery()
        gallery.enroll(vec)

        # Threshold exactly at max cosine so any ≤1.0 result gets dropped
        results = gallery.match(vec, confidence_threshold=1.0 + 1e-9)
        assert len(results) == 0

    def test_match_returns_controlled_enumeration(self):
        """Results must use controlled categories: high/medium/nomatch."""
        vec = np.array([0.82, 0.57, 0.04]) / np.linalg.norm([0.82, 0.57, 0.04])
        gallery = LocalGallery()
        pid = gallery.enroll(vec)

        close_vec = vec + np.array([0.01, -0.01, 0.0])
        results = gallery.match(close_vec)
        assert len(results) >= 1
        for res in results:
            assert res.score_category in ("high", "medium", "nomatch")

    def test_multiple_candidates(self):
        """Multiple enrollments should allow matching against all profiles."""
        v1 = np.array([1.0, 0.0, 0.0]) / np.linalg.norm([1.0, 0.0, 0.0])
        v2 = np.array([0.7, 0.7, 0.0]) / np.linalg.norm([0.7, 0.7, 0.0])  # ~45deg rotation
        gallery = LocalGallery()
        pid1 = gallery.enroll(v1)
        pid2 = gallery.enroll(v2)

        results = gallery.match(v1, top_k=2)
        assert len(results) == 2
        assert results[0].profile_id == pid1  # highest similarity first


class TestPersistence:
    """Gallery save/load round-trip and serialization integrity."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """save() + load() should preserve all profile data."""
        gallery = LocalGallery()
        v1 = np.array([0.6, 0.8, 0.0]) / np.linalg.norm([0.6, 0.8, 0.0])
        pid1 = gallery.enroll(v1)

        gallery_path = tmp_path / "gallery.json"
        gallery.save(gallery_path)

        new_gallery = LocalGallery()
        new_gallery.load(gallery_path)
        assert new_gallery.profile_count == 1
        assert new_gallery.profile_ids == [pid1]


class TestSerialization:
    """from_dict/to_dict round-trip for test fixtures."""

    def test_to_dict_returns_expected_structure(self):
        """to_dict should return a dict with 'version', 'next_profile_counter', 'profiles'."""
        vec = np.array([0.6, 0.8, 0.0]) / np.linalg.norm([0.6, 0.8, 0.0])
        gallery = LocalGallery()
        gallery.enroll(vec)

        d = gallery.to_dict()
        assert "version" in d
        assert "next_profile_counter" in d
        assert "profiles" in d


class TestDeterministicOrdering:
    """Profiles must be returned in registration order for deterministic behavior."""

    def test_profile_ids_returns_registration_order(self):
        """profile_ids should maintain insertion order."""
        gallery = LocalGallery()
        v1 = np.array([1.0, 0.0, 0.0]) / np.linalg.norm([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0]) / np.linalg.norm([0.0, 1.0, 0.0])

        pid1 = gallery.enroll(v1)
        pid2 = gallery.enroll(v2)

        ids = gallery.profile_ids
        assert len(ids) == 2
        assert ids[0] < ids[1]  # profiles numbered sequentially


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
