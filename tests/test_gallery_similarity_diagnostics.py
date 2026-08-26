"""Batch gallery enrollment and diagnostic similarity tests."""

from __future__ import annotations

import numpy as np
import pytest

from consented_face_redactor.gallery import EnrollmentValidationError, LocalGallery


def _unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_profile_similarity_details_include_best_reference_and_centroid():
    gallery = LocalGallery()
    profile_id = gallery.enroll(_unit([1.0, 0.0]))
    gallery.add_reference(profile_id, _unit([0.0, 1.0]))

    details = gallery.profile_similarity_details(_unit([0.0, 1.0]), profile_id)

    assert details.best_reference_index == 1
    assert details.best_similarity == pytest.approx(1.0)
    assert details.centroid_similarity < 1.0


def test_enroll_many_is_atomic_when_later_vector_is_invalid():
    gallery = LocalGallery()
    existing = gallery.enroll(_unit([1.0, 0.0]))
    before = gallery.to_dict()

    with pytest.raises(EnrollmentValidationError):
        gallery.enroll_many([_unit([0.0, 1.0]), np.zeros(2, dtype=np.float32)])

    assert gallery.to_dict() == before
    assert gallery.profile_ids == [existing]


def test_enroll_many_adds_multiple_references_to_one_profile():
    gallery = LocalGallery()
    profile_id = gallery.enroll_many([_unit([1.0, 0.0]), _unit([0.0, 1.0])])

    assert gallery.profile_count == 1
    assert gallery.to_dict()["profiles"][profile_id]["v_count"] == 2
