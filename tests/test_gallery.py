"""Tests for validated enrollment, matching, and gallery persistence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from consented_face_redactor.gallery import (
    EnrollmentValidationError,
    LocalGallery,
    LocalGalleryError,
    VectorCollisionError,
)


def _unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class TestEnrollment:
    def test_normalizes_before_storing(self):
        gallery = LocalGallery()
        profile_id = gallery.enroll(np.array([3.0, 4.0, 0.0]))

        stored = np.asarray(gallery.to_dict()["profiles"][profile_id]["vectors"][0])
        np.testing.assert_allclose(stored, [0.6, 0.8, 0.0])
        assert gallery.embedding_dimension == 3

    def test_adds_multiple_references_to_one_profile(self):
        gallery = LocalGallery()
        profile_id = gallery.enroll(_unit([1.0, 0.0, 0.0]))

        returned_id = gallery.add_reference(profile_id, _unit([0.98, 0.2, 0.0]))

        profile = gallery.to_dict()["profiles"][profile_id]
        assert returned_id == profile_id
        assert gallery.profile_count == 1
        assert profile["v_count"] == 2
        assert len(profile["vectors"]) == 2

    @pytest.mark.parametrize(
        "embedding,reason",
        [
            (np.array([[0.6, 0.8]]), "invalid_shape"),
            (np.array([]), "invalid_shape"),
            (np.array([np.nan, 0.8]), "non_finite_vector"),
            (np.array([np.inf, 0.8]), "non_finite_vector"),
            (np.zeros(3), "zero_norm_vector"),
            (np.array([True, False]), "invalid_dtype"),
        ],
    )
    def test_rejects_malformed_enrollment(self, embedding, reason):
        with pytest.raises(EnrollmentValidationError) as error:
            LocalGallery().enroll(embedding)
        assert error.value.reason == reason

    def test_rejects_non_array_input(self):
        with pytest.raises(EnrollmentValidationError) as error:
            LocalGallery().enroll([1.0, 0.0])
        assert error.value.reason == "invalid_type"

    def test_rejects_dimension_change(self):
        gallery = LocalGallery()
        gallery.enroll(_unit([1.0, 0.0, 0.0]))
        with pytest.raises(EnrollmentValidationError) as error:
            gallery.enroll(_unit([1.0, 0.0]))
        assert error.value.reason == "incompatible_embedding"

    def test_rejects_duplicate_reference_in_same_profile(self):
        gallery = LocalGallery()
        vector = _unit([1.0, 0.0, 0.0])
        profile_id = gallery.enroll(vector)
        with pytest.raises(EnrollmentValidationError) as error:
            gallery.add_reference(profile_id, vector.copy())
        assert error.value.reason == "duplicate_vector"

    def test_rejects_cross_profile_collision_without_consuming_id(self):
        gallery = LocalGallery()
        gallery.enroll(_unit([1.0, 0.0, 0.0]))

        with pytest.raises(VectorCollisionError):
            gallery.enroll(_unit([1.0, 0.01, 0.0]))

        second_id = gallery.enroll(_unit([0.0, 1.0, 0.0]))
        assert second_id == "prof-00000001"

    def test_invalid_profile_does_not_set_gallery_dimension(self):
        gallery = LocalGallery()
        with pytest.raises(EnrollmentValidationError):
            gallery.add_reference("not-a-profile", _unit([1.0, 0.0]))
        assert gallery.embedding_dimension is None


class TestMatch:
    def test_exact_reference_match_is_high(self):
        gallery = LocalGallery()
        vector = _unit([0.6, 0.8, 0.0])
        profile_id = gallery.enroll(vector)

        result = gallery.match(vector)[0]

        assert result.profile_id == profile_id
        assert result.confidence == pytest.approx(1.0)
        assert result.score_category == "high"
        assert result.is_match is True

    def test_compares_individual_references_as_well_as_centroid(self):
        gallery = LocalGallery()
        profile_id = gallery.enroll(_unit([1.0, 0.0, 0.0]))
        second = _unit([0.8, 0.6, 0.0])
        gallery.add_reference(profile_id, second)

        result = gallery.match(second)[0]

        assert result.confidence == pytest.approx(1.0)

    def test_filters_below_threshold(self):
        gallery = LocalGallery()
        gallery.enroll(_unit([1.0, 0.0, 0.0]))
        assert gallery.match(_unit([0.0, 1.0, 0.0]), confidence_threshold=0.5) == []

    def test_uses_calibrated_categories(self):
        gallery = LocalGallery(high_threshold=0.9, medium_threshold=0.5)
        gallery.enroll(_unit([1.0, 0.0]))

        result = gallery.match(_unit([0.8, 0.6]))[0]

        assert result.confidence == pytest.approx(0.8)
        assert result.score_category == "medium"

    def test_orders_ties_by_opaque_profile_id(self):
        gallery = LocalGallery(profile_collision_threshold=0.99)
        first = gallery.enroll(_unit([1.0, 0.0, 0.0]))
        second = gallery.enroll(_unit([0.0, 1.0, 0.0]))

        results = gallery.match(_unit([1.0, 1.0, 0.0]), top_k=2)

        assert [result.profile_id for result in results] == [first, second]

    @pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
    def test_rejects_invalid_top_k(self, top_k):
        with pytest.raises(ValueError):
            LocalGallery().match(_unit([1.0, 0.0]), top_k=top_k)

    @pytest.mark.parametrize("threshold", [float("nan"), -1.01, 1.01, True])
    def test_rejects_invalid_confidence_threshold(self, threshold):
        with pytest.raises((TypeError, ValueError)):
            LocalGallery().match(
                _unit([1.0, 0.0]), confidence_threshold=threshold
            )

    def test_empty_match_does_not_mutate_dimension(self):
        gallery = LocalGallery()
        assert gallery.match(_unit([1.0, 0.0])) == []
        assert gallery.embedding_dimension is None


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        gallery = LocalGallery(high_threshold=0.9, medium_threshold=0.4)
        profile_id = gallery.enroll(_unit([1.0, 0.0, 0.0]))
        gallery.add_reference(profile_id, _unit([0.8, 0.6, 0.0]))
        path = tmp_path / "gallery.json"

        gallery.save(path)
        restored = LocalGallery()
        restored.load(path)

        assert restored.to_dict() == gallery.to_dict()
        assert restored.match(_unit([0.8, 0.6, 0.0]))[0].confidence == 1.0
        assert not list(tmp_path.glob(".gallery.json.*.tmp"))

    def test_serialization_contains_only_approved_profile_fields(self):
        gallery = LocalGallery()
        gallery.enroll(_unit([1.0, 0.0]))
        payload = gallery.to_dict()

        profile = next(iter(payload["profiles"].values()))
        assert set(profile) == {"version", "v_count", "centroid", "vectors"}
        serialized = json.dumps(payload).lower()
        for forbidden in ("human_name", "source_path", "raw_crop", "debug_frame"):
            assert forbidden not in serialized

    def test_to_dict_is_detached_from_internal_state(self):
        gallery = LocalGallery()
        profile_id = gallery.enroll(_unit([1.0, 0.0]))
        payload = gallery.to_dict()
        payload["profiles"][profile_id]["vectors"][0][0] = 0.0

        assert gallery.match(_unit([1.0, 0.0]))[0].confidence == 1.0

    @pytest.mark.parametrize(
        "mutator",
        [
            lambda data: data.update(version=999),
            lambda data: data.update(extra_field="PII"),
            lambda data: data.update(next_profile_counter=0),
            lambda data: data["profiles"]["prof-00000000"].update(human_name="x"),
            lambda data: data["profiles"]["prof-00000000"].update(
                centroid=[0.0, 1.0, 0.0]
            ),
            lambda data: data["profiles"]["prof-00000000"]["vectors"][0].__setitem__(
                0, 2.0
            ),
        ],
    )
    def test_rejects_tampered_payload(self, mutator):
        gallery = LocalGallery()
        gallery.enroll(_unit([1.0, 0.0, 0.0]))
        payload = gallery.to_dict()
        mutator(payload)
        with pytest.raises(LocalGalleryError):
            LocalGallery.from_dict(payload)

    def test_load_is_atomic_on_validation_failure(self, tmp_path):
        gallery = LocalGallery()
        profile_id = gallery.enroll(_unit([1.0, 0.0]))
        path = tmp_path / "gallery.json"
        path.write_text('{"version": 999}', encoding="utf-8")

        with pytest.raises(LocalGalleryError):
            gallery.load(path)

        assert gallery.profile_ids == [profile_id]

    def test_load_rejects_duplicate_json_keys(self, tmp_path):
        path = tmp_path / "gallery.json"
        path.write_text('{"version": 2, "version": 2}', encoding="utf-8")
        with pytest.raises(LocalGalleryError, match="duplicate key"):
            LocalGallery().load(path)

    def test_save_requires_existing_parent(self, tmp_path):
        gallery = LocalGallery()
        with pytest.raises(LocalGalleryError, match="directory"):
            gallery.save(tmp_path / "missing" / "gallery.json")
