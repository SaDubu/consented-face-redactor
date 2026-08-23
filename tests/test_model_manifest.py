"""Tests for strict model manifest and checksum validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from consented_face_redactor.model_manifest import (
    ManifestValidationError,
    VALID_ROLES,
    load_manifest_from_json,
    validate_manifest,
    verify_model_file,
)


def _make_entry(**overrides) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "model_id": "test-model-1",
        "role": "detector",
        "source": "Shangjin-Xing/YuNet",
        "filename": "yunet.onnx",
        "sha256": "a" * 64,
        "license": "Apache-2.0",
        "input_shape": [1, 320, 320],
        "preprocessing_revision": 1,
        "provider": "OpenCV",
    }
    entry.update(overrides)
    return entry


class TestValidateManifest:
    def test_accepts_and_canonicalizes_clean_entry(self):
        result = validate_manifest(_make_entry(provider="opencv", sha256="A" * 64))
        assert result["role"] == "detector"
        assert result["provider"] == "OpenCV"
        assert result["sha256"] == "a" * 64

    @pytest.mark.parametrize("role", ["invalid", "DETECTOR ", 123, None, ""])
    def test_rejects_invalid_role(self, role):
        with pytest.raises(ManifestValidationError):
            validate_manifest(_make_entry(role=role))

    @pytest.mark.parametrize("sha", ["short", "a" * 63, "a" * 65, "g" * 64])
    def test_rejects_invalid_sha256(self, sha):
        with pytest.raises(ManifestValidationError):
            validate_manifest(_make_entry(sha256=sha))

    @pytest.mark.parametrize("revision", [-1, 0, True, 1.5])
    def test_rejects_invalid_preprocessing_revision(self, revision):
        with pytest.raises(ManifestValidationError):
            validate_manifest(_make_entry(preprocessing_revision=revision))

    @pytest.mark.parametrize(
        "shape",
        [[], [1, True, 320], [1, -1, 320], "1,320,320"],
    )
    def test_rejects_invalid_input_shape(self, shape):
        with pytest.raises(ManifestValidationError):
            validate_manifest(_make_entry(input_shape=shape))

    @pytest.mark.parametrize(
        "filename",
        ["../yunet.onnx", "models/yunet.onnx", "C:\\models\\yunet.onnx", 123],
    )
    def test_rejects_non_local_filename(self, filename):
        with pytest.raises(ManifestValidationError):
            validate_manifest(_make_entry(filename=filename))

    def test_rejects_unsupported_provider(self):
        with pytest.raises(ManifestValidationError, match="provider"):
            validate_manifest(_make_entry(provider="TensorRT"))

    def test_rejects_missing_and_unknown_keys(self):
        missing = _make_entry()
        del missing["model_id"]
        with pytest.raises(ManifestValidationError, match="missing required key"):
            validate_manifest(missing)

        extra = _make_entry(unexpected_field="x")
        with pytest.raises(ManifestValidationError, match="unknown keys"):
            validate_manifest(extra)


class TestLoadManifest:
    def test_loads_multiple_unique_entries(self, tmp_path):
        path = tmp_path / "manifest.json"
        entries = [
            _make_entry(model_id=f"model-{index}", filename=f"model-{index}.onnx")
            for index in range(3)
        ]
        path.write_text(json.dumps(entries), encoding="utf-8")

        result = load_manifest_from_json(path)

        assert [item["model_id"] for item in result] == [
            "model-0",
            "model-1",
            "model-2",
        ]

    @pytest.mark.parametrize("payload", [{}, "entry", 3, None])
    def test_requires_array_envelope(self, tmp_path, payload):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ManifestValidationError, match="JSON array"):
            load_manifest_from_json(path)

    def test_rejects_non_object_entry(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(["bad"]), encoding="utf-8")
        with pytest.raises(ManifestValidationError, match="not an object"):
            load_manifest_from_json(path)

    @pytest.mark.parametrize("duplicate_field", ["model_id", "filename"])
    def test_rejects_duplicate_identity_fields(self, tmp_path, duplicate_field):
        first = _make_entry()
        second = _make_entry(model_id="other", filename="other.onnx")
        second[duplicate_field] = first[duplicate_field]
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps([first, second]), encoding="utf-8")
        with pytest.raises(ManifestValidationError, match="Duplicate"):
            load_manifest_from_json(path)

    def test_rejects_duplicate_json_key(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text('[{"model_id":"a","model_id":"b"}]', encoding="utf-8")
        with pytest.raises(ManifestValidationError, match="duplicate key"):
            load_manifest_from_json(path)

    def test_rejects_invalid_json_without_leaking_parent_path(self, tmp_path):
        path = tmp_path / "secret-parent" / "manifest.json"
        path.parent.mkdir()
        path.write_text("[", encoding="utf-8")
        with pytest.raises(ManifestValidationError) as error:
            load_manifest_from_json(path)
        assert str(path.parent) not in str(error.value)


class TestVerifyModelFile:
    def test_accepts_matching_filename_and_digest(self, tmp_path):
        model = tmp_path / "yunet.onnx"
        model.write_bytes(b"model-content")
        entry = _make_entry(sha256=hashlib.sha256(model.read_bytes()).hexdigest())

        verify_model_file(entry, model)

    def test_rejects_filename_mismatch(self, tmp_path):
        model = tmp_path / "different.onnx"
        model.write_bytes(b"model-content")
        with pytest.raises(ManifestValidationError, match="filename"):
            verify_model_file(_make_entry(), model)

    def test_rejects_checksum_mismatch(self, tmp_path):
        model = tmp_path / "yunet.onnx"
        model.write_bytes(b"model-content")
        with pytest.raises(ManifestValidationError, match="checksum mismatch"):
            verify_model_file(_make_entry(), model)

    def test_rejects_missing_file_with_basename_only(self, tmp_path):
        model = tmp_path / "private" / "yunet.onnx"
        with pytest.raises(ManifestValidationError) as error:
            verify_model_file(_make_entry(), model)
        assert str(model.parent) not in str(error.value)
        assert model.name in str(error.value)


def test_all_controlled_roles_are_present():
    assert VALID_ROLES == {"detector", "embedder", "tracker", "renderer"}
