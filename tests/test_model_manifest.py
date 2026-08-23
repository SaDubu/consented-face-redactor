"""Tests for model_manifest module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest


def _make_entry(**overrides) -> dict[str, Any]:
    base: dict[str, Any] = {
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
    base.update(overrides)
    return base


def _write_manifest(entries: list[dict[str, Any]]) -> Path:
    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False, encoding="utf-8"
    ) as fh:
        json.dump(entries, fh)
        return Path(fh.name)


# ---- manifest validation ------------------------------------------ #


class TestValidateManifest:
    def test_valid_manifest_accepts_clean_entry(self):
        from consented_face_redactor.model_manifest import validate_manifest

        entry = _make_entry()
        result = validate_manifest(entry)
        assert isinstance(result, dict)
        assert result["model_id"] == "test-model-1"
        assert result["role"] in {"detector"}

    def test_validates_role_is_one_of_choices(self):
        from consented_face_redactor.model_manifest import ManifestValidationError, validate_manifest

        for bad_role in ["invalid", "DETECTOR ", 123, None]:  # noqa: PYI051 — bad types (test intentional)
            entry = _make_entry(role=bad_role)
            with pytest.raises(ManifestValidationError):
                validate_manifest(entry)

    def test_rejects_sha_with_wrong_length(self):
        from consented_face_redactor.model_manifest import ManifestValidationError, validate_manifest

        for bad in ["short", "a" * 63, "a" * 65]:
            entry = _make_entry(sha256=bad)
            with pytest.raises(ManifestValidationError, match="64 hex characters"):
                validate_manifest(entry)

    def test_rejects_non_hex_sha(self):
        from consented_face_redactor.model_manifest import ManifestValidationError, validate_manifest

        entry = _make_entry(sha256="a" * 64)
        entry["sha256"] = "g" * 48 + "a" * 16  # len=64 but 'g' is not valid hex
        assert len(entry["sha256"]) == 64
        with pytest.raises(ManifestValidationError, match="non-hex"):
            validate_manifest(entry)

    def test_rejects_negative_preprocessing_revision(self):
        from consented_face_redactor.model_manifest import ManifestValidationError, validate_manifest

        entry = _make_entry(preprocessing_revision=-1)
        with pytest.raises(ManifestValidationError):
            validate_manifest(entry)

    def test_rejects_zero_preprocessing_revision(self):
        from consented_face_redactor.model_manifest import ManifestValidationError, validate_manifest

        entry = _make_entry(preprocessing_revision=0)
        with pytest.raises(ManifestValidationError):
            validate_manifest(entry)

    def test_rejects_empty_string_role(self):
        from consented_face_redactor.model_manifest import ManifestValidationError, validate_manifest

        entry = _make_entry(role="")
        with pytest.raises(ManifestValidationError):
            validate_manifest(entry)

    def test_rejects_missing_required_key_model_id(self):
        from consented_face_redactor.model_manifest import ManifestValidationError, validate_manifest

        entry = {k: v for k, v in _make_entry().items() if k != "model_id"}
        with pytest.raises(ManifestValidationError, match="missing required key"):  # noqa: PYI051
            validate_manifest(entry)

    def test_rejects_unknown_keys(self):
        from consented_face_redactor.model_manifest import ManifestValidationError, validate_manifest

        entry = _make_entry()
        entry["unexpected_field"] = "should_not_be_here"
        with pytest.raises(ManifestValidationError, match="unknown keys"):
            validate_manifest(entry)


# ---- load_manifest_from_json -------------------------------------- #


class TestLoadManifestFromJson:
    def test_load_valid_manifest(self):
        from consented_face_redactor.model_manifest import load_manifest_from_json

        entries = [_make_entry()]
        path = _write_manifest(entries)
        try:
            result = load_manifest_from_json(path)
            assert len(result) == 1
            assert result[0]["model_id"] == "test-model-1"
        finally:
            path.unlink()

    def test_load_rejects_single_entry_not_array(self):
        from consented_face_redactor.model_manifest import ManifestValidationError, load_manifest_from_json

        path = _write_manifest(_make_entry())
        try:
            with pytest.raises(ManifestValidationError, match="JSON array"):
                load_manifest_from_json(path)
        finally:
            path.unlink()

    def test_load_rejects_invalid_entry_in_array(self):
        from consented_face_redactor.model_manifest import ManifestValidationError, load_manifest_from_json

        good = _make_entry()
        bad_entry = str(good)  # not a dict — intentional (invalid entry)
        path = _write_manifest([good, bad_entry])
        try:
            with pytest.raises(ManifestValidationError):
                load_manifest_from_json(path)
        finally:
            path.unlink()

    def test_load_multiple_entries(self):
        from consented_face_redactor.model_manifest import load_manifest_from_json

        entries = [_make_entry(model_id=f"model-{i}") for i in range(3)]
        path = _write_manifest(entries)
        try:
            result = load_manifest_from_json(path)
            assert len(result) == 3
            assert result[0]["model_id"] == "model-0"
            assert result[2]["model_id"] == "model-2"
        finally:
            path.unlink()


# ---- verify_model_file -------------------------------------------- #


class TestVerifyModelFile:
    def test_verify_existing_file_passes(self, tmp_path):
        from consented_face_redactor.model_manifest import ManifestValidationError, verify_model_file

        entry = _make_entry()
        expected_sha = "a" * 64

        # Create dummy file with correct checksum
        dummy = tmp_path / "dummy.onnx"
        dummy.write_bytes(b"\x00" * 1024)
        import hashlib

        actual_sha = hashlib.sha256(dummy.read_bytes()).hexdigest()
        entry["sha256"] = actual_sha

        verify_model_file(entry, dummy)  # should NOT raise


# ---- fail-closed -------------------------------------------------- #


class TestFailClosed:
    def test_all_roles_in_list(self):
        from consented_face_redactor.model_manifest import VALID_ROLES

        expected_roles = {"detector", "embedder", "tracker", "renderer"}
        assert VALID_ROLES == expected_roles

    def test_sha256_with_uppercase_is_accepted(self):
        """SHA-256 can be uppercase hex (normalized to lowercase internally)."""
        from consented_face_redactor.model_manifest import validate_manifest

        entry = _make_entry(sha256="A" * 64)
        result = validate_manifest(entry)
        assert result["sha256"] == "a" * 64


# ---- edge cases for detection_iface ------------------------------- #


class TestDetectionIface:
    def test_bbox_properties(self):
        from consented_face_redactor.adapters.detection_iface import BoundingBox

        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=15)
        assert bbox.width == 11  # x2 - x1 + 1 (inclusive coords)
        assert bbox.height == 16
        assert bbox.area == 176


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
