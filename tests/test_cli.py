"""Tests for CLI validation commands and privacy-safe output."""

from __future__ import annotations

import hashlib
import json

from consented_face_redactor.cli import main


def _manifest_entry(filename: str, content: bytes) -> dict:
    return {
        "model_id": "detector-1",
        "role": "detector",
        "source": "Shangjin-Xing/YuNet",
        "filename": filename,
        "sha256": hashlib.sha256(content).hexdigest(),
        "license": "Apache-2.0",
        "input_shape": [1, 320, 320],
        "preprocessing_revision": 1,
        "provider": "OpenCV",
    }


class TestInspectConfig:
    def test_default_verbose_output_is_cp949_safe(self, capsys):
        result = main(["inspect-config", "--verbose"])
        captured = capsys.readouterr()

        assert result == 0
        assert "Config schema inspection" in captured.out
        assert "UncertainPolicy: ['privacy_safe', 'precision']" in captured.out
        captured.out.encode("cp949")
        captured.err.encode("cp949")

    def test_loads_json_and_redacts_local_paths(self, tmp_path, capsys):
        private_path = tmp_path / "private-input.mp4"
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"input_path": str(private_path)}), encoding="utf-8"
        )

        result = main(["inspect-config", "--file", str(config_path)])
        captured = capsys.readouterr()

        assert result == 0
        assert "input_path: '<configured>'" in captured.out
        assert str(private_path) not in captured.out

    def test_invalid_config_does_not_echo_token(self, tmp_path, capsys):
        secret = "A" * 48
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"effect_mode": secret}), encoding="utf-8")

        result = main(["inspect-config", "--file", str(path)])
        captured = capsys.readouterr()

        assert result == 2
        assert secret not in captured.err
        assert path.name in captured.err


class TestValidateModels:
    def test_validates_manifest_and_colocated_binary(self, tmp_path, capsys):
        content = b"model-content"
        filename = "yunet.onnx"
        (tmp_path / filename).write_bytes(content)
        (tmp_path / "models.json").write_text(
            json.dumps([_manifest_entry(filename, content)]), encoding="utf-8"
        )

        result = main(["validate-models", "--manifest-dir", str(tmp_path)])
        captured = capsys.readouterr()

        assert result == 0
        assert "Validated 1 model file(s)" in captured.out

    def test_rejects_empty_manifest(self, tmp_path, capsys):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        result = main(["validate-models", "--manifest-dir", str(tmp_path)])
        assert result == 2
        assert "no model entries" in capsys.readouterr().err

    def test_rejects_duplicates_across_manifest_files(self, tmp_path, capsys):
        content = b"model-content"
        filename = "yunet.onnx"
        entry = _manifest_entry(filename, content)
        (tmp_path / filename).write_bytes(content)
        (tmp_path / "a.json").write_text(json.dumps([entry]), encoding="utf-8")
        (tmp_path / "b.json").write_text(json.dumps([entry]), encoding="utf-8")

        result = main(["validate-models", "--manifest-dir", str(tmp_path)])

        assert result == 2
        assert "Duplicate model_id" in capsys.readouterr().err

    def test_rejects_missing_manifest_directory(self, tmp_path, capsys):
        result = main(
            ["validate-models", "--manifest-dir", str(tmp_path / "missing")]
        )
        assert result == 2
        assert "unavailable" in capsys.readouterr().err


def test_no_command_prints_help(capsys):
    assert main([]) == 1
    assert "usage:" in capsys.readouterr().out
