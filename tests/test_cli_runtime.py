"""Runtime CLI contracts: no partial model wiring and opt-in evidence only."""

from __future__ import annotations

from argparse import Namespace
import json

import pytest

from consented_face_redactor.cli import (
    _build_processing_summary,
    _load_runtime_components,
    _reject_output_overwrite,
    _write_evidence,
)


def test_runtime_without_all_options_is_explicit_safe_stub():
    detector, gallery, metadata = _load_runtime_components(
        Namespace(model_dir=None, manifest_dir=None, gallery_db=None, approval_db=None)
    )

    assert detector is None
    assert gallery is None
    assert metadata == {"runtime_mode": "stub_no_redaction"}


def test_partial_runtime_options_are_rejected():
    with pytest.raises(ValueError, match="must be supplied together"):
        _load_runtime_components(
            Namespace(model_dir=None, manifest_dir=None, gallery_db="gallery.json", approval_db=None)
        )


def test_evidence_is_written_only_to_explicit_destination(tmp_path):
    path = tmp_path / "evidence.json"
    rows = [{"frame_index": 0, "track_state": "candidate", "is_redacted": False, "review_required": True, "approval_reason_codes": ["empty_gallery"], "approved_face_count": 0}]

    _write_evidence(path, input_path=tmp_path / "input.mp4", runtime={"runtime_mode": "test"}, rows=rows)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["frame_count"] == 1
    assert payload["frames"] == rows


def test_processing_summary_reports_fhd_and_latency_without_a_threshold():
    rows = [
        {"is_redacted": True, "review_required": False, "duration_ms": 4.0, "approval_reason_codes": ["test_consent"]},
        {"is_redacted": False, "review_required": True, "duration_ms": 8.0, "approval_reason_codes": ["similarity_insufficient"]},
    ]
    summary = _build_processing_summary(
        rows, frame_shape=(1080, 1920, 3), elapsed_seconds=0.02, source_fps=30.0
    )

    assert summary["is_fhd_or_higher"] is True
    assert summary["p50_latency_ms"] == pytest.approx(6.0)
    assert summary["approval_reason_counts"] == {"test_consent": 1, "similarity_insufficient": 1}


def test_output_path_cannot_equal_input_path(tmp_path):
    path = tmp_path / "input.mp4"
    with pytest.raises(ValueError, match="must differ"):
        _reject_output_overwrite(path, path)


def test_audio_preservation_is_not_allowed_for_dry_run():
    from consented_face_redactor.cli import main

    assert main([
        "process-video", "--input", "missing.mp4", "--dry-run",
        "--preserve-audio", "--ffmpeg-path", "ffmpeg.exe",
    ]) == 2
