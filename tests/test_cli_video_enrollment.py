"""CLI tests for video enrollment dry-run and local persistence behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection


class _Source:
    fps = 30.0
    frame_count = 3

    def __init__(self):
        self._index = 0

    def open(self):
        return None

    def close(self):
        return None

    def read(self):
        if self._index >= 3:
            return False, None
        self._index += 1
        return True, np.zeros((80, 80, 3), dtype=np.uint8)


class _Detector:
    def detect(self, frame):
        return [FaceDetection(BoundingBox(5, 5, 50, 50), np.zeros((5, 2), dtype=np.float32), 0.99)]


class _Embedder:
    def embed(self, frame, detection):
        return np.array([1.0, 0.0], dtype=np.float32), 1


def _args(tmp_path: Path, *, dry_run: bool) -> list[str]:
    input_path = tmp_path / "target.mp4"
    input_path.write_bytes(b"placeholder")
    args = [
        "gallery-enroll-video", "--input", str(input_path), "--gallery-db", str(tmp_path / "gallery.json"),
        "--approval-db", str(tmp_path / "approvals.json"), "--model-dir", str(tmp_path),
        "--manifest-dir", str(tmp_path), "--report-out", str(tmp_path / "report.json"),
    ]
    return args + (["--dry-run"] if dry_run else ["--approve", "--approval-reason", "test_consent"])


@patch("consented_face_redactor.cli.OpenCvFrameSource", _Source, create=True)
def test_video_enrollment_dry_run_writes_report_not_gallery(tmp_path, monkeypatch):
    from consented_face_redactor import cli

    monkeypatch.setattr(cli, "_build_enrollment_runtime", lambda args: (_Detector(), _Embedder()))
    monkeypatch.setattr("consented_face_redactor.media.frame_source.OpenCvFrameSource", lambda path: _Source())

    assert cli.main(_args(tmp_path, dry_run=True)) == 0
    assert (tmp_path / "report.json").is_file()
    assert not (tmp_path / "gallery.json").exists()
    assert not (tmp_path / "approvals.json").exists()
