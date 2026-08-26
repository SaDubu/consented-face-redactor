"""Safety tests for optional local FFmpeg remuxing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from consented_face_redactor.media.remux import AudioRemuxError, remux_original_audio


def test_remux_rejects_existing_destination(tmp_path):
    original = tmp_path / "original.mp4"
    processed = tmp_path / "processed.mp4"
    destination = tmp_path / "destination.mp4"
    executable = tmp_path / "ffmpeg.exe"
    for path in (original, processed, destination, executable):
        path.write_bytes(b"x")

    with pytest.raises(AudioRemuxError, match="already exists"):
        remux_original_audio(original_video=original, processed_video=processed, destination=destination, ffmpeg_path=executable)


def test_remux_uses_temporary_output_then_replaces(monkeypatch, tmp_path):
    original = tmp_path / "original.mp4"
    processed = tmp_path / "processed.mp4"
    destination = tmp_path / "destination.mp4"
    executable = tmp_path / "ffmpeg.exe"
    for path in (original, processed, executable):
        path.write_bytes(b"x")

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"remuxed")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("consented_face_redactor.media.remux.subprocess.run", fake_run)
    remux_original_audio(original_video=original, processed_video=processed, destination=destination, ffmpeg_path=executable)

    assert destination.read_bytes() == b"remuxed"
