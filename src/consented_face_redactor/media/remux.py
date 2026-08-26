"""Optional local FFmpeg audio remuxing after frame redaction."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class AudioRemuxError(RuntimeError):
    """Raised when a requested audio remux cannot safely complete."""


def remux_original_audio(
    *,
    original_video: Path,
    processed_video: Path,
    destination: Path,
    ffmpeg_path: Path,
) -> None:
    """Copy processed video with original audio into a new destination.

    The function never overwrites either input and refuses an existing
    destination. FFmpeg writes to a sibling temporary file which is atomically
    moved into place only after a successful command.
    """
    original = Path(original_video).resolve(strict=False)
    processed = Path(processed_video).resolve(strict=False)
    target = Path(destination).resolve(strict=False)
    executable = Path(ffmpeg_path).resolve(strict=False)
    if not original.is_file() or not processed.is_file() or not executable.is_file():
        raise AudioRemuxError("original video, processed video, and ffmpeg executable must exist")
    if target in {original, processed}:
        raise AudioRemuxError("remux destination must differ from both inputs")
    if target.exists():
        raise AudioRemuxError("remux destination already exists")
    if not target.parent.is_dir():
        raise AudioRemuxError("remux destination directory is unavailable")
    temporary = target.with_name(f".{target.stem}.remux{target.suffix}")
    if temporary.exists():
        raise AudioRemuxError("remux temporary destination already exists")
    command = [
        str(executable), "-nostdin", "-y", "-i", str(processed), "-i", str(original),
        "-map", "0:v:0", "-map", "1:a?", "-c", "copy", str(temporary),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not temporary.is_file():
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown ffmpeg failure"
            raise AudioRemuxError(f"ffmpeg remux failed: {detail}")
        os.replace(temporary, target)
    except OSError as exc:
        raise AudioRemuxError("ffmpeg remux could not be started") from exc
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
