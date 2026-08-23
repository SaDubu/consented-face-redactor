"""Tests for media.frame_source module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing src without install
_HERE = Path(__file__).resolve()
_src = _HERE.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


class TestFakeFrameReader:
    def test_creates_with_defaults(self):
        """Creating FakeFrameReader should work with defaults."""
        from consented_face_redactor.media import FakeFrameReader

        fr = FakeFrameReader()
        assert fr.fps == 30.0
        assert fr.height == 240
        assert fr.width == 320
        assert fr.frame_count == 60
        assert fr.current_frame_index == 0

    def test_read_returns_frames_in_order(self):
        """read() should return frames sequentially and increment index."""
        from consented_face_redactor.media import FakeFrameReader

        fr = FakeFrameReader(width=10, height=8, n_frames=5, ch=1)
        fr.open()
        for i in range(5):
            ok, frame = fr.read()
            assert ok is True
            assert frame is not None
            assert frame.shape == (8, 10, 1)
            assert frame[:,:,0][0,0] == (i * 5) % 256
        # After exhausting frames
        ok, frame = fr.read()
        assert ok is False
        assert frame is None

    def test_seeks_to_frame(self):
        """seek() should jump back to a specific frame without closing."""
        from consented_face_redactor.media import FakeFrameReader

        fr = FakeFrameReader(width=4, height=2, n_frames=10)
        fr.open()
        # Read 3 frames forward
        for _ in range(3):
            fr.read()
        assert fr.current_frame_index == 3
        # Seek back to frame 0
        result = fr.seek(0)
        assert result is True
        ok, _ = fr.read()
        assert ok is True
        assert fr.current_frame_index == 1

    def test_close_then_open_resets_state(self):
        """close() followed by open() should fully reset the reader."""
        from consented_face_redactor.media import FakeFrameReader

        fr = FakeFrameReader(n_frames=3)
        fr.open()
        fr.read()  # consume one frame
        assert fr.current_frame_index == 1
        fr.close()
        assert fr.current_frame_index == 0
        fr.open()
        ok, _ = fr.read()
        assert ok is True
        assert fr.current_frame_index == 1

    def test_read_after_close_fails(self):
        """read() should return (False, None) when closed."""
        from consented_face_redactor.media import FakeFrameReader

        fr = FakeFrameReader()
        ok, frame = fr.read()
        assert ok is False
        assert frame is None


class TestOpenCvFrameSourceProperties:
    def test_constructor_initializes_without_loading(self):
        """Constructing OpenCvFrameSource should never load the file."""
        from consented_face_redactor.media.frame_source import OpenCvFrameSource

        # Constructor only stores path; open() is what loads
        src = OpenCvFrameSource("/nonexistent/file.mp4")
        # The constructor shouldn't fail — only open() validates the path exists
        assert src.path is not None

    def test_raises_on_nonexistent_file(self, tmp_path):
        """open() should raise for missing files."""
        from consented_face_redactor.media.frame_source import OpenCvFrameSource

        bad_path = tmp_path / "does_not_exist.mp4"
        src = OpenCvFrameSource(bad_path)
        with pytest.raises((FileNotFoundError, OSError)):
            src.open()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
