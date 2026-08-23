"""Tests for deterministic frame readers and OpenCV media boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from consented_face_redactor.media import FakeFrameReader
from consented_face_redactor.media import frame_source as media_module
from consented_face_redactor.media.frame_source import OpenCvFrameSource, OpenCvFrameWriter


class TestFakeFrameReader:
    def test_reads_and_seeks_deterministic_frames(self):
        reader = FakeFrameReader(width=10, height=8, n_frames=5, ch=1)
        reader.open()
        for index in range(5):
            ok, frame = reader.read()
            assert ok is True
            assert frame is not None
            assert frame.shape == (8, 10, 1)
            assert frame[0, 0, 0] == (index * 5) % 256
        assert reader.read() == (False, None)
        assert reader.seek(0) is True
        assert reader.read()[0] is True

    def test_close_and_open_reset_state(self):
        reader = FakeFrameReader(n_frames=3)
        reader.open()
        reader.read()
        reader.close()
        assert reader.current_frame_index == 0
        assert reader.read() == (False, None)
        reader.open()
        assert reader.read()[0] is True


class TestOpenCvImageSource:
    def test_reads_still_image_once_and_can_seek(self, tmp_path):
        import cv2

        image = np.zeros((7, 9, 3), dtype=np.uint8)
        image[:, :, 1] = 123
        path = tmp_path / "input.png"
        assert cv2.imwrite(str(path), image)
        source = OpenCvFrameSource(path)

        source.open()
        assert (source.width, source.height, source.frame_count, source.fps) == (
            9,
            7,
            1,
            -1.0,
        )
        ok, decoded = source.read()
        assert ok is True
        np.testing.assert_array_equal(decoded, image)
        assert source.read() == (False, None)
        assert source.seek(0) is True
        assert source.read()[0] is True
        source.close()
        with pytest.raises(RuntimeError, match="not open"):
            source.read()

    def test_constructor_is_lazy_and_missing_file_error_is_private(self, tmp_path):
        path = tmp_path / "private" / "missing.mp4"
        source = OpenCvFrameSource(path)
        assert source.path == str(path.resolve())
        with pytest.raises(FileNotFoundError) as error:
            source.open()
        assert str(path.parent) not in str(error.value)


class FakeCapture:
    def __init__(self, frames, properties):
        self.frames = frames
        self.properties = properties
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def get(self, prop):
        return self.properties[prop]

    def set(self, prop, value):
        self.position = int(value)
        return True

    def release(self):
        self.released = True


def _fake_capture_cv(capture):
    return SimpleNamespace(
        VideoCapture=lambda path: capture,
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_POS_FRAMES=5,
    )


class TestOpenCvVideoSource:
    def test_seek_does_not_consume_target_frame(self, tmp_path, monkeypatch):
        path = tmp_path / "input.mp4"
        path.touch()
        frames = [np.full((4, 6, 3), index, dtype=np.uint8) for index in range(4)]
        capture = FakeCapture(frames, {1: 25.0, 2: 4.0, 3: 6.0, 4: 4.0})
        fake_cv = _fake_capture_cv(capture)
        monkeypatch.setattr(media_module, "_load_cv2", lambda: fake_cv)
        source = OpenCvFrameSource(path)
        source.open()

        assert source.seek(2) is True
        assert source.current_frame_index == 2
        ok, frame = source.read()

        assert ok is True
        assert frame is not None and frame[0, 0, 0] == 2
        assert source.current_frame_index == 3

    def test_invalid_reported_dimensions_fail_closed(self, tmp_path, monkeypatch):
        path = tmp_path / "input.mp4"
        path.touch()
        capture = FakeCapture([], {1: float("nan"), 2: float("nan"), 3: 0, 4: 0})
        monkeypatch.setattr(
            media_module, "_load_cv2", lambda: _fake_capture_cv(capture)
        )
        with pytest.raises(ValueError, match="dimensions"):
            OpenCvFrameSource(path).open()
        assert capture.released is True


class FakeVideoWriter:
    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())
        return None

    def release(self):
        self.released = True


class FakeWriterCv:
    def __init__(self):
        self.writers = []

    @staticmethod
    def VideoWriter_fourcc(*codec):
        return "".join(codec)

    def VideoWriter(self, path, fourcc, fps, size):
        writer = FakeVideoWriter(path, fourcc, fps, size)
        self.writers.append(writer)
        return writer


class TestOpenCvFrameWriter:
    def test_defers_creation_until_first_frame_and_keeps_dimensions(
        self, tmp_path, monkeypatch
    ):
        fake_cv = FakeWriterCv()
        monkeypatch.setattr(media_module, "_load_cv2", lambda: fake_cv)
        writer = OpenCvFrameWriter(tmp_path / "output.mp4", fps=24.0, codec="mp4v")

        writer.open()
        assert writer.is_open is True
        assert fake_cv.writers == []
        frame = np.zeros((7, 11, 3), dtype=np.uint8)
        assert writer.write(frame) is True

        backend = fake_cv.writers[0]
        assert backend.size == (11, 7)
        assert backend.fps == 24.0
        assert backend.fourcc == "mp4v"
        assert len(backend.frames) == 1
        with pytest.raises(ValueError, match="identical dimensions"):
            writer.write(np.zeros((8, 11, 3), dtype=np.uint8))
        writer.close()
        assert backend.released is True
        assert writer.is_open is False

    def test_refuses_existing_output_by_default(self, tmp_path, monkeypatch):
        fake_cv = FakeWriterCv()
        monkeypatch.setattr(media_module, "_load_cv2", lambda: fake_cv)
        output = tmp_path / "output.mp4"
        output.touch()
        with pytest.raises(FileExistsError):
            OpenCvFrameWriter(output).open()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"fps": True},
            {"fps": 0},
            {"fps": float("nan")},
            {"codec": "abc"},
            {"codec": "abcae"},
            {"codec": "\u00e9abc"},
        ],
    )
    def test_rejects_invalid_writer_configuration(self, tmp_path, monkeypatch, kwargs):
        monkeypatch.setattr(media_module, "_load_cv2", lambda: FakeWriterCv())
        with pytest.raises((TypeError, ValueError)):
            writer = OpenCvFrameWriter(tmp_path / "output.mp4", **kwargs)
            writer.open()

    def test_rejects_invalid_frame(self, tmp_path, monkeypatch):
        monkeypatch.setattr(media_module, "_load_cv2", lambda: FakeWriterCv())
        writer = OpenCvFrameWriter(tmp_path / "output.mp4")
        writer.open()
        with pytest.raises(ValueError, match="uint8 BGR"):
            writer.write(np.zeros((4, 4), dtype=np.uint8))
