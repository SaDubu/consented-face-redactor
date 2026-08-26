"""Phase 9: CLI command integration tests for process-image, process-video and gallery-enroll."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_png_bytes():
    """Create minimal valid PNG bytes (8x8 black)."""
    try:
        import cv2
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        tmp_path = Path(tempfile.gettempdir()) / "p9_min.png"
        cv2.imwrite(str(tmp_path), img)
        return tmp_path.read_bytes()
    except Exception:
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _make_png_with_mean():
    """Create a small PNG that is all white (255) so mean ≠ zero."""
    try:
        import cv2
        img = np.full((8, 8, 3), 255, dtype=np.uint8)
        tmp_path = Path(tempfile.gettempdir()) / "p9_mean.png"
        cv2.imwrite(str(tmp_path), img)
        return tmp_path.read_bytes()
    except Exception:
        # fallback — a plain PNG won't help mean; we rely on mock anyway
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _write_config(tmp_path, extra=None):
    cfg = {"effect_mode": "mosaic", "uncertain_policy": "privacy_safe"}
    if extra:
        cfg.update(extra)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


# ------------------------------------------------------------------ #
# process-image tests
# ------------------------------------------------------------------ #


def test_process_image_no_subparser_args():
    """When no subparser args are given, argparse prints help and exits 2."""
    from consented_face_redactor.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["process-image"])
    assert exc.value.code == 2


@patch("consented_face_redactor.pipeline.RedactionPipeline")
@patch("consented_face_redactor.media.frame_source.OpenCvFrameSource")
def test_process_image_basic_flow(mock_src_cls, mock_pipe_cls):
    """Verify that process-image opens source, calls pipeline, and returns 0."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tmp_dir = Path(tempfile.gettempdir())
    input_path = tmp_dir / "test_img.png"
    output_path = tmp_dir / "test_out_redacted.png"
    input_path.write_bytes(_make_png_bytes())

    src_inst = MagicMock()
    src_inst.path = str(input_path)
    src_inst.width = 320
    src_inst.height = 240
    src_inst.fps = 0.0
    src_inst.frame_count = -1
    src_inst.current_frame_index = 0
    src_inst.read.return_value = (True, frame.copy())
    mock_src_cls.return_value = src_inst

    result_pipe = MagicMock()
    result_pipe.result_frame = frame.copy()
    result_pipe.track_state.value = "CONFIRMED"
    result_pipe.is_redacted = False
    result_pipe.review_required = False

    mock_pipe_cls.return_value.process_frame.return_value = result_pipe
    mock_pipe_cls.return_value.load_track_state.return_value = None
    # _save_track_state calls pipe.save_track_state() → json.dumps → needs serializable data
    mock_pipe_cls.return_value.save_track_state.return_value = {}

    from consented_face_redactor.cli import main

    result = main(["process-image", "--input", str(input_path), "--output", str(output_path)])
    assert result == 0
    src_inst.open.assert_called_once()


# ------------------------------------------------------------------ #
# process-video tests
# ------------------------------------------------------------------ #


def test_process_video_no_subparser_args():
    """When no subparser args are given, argparse prints help and exits 2."""
    from consented_face_redactor.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["process-video"])
    assert exc.value.code == 2


@patch("consented_face_redactor.pipeline.RedactionPipeline")
@patch("consented_face_redactor.media.frame_source.OpenCvFrameSource")
def test_process_video_basic_flow(mock_src_cls, mock_pipe_cls):
    """Verify that process-video reads frames and returns 0."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tmp_dir = Path(tempfile.gettempdir())
    input_path = tmp_dir / "test_vid.mp4"
    input_path.write_bytes(b"")  # dummy — source will be mocked anyway

    src_inst = MagicMock()
    src_inst.path = str(input_path)
    src_inst.width = 320
    src_inst.height = 240
    src_inst.fps = 30.0
    src_inst.frame_count = 60
    src_inst.current_frame_index = 0

    # Return (True, frame) then (False, None) to simulate 1 frame read
    src_inst.read.side_effect = [(True, frame.copy()), (False, None)]

    mock_src_cls.return_value = src_inst

    result_pipe = MagicMock()
    result_pipe.result_frame = frame.copy()

    mock_pipe_cls.return_value.process_frame.return_value = result_pipe
    mock_pipe_cls.return_value.load_track_state.return_value = None
    # _save_track_state calls pipe.save_track_state() → json.dumps → needs serializable data
    mock_pipe_cls.return_value.save_track_state.return_value = {}

    output_path = tmp_dir / "test_out_processed.mp4"
    # Ensure output directory exists so cv2.VideoWriter doesn't fail
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from consented_face_redactor.cli import main

    result = main(["process-video", "--input", str(input_path), "--output", str(output_path)])
    assert result == 0
    src_inst.open.assert_called_once()


# ------------------------------------------------------------------ #
# gallery-enroll tests
# ------------------------------------------------------------------ #


def test_gallery_enroll_no_subparser_args():
    """When no subparser args are given, argparse prints help and exits 2."""
    from consented_face_redactor.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["gallery-enroll"])
    assert exc.value.code == 2


@patch("consented_face_redactor.media.frame_source.OpenCvFrameSource")
def test_gallery_enroll_basic_flow(mock_src_cls, tmp_path):
    """Enrollment requires a real detector/embedder path and explicit approval."""
    tmp_dir = tmp_path
    input_img = tmp_dir / "test_enroll.png"
    input_img.write_bytes(_make_png_with_mean())

    src_inst = MagicMock()
    src_inst.open.side_effect = None
    src_inst.read.return_value = (True, np.full((240, 320, 3), 255, dtype=np.uint8))
    mock_src_cls.return_value = src_inst

    from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection
    from consented_face_redactor.cli import main
    from consented_face_redactor.gallery import LocalGallery

    gallery_db_path = tmp_dir / "test_gallery.json"
    approval_db_path = tmp_dir / "approvals.json"
    detector_entry = {"filename": "yunet.onnx", "model_id": "yunet"}
    embedder_entry = {"filename": "sface.onnx", "model_id": "sface", "preprocessing_revision": 1}
    detection = FaceDetection(
        BoundingBox(1, 1, 20, 20), np.zeros((5, 2), dtype=np.float32), 0.99
    )

    with patch("consented_face_redactor.cli._load_verified_model_entries", return_value=(detector_entry, embedder_entry)), \
         patch("consented_face_redactor.adapters.opencv.OpenCvYuNetDetector") as detector_cls, \
         patch("consented_face_redactor.adapters.opencv.OpenCvSFaceEmbedder") as embedder_cls, \
         patch.object(LocalGallery, 'enroll', return_value="prof-00000000"), \
         patch.object(LocalGallery, 'save', return_value=None), \
         patch("consented_face_redactor.approval_store.ApprovalStore.save", return_value=None):
        detector_cls.return_value.detect.return_value = [detection]
        embedder_cls.return_value.embed.return_value = (np.array([1.0, 0.0], dtype=np.float32), 1)
        result = main([
            "gallery-enroll", "--input", str(input_img), "--gallery-db", str(gallery_db_path),
            "--approval-db", str(approval_db_path), "--model-dir", str(tmp_dir),
            "--manifest-dir", str(tmp_dir), "--approve", "--approval-reason", "test_consent",
        ])

    assert result == 0


# ------------------------------------------------------------------ #
# config loading tests
# ------------------------------------------------------------------ #


def test_load_config_from_file(tmp_path):
    cfg_path = _write_config(tmp_path)
    from consented_face_redactor.cli import _load_config

    cfg = _load_config(cfg_path)
    assert cfg is not None


def test_load_config_default():
    from consented_face_redactor.cli import _load_config

    cfg = _load_config(None)
    assert cfg is not None


def test_load_config_invalid_json(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not valid json {{{", encoding="utf-8")
    from consented_face_redactor.cli import _load_config

    with pytest.raises(SystemExit) as exc:
        _load_config(bad_path)
    assert exc.value.code == 2


def test_load_config_non_object_raises(tmp_path):
    arr_path = tmp_path / "arr.json"
    arr_path.write_text("[1, 2, 3]", encoding="utf-8")
    from consented_face_redactor.cli import _load_config

    with pytest.raises(SystemExit) as exc:
        _load_config(arr_path)
    assert exc.value.code == 2


# ------------------------------------------------------------------ #
# track state I/O tests
# ------------------------------------------------------------------ #


def test_load_track_state_no_file():
    from consented_face_redactor.cli import _load_track_state

    result = _load_track_state(Path("/nonexistent/path"))
    assert result is None
