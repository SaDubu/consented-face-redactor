"""Contract tests for OpenCV YuNet and SFace adapters."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from consented_face_redactor.adapters import opencv as adapter_module
from consented_face_redactor.adapters.detection_iface import BoundingBox, FaceDetection
from consented_face_redactor.adapters.opencv import OpenCvSFaceEmbedder, OpenCvYuNetDetector


class FakeYuNet:
    def __init__(self, faces):
        self.faces = faces
        self.input_size = None
        self.last_image = None

    def setInputSize(self, size):
        self.input_size = size

    def detect(self, image):
        self.last_image = image
        return 1, self.faces


def _face_row() -> np.ndarray:
    return np.array(
        [
            10.2,
            20.4,
            30.3,
            40.2,
            11,
            12,
            21,
            22,
            16,
            27,
            12,
            35,
            24,
            36,
            0.91,
        ],
        dtype=np.float32,
    )


class TestYuNetDetector:
    def test_uses_official_row_layout_and_original_bgr_shape(self, tmp_path, monkeypatch):
        model = tmp_path / "yunet.onnx"
        model.touch()
        fake_detector = FakeYuNet(np.stack([_face_row()]))
        creator_args = []

        def creator(*args):
            creator_args.append(args)
            return fake_detector

        monkeypatch.setattr(
            adapter_module,
            "_load_cv2",
            lambda: SimpleNamespace(FaceDetectorYN_create=creator),
        )
        adapter = OpenCvYuNetDetector(model, score_threshold=0.8, top_k=42)
        image = np.zeros((80, 100, 3), dtype=np.uint8)

        detections = adapter.detect(image)

        assert len(creator_args) == 1
        assert creator_args[0][1:] == ("", (320, 320), 0.8, 0.3, 42)
        assert fake_detector.input_size == (100, 80)
        assert fake_detector.last_image is image
        assert len(detections) == 1
        detection = detections[0]
        assert detection.bbox == BoundingBox(10, 20, 41, 61)
        np.testing.assert_array_equal(detection.landmarks[0], [11, 12])
        assert detection.confidence == pytest.approx(0.91)

    def test_filters_invalid_rows_and_clamps_box(self, tmp_path, monkeypatch):
        model = tmp_path / "yunet.onnx"
        model.touch()
        valid = _face_row()
        valid[:4] = [-5.0, -2.0, 20.0, 20.0]
        non_finite = _face_row()
        non_finite[6] = np.nan
        bad_confidence = _face_row()
        bad_confidence[14] = 2.0
        bad_box = _face_row()
        bad_box[2] = -5.0
        fake_detector = FakeYuNet(
            np.stack([valid, non_finite, bad_confidence, bad_box])
        )
        monkeypatch.setattr(
            adapter_module,
            "_load_cv2",
            lambda: SimpleNamespace(FaceDetectorYN_create=lambda *args: fake_detector),
        )

        detections = OpenCvYuNetDetector(model).detect(
            np.zeros((30, 30, 3), dtype=np.uint8)
        )

        assert [item.bbox for item in detections] == [BoundingBox(0, 0, 15, 18)]

    def test_initializes_model_lazily_only_once(self, tmp_path, monkeypatch):
        model = tmp_path / "yunet.onnx"
        model.touch()
        calls = []
        fake_detector = FakeYuNet(None)

        def creator(*args):
            calls.append(args)
            return fake_detector

        monkeypatch.setattr(
            adapter_module,
            "_load_cv2",
            lambda: SimpleNamespace(FaceDetectorYN_create=creator),
        )
        adapter = OpenCvYuNetDetector(model)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        assert adapter.detect(image) == []
        assert adapter.detect(image) == []
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"input_size": (320.0, 320)},
            {"score_threshold": True},
            {"score_threshold": float("nan")},
            {"nms_threshold": 2.0},
            {"top_k": True},
            {"model_id": 3},
        ],
    )
    def test_rejects_invalid_constructor_values(self, tmp_path, kwargs):
        with pytest.raises((TypeError, ValueError)):
            OpenCvYuNetDetector(tmp_path / "model.onnx", **kwargs)

    def test_missing_model_error_uses_basename(self, tmp_path):
        path = tmp_path / "private" / "yunet.onnx"
        adapter = OpenCvYuNetDetector(path)
        with pytest.raises(FileNotFoundError) as error:
            adapter.detect(np.zeros((10, 10, 3), dtype=np.uint8))
        assert str(path.parent) not in str(error.value)


class FakeRecognizer:
    def __init__(self, feature):
        self.returned_feature = feature
        self.row = None
        self.image = None

    def alignCrop(self, image, row):
        self.image = image
        self.row = row.copy()
        return np.ones((112, 112, 3), dtype=np.uint8)

    def feature(self, aligned):
        assert aligned.shape == (112, 112, 3)
        return self.returned_feature


class TestSFaceEmbedder:
    def test_aligns_with_yunet_row_and_normalizes_feature(self, tmp_path, monkeypatch):
        model = tmp_path / "sface.onnx"
        model.touch()
        recognizer = FakeRecognizer(np.array([[3.0, 4.0]], dtype=np.float32))
        calls = []

        def creator(*args):
            calls.append(args)
            return recognizer

        monkeypatch.setattr(
            adapter_module,
            "_load_cv2",
            lambda: SimpleNamespace(FaceRecognizerSF_create=creator),
        )
        adapter = OpenCvSFaceEmbedder(model, preprocessing_revision=3)
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        detection = FaceDetection(
            BoundingBox(1, 2, 11, 17),
            np.arange(10, dtype=np.float32).reshape(5, 2),
            0.9,
        )

        vector, revision = adapter.embed(image, detection)

        assert calls == [(str(model.resolve()), "")]
        assert recognizer.image is image
        np.testing.assert_array_equal(recognizer.row, detection.as_yunet_row())
        np.testing.assert_allclose(vector, [0.6, 0.8])
        assert revision == 3

    @pytest.mark.parametrize(
        "feature",
        [
            np.array([[0.0, 0.0]], dtype=np.float32),
            np.array([[np.nan, 1.0]], dtype=np.float32),
            np.array([], dtype=np.float32),
        ],
    )
    def test_rejects_invalid_model_feature(self, tmp_path, monkeypatch, feature):
        model = tmp_path / "sface.onnx"
        model.touch()
        recognizer = FakeRecognizer(feature)
        monkeypatch.setattr(
            adapter_module,
            "_load_cv2",
            lambda: SimpleNamespace(
                FaceRecognizerSF_create=lambda *args: recognizer
            ),
        )
        detection = FaceDetection(
            BoundingBox(0, 0, 10, 10),
            np.zeros((5, 2), dtype=np.float32),
            0.9,
        )
        with pytest.raises(ValueError, match="SFace"):
            OpenCvSFaceEmbedder(model).embed(
                np.zeros((10, 10, 3), dtype=np.uint8), detection
            )
