"""OpenCV YuNet detector and SFace embedder adapters.

Model paths passed here must already have passed manifest and digest validation.
The adapters load model binaries lazily on first inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .detection_iface import BoundingBox, DetectorAdapter, EmbedderAdapter, FaceDetection


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "OpenCV inference requires the 'inference' project extra"
        ) from exc
    return cv2


def _validate_bgr_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a uint8 BGR array with shape (H, W, 3)")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image dimensions must be positive")


class OpenCvYuNetDetector(DetectorAdapter):
    """Detect faces and five landmarks with OpenCV FaceDetectorYN."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str = "yunet",
        input_size: tuple[int, int] = (320, 320),
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if (
            not isinstance(input_size, tuple)
            or len(input_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in input_size
            )
        ):
            raise ValueError("input_size must contain two positive integers")
        if isinstance(score_threshold, bool) or not isinstance(
            score_threshold, (int, float)
        ):
            raise TypeError("score_threshold must be numeric")
        if not np.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if isinstance(nms_threshold, bool) or not isinstance(
            nms_threshold, (int, float)
        ):
            raise TypeError("nms_threshold must be numeric")
        if not np.isfinite(nms_threshold) or not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be in [0, 1]")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        self._model_path = Path(model_path).resolve(strict=False)
        self._model_id = model_id.strip()
        self._input_size = (int(input_size[0]), int(input_size[1]))
        self._score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)
        self._top_k = top_k
        self._detector: Any = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _ensure_initialized(self) -> None:
        if self._detector is not None:
            return
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Detector model is unavailable: {self._model_path.name}")
        cv2 = _load_cv2()
        creator = getattr(cv2, "FaceDetectorYN_create", None)
        if creator is None and hasattr(cv2, "FaceDetectorYN"):
            creator = getattr(cv2.FaceDetectorYN, "create", None)
        if creator is None:
            raise RuntimeError("Installed OpenCV does not provide FaceDetectorYN")
        self._detector = creator(
            str(self._model_path),
            "",
            self._input_size,
            self._score_threshold,
            self._nms_threshold,
            self._top_k,
        )

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        _validate_bgr_image(image)
        self._ensure_initialized()

        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        raw_result = self._detector.detect(image)
        faces = raw_result[1] if isinstance(raw_result, tuple) else raw_result
        if faces is None:
            return []

        detections: list[FaceDetection] = []
        for face in np.asarray(faces):
            if face.size < 15 or not np.isfinite(face[:15]).all():
                continue
            x, y, box_width, box_height = map(float, face[:4])
            x1 = max(0, int(np.floor(x)))
            y1 = max(0, int(np.floor(y)))
            x2 = min(width, int(np.ceil(x + box_width)))
            y2 = min(height, int(np.ceil(y + box_height)))
            if x2 <= x1 or y2 <= y1:
                continue
            confidence = float(face[14])
            if not 0.0 <= confidence <= 1.0:
                continue
            detections.append(
                FaceDetection(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    landmarks=np.asarray(face[4:14], dtype=np.float32).reshape(5, 2),
                    confidence=confidence,
                )
            )
        return detections


class OpenCvSFaceEmbedder(EmbedderAdapter):
    """Align faces and extract normalized SFace features."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str = "sface",
        preprocessing_revision: int = 1,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if (
            isinstance(preprocessing_revision, bool)
            or not isinstance(preprocessing_revision, int)
            or preprocessing_revision < 1
        ):
            raise ValueError("preprocessing_revision must be a positive integer")
        self._model_path = Path(model_path).resolve(strict=False)
        self._model_id = model_id.strip()
        self._preprocessing_revision = preprocessing_revision
        self._recognizer: Any = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _ensure_initialized(self) -> None:
        if self._recognizer is not None:
            return
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Embedder model is unavailable: {self._model_path.name}")
        cv2 = _load_cv2()
        creator = getattr(cv2, "FaceRecognizerSF_create", None)
        if creator is None and hasattr(cv2, "FaceRecognizerSF"):
            creator = getattr(cv2.FaceRecognizerSF, "create", None)
        if creator is None:
            raise RuntimeError("Installed OpenCV does not provide FaceRecognizerSF")
        self._recognizer = creator(str(self._model_path), "")

    def embed(
        self,
        image: np.ndarray,
        detection: FaceDetection,
    ) -> tuple[np.ndarray, int]:
        _validate_bgr_image(image)
        if not isinstance(detection, FaceDetection):
            raise TypeError("detection must be a FaceDetection")
        self._ensure_initialized()

        aligned = self._recognizer.alignCrop(image, detection.as_yunet_row())
        feature = self._recognizer.feature(aligned)
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("SFace returned an invalid embedding")
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float32).eps:
            raise ValueError("SFace returned a zero-norm embedding")
        return vector / norm, self._preprocessing_revision
