"""YuNet face detection adapter via OpenCV DNN.

This adapter implements the DetectorAdapter contract using OpenCV's YuNet model.
The raw vendor output is normalized at the boundary so the pipeline never
depends on vendor-specific formats.

No download code exists — the operator must place the ONNX model file manually.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .detection_iface import BoundingBox, FaceDetection  # noqa: F401 (re-export)
from .detection_iface import DetectorAdapter as _DetectorBase
from .detection_iface import EmbedderAdapter as _EmbedderBase


class DetectorAdapter(_DetectorBase):  # type: ignore[misc] (abstract base inheritance)
    """YuNet face detector adapter via OpenCV DNN."""

    REQUIRED_MODEL_KEYS = [
        "model_id",
        "source",
        "filename",
        "sha256",
        "license",
        "provider",
        "input_shape",
        "preprocessing_revision",
    ]

    def __init__(
        self,
        model_path: str | "os.PathLike[str]",
        *,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """Initialize YuNet detector.

        Parameters
        ----------
        model_path : str
            Path to .onnx YuNet model file. Model is not loaded here (lazy).
        input_size : tuple[int, int] or None
            Resize target for detection. Default = model_default.
        """
        import os  # noqa: F811 — used only within scope (str/PathLike dispatch)

        self._model_path = str(model_path)
        self.input_size = input_size
        self._initialized = False
        self._net = None  # OpenCV DNN Net — opaque handle

    @property
    def model_id(self) -> str:
        return "yunet"

    @property
    def model_path(self) -> str:
        return self._model_path

    def _ensure_initialized(self):
        """Lazy load the detector on first call to detect()."""
        if self._initialized:
            return

        try:
            import cv2  # type: ignore[import-untyped]  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "YuNet detection requires opencv-python. Install via: pip install opencv-python"
            ) from exc

        if not hasattr(cv2, "FaceDetectorYN"):
            raise ValueError(
                "OpenCV version does not support FaceDetectorYN. "
                "Requires OpenCV-contrib-python >= 4.5.5."
            )

        self._net = cv2.FaceDetectorYN.create(
            str(self._model_path),
            "",  # config_dir — empty when using defaults
        )

        if self._net is None:
            raise ValueError("YuNet model load failed — returned None")

        self._initialized = True

    def detect(self, image):  # noqa: ANN001 — simple array arg (matches abstract)
        """Detect faces in input image using YuNet.

        Parameters (per DetectorAdapter contract):
            image : numpy.ndarray — uint8, BGR or RGB; shape (H,W,C). Never mutated.

        Returns:
            Sequence[FaceDetection] — zero or more detections
        """
        self._ensure_initialized()

        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("image must be a 3D numpy array (H,W,C)")

        # Convert to BGR if input is RGB — YuNet expects BGR
        bgr = image
        if image.shape[2] == 3 and image.dtype == np.uint8:
            import cv2 as _cv2  # noqa: F811

            bgr = _cv2.cvtColor(image, _cv2.COLOR_RGB2BGR)

        faces = self._net.detect(bgr.T)[0]  # returns numpy array of shape (N,6) or None
        detections: list[FaceDetection] = []
        if faces is not None and len(faces) > 0:
            for face in faces:
                x1 = int(face[0])
                y1 = int(face[1])
                w = int(face[2])
                h = int(face[3])
                conf = float(face[4]) if len(face) > 4 else 0.0

                bbox = BoundingBox(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h)
                # YuNet does not provide landmarks in default config; stub zeroed
                lm = np.zeros((5, 2), dtype=np.float32)
                detections.append(FaceDetection(bbox=bbox, landmarks=lm, confidence=conf))

        return detections


class EmbedderAdapter(_EmbedderBase):  # type: ignore[misc] (abstract base inheritance)
    """SFace embedder adapter via OpenCV DNN."""

    REQUIRED_MODEL_KEYS = [
        "model_id",
        "role",
        "source",
        "filename",
        "sha256",
        "license",
        "input_shape",
        "preprocessing_revision",
        "provider",
    ]

    def __init__(
        self,
        model_path: str | "os.PathLike[str]",  # noqa: ANN001 — PathLike accepted
    ) -> None:
        """Initialize SFace embedding model.

        Parameters
        ----------
        model_path : str
            Path to .onnx SFace model file. Lazy-loaded in embed().
        """
        self._model_path = str(model_path)
        self._initialized = False
        self._net = None  # OpenCV face.SFaceRecognizer — opaque handle

    @property
    def model_id(self) -> str:
        return "sface"

    @property
    def model_path(self) -> str:
        return self._model_path

    def _ensure_initialized(self):
        """Lazy load the embedder."""
        if self._initialized:
            return

        try:
            import cv2  # type: ignore[import-untyped]  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "SFace embedding requires opencv-python. Install via: pip install opencv-python"
            ) from exc

        if not hasattr(cv2, "face") or not hasattr(cv2.face, "SFaceRecognizer"):
            raise ValueError(
                "OpenCV does not have face.SFaceRecognizer. Requires opencv-contrib-python >= 4.5.0."
            )

        self._net = cv2.face.SFaceRecognizer.create(
            str(self._model_path),
            "",  # config_path — optional (empty when using defaults)
            confidence_threshold=0.6,
            nms_threshold=0.4,
        )

        if self._net is None:
            raise ValueError("SFaceRecognizer create returned None — model load failed")

        self._initialized = True

    def embed(self, face_crop, landmarks):  # noqa: ANN001 — simple inputs (per abstract sig)
        """Extract normalized embedding from an aligned face crop using SFace.

        Parameters (per contract):
            face_crop : np.ndarray — aligned face image; typically (112,112,3)
            landmarks : np.ndarray — five-point landmarks used for alignment (5,2)

        Returns:
            tuple[np.ndarray, int] — embedding vector and model_version
        """
        self._ensure_initialized()

        if not isinstance(face_crop, np.ndarray):
            raise ValueError("face_crop must be a numpy array")

        import cv2 as _cv2  # noqa: F811 — face crop resizer (112x112)

        # Resize to model expected size (typically 112x112); SFace normalizes internally
        face_112 = _cv2.resize(face_crop, (112, 112))  # type: ignore[arg-type] — cv2 resizer

        if face_112.shape[2] == 3 and face_112.dtype == np.uint8:
            face_112 = _cv2.cvtColor(face_112, _cv2.COLOR_RGB2BGR)

        emb = self._net.frontal(face_112)  # returns float32 embeddings (N,D)
        if emb is not None and len(emb) > 0:
            return (np.array(emb[0], dtype=np.float32), 1)  # model_version = 1

        raise ValueError("SFace returned no embedding")
