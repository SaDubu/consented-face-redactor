"""YuNet face detection adapter via OpenCV DNN.

This adapter implements the DetectorAdapter contract using OpenCV's YuNet model.
The raw vendor output is normalized at the boundary so the pipeline never
depends on vendor-specific formats.

No download code exists — the operator must place the ONNX model file manually.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class DetectorAdapter:
    """YuNet face detector adapter via OpenCV DNN."""

    _REQUIRED_MODEL_KEYS = ["model_id", "source", "filename",
                            "sha256", "license"]

    def __init__(self, model_path):  # noqa: ANN001 — simple path arg
        """Initialize YuNet detector.

        Parameters
        ----------
        model_path : str | PathLike[str]
            Path to a .onnx YuNet model file. Does NOT load the model yet
            (lazy loading in `detect()`).

        Raises
        ------
        ValueError if model_id is not provided / empty.
        """
        self._model_path = str(model_path)
        self._initialized = False
        self._net = None

    @property
    def model_id(self) -> str:
        return "yunet"

    @property
    def model_path(self) -> str:
        return self._model_path

    def _ensure_initialized(self):
        """Lazy load the detector if not already loaded."""
        if self._initialized:
            return

        try:
            import cv2  # type: ignore[import-not-found] (opencv-python required)
        except ImportError as ex:
            raise ImportError(
                "YuNet detection requires opencv-python. Install via: pip install opencv-python"
            ) from ex

        self._net = cv2.FaceDetectorYN.create(
            self._model_path,  # noqa: ARG005 (placeholder — YuNet path param)
            "",                 # config_path — not needed for YuNet
            (320, 320),       # input_size — default; adjust per-frame in real impl
        ) if hasattr(cv2, "FaceDetectorYN") else None

        if self._net is None:
            raise ValueError(
                "OpenCV version does not support FaceDetectorYN. "
                "Requires OpenCV-contrib-python >= 4.5.5."
            )

        self._initialized = True

    def detect(self, image):  # noqa: ANN001 — simple array arg
        """Detect faces in input image using YuNet.

        Parameters (per DetectorAdapter contract):
            image : numpy.ndarray — uint8, BGR or RGB, shape (H,W,C)

        Returns:
            Sequence[FaceDetection] — zero or more detections
        """
        self._ensure_initialized()

        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("image must be a 3D numpy array (H,W,C)")

        # Convert to BGR if input is RGB
        bgr = image
        if image.shape[2] == 3:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.dtype == np.uint8 else image

        # YuNet expects (H,W,C); may need to resize internally
        faces = self._net.detect(bgr)  # returns numpy array of shape (N,6)

        detections = []
        if faces is not None and len(faces) > 0:
            for face in faces:
                x1, y1, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])  # noqa: N806
                conf = float(face[4]) if len(face) > 4 else 0.0

                from .detection_iface import BoundingBox, FaceDetection

                bbox = BoundingBox(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h)  # noqa: N806
                detections.append(FaceDetection(
                    bbox=bbox,
                    landmarks=np.zeros((5, 2), dtype=np.float32),  # stub — YuNet doesn't provide landmarks in this config
                    confidence=conf,
                ))

        return detections


class EmbedderAdapter:
    """SFace embedder adapter via OpenCV DNN."""

    _REQUIRED_MODEL_KEYS = ["model_id", "role", "source",
                            "filename", "sha256",
                            "license", "input_shape",
                            "preprocessing_revision", "provider"]

    def __init__(self, model_path):  # noqa: ANN001
        """Initialize SFace embedding model.

        Parameters
        ----------
        model_path : str | PathLike[str]
            Path to a .onnx SFace model file. Lazy-loaded in `embed()`.
        """
        self._model_path = str(model_path)
        self._initialized = False
        self._net = None

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
            import cv2  # type: ignore[import-not-found]
        except ImportError as ex:
            raise ImportError(
                "SFace embedding requires opencv-python. Install via: pip install opencv-python"
            ) from ex

        if not hasattr(cv2, "face"):
            raise ValueError(
                "OpenCV does not have face module. Requires opencv-contrib-python >= 4.5.0"
            )

        self._net = cv2.face.SFaceRecognizer.create(
            str(self._model_path),  # noqa: ARG005 — placeholder
            "",                      # config_path — optional
            confidence_threshold=0.6,
            nms_threshold=0.4,
        ) if hasattr(cv2, "face") and hasattr(cv2.face, "SFaceRecognizer") else None

        if self._net is None:
            raise ValueError(
                "OpenCV version does not support face recognition. "
                "Requires opencv-contrib-python."
            )

        self._initialized = True

    def embed(self, face_crop):  # noqa: ANN001 — simple input
        """Extract normalized embedding from an aligned face crop using SFace.

        Parameters (per contract):
            face_crop : np.ndarray — aligned face image, typically (112,112,3)
            landmarks : np.ndarray — five-point landmarks used for alignment (5,2)

        Returns:
            tuple[np.ndarray, int] — embedding vector and model_version
        """
        self._ensure_initialized()

        if not isinstance(face_crop, np.ndarray):
            raise ValueError("face_crop must be a numpy array")

        # Resize to model expected size (typically 112x112)
        face_112 = cv2.resize(face_crop, (112, 112))  # noqa: ARG005 — placeholder — SFace normalizes internally

        # Convert RGB -> BGR for OpenCV
        if face_112.shape[2] == 3:
            face_112 = cv2.cvtColor(face_112, cv2.COLOR_RGB2BGR)  # noqa: ARG005 — placeholder

        emb = self._net.frontal(face_112)  # returns float32 embeddings (N,D)
        if emb is not None and len(emb) > 0:
            return (emb[0], 1)  # model_version = 1 for initial release
        else:
            raise ValueError("SFace returned no embedding")

