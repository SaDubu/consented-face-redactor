"""Verified, lazy-loaded TAPNext++ adapter.

The adapter exposes only point localization.  Its output must never be used as
identity approval; gallery authorization remains a separate pipeline concern.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from consented_face_redactor.model_manifest import verify_model_file

from .types import PointTrackResult


BackendFactory = Callable[..., Any]


def _load_tapnextpp_factory(vendor_source_dir: Path) -> BackendFactory:
    """Import the optional official TAPNet source without importing it at package load."""
    source = Path(vendor_source_dir).resolve()
    if not (source / "tapnet" / "tapnextpp" / "votsp2026" / "model.py").is_file():
        raise RuntimeError("TAPNext++ source directory is unavailable or incomplete")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        # Import torchvision first so its operators are registered before TAPNet.
        importlib.import_module("torch")
        importlib.import_module("torchvision")
        module = importlib.import_module("tapnet.tapnextpp.votsp2026.model")
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError("TAPNext++ runtime dependencies could not be loaded") from exc
    return module.TAPNextPP.from_checkpoint


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("frame must be an HxWx3 uint8 BGR array")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("frame dimensions must be at least 2x2")
    return np.ascontiguousarray(array)


def _validate_queries(query_points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    points = np.asarray(query_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
        raise ValueError("query_points must have shape (N, 2) with N >= 2")
    if not np.isfinite(points).all():
        raise ValueError("query_points must be finite")
    height, width = frame.shape[:2]
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width):
        raise ValueError("query point x coordinate is outside the frame")
    if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
        raise ValueError("query point y coordinate is outside the frame")
    return np.ascontiguousarray(points)


class TapNextPlusPlusAdapter:
    """Online TAPNext++ point tracker implementing :class:`PointTracker`."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        vendor_source_dir: Path,
        manifest_entry: dict[str, Any],
        device: str = "cuda",
        input_resolution: int = 512,
        half_precision: bool = False,
        compile_model: bool = False,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        if input_resolution <= 0 or input_resolution % 16 != 0:
            raise ValueError("input_resolution must be a positive multiple of 16")
        if manifest_entry.get("role") != "tracker" or manifest_entry.get("provider") != "PyTorch":
            raise ValueError("manifest entry must describe a PyTorch tracker")
        checkpoint = Path(checkpoint_path)
        verify_model_file(manifest_entry, checkpoint)
        factory = backend_factory or _load_tapnextpp_factory(Path(vendor_source_dir))
        try:
            self._backend = factory(
                checkpoint,
                device=device,
                half_precision=half_precision,
                compile_model=compile_model,
                input_resolution=input_resolution,
            )
        except Exception as exc:
            raise RuntimeError("TAPNext++ checkpoint could not be loaded") from exc
        self._model_id = str(manifest_entry["model_id"])
        self._model_revision = (
            f"{self._model_id}:preprocess-v{manifest_entry['preprocessing_revision']}"
        )
        self._state: Any | None = None
        self._last_frame_index: int | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def reset(self) -> None:
        self._state = None
        self._last_frame_index = None

    def initialize(
        self,
        frame: np.ndarray,
        *,
        frame_index: int,
        query_points: np.ndarray,
    ) -> PointTrackResult:
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        clean_frame = _validate_frame(frame)
        clean_queries = _validate_queries(query_points, clean_frame)
        self.reset()
        positions, visible, state = self._backend.track_frame(
            clean_frame,
            query_points_xy=clean_queries,
        )
        result = self._make_result(frame_index, positions, visible)
        self._state = state
        self._last_frame_index = frame_index
        return result

    def update(self, frame: np.ndarray, *, frame_index: int) -> PointTrackResult:
        if self._state is None or self._last_frame_index is None:
            raise RuntimeError("tracker must be initialized before update")
        if frame_index != self._last_frame_index + 1:
            raise ValueError("TAPNext++ updates must use consecutive frame indices")
        clean_frame = _validate_frame(frame)
        positions, visible, state = self._backend.track_frame(
            clean_frame,
            state=self._state,
        )
        result = self._make_result(frame_index, positions, visible)
        self._state = state
        self._last_frame_index = frame_index
        return result

    def _make_result(
        self,
        frame_index: int,
        positions: np.ndarray,
        visible: np.ndarray,
    ) -> PointTrackResult:
        points = np.asarray(positions, dtype=np.float32)
        visibility = np.asarray(visible, dtype=np.float32)
        return PointTrackResult(
            frame_index=frame_index,
            points_xy=points,
            visibility=visibility,
            model_revision=self._model_revision,
        )
