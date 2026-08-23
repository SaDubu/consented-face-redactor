"""Mosaic renderer — downscale/upscale ROI with configurable block size."""

from __future__ import annotations

import cv2
import numpy as np

from ..domain.types import FaceBox, MosaicConfig


class MosaicEffect:
    """Apply a mosaic/pixelate effect to a face ROI in an image.

    Never mutates the input array — always returns a copy with the ROI applied.
    Degenerate ROIs (width/height <= 0) are returned unmodified.
    """

    def __init__(self, config: MosaicConfig | None = None) -> None:
        self.config = config or MosaicConfig()

    def render(self, frame: np.ndarray, roi: FaceBox) -> np.ndarray:
        """Return a copy of *frame* with mosaic applied to the face bounding box.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, C) or grayscale (H, W).  Not mutated.
        roi : FaceBox
            Face bounding box in pixel coordinates [[x1, y1), [x2, y2)).

        Returns
        -------
        np.ndarray
            New array with mosaic applied. Shape/dtype identical to *frame*.
        """
        if roi.width < 1 or roi.height < 1:
            return frame.copy()

        # Clamp ROI to image bounds
        h, w = frame.shape[:2]
        x1 = max(roi.x1, 0)
        y1 = max(roi.y1, 0)
        x2 = min(roi.x2, w)
        y2 = min(roi.y2, h)

        if x2 <= x1 or y2 <= y1:
            return frame.copy()

        # Auto-size block if not overridden
        block_size = self.config.force_block_size or (max(roi.width, roi.height) // 8)
        block_size = max(block_size, self.config.min_block_px)

        # Extract ROI and clone to avoid mutating input
        roi_img = frame[y1:y2, x1:x2].copy()

        # Downscale → upscale with nearest-neighbor interpolation (mosaic effect)
        small_h = max((y2 - y1) // block_size, 1)
        small_w = max((x2 - x1) // block_size, 1)
        small = cv2.resize(roi_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        # Compose output
        out = frame.copy()
        out[y1:y2, x1:x2] = mosaic
        return out  # type: ignore[return-value]


__all__ = ["MosaicEffect"]
