"""Mosaic renderer — downscale/upscale ROI with configurable block size."""

from __future__ import annotations

import cv2
import numpy as np

from ..domain.types import FaceBox, MosaicConfig


def mosaic_block_size(
    face_width: int,
    face_height: int,
    *,
    grid_cells: int,
    min_block_px: int,
) -> int:
    """Return an adaptive block size based on the shorter face dimension."""
    if face_width < 1 or face_height < 1:
        raise ValueError("face dimensions must be positive")
    if isinstance(grid_cells, bool) or not isinstance(grid_cells, int) or grid_cells < 2:
        raise ValueError("grid_cells must be an integer >= 2")
    if isinstance(min_block_px, bool) or not isinstance(min_block_px, int) or min_block_px < 1:
        raise ValueError("min_block_px must be a positive integer")
    return max(int(round(min(face_width, face_height) / grid_cells)), min_block_px)


def expand_bbox(
    bbox: tuple[float, float, float, float],
    *,
    frame_shape: tuple[int, ...],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Expand a bbox on every side and clip it to the frame."""
    if len(frame_shape) < 2 or frame_shape[0] < 1 or frame_shape[1] < 1:
        raise ValueError("frame_shape must contain positive height and width")
    if not np.isfinite(padding_ratio) or not 0.0 <= padding_ratio <= 0.5:
        raise ValueError("padding_ratio must be finite and in [0, 0.5]")
    x1, y1, x2, y2 = map(float, bbox)
    if not np.isfinite((x1, y1, x2, y2)).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must contain finite ordered coordinates")
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    height, width = int(frame_shape[0]), int(frame_shape[1])
    return (
        max(0, int(np.floor(x1 - pad_x))),
        max(0, int(np.floor(y1 - pad_y))),
        min(width, int(np.ceil(x2 + pad_x))),
        min(height, int(np.ceil(y2 + pad_y))),
    )


def ellipse_bounds(
    bbox: tuple[float, float, float, float],
    *,
    frame_shape: tuple[int, ...],
    horizontal_scale: float,
    vertical_scale: float,
) -> tuple[int, int, int, int]:
    """Return clipped bounds of a vertical ellipse enclosing ``bbox``."""
    if len(frame_shape) < 2 or frame_shape[0] < 1 or frame_shape[1] < 1:
        raise ValueError("frame_shape must contain positive height and width")
    x1, y1, x2, y2 = map(float, bbox)
    if not np.isfinite((x1, y1, x2, y2)).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must contain finite ordered coordinates")
    scales = (float(horizontal_scale), float(vertical_scale))
    if not np.isfinite(scales).all() or min(scales) < 1.0:
        raise ValueError("ellipse scales must be finite and at least 1.0")
    if scales[1] < scales[0]:
        raise ValueError("ellipse must be at least as tall as it is wide")
    if 1.0 / scales[0] ** 2 + 1.0 / scales[1] ** 2 > 1.0:
        raise ValueError("ellipse scales must enclose the bbox corners")
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    radius_x = (x2 - x1) * scales[0] / 2.0
    radius_y = (y2 - y1) * scales[1] / 2.0
    height, width = int(frame_shape[0]), int(frame_shape[1])
    return (
        max(0, int(np.floor(center_x - radius_x))),
        max(0, int(np.floor(center_y - radius_y))),
        min(width, int(np.ceil(center_x + radius_x))),
        min(height, int(np.ceil(center_y + radius_y))),
    )


def ellipse_mask(
    bbox: tuple[float, float, float, float],
    *,
    frame_shape: tuple[int, ...],
    horizontal_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    """Return a full-frame boolean mask for the configured face ellipse."""
    bounds = ellipse_bounds(
        bbox,
        frame_shape=frame_shape,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
    )
    x1, y1, x2, y2 = map(float, bbox)
    center = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))
    axes = (
        max(1, int(np.ceil((x2 - x1) * horizontal_scale / 2.0))),
        max(1, int(np.ceil((y2 - y1) * vertical_scale / 2.0))),
    )
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 255, thickness=-1)
    bx1, by1, bx2, by2 = bounds
    clipped = np.zeros_like(mask)
    clipped[by1:by2, bx1:bx2] = mask[by1:by2, bx1:bx2]
    return clipped.astype(bool)


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

        if self.config.shape not in {"rectangle", "ellipse"}:
            raise ValueError("mosaic shape must be 'rectangle' or 'ellipse'")

        raw_bbox = (float(roi.x1), float(roi.y1), float(roi.x2), float(roi.y2))
        if self.config.shape == "ellipse":
            effect_bounds = ellipse_bounds(
                raw_bbox,
                frame_shape=frame.shape,
                horizontal_scale=self.config.ellipse_horizontal_scale,
                vertical_scale=self.config.ellipse_vertical_scale,
            )
        else:
            effect_bounds = (roi.x1, roi.y1, roi.x2, roi.y2)

        # Clamp effect bounds to image bounds
        h, w = frame.shape[:2]
        x1 = max(effect_bounds[0], 0)
        y1 = max(effect_bounds[1], 0)
        x2 = min(effect_bounds[2], w)
        y2 = min(effect_bounds[3], h)

        if x2 <= x1 or y2 <= y1:
            return frame.copy()

        # Auto-size from the shorter side so production strength stays
        # consistent across small/large and portrait/landscape face boxes.
        block_size = self.config.force_block_size or mosaic_block_size(
            roi.width,
            roi.height,
            grid_cells=self.config.grid_cells,
            min_block_px=self.config.min_block_px,
        )
        block_size = max(int(block_size), self.config.min_block_px)

        # Extract ROI and clone to avoid mutating input
        roi_img = frame[y1:y2, x1:x2].copy()

        # Downscale → upscale with nearest-neighbor interpolation (mosaic effect)
        small_h = max((y2 - y1) // block_size, 1)
        small_w = max((x2 - x1) // block_size, 1)
        small = cv2.resize(roi_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        # Compose output
        out = frame.copy()
        if self.config.shape == "ellipse":
            mask = ellipse_mask(
                raw_bbox,
                frame_shape=frame.shape,
                horizontal_scale=self.config.ellipse_horizontal_scale,
                vertical_scale=self.config.ellipse_vertical_scale,
            )[y1:y2, x1:x2]
            out_region = out[y1:y2, x1:x2]
            out_region[mask] = mosaic[mask]
        else:
            out[y1:y2, x1:x2] = mosaic
        return out  # type: ignore[return-value]


__all__ = [
    "MosaicEffect",
    "ellipse_bounds",
    "ellipse_mask",
    "expand_bbox",
    "mosaic_block_size",
]
