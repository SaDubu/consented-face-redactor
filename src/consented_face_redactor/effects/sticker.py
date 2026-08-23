"""Sticker renderer — alpha-blended PNG overlay over a face ROI."""

from __future__ import annotations

import math
import cv2
import numpy as np

from ..domain.types import FaceBox, FiveLandmarks


class StickerEffect:
    """Alpha-blend a transparent sticker (PNG) over an image ROI.

    Never mutates input arrays — always returns a new output array.
    Degenerate ROIs are passed through unmodified.
    """

    def __init__(
        self,
        png_bytes: bytes,
        *,
        scale_factor: float = 1.0,
        anchor: str = "center",
        eye_rotation: bool = True,
    ) -> None:
        """Load a sticker PNG and cache default override parameters."""
        raw = np.frombuffer(png_bytes, dtype=np.uint8)
        self._sticker = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)  # RGBA
        if self._sticker is None:
            raise ValueError("StickerEffect received invalid PNG bytes.")
        self._scale_factor = scale_factor
        self._anchor = anchor
        self._eye_rotation = eye_rotation

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def render(
        self,
        frame: np.ndarray,
        box: FaceBox,
        landmarks: FiveLandmarks,
        *,
        scale_factor: float | None = None,
        anchor: str | None = None,
    ) -> np.ndarray:
        """Alpha-blend the sticker onto an RGB image, rotated to eye-angle."""
        out = frame.copy()

        # Use instance defaults when kwargs are not provided
        scale_value = scale_factor if scale_factor is not None else self._scale_factor
        anchor_value = anchor if anchor is not None else self._anchor
        h = box.height or 1
        sticker_w = max(int(self._sticker.shape[1] * scale_value), 1)
        sticker_h = max(int(self._sticker.shape[0] * scale_value), 1)

        sticker_resized = cv2.resize(
            self._sticker, (sticker_w, sticker_h), interpolation=cv2.INTER_AREA
        )

        # Compute placement rectangle inside-frame
        cx, cy = box.center_x, box.center_y
        place_rect = self._place(anchor_value, x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2, h=sticker_h, w=sticker_w)

        # Extract the canvas ROI (clipped to image bounds)
        h_img, w_img = out.shape[:2]
        roi_y1 = max(0, place_rect.y1)
        roi_y2 = min(h_img, place_rect.y2)
        roi_x1 = max(0, place_rect.x1)
        roi_x2 = min(w_img, place_rect.x2)

        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return out  # clipped away; nothing to do

        canvas_roi = out[roi_y1:roi_y2, roi_x1:roi_x2]  # shape (H', W', C)

        # Sticker ROI (clipped)
        sx1 = max(0, -place_rect.x1)
        sy1 = max(0, -place_rect.y1)
        sw = min(sticker_w, place_rect.x2 - roi_x1) - 0
        sh = min(sticker_h, place_rect.y2 - roi_y1) - 0
        sticker_roi = sticker_resized[sy1 : sy1 + sh, sx1 : sx1 + sw]

        if sticker_roi.size == 0 or canvas_roi.size == 0:
            return out

        # Eye-angle rotation (when enabled)
        angle = math.degrees(landmarks.eye_angle)
        sticker_roi_rotated = None
        if abs(angle) > 0.5:
            center = (sticker_roi.shape[1] / 2, sticker_roi.shape[0] / 2)
            mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            sticker_roi_rotated = cv2.warpAffine(sticker_roi, mat, (sticker_roi.shape[1], sticker_roi.shape[0]))
        else:
            sticker_roi_rotated = sticker_roi

        # Alpha-blend (RGBA → RGB on canvas_roi which is RGB)
        # sticker ROI has 4 channels (A); canvas has 3.
        # Pad or crop to match sizes before blending.
        if canvas_roi.shape[2] == 3 and sticker_roi_rotated.ndim == 3:
            src_h, src_w = sticker_roi_rotated.shape[:2]
            tgt_h, tgt_w = canvas_roi.shape[:2]

            # Build aligned RGBA on the target size by cropping or padding
            align_h = min(src_h, tgt_h)
            align_w = min(src_w, tgt_w)
            aligned = np.zeros((tgt_h, tgt_w, 4), dtype=np.float32)

            # Cropping sticker ROI to target grid size so shapes match exactly
            blended_sticker = sticker_roi_rotated[:align_h, :align_w].astype(np.float32)
            aligned[:align_h, :align_w, :3] = blended_sticker[:, :, :3]
            aligned[:align_h, :align_w, 3]  = blended_sticker[:, :, 3]

            alpha = aligned[:, :, 3:4] / 255.0
            color = aligned[:, :, :3]
            canvas_roi_f = canvas_roi.astype(np.float32)

            # Alpha is per-pixel — need manual blending (cv2.addWeighted doesn't accept array alpha)
            blended = color * alpha + canvas_roi_f * (1.0 - alpha)
            out[roi_y1:roi_y1 + tgt_h, roi_x1:roi_x1 + tgt_w] = np.clip(blended, 0, 255).astype(np.uint8)

        return out

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _place(anchor, x1, y1, x2, y2, h: int, w: int):
        """Return a placement box based on the given anchor relative to the face box."""
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if anchor == "left":
            px, py = x1 - w, cy - h // 2
        elif anchor == "right":
            px, py = x2, cy - h // 2
        elif anchor == "top":
            px = cx - w / 2
            py = y1 - h
        else:  # center / fallback
            px, py = cx - w // 2, cy - h // 2
        return type("Rect", (), {"x1": int(px), "y1": int(py), "x2": int(px) + w, "y2": int(py) + h})


__all__ = ["StickerEffect"]
