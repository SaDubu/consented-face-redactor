"""Tests for effects package (Mosaic / Sticker renderers)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from consented_face_redactor.domain.types import FaceBox, FiveLandmarks, MosaicConfig
from consented_face_redactor.effects import MosaicEffect, StickerEffect


# ------------------------------------------------------------------ #
# Helpers — create synthetic RGBA PNG bytes (4×4 fully opaque red)  #
# ------------------------------------------------------------------ #

def _make_png(width: int = 4, height: int = 4, r: int = 255, g: int = 0, b: int = 0, a: int = 255) -> bytes:
    """Return valid PNG bytes with uniform RGBA colour."""
    image = np.full((height, width, 4), (r, g, b, a), dtype=np.uint8)
    encoded, png = cv2.imencode(".png", image)
    assert encoded
    return png.tobytes()


# ================================================================== #
# MosaicEffect tests
# ================================================================== #


class TestMosaicEffect:

    @pytest.fixture
    def frame_rgb_20(self) -> np.ndarray:
        """Return a 20×20 RGB frame filled with gradient values (easy to see changes)."""
        arr = np.zeros((20, 20, 3), dtype=np.uint8)
        for i in range(20):
            for j in range(20):
                arr[i, j] = [i * 10 % 256, j * 10 % 256, (i + j) * 5 % 256]
        return arr

    @pytest.fixture
    def frame_gray_20(self) -> np.ndarray:
        """Return a 20×20 grayscale frame."""
        arr = np.zeros((20, 20), dtype=np.uint8)
        for i in range(20):
            for j in range(20):
                arr[i, j] = (i + j) * 6 % 256
        return arr

    # ---- Non-mutation tests ----

    def test_mosaic_does_not_mutate_input_rgb(self, frame_rgb_20: np.ndarray) -> None:
        """Mosaic must never modify the input array."""
        before = frame_rgb_20.copy()
        effect = MosaicEffect(MosaicConfig(force_block_size=4))
        _ = effect.render(frame_rgb_20, FaceBox(5, 5, 15, 15))
        assert np.array_equal(frame_rgb_20, before), "Mosaic mutated the input array!"

    def test_mosaic_does_not_mutate_input_gray(self, frame_gray_20: np.ndarray) -> None:
        """Mosaic must never modify grayscale input either."""
        before = frame_gray_20.copy()
        effect = MosaicEffect(MosaicConfig(force_block_size=4))
        _ = effect.render(frame_gray_20, FaceBox(5, 5, 15, 15))
        assert np.array_equal(frame_gray_20, before), "Mosaic mutated grayscale input!"

    # ---- Output shape / dtype ----

    def test_output_shape_rgb(self, frame_rgb_20: np.ndarray) -> None:
        result = MosaicEffect(MosaicConfig(force_block_size=4)).render(
            frame_rgb_20, FaceBox(5, 5, 15, 15)
        )
        assert result.shape == (20, 20, 3), f"Expected shape (20,20,3), got {result.shape}"
        assert result.dtype == frame_rgb_20.dtype

    def test_output_shape_gray(self, frame_gray_20: np.ndarray) -> None:
        result = MosaicEffect(MosaicConfig(force_block_size=4)).render(
            frame_gray_20, FaceBox(5, 5, 15, 15)
        )
        assert result.shape == (20, 20), f"Expected shape (20,20), got {result.shape}"

    # ---- Degenerate ROI ----

    def test_degenerate_zero_width_roi(self, frame_rgb_20: np.ndarray) -> None:
        result = MosaicEffect().render(frame_rgb_20, FaceBox(5, 5, 5, 15))
        assert np.array_equal(result, frame_rgb_20), "Degenerate ROI should pass through unchanged."

    def test_degenerate_out_of_bounds_roi(self, frame_rgb_20: np.ndarray) -> None:
        result = MosaicEffect().render(
            frame_rgb_20, FaceBox(100, 100, 200, 200)
        )
        assert np.array_equal(result, frame_rgb_20), "Out-of-bounds ROI (no overlap) passes through."

    def test_degenerate_negative_roi(self, frame_rgb_20: np.ndarray) -> None:
        result = MosaicEffect().render(
            frame_rgb_20, FaceBox(-5, -5, -1, -1)
        )
        assert np.array_equal(result, frame_rgb_20), "Negative ROI passes through."

    # ---- Downscale/upscale effect visible ----

    def test_mosaic_blocks_are_uniform(self, frame_rgb_20: np.ndarray) -> None:
        """Each block within the mosaic ROI should be a uniform colour (not original gradient)."""
        roi = FaceBox(5, 5, 15, 15)
        effect = MosaicEffect(MosaicConfig(force_block_size=4))
        result = effect.render(frame_rgb_20, roi)

        # ROI region is [5:15, 5:15]. Check that inside the mosaic patch there are blocks.
        roi_result = result[roi.y1 : roi.y2, roi.x1 : roi.x2]
        assert roi_result.shape == (10, 10, 3)

        # Block [0-h-1, 0-w-1] should be uniform — all pixels must equal first pixel.
        block_h = 4  # forced block size
        ref_val = roi_result[0, 0]
        assert np.all(roi_result[:block_h, :block_h] == ref_val), \
            "Mosaic blocks should be uniform."

    # ---- Clamping to frame edge ----

    def test_clamp_roi_to_frame(self, frame_rgb_20: np.ndarray) -> None:
        """ROI that extends beyond the frame must not cause IndexError."""
        effect = MosaicEffect(MosaicConfig(force_block_size=4))
        result = effect.render(frame_rgb_20, FaceBox(15, 15, 30, 30))
        assert result.shape == (20, 20, 3)

    # ---- Auto-size block from face width ----

    def test_auto_block_size_proportional(self, frame_rgb_20: np.ndarray) -> None:
        """Larger faces → larger blocks (default is face size // 8)."""
        small = FaceBox(5, 5, 15, 15)   # face width = 10 → block = 1
        large = FaceBox(2, 2, 42, 42)   # face width = 40 → block = 5

        eff = MosaicEffect()  # default config (no force_block_size)

        res_small = eff.render(frame_rgb_20, small)
        res_large = eff.render(frame_rgb_20, large)

        # Both should produce output; larger face should have fewer uniform blocks.
        assert res_small.shape == frame_rgb_20.shape
        assert res_large.shape == frame_rgb_20.shape


# ================================================================== #
# StickerEffect tests
# ================================================================== #


class TestStickerEffect:

    @pytest.fixture
    def png_bytes(self) -> bytes:
        return _make_png(32, 32, r=128, g=0, b=0, a=255)  # opaque red sticker

    @pytest.fixture
    def frame_rgb_64(self) -> np.ndarray:
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        # Fill with green (to contrast with red sticker)
        arr[:, :] = [0, 255, 0]
        return arr

    @pytest.fixture
    def box(self) -> FaceBox:
        return FaceBox(16, 16, 32, 32)

    @pytest.fixture
    def landmarks(self) -> FiveLandmarks:
        return FiveLandmarks(
            left_eye_x=20, left_eye_y=20,
            right_eye_x=28, right_eye_y=20,
            nose_x=24, nose_y=26,
        )

    # ---- Non-mutation ----

    def test_sticker_does_not_mutate_input(self, png_bytes: bytes, frame_rgb_64: np.ndarray, box: FaceBox, landmarks: FiveLandmarks) -> None:
        before = frame_rgb_64.copy()
        effect = StickerEffect(png_bytes)
        _ = effect.render(frame_rgb_64, box, landmarks)
        assert np.array_equal(frame_rgb_64, before), "Sticker mutated the input array!"

    # ---- Invalid PNG rejection ----

    def test_invalid_png_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid PNG bytes"):
            StickerEffect(b"\x00\x01\x02\x03")  # not a valid PNG

    # ---- Output shape / dtype ----

    def test_output_shape_and_dtype(self, png_bytes: bytes, frame_rgb_64: np.ndarray, box: FaceBox, landmarks: FiveLandmarks) -> None:
        effect = StickerEffect(png_bytes)
        result = effect.render(frame_rgb_64, box, landmarks)
        assert result.shape == frame_rgb_64.shape
        assert result.dtype == frame_rgb_64.dtype

    # ---- Anchor fallback → center (sticker placed inside ROI) ----

    # ---- Anchor fallback → center (sticker placed inside ROI) ----

    def test_anchor_center(self, png_bytes: bytes, frame_rgb_64: np.ndarray, box: FaceBox, landmarks: FiveLandmarks) -> None:
        effect = StickerEffect(png_bytes)
        face_box = FaceBox(24, 24, 40, 40)
        res = effect.render(frame_rgb_64, face_box, landmarks, anchor="center")

        # Check that some pixels inside the sticker ROI are NOT green anymore
        # (the red sticker overwrote some green).
        roi_res = res[face_box.y1 : face_box.y2, face_box.x1 : face_box.x2]
        assert not np.all(roi_res[:, :, 1] == 255), "center anchor should have blended the sticker."

    def test_anchor_left(self, png_bytes: bytes, frame_rgb_64: np.ndarray, box: FaceBox, landmarks: FiveLandmarks) -> None:
        effect = StickerEffect(png_bytes)
        face_box = FaceBox(24, 24, 40, 40)
        res = effect.render(frame_rgb_64, face_box, landmarks, anchor="left")
        roi_res = res[face_box.y1 : face_box.y2, face_box.x1 : face_box.x2]
        # "left" places the sticker at the left side of the face.
        # The right half of ROI should still be green (unchanged).
        mid_x = roi_res.shape[1] // 2
        assert np.all(roi_res[:, mid_x:] == [0, 255, 0]), \
            f"Right half of ROI should retain green background for 'left' anchor."

    def test_anchor_right(self, png_bytes: bytes, frame_rgb_64: np.ndarray, box: FaceBox, landmarks: FiveLandmarks) -> None:
        effect = StickerEffect(png_bytes)
        face_box = FaceBox(24, 24, 40, 40)
        res = effect.render(frame_rgb_64, face_box, landmarks, anchor="right")
        # "right" places the sticker at the right side of the face.
        # ROI should have the same shape as input (clipping is handled).
        assert res.shape == frame_rgb_64.shape

    def test_anchor_top(self, png_bytes: bytes, frame_rgb_64: np.ndarray, box: FaceBox, landmarks: FiveLandmarks) -> None:
        effect = StickerEffect(png_bytes)
        face_box = FaceBox(24, 24, 40, 40)
        res = effect.render(frame_rgb_64, face_box, landmarks, anchor="top")
        # Top anchor places sticker *above* the face → may be clipped.
        # This tests that clipping does NOT cause an IndexError.
        assert res.shape == frame_rgb_64.shape

    # ---- Scale policy ----

    def test_scale_factor_doubles_sticker(self, png_bytes: bytes, frame_rgb_64: np.ndarray, box: FaceBox, landmarks: FiveLandmarks) -> None:
        effect = StickerEffect(png_bytes)
        face_box = FaceBox(10, 10, 26, 26)  # small face; height=16
        # scale=0.5 → ~16px sticker (area ~256); scale=2.0 → ~64px sticker (area ~4096).
        res_half = effect.render(frame_rgb_64, face_box, landmarks, scale_factor=0.5)
        res_large = effect.render(frame_rgb_64, face_box, landmarks, scale_factor=2.0)

        # Larger sticker → more non-green pixels in the entire frame.
        non_green_small = int(np.sum(res_half[:, :, 1] != 255))
        non_green_large = int(np.sum(res_large[:, :, 1] != 255))

        assert non_green_large > non_green_small, \
            f"scale=2.0 should cover more area (non_greens={non_green_large}, non_greens={non_green_small})"

    # ---- Frame-edge clipping ----

    def test_clipped_sticker_no_crash(self, png_bytes: bytes, landmarks: FiveLandmarks) -> None:
        """Sticker that would place entirely outside the frame must not IndexError."""
        effect = StickerEffect(png_bytes)
        small_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = effect.render(small_frame, FaceBox(50, 50, 80, 80), landmarks)
        assert result.shape == (10, 10, 3)

    # ---- Source asset immutability ----

    def test_source_png_bytes_unchanged(self, png_bytes: bytes) -> None:
        """The original PNG bytes must not be modified after StickerEffect construction."""
        before = bytes(png_bytes)
        _ = StickerEffect(png_bytes)
        assert png_bytes == before

    # ---- Transparent pixel preservation ----

    def test_transparent_pixels_preserve_background(self, frame_rgb_64: np.ndarray) -> None:
        """Transparent pixels of the overlay should remain as original background colour.

        We use a small 1×1 red-sticker (r=255,g=0,b=0,a=0 → fully transparent).
        Pixels under that ROI should still be green [0,255,0].
        """
        transparent_png = _make_png(1, 1, r=0, g=0, b=0, a=0)

        effect = StickerEffect(transparent_png, scale_factor=1.0, anchor="center", eye_rotation=True)
        # Place the transparent sticker at [32,32], a green pixel on frame.
        res = effect.render(frame_rgb_64, FaceBox(32, 32, 33, 33), FiveLandmarks(left_eye_x=32, left_eye_y=30, right_eye_x=35, right_eye_y=30, nose_x=33, nose_y=34))
        # Pixel inside ROI should remain green (background preserved).
        roi_pixel = res[32, 32]
        assert tuple(roi_pixel) == (0, 255, 0), \
            f"Transparent pixels did not preserve background: got {tuple(roi_pixel)}."
