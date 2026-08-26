"""Contracts for strong adaptive mosaic and padded face coverage."""

from __future__ import annotations

import numpy as np

from consented_face_redactor.config import Config
from consented_face_redactor.effects.mosaic import (
    MosaicEffect,
    ellipse_bounds,
    ellipse_mask,
    expand_bbox,
    mosaic_block_size,
)
from consented_face_redactor.domain.types import FaceBox, MosaicConfig
from consented_face_redactor.pipeline import _apply_effect_to_bbox


def _noise_frame(height: int = 240, width: int = 320) -> np.ndarray:
    return np.random.default_rng(20260826).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )


def test_adaptive_block_size_uses_short_face_side() -> None:
    assert mosaic_block_size(320, 240, grid_cells=12, min_block_px=10) == 20
    assert mosaic_block_size(80, 120, grid_cells=12, min_block_px=10) == 10


def test_expand_bbox_is_symmetric_and_clipped() -> None:
    assert expand_bbox(
        (10.0, 20.0, 110.0, 120.0),
        frame_shape=(200, 200, 3),
        padding_ratio=0.2,
    ) == (0, 0, 130, 140)


def test_moderate_ellipse_mosaic_changes_only_vertical_oval() -> None:
    source = _noise_frame()
    roi = FaceBox(40, 40, 280, 200)
    config = MosaicConfig(
        grid_cells=12,
        min_block_px=10,
        shape="ellipse",
        ellipse_horizontal_scale=1.4,
        ellipse_vertical_scale=1.5,
    )
    result = MosaicEffect(config).render(source, roi)
    mask = ellipse_mask(
        (40.0, 40.0, 280.0, 200.0),
        frame_shape=source.shape,
        horizontal_scale=1.4,
        vertical_scale=1.5,
    )
    unique_colors = np.unique(result[mask].reshape(-1, 3), axis=0)

    x1, y1, x2, y2 = ellipse_bounds(
        (40.0, 40.0, 280.0, 200.0),
        frame_shape=source.shape,
        horizontal_scale=1.4,
        vertical_scale=1.5,
    )
    block_size = mosaic_block_size(240, 160, grid_cells=12, min_block_px=10)
    maximum_cells = max((x2 - x1) // block_size, 1) * max(
        (y2 - y1) // block_size, 1
    )
    assert len(unique_colors) <= maximum_cells
    assert not np.array_equal(result[mask], source[mask])
    assert np.array_equal(result[~mask], source[~mask])


def test_pipeline_mosaic_uses_circumscribed_vertical_ellipse() -> None:
    source = _noise_frame()
    config = Config()
    bbox = (100.0, 80.0, 200.0, 160.0)
    result = _apply_effect_to_bbox(
        source,
        bbox,
        "mosaic",
        config,
        None,
    )
    bounds = ellipse_bounds(
        bbox,
        frame_shape=source.shape,
        horizontal_scale=1.4,
        vertical_scale=1.5,
    )
    mask = ellipse_mask(
        bbox,
        frame_shape=source.shape,
        horizontal_scale=1.4,
        vertical_scale=1.5,
    )
    x1, y1, x2, y2 = bounds

    assert (y2 - y1) / 80.0 > (x2 - x1) / 100.0
    assert all(mask[y, x] for x, y in ((100, 80), (199, 80), (100, 159), (199, 159)))
    assert not np.array_equal(result[mask], source[mask])
    assert np.array_equal(result[~mask], source[~mask])
