"""Versioned, backwards-compatible configuration for the redaction pipeline."""

from __future__ import annotations

import math
from typing import Any


class Config:
    """Pipeline configuration with a compatibility-first dictionary loader."""

    SCHEMA_VERSION = 4
    _FIELDS = (
        "effect_mode",
        "t_confirm",
        "t_keep",
        "track_lost_ttl_frames",
        "recheck_interval_frames",
        "mosaic_grid_cells",
        "mosaic_padding_ratio",
        "mosaic_min_block_px",
        "mosaic_shape",
        "mosaic_ellipse_horizontal_scale",
        "mosaic_ellipse_vertical_scale",
    )

    def __init__(
        self,
        effect_mode: str = "mosaic",
        *,
        t_confirm: float = 0.65,
        t_keep: float = 0.55,
        track_lost_ttl_frames: int = 10,
        recheck_interval_frames: int = 30,
        mosaic_grid_cells: int = 12,
        mosaic_padding_ratio: float = 0.18,
        mosaic_min_block_px: int = 10,
        mosaic_shape: str = "ellipse",
        mosaic_ellipse_horizontal_scale: float = 1.40,
        mosaic_ellipse_vertical_scale: float = 1.50,
        schema_version: int = SCHEMA_VERSION,
    ) -> None:
        if isinstance(mosaic_grid_cells, bool) or not isinstance(mosaic_grid_cells, int):
            raise TypeError("mosaic_grid_cells must be an integer")
        if not 2 <= mosaic_grid_cells <= 64:
            raise ValueError("mosaic_grid_cells must be in [2, 64]")
        if isinstance(mosaic_padding_ratio, bool) or not isinstance(
            mosaic_padding_ratio, (int, float)
        ):
            raise TypeError("mosaic_padding_ratio must be numeric")
        if not math.isfinite(float(mosaic_padding_ratio)) or not 0.0 <= float(
            mosaic_padding_ratio
        ) <= 0.5:
            raise ValueError("mosaic_padding_ratio must be finite and in [0, 0.5]")
        if isinstance(mosaic_min_block_px, bool) or not isinstance(mosaic_min_block_px, int):
            raise TypeError("mosaic_min_block_px must be an integer")
        if not 1 <= mosaic_min_block_px <= 256:
            raise ValueError("mosaic_min_block_px must be in [1, 256]")
        if mosaic_shape not in {"rectangle", "ellipse"}:
            raise ValueError("mosaic_shape must be 'rectangle' or 'ellipse'")
        ellipse_scales = {
            "mosaic_ellipse_horizontal_scale": mosaic_ellipse_horizontal_scale,
            "mosaic_ellipse_vertical_scale": mosaic_ellipse_vertical_scale,
        }
        for name, value in ellipse_scales.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or not 1.0 <= float(value) <= 3.0:
                raise ValueError(f"{name} must be finite and in [1.0, 3.0]")
        horizontal_scale = float(mosaic_ellipse_horizontal_scale)
        vertical_scale = float(mosaic_ellipse_vertical_scale)
        if vertical_scale < horizontal_scale:
            raise ValueError("mosaic ellipse must be at least as tall as it is wide")
        if 1.0 / horizontal_scale**2 + 1.0 / vertical_scale**2 > 1.0:
            raise ValueError("mosaic ellipse scales must enclose the source bbox corners")
        self.effect_mode = effect_mode
        self.t_confirm = t_confirm
        self.t_keep = t_keep
        self.track_lost_ttl_frames = track_lost_ttl_frames
        self.recheck_interval_frames = recheck_interval_frames
        self.mosaic_grid_cells = mosaic_grid_cells
        self.mosaic_padding_ratio = float(mosaic_padding_ratio)
        self.mosaic_min_block_px = mosaic_min_block_px
        self.mosaic_shape = mosaic_shape
        self.mosaic_ellipse_horizontal_scale = horizontal_scale
        self.mosaic_ellipse_vertical_scale = vertical_scale
        self.schema_version = schema_version

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, strict: bool = False) -> "Config":
        """Load v1/v2/v3/v4 data; strict mode rejects unknown keys.

        Omitted ``schema_version`` denotes legacy v1 input. It is accepted and
        represented as v4 in memory so its next serialization is migrated.
        """
        if not isinstance(data, dict):
            raise ValueError("configuration must be an object")
        raw_version = data.get("schema_version", 1)
        if (
            isinstance(raw_version, bool)
            or not isinstance(raw_version, int)
            or raw_version not in (1, 2, 3, cls.SCHEMA_VERSION)
        ):
            raise ValueError("unsupported config schema_version")
        allowed = set(cls._FIELDS) | {"schema_version"}
        unknown = set(data) - allowed
        if strict and unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
        values = {key: data[key] for key in cls._FIELDS if key in data}
        values["schema_version"] = cls.SCHEMA_VERSION
        return cls(**values)

    @classmethod
    def default(cls) -> "Config":
        return cls()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "effect_mode": self.effect_mode,
            "t_confirm": self.t_confirm,
            "t_keep": self.t_keep,
            "track_lost_ttl_frames": self.track_lost_ttl_frames,
            "recheck_interval_frames": self.recheck_interval_frames,
            "mosaic_grid_cells": self.mosaic_grid_cells,
            "mosaic_padding_ratio": self.mosaic_padding_ratio,
            "mosaic_min_block_px": self.mosaic_min_block_px,
            "mosaic_shape": self.mosaic_shape,
            "mosaic_ellipse_horizontal_scale": self.mosaic_ellipse_horizontal_scale,
            "mosaic_ellipse_vertical_scale": self.mosaic_ellipse_vertical_scale,
        }
