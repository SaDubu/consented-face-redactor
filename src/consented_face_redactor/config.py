"""Strict configuration validation for consented-face-redactor.

All configuration must be validated before use. Unknown keys, URLs, token-like
fields, NaN/Inf values, invalid ranges, and equal input/output paths are rejected.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


# ------------------------------------------------------------------ #
# Enums & constraints
# ------------------------------------------------------------------ #


class EffectMode(str, Enum):
    MOSAIC = "mosaic"
    STICKER = "sticker"
    NONE = "none"


class UncertainPolicy(str, Enum):
    PRIVACY_SAFE = "privacy_safe"  # default: keep bounded ROI short TTL
    PRECISION = "precision"          # stop redaction immediately


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

_URL_RE = re.compile(r"^https?://")
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=]{40,}")


def _validate_path(p: Any, label: str) -> Path:
    """Convert to absolute path; reject non-path values."""
    if not isinstance(p, (str, Path)):
        raise ValueError(f"{label} must be a string or Path, got {type(p).__name__!r}")
    raw = str(p).strip()
    if not raw:
        raise ValueError(f"{label} is empty")
    if _URL_RE.match(raw) or "://" in raw:
        raise ValueError(f"{label} must be a local filesystem path")
    return Path(raw).resolve(strict=False)


# ------------------------------------------------------------------ #
# Config dataclass (manual to avoid runtime dependency on 'dataclasses')
# ------------------------------------------------------------------ #


class Config:
    """Immutable configuration with strict field validation."""

    __slots__ = (
        "effect_mode",
        "mosaic_block_size_px",
        "mosaic_padding_px",
        "sticker_scale_f",
        "t_confirm",
        "t_keep",
        "uncertain_policy",
        "track_lost_ttl_frames",
        "recheck_interval_frames",
        "input_path",
        "output_path",
        "max_face_area_ratio",
        "min_face_area_px",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self.__slots__:
            raise AttributeError(f"Unknown configuration field: {name}")
        try:
            object.__getattribute__(self, name)
        except AttributeError:
            object.__setattr__(self, name, value)
            return
        raise AttributeError("Config instances are immutable")

    def __init__(
        self,
        *,
        effect_mode: str = EffectMode.MOSAIC.value,
        mosaic_block_size_px: int = 8,
        mosaic_padding_px: int = 0,
        sticker_scale_f: float = 1.0,
        t_confirm: float = 0.65,
        t_keep: float = 0.55,
        uncertain_policy: str = UncertainPolicy.PRIVACY_SAFE.value,
        track_lost_ttl_frames: int = 3,
        recheck_interval_frames: int = 2,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        max_face_area_ratio: float = 0.85,
        min_face_area_px: int = 16 * 16,
    ) -> None:
        _validate_effect_mode(effect_mode)
        _validate_int_positive(mosaic_block_size_px, "mosaic_block_size_px")
        _validate_int_ge0(mosaic_padding_px, "mosaic_padding_px")
        sticker_scale_f = _validate_float_range(
            sticker_scale_f, 0.1, 5.0, "sticker_scale_f"
        )
        t_confirm = _validate_float_range(t_confirm, 0.0, 1.0, "t_confirm")
        t_keep = _validate_float_range(t_keep, 0.0, 1.0, "t_keep")
        if t_keep >= t_confirm:
            raise ValueError("t_keep must be strictly less than t_confirm")
        _validate_uncertain_policy(uncertain_policy)
        _validate_int_positive(track_lost_ttl_frames, "track_lost_ttl_frames")
        _validate_int_positive(recheck_interval_frames, "recheck_interval_frames")

        self.input_path: Path | None = (
            _validate_path(input_path, "input_path") if input_path is not None else None
        )
        self.output_path: Path | None = (
            _validate_path(output_path, "output_path") if output_path is not None else None
        )
        if self.input_path is not None and self.output_path is not None:
            if self.input_path == self.output_path:
                raise ValueError("input_path and output_path must be distinct")

        max_face_area_ratio = _validate_float_range(
            max_face_area_ratio, 0.1, 1.0, "max_face_area_ratio"
        )
        _validate_int_positive(min_face_area_px, "min_face_area_px")

        # Freeze values
        self.effect_mode = EffectMode(effect_mode).value
        self.mosaic_block_size_px = mosaic_block_size_px
        self.mosaic_padding_px = mosaic_padding_px
        self.sticker_scale_f = sticker_scale_f
        self.t_confirm = t_confirm
        self.t_keep = t_keep
        self.uncertain_policy = UncertainPolicy(uncertain_policy).value
        self.track_lost_ttl_frames = track_lost_ttl_frames
        self.recheck_interval_frames = recheck_interval_frames
        self.max_face_area_ratio = max_face_area_ratio
        self.min_face_area_px = min_face_area_px

    # ---- serialization / reconstruction helpers for testing ---------- #

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for slot in self.__slots__:
            val = getattr(self, slot)
            if isinstance(val, Path):
                d[slot] = str(val)
            elif isinstance(val, Enum):
                d[slot] = val.value
            else:
                d[slot] = val
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        if not isinstance(data, dict):
            raise ValueError("Config payload must be an object")
        # Reject unknown keys
        allowed = set(cls.__slots__)
        unknown = set(data.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

        # Reject any string values that look like URLs or tokens
        for key, val in data.items():
            if isinstance(val, str):
                _URL_RE.match(val) is None or _raise_val_err(key)
                _TOKEN_RE.search(val) is None or _raise_val_err(key)

        # Reject NaN / Inf in float values
        for key, val in data.items():
            if isinstance(val, float):
                if not np.isfinite(val):
                    raise ValueError(f"{key} must be finite, got {val}")

        return cls(**data)

    @classmethod
    def default(cls) -> Config:
        return cls()


# ------------------------------------------------------------------ #
# Validators (private helpers)
# ------------------------------------------------------------------ #


def _validate_effect_mode(v: str) -> None:
    valid = [e.value for e in EffectMode]
    if not isinstance(v, str) or v not in valid:
        raise ValueError(f"effect_mode must be one of {valid}")


def _validate_float_range(v: float, lo: float, hi: float, key: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{key} must be numeric, got {type(v).__name__!r}")
    try:
        value = float(v)
    except OverflowError:
        _raise_val_err(key)
    if not np.isfinite(value):
        _raise_val_err(key)
    if not (lo <= value <= hi):
        raise ValueError(f"{key} must be in [{lo}, {hi}], got {value}")
    return value


def _validate_int_positive(v: int, key: str) -> None:
    _validate_int_ge1(v, key)


def _validate_int_ge0(v: int, key: str) -> None:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{key} must be integer, got {type(v).__name__!r}")
    if v < 0:
        raise ValueError(f"{key} must be >= 0, got {v}")


def _validate_int_ge1(v: int, key: str) -> None:
    _validate_int_ge0(v, key)  # type: ignore[arg-type]  (called above)
    if v < 1:
        raise ValueError(f"{key} must be >= 1, got {v}")


def _validate_uncertain_policy(v: str) -> None:
    valid = [e.value for e in UncertainPolicy]
    if not isinstance(v, str) or v not in valid:
        raise ValueError(f"uncertain_policy must be one of {valid}")


def _raise_val_err(key: str) -> None:
    raise ValueError(f"{key} contains a disallowed value")
