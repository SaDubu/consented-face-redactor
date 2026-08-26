"""Implementation-neutral point-tracker protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import PointTrackResult


@runtime_checkable
class PointTracker(Protocol):
    """Track query pixels only; this protocol carries no identity authority."""

    @property
    def model_id(self) -> str: ...

    def reset(self) -> None: ...

    def initialize(
        self,
        frame: np.ndarray,
        *,
        frame_index: int,
        query_points: np.ndarray,
    ) -> PointTrackResult: ...

    def update(
        self,
        frame: np.ndarray,
        *,
        frame_index: int,
    ) -> PointTrackResult: ...
