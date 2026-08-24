"""Minimal fake detector for benchmark runner (synthetic data only)."""

from __future__ import annotations


class FakeDetector:
    """Stub detector that returns supplied bboxes/confidences."""

    def __init__(self, bboxes: list[tuple[float, float, float, float]] | None = None, confs: list[float] | None = None) -> None:
        self._bboxes = bboxes or []
        self._confs = confs or []

    def detect(self, _frame):  # noqa: ANN001
        n = min(len(self._bboxes), len(self._confs))
        return [type("FD", (), {"bbox": self._bboxes[i], "confidence": self._confs[i]})() for i in range(n)]
