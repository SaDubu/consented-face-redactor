"""Minimal fake gallery matcher for benchmark runner (synthetic data only)."""

from __future__ import annotations


class FakeGallery:
    """Stub gallery that returns controlled match results."""

    def __init__(self, matches: list[tuple[str, float]] | None = None) -> None:
        self._matches = tuple(matches or [])

    def embed(self, _frame=None):  # noqa: ANN001
        return b"dummy_embedding"

    def match(self, vector):  # noqa: ANN001
        if not hasattr(vector, "__len__"):
            return []
        return list(self._matches)
