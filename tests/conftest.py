"""Fixtures shared across all test modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the src package is on sys.path for 'import consented_face_redactor'
_THIS = Path(__file__).resolve().parents[0]  # tests/
_PROJECT = _THIS.parents[0]  # repo root
_SRC = _PROJECT / "src"

# Insert before any system paths (if not already present)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def config() -> "config":  # type: ignore[misc] (eval forward-reference)
    """Return a default Config instance for use across tests."""
    from consented_face_redactor.config import Config

    return Config.default()


@pytest.fixture
def pipeline(config):
    """Return a RedactionPipeline backed by config and fake adapters."""
    from consented_face_redactor.pipeline import RedactionPipeline

    return RedactionPipeline(config)


@pytest.fixture
def sample_frame() -> "np.ndarray":  # type: ignore[misc] (numpy type-hint)
    """Create a small synthetic RGB frame filled with zeros."""
    import numpy as np

    return np.zeros((48, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_gray() -> "np.ndarray":  # type: ignore[misc]
    """Create a small synthetic grayscale frame (H, W)."""
    import numpy as np

    return np.zeros((48, 64), dtype=np.uint8)
