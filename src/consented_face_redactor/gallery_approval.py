"""Explicit identity-approval contract consumed by the redaction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class GalleryApproval:
    """A gallery decision; only ``approved=True`` authorizes redaction."""

    approved: bool
    profile_id: str | None
    similarity: float | None
    reason_code: str
    gallery_revision: str | None

    @classmethod
    def denied(
        cls,
        reason_code: str,
        *,
        profile_id: str | None = None,
        similarity: float | None = None,
        gallery_revision: str | None = None,
    ) -> "GalleryApproval":
        return cls(False, profile_id, similarity, reason_code, gallery_revision)


class GalleryApprovalProtocol(Protocol):
    """Minimal gallery dependency required by :class:`RedactionPipeline`."""

    def embed(self, frame: Any) -> Any | None:
        """Produce a local embedding, or ``None`` when no candidate is available."""

    def match(self, embedding: Any) -> GalleryApproval:
        """Return one explicit approval or denial decision for an embedding."""


__all__ = ["GalleryApproval", "GalleryApprovalProtocol"]
