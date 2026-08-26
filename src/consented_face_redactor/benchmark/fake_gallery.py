"""Controlled local gallery adapter for synthetic benchmark scenarios."""

from __future__ import annotations

from typing import Any

from consented_face_redactor.gallery_approval import GalleryApproval


class FakeGallery:
    """Return one supplied explicit approval, denial, or deterministic failure."""

    def __init__(
        self,
        approval: GalleryApproval | list[tuple[str, float]] | None = None,
        *,
        embed_error: Exception | None = None,
        match_error: Exception | None = None,
        malformed_result: bool = False,
    ) -> None:
        if isinstance(approval, list):
            self._approval = (
                GalleryApproval(True, approval[0][0], approval[0][1], "explicit_approval", "synthetic-v1")
                if approval else GalleryApproval.denied("empty_gallery", gallery_revision="synthetic-v1")
            )
        else:
            self._approval = approval or GalleryApproval.denied("empty_gallery", gallery_revision="synthetic-v1")
        self._embed_error = embed_error
        self._match_error = match_error
        self._malformed_result = malformed_result

    def embed(self, _frame: Any = None) -> bytes:
        if self._embed_error is not None:
            raise self._embed_error
        return b"synthetic_embedding"

    def match(self, _embedding: Any) -> GalleryApproval | object:
        if self._match_error is not None:
            raise self._match_error
        if self._malformed_result:
            return {"approved": True}
        return self._approval
