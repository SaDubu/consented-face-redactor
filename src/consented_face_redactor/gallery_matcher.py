"""Gallery matcher for face identity verification."""

from __future__ import annotations

from consented_face_redactor.gallery_approval import GalleryApproval


class GalleryMatcher:
    """Match extracted embeddings against a gallery of known profiles.

    This is a thin adapter that provides embed/match semantics required by the
    pipeline without embedding any real model code (that comes in later phases).
    """

    def __init__(self, gallery_db: dict[str, list[float]] | None = None) -> None:
        """Initialise with an optional gallery database.

        Parameters
        ----------
        gallery_db : dict[str, list[float]] | None
            Mapping of profile_id → normalised embedding vector (as a Python
            list).  If ``None`` the matcher simply returns empty results.
        """
        self._gallery_db: dict[str, list[float]] = gallery_db or {}

    def embed(self, frame) -> list[float] | None:
        """Return an embedding for *frame*.

        Returns ``None`` when no real model is wired in (stub behaviour).
        """
        return None  # stub — no embeddings produced without a model backend

    def match(self, embedding: list[float] | None) -> GalleryApproval:
        """Match *embedding* against the gallery.

        Returns
        -------
        GalleryApproval
            One explicit approved or denied decision. The pipeline never infers
            authority from the similarity field itself.
        """
        if embedding is None:
            return GalleryApproval.denied("embedding_unavailable", gallery_revision="stub-v1")

        if not self._gallery_db:
            return GalleryApproval.denied("empty_gallery", gallery_revision="stub-v1")

        # Stub matching — in a real implementation this would compute cosine
        # similarity against every gallery profile and return the top N.
        return GalleryApproval.denied(
            "similarity_not_evaluated",
            similarity=0.0,
            gallery_revision="stub-v1",
        )
