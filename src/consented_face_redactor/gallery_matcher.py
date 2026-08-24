"""Gallery matcher for face identity verification."""

from __future__ import annotations


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

    def match(self, embedding: list[float] | None) -> list[tuple[str, float]] | None:
        """Match *embedding* against the gallery.

        Returns
        -------
        list[tuple[str, float]] | None
            A sequence of ``(profile_id, cosine_similarity)`` tuples sorted by
            similarity descending when an embedding is provided and matches exist;
            ``None`` otherwise.
        """
        if embedding is None:
            return None

        if not self._gallery_db:
            return []

        # Stub matching — in a real implementation this would compute cosine
        # similarity against every gallery profile and return the top N.
        results: list[tuple[str, float]] = []
        for profile_id, vec in self._gallery_db.items():
            results.append((profile_id, 0.0))  # stub similarity score

        results.sort(key=lambda x: x[1], reverse=True)
        return results
