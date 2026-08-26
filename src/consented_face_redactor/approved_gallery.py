"""Production adapter joining embeddings, local similarity, and approval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from consented_face_redactor.adapters.detection_iface import FaceDetection
from consented_face_redactor.approval_store import ApprovalStore
from consented_face_redactor.gallery import LocalGallery
from consented_face_redactor.gallery_approval import GalleryApproval


@dataclass(frozen=True, slots=True)
class MatchObservation:
    """Non-authorizing match diagnostics for local evaluation reports."""

    profile_id: str | None
    best_similarity: float | None
    best_reference_index: int | None
    centroid_similarity: float | None
    reason_code: str


class ApprovedLocalGalleryAdapter:
    """Return approval only when a current explicit record backs a match.

    The adapter deliberately makes the vector match and the authorization two
    distinct steps. It never infers approval from a similarity score.
    """

    def __init__(self, *, embedder: Any, gallery: LocalGallery, approvals: ApprovalStore) -> None:
        self._embedder = embedder
        self._gallery = gallery
        self._approvals = approvals
        self._last_observation = MatchObservation(None, None, None, None, "not_checked")

    @property
    def last_observation(self) -> MatchObservation:
        """Return latest diagnostic data without granting any authority."""
        return self._last_observation

    def evaluate(self, frame: np.ndarray, detection: FaceDetection) -> GalleryApproval:
        """Embed one detected face, match it, then require explicit approval."""
        vector, _revision = self._embedder.embed(frame, detection)
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            self._last_observation = MatchObservation(None, None, None, None, "invalid_embedding")
            return GalleryApproval.denied(
                "invalid_embedding", gallery_revision=self._approvals.gallery_revision
            )
        matches = self._gallery.match(vector, top_k=1)
        if not matches or not matches[0].is_match:
            similarity = matches[0].confidence if matches else None
            self._last_observation = MatchObservation(
                matches[0].profile_id if matches else None, similarity, None, None,
                "similarity_insufficient",
            )
            return GalleryApproval.denied(
                "similarity_insufficient", similarity=similarity,
                gallery_revision=self._approvals.gallery_revision,
            )
        match = matches[0]
        details = self._gallery.profile_similarity_details(vector, match.profile_id)
        record = self._approvals.get(match.profile_id)
        if record is None:
            self._last_observation = MatchObservation(
                match.profile_id, details.best_similarity, details.best_reference_index,
                details.centroid_similarity, "profile_not_explicitly_approved",
            )
            return GalleryApproval.denied(
                "profile_not_explicitly_approved",
                profile_id=match.profile_id,
                similarity=match.confidence,
                gallery_revision=self._approvals.gallery_revision,
            )
        if not record.approved:
            self._last_observation = MatchObservation(
                match.profile_id, details.best_similarity, details.best_reference_index,
                details.centroid_similarity, "profile_not_approved",
            )
            return GalleryApproval.denied(
                "profile_not_approved",
                profile_id=match.profile_id,
                similarity=match.confidence,
                gallery_revision=self._approvals.gallery_revision,
            )
        if not record.is_current():
            self._last_observation = MatchObservation(
                match.profile_id, details.best_similarity, details.best_reference_index,
                details.centroid_similarity, "approval_expired",
            )
            return GalleryApproval.denied(
                "approval_expired",
                profile_id=match.profile_id,
                similarity=match.confidence,
                gallery_revision=self._approvals.gallery_revision,
            )
        self._last_observation = MatchObservation(
            match.profile_id, details.best_similarity, details.best_reference_index,
            details.centroid_similarity, record.reason_code,
        )
        return GalleryApproval(
            approved=True,
            profile_id=match.profile_id,
            similarity=match.confidence,
            reason_code=record.reason_code,
            gallery_revision=self._approvals.gallery_revision,
        )
