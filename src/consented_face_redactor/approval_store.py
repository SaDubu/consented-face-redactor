"""Explicit, local-only approval records for gallery profiles.

Similarity can select a profile candidate, but only an approval record grants
the authority represented by :class:`GalleryApproval`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ApprovalStoreError(ValueError):
    """Raised when an approval payload is malformed or unavailable."""


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """A human-controlled permission record for one opaque profile ID."""

    approved: bool
    reason_code: str
    expires_at: str | None = None

    def is_current(self, now: datetime | None = None) -> bool:
        """Return whether the record is explicitly approved and unexpired."""
        if not self.approved:
            return False
        if self.expires_at is None:
            return True
        instant = now or datetime.now(UTC)
        try:
            expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        if expiry.tzinfo is None:
            return False
        return instant < expiry.astimezone(UTC)

    def to_dict(self) -> dict[str, object]:
        return {
            "approved": self.approved,
            "reason_code": self.reason_code,
            "expires_at": self.expires_at,
        }


class ApprovalStore:
    """Validated approval mapping, stored separately from biometric vectors."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        records: dict[str, ApprovalRecord] | None = None,
        *,
        gallery_revision: str = "local-v1",
    ) -> None:
        if not isinstance(gallery_revision, str) or not gallery_revision.strip():
            raise ApprovalStoreError("gallery_revision must be a non-empty string")
        self._gallery_revision = gallery_revision.strip()
        self._records = dict(records or {})
        if any(not isinstance(profile_id, str) or not profile_id for profile_id in self._records):
            raise ApprovalStoreError("approval profile IDs must be non-empty strings")
        if any(not isinstance(record, ApprovalRecord) for record in self._records.values()):
            raise ApprovalStoreError("approval records are invalid")

    @property
    def gallery_revision(self) -> str:
        return self._gallery_revision

    def get(self, profile_id: str) -> ApprovalRecord | None:
        """Return a detached approval record for *profile_id*, when present."""
        return self._records.get(profile_id)

    def set(self, profile_id: str, record: ApprovalRecord) -> None:
        """Set a record after validating its opaque ID and explicit fields."""
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ApprovalStoreError("profile_id must be a non-empty string")
        if not isinstance(record, ApprovalRecord):
            raise ApprovalStoreError("record must be an ApprovalRecord")
        self._records[profile_id] = record

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "gallery_revision": self._gallery_revision,
            "profiles": {
                profile_id: self._records[profile_id].to_dict()
                for profile_id in sorted(self._records)
            },
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "ApprovalStore":
        """Construct a store from an exact, fail-closed JSON payload."""
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version", "gallery_revision", "profiles"
        }:
            raise ApprovalStoreError("approval payload has missing or unknown fields")
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise ApprovalStoreError("unsupported approval schema_version")
        profiles = payload["profiles"]
        if not isinstance(profiles, dict):
            raise ApprovalStoreError("approval profiles must be an object")
        records: dict[str, ApprovalRecord] = {}
        for profile_id, raw_record in profiles.items():
            if not isinstance(profile_id, str) or not profile_id:
                raise ApprovalStoreError("approval profile IDs must be non-empty strings")
            if not isinstance(raw_record, dict) or set(raw_record) != {
                "approved", "reason_code", "expires_at"
            }:
                raise ApprovalStoreError("approval record has missing or unknown fields")
            approved = raw_record["approved"]
            reason_code = raw_record["reason_code"]
            expires_at = raw_record["expires_at"]
            if not isinstance(approved, bool):
                raise ApprovalStoreError("approval flag must be boolean")
            if not isinstance(reason_code, str) or not reason_code.strip():
                raise ApprovalStoreError("approval reason_code must be non-empty")
            if expires_at is not None:
                if not isinstance(expires_at, str):
                    raise ApprovalStoreError("approval expires_at must be a string or null")
                try:
                    parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ApprovalStoreError("approval expires_at is invalid") from exc
                if parsed.tzinfo is None:
                    raise ApprovalStoreError("approval expires_at must include a timezone")
            records[profile_id] = ApprovalRecord(approved, reason_code.strip(), expires_at)
        return cls(records, gallery_revision=payload["gallery_revision"])

    @classmethod
    def load(cls, path: str | Path) -> "ApprovalStore":
        """Read and validate a local approval file without partial recovery."""
        source = Path(path).expanduser().resolve(strict=False)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ApprovalStoreError(f"approval file is unreadable or invalid: {source.name}") from exc
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> None:
        """Atomically save records; no implicit directory creation occurs."""
        target = Path(path).expanduser().resolve(strict=False)
        if not target.parent.is_dir():
            raise ApprovalStoreError("approval directory is unavailable")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=target.parent,
                prefix=f".{target.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, target)
            temporary = None
        except OSError as exc:
            raise ApprovalStoreError(f"approval file could not be saved: {target.name}") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
