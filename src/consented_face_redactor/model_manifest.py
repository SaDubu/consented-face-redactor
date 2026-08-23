"""Model manifest schema and validation for consented-face-redactor.

The manifest defines which models are permitted and their provenance.
Every model binary must have a matching manifest entry with:
  - model_id (opaque identifier)
  - role (detector | embedder | tracker | renderer)
  - source (organization / paper reference)
  - filename (local filename, NOT the absolute path)
  - sha256 (expected checksum — fail-closed on mismatch)
  - license (SPDX identifier or explicit text)
  - input_shape (list of ints: [channels, height, width] or similar)
  - preprocessing_revision (integer version)
  - provider (one of: ONNXRuntime, OpenCV, etc.)

Checksum or license mismatch → immediate fail-closed.
No code to download models — manifest only lists pre-acquired assets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# ---- fixed schema constants ------------------------------------------ #

REQUIRED_KEYS = [
    "model_id",
    "role",
    "source",
    "filename",
    "sha256",
    "license",
    "input_shape",
    "preprocessing_revision",
    "provider",
]

VALID_ROLES = {"detector", "embedder", "tracker", "renderer"}


class ManifestValidationError(Exception):
    """Raised when a manifest entry fails validation."""

    def __init__(self, message: str, *, key: str | None = None) -> None:
        super().__init__(message)
        self.key = key


# ---- single-entry validation ----------------------------------------- #


def _assert_str(val: Any, key: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise ManifestValidationError(
            f"Manifest entry '{key}' must be a non-empty string",
            key=key,
        )
    return val.strip()


def _assert_bool(val: Any, key: str) -> bool:
    if not isinstance(val, bool):
        raise ManifestValidationError(
            f"Manifest entry '{key}' must be a boolean",
            key=key,
        )
    return val


def _assert_non_empty_str(val: Any, key: str) -> str:
    s = _assert_str(val, key)
    if not s:
        raise ManifestValidationError(
            f"Manifest entry '{key}' must be non-empty",
            key=key,
        )
    return s


def _assert_list_of_nonneg_ints(val: Any, key: str) -> list[int]:
    if not isinstance(val, (list, tuple)):
        raise ManifestValidationError(
            f"Manifest entry '{key}' must be a non-empty array of integers",
            key=key,
        )
    out: list[int] = []
    for i, item in enumerate(val):
        if not isinstance(item, int) or item < 0:
            raise ManifestValidationError(
                f"Manifest entry '{key}[{i}]' must be a non-negative integer, "
                f"got {type(item).__name__!r}",
                key=key,
            )
        out.append(item)
    if not out:
        raise ManifestValidationError(
            f"Manifest entry '{key}' must have at least one element",
            key=key,
        )
    return out


def _assert_pos_int(val: Any, key: str) -> int:
    if not isinstance(val, int):
        raise ManifestValidationError(
            f"Manifest entry '{key}' must be an integer",
            key=key,
        )
    if val < 1:
        raise ManifestValidationError(
            f"Manifest entry '{key}' must be >= 1, got {val}",
            key=key,
        )
    return val


# ---- public API ------------------------------------------------------ #


def validate_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a raw manifest JSON/dict in-place.

    Returns the validated (and potentially cleaned) entry dict on success;
    raises ManifestValidationError on any failure (fail-closed).
    """
    # All required keys present?
    for rk in REQUIRED_KEYS:
        if rk not in raw:
            raise ManifestValidationError(
                f"Manifest entry missing required key '{rk}'",
                key=rk,
            )

    extra = set(raw.keys()) - {"version"} - set(REQUIRED_KEYS)
    if extra:
        raise ManifestValidationError(
            f"Manifest entry has unknown keys: {sorted(extra)}",
            key="__unknown__",
        )

    model_id  = _assert_non_empty_str(raw["model_id"], "model_id")
    role      = raw["role"]
    source    = _assert_non_empty_str(raw["source"], "source")
    filename  = str(raw["filename"])
    sha256_raw= raw["sha256"]
    license_  = _assert_non_empty_str(raw["license"], "license")
    input_shape = raw["input_shape"]
    preproc     = raw["preprocessing_revision"]
    provider    = raw["provider"]

    # role must be one of the valid ones
    if not isinstance(role, str) or role.lower() not in VALID_ROLES:
        raise ManifestValidationError(
            f"Manifest entry 'role' must be one of {VALID_ROLES}, got {role!r}",
            key="role",
        )

    # sha256 must be a 64-char hex string and valid hex
    sha256 = _assert_non_empty_str(sha256_raw, "sha256")
    if len(sha256) != 64:
        raise ManifestValidationError(
            f"Manifest entry 'sha256' must be 64 hex characters, got length={len(sha256)}",
            key="sha256",
        )
    try:
        int(sha256, 16)
    except ValueError:
        raise ManifestValidationError(
            f"Manifest entry 'sha256' contains non-hex characters",
            key="sha256",
        )

    input_shape = _assert_list_of_nonneg_ints(input_shape, "input_shape")
    preproc  = _assert_pos_int(preproc, "preprocessing_revision")
    provider = _assert_non_empty_str(provider, "provider")

    return {
        "model_id": model_id,
        "role": role.lower(),
        "source": source,
        "filename": filename,
        "sha256": sha256.lower(),
        "license": license_,
        "input_shape": input_shape,
        "preprocessing_revision": preproc,
        "provider": provider,
    }


def verify_model_file(manifest_entry: dict[str, Any], file_path: Path) -> None:
    """Compute sha256 of a real model binary and compare with manifest.

    Raises ManifestValidationError on mismatch (fail-closed).  Never
    produces partial state — the caller should discard any loaded object
    if this raises.
    """
    if not file_path.exists():
        raise ManifestValidationError(
            f"Model file does not exist: {file_path}",
            key="sha256",
        )

    sha = hashlib.sha256()
    buf_size = 1 << 16  # 64 KB
    with file_path.open("rb") as fh:
        while True:
            chunk = fh.read(buf_size)
            if not chunk:
                break
            sha.update(chunk)

    actual_sha = sha.hexdigest()
    expected_sha = manifest_entry["sha256"].lower()

    if actual_sha != expected_sha:
        raise ManifestValidationError(
            f"Model checksum mismatch — expected {expected_sha}, "
            f"got {actual_sha}",
            key="sha256",
        )


def load_manifest_from_json(json_path: Path) -> list[dict[str, Any]]:
    """Load a manifest file (array of entries) and validate all.

    Raises ManifestValidationError if any entry or the envelope is invalid.
    Returns a list of validated entry dicts.
    """
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ManifestValidationError(
            "Manifest file must be a JSON array",
            key="__envelope__",
        )

    results: list[dict[str, Any]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ManifestValidationError(
                f"Manifest entry at index {i} is not an object",
                key=f"[{i}]",
            )
        results.append(validate_manifest(entry))

    return results
