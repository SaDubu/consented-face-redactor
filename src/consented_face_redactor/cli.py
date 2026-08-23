"""Command-line validation entry points for consented-face-redactor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="consented-face-redactor",
        description="Consented face redaction CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # inspect-config
    cfg = sub.add_parser("inspect-config", help="Validate and print the configuration schema")
    cfg.add_argument("--file", "-f", default=None, help="Path to a JSON config file")
    cfg.add_argument("--verbose", action="store_true", help="Print field descriptions")

    # validate-models
    val = sub.add_parser("validate-models", help="Validate model manifests without loading models")
    val.add_argument("--manifest-dir", required=True, help="Directory containing model manifest JSON files")

    return parser


def _cmd_inspect_config(args: argparse.Namespace) -> int:
    print("=" * 60)
    print("Config schema inspection")
    print("=" * 60)

    from consented_face_redactor.config import Config, EffectMode, UncertainPolicy

    if args.file is None:
        cfg = Config.default()
    else:
        config_path = Path(args.file)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("configuration must be a JSON object")
            cfg = Config.from_dict(raw)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Error: invalid config '{config_path.name}': {exc}", file=sys.stderr)
            return 2

    d = cfg.to_dict()

    for slot in Config.__slots__:
        val = d[slot]
        if slot.endswith("_path") and val is not None:
            val = "<configured>"
        print(f"  {slot}: {val!r}")

    if args.verbose:
        print("\n--- Enum values ---")
        print(f"  EffectMode: {[e.value for e in EffectMode]}")
        print(f"  UncertainPolicy: {[e.value for e in UncertainPolicy]}")

    print()
    print("Schema is valid. No files created.")
    return 0


def _cmd_validate_models(args: argparse.Namespace) -> int:
    """Validate every JSON manifest and its colocated model binary."""
    from consented_face_redactor.model_manifest import (
        ManifestValidationError,
        load_manifest_from_json,
        verify_model_file,
    )

    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_dir():
        print("Error: manifest directory is unavailable", file=sys.stderr)
        return 2

    manifest_paths = sorted(manifest_dir.glob("*.json"))
    if not manifest_paths:
        print("Error: no JSON manifests found", file=sys.stderr)
        return 2

    verified = 0
    seen_model_ids: set[str] = set()
    seen_filenames: set[str] = set()
    try:
        for manifest_path in manifest_paths:
            entries = load_manifest_from_json(manifest_path)
            if not entries:
                raise ManifestValidationError("Manifest contains no model entries")
            for entry in entries:
                if entry["model_id"] in seen_model_ids:
                    raise ManifestValidationError("Duplicate model_id across manifests")
                normalized_filename = entry["filename"].casefold()
                if normalized_filename in seen_filenames:
                    raise ManifestValidationError("Duplicate filename across manifests")
                seen_model_ids.add(entry["model_id"])
                seen_filenames.add(normalized_filename)
                verify_model_file(entry, manifest_dir / entry["filename"])
                verified += 1
    except (OSError, ManifestValidationError) as exc:
        print(f"Error: model validation failed: {exc}", file=sys.stderr)
        return 2

    if verified == 0:
        print("Error: manifests contain no model entries", file=sys.stderr)
        return 2
    print(f"Validated {verified} model file(s) from {len(manifest_paths)} manifest(s).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    handlers = {
        "inspect-config": _cmd_inspect_config,
        "validate-models": _cmd_validate_models,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Error: unknown command '{args.command}'", file=sys.stderr)
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
