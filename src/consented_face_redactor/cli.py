"""Command-line interface skeleton for consented-face-redactor.

Current commands (stub):
    validate-models
    inspect-config
    process-image   (placeholder)
    process-video   (placeholder)
    enroll          (placeholder)

Only 'inspect-config' is fully implemented in this phase — it reads and prints the config dict.
"""

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
    cfg.add_argument("--file", "-f", default=None, help="Path to config file (JSON or Python script)")
    cfg.add_argument("--verbose", action="store_true", help="Print field descriptions")

    # validate-models
    val = sub.add_parser("validate-models", help="Validate model manifests without loading models")
    val.add_argument("--manifest-dir", required=True, help="Directory containing model manifest JSON files")

    return parser


def _cmd_inspect_config(args: argparse.Namespace) -> int:
    print("=" * 60)
    print("Config schema inspection (Phase 1 — skeleton)")
    print("=" * 60)

    # Show all available fields and defaults
    from consented_face_redactor.config import Config, EffectMode, UncertainPolicy

    cfg = Config.default()
    d = cfg.to_dict()

    for slot in Config.__slots__:
        val = d[slot]
        print(f"  {slot}: {val!r}")

    if args.verbose:
        print("\n--- Enum values ---")
        print(f"  EffectMode: {[e.value for e in EffectMode]}")
        print(f"  UncertainPolicy: {{[e.value for e in UncertainPolicy]}}")

    print()
    print("Schema is valid. No files created.")
    return 0


def _cmd_validate_models(args: argparse.Namespace) -> int:
    """Stub — real implementation in Phase 2."""
    print("[WARN] validate-models: stub (Phase 1 skeleton)")
    print(f"  manifest dir: {args.manifest_dir}")
    print("Expected format: JSON with keys [model_id, role, source, filename, sha256, license]")
    return 0


def main(argv=None) -> int:
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
