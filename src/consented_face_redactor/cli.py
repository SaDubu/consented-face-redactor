"""Command-line entry points for consented-face-redactor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


# ------------------------------------------------------------------ #
# Parser construction
# ------------------------------------------------------------------ #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="consented-face-redactor",
        description="Consented face redaction CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # --------------------------------------------------------------- #
    # inspect-config
    # --------------------------------------------------------------- #
    cfg_help = sub.add_parser("inspect-config", help="Validate and print the configuration schema")
    cfg_help.add_argument("--file", "-f", default=None, help="Path to a JSON config file")
    cfg_help.add_argument("--verbose", action="store_true", help="Print field descriptions")

    # --------------------------------------------------------------- #
    # validate-models
    # --------------------------------------------------------------- #
    val_help = sub.add_parser(
        "validate-models",
        help="Validate model manifests without loading models",
    )
    val_help.add_argument("--manifest-dir", required=True, help="Directory containing model manifest JSON files")

    # --------------------------------------------------------------- #
    # process-image
    # --------------------------------------------------------------- #
    img_sub = sub.add_parser(
        "process-image",
        help="Process a single image through the redaction pipeline",
    )
    img_sub.add_argument("--input", "-i", required=True, type=Path, help="Path to source image")
    img_sub.add_argument("--output", "-o", required=False, type=Path, help="Output path for redacted image")
    img_sub.add_argument(
        "--config", "-c",
        default=None,
        type=Path,
        help="Path to JSON config; defaults to <cwd>/config.json",
    )
    img_sub.add_argument(
        "--state-dir",
        default=None,
        type=Path,
        help="Directory to persist/load track state JSON (default: same dir as input)",
    )

    # --------------------------------------------------------------- #
    # process-video
    # --------------------------------------------------------------- #
    vid_sub = sub.add_parser(
        "process-video",
        help="Process video file through the redaction pipeline",
    )
    vid_sub.add_argument("--input", "-i", required=True, type=Path, help="Path to source video")
    vid_sub.add_argument("--output", "-o", required=False, type=Path, help="Output path for processed video")
    vid_sub.add_argument(
        "--config", "-c",
        default=None,
        type=Path,
        help="Path to JSON config; defaults to <cwd>/config.json",
    )
    vid_sub.add_argument(
        "--state-dir",
        default=None,
        type=Path,
        help="Directory to persist/load track state JSON (default: same dir as input)",
    )

    # --------------------------------------------------------------- #
    # gallery enroll
    # --------------------------------------------------------------- #
    gal_sub = sub.add_parser(
        "gallery-enroll",
        help="Enroll a face into the local gallery from an image snapshot",
    )
    gal_sub.add_argument("--input", "-i", required=True, type=Path, help="Snapshot image with face to enroll")
    gal_sub.add_argument("--gallery-db", required=True, type=Path, help="Path to gallery JSON database file")
    gal_sub.add_argument(
        "--config", "-c",
        default=None,
        type=Path,
        help="Path to JSON config for embedding parameters (optional)",
    )

    return parser


# ------------------------------------------------------------------ #
# Existing commands
# ------------------------------------------------------------------ #


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


# ------------------------------------------------------------------ #
# Helpers used by the new commands
# ------------------------------------------------------------------ #


def _load_config(config_path: Optional[Path]) -> "Config":
    """Load Config from a JSON file or fall back to defaults."""
    from consented_face_redactor.config import Config

    if config_path is None:
        return Config.default()

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"Error: cannot read config file '{config_path.name}': {exc}", file=sys.stderr)
        sys.exit(2)
    try:
        if not isinstance(raw, dict):
            raise ValueError("configuration must be a JSON object")
        return Config.from_dict(raw)
    except (ValueError, TypeError) as exc:
        print(f"Error: invalid config in '{config_path.name}': {exc}", file=sys.stderr)
        sys.exit(2)


def _load_track_state(state_dir: Optional[Path]) -> Optional[Any]:
    """Load previous track state from JSON if it exists."""
    if state_dir is None:
        return None
    state_file = state_dir / "track_state.json"
    if not state_file.exists():
        return None
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw


def _save_track_state(pipeline: Any, state_dir: Optional[Path]) -> None:
    """Persist current track state to JSON."""
    if state_dir is None or pipeline is None:
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    snapshot = pipeline.save_track_state()
    state_file = state_dir / "track_state.json"
    state_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ #
# process-image command
# ------------------------------------------------------------------ #


def _cmd_process_image(args: argparse.Namespace) -> int:
    """Process a single image through the redaction pipeline."""
    import numpy as np

    from consented_face_redactor.pipeline import RedactionPipeline, ProcessResult
    from consented_face_redactor.media.frame_source import OpenCvFrameSource

    # 1. Load config and state
    cfg = _load_config(args.config)
    state_dir = args.state_dir or args.input.parent
    prev_state = _load_track_state(state_dir)

    # 2. Instantiate pipeline
    pipe = RedactionPipeline(cfg)  # no detector attached → stub path
    if prev_state is not None:
        pipe.load_track_state(prev_state)

    # 3. Read input image via frame source abstraction
    src = OpenCvFrameSource(args.input)
    try:
        src.open()
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: cannot open image '{args.input.name}': {exc}", file=sys.stderr)
        return 2

    success, frame = src.read()
    if not success or frame is None:
        print(f"Error: failed to read frame from '{args.input.name}'", file=sys.stderr)
        return 2

    # 4. Process through pipeline
    result: Optional[ProcessResult] = None
    try:
        result = pipe.process_frame(
            frame=frame,
            frame_index=src.current_frame_index,
            timestamp=float(src.current_frame_index),
            state=None,  # already loaded above
        )
    except (TypeError, ValueError) as exc:
        print(f"Error: pipeline processing failed: {exc}", file=sys.stderr)
        return 2

    if result is None:
        print("Error: pipeline returned no result", file=sys.stderr)
        return 2

    # 5. Write output (default path = input name with _redacted suffix)
    out_path = args.output or (args.input.parent / f"{args.input.stem}_redacted{args.input.suffix}")
    try:
        import cv2 as _cv2
        _cv2.imwrite(str(out_path), result.result_frame)
    except Exception as exc:
        print(f"Error: failed to write output '{out_path.name}': {exc}", file=sys.stderr)
        return 2

    # 6. Persist track state
    _save_track_state(pipe, state_dir)

    # 7. Summary
    print(f"Processed '{args.input.name}' → '{out_path.name}'")
    print(f"  Track state: {result.track_state.value}")
    print(f"  Is redacted: {result.is_redacted}")
    print(f"  Review required: {result.review_required}")
    return 0


# ------------------------------------------------------------------ #
# process-video command
# ------------------------------------------------------------------ #


def _cmd_process_video(args: argparse.Namespace) -> int:
    """Process every frame of a video through the redaction pipeline."""
    import numpy as np

    from consented_face_redactor.pipeline import RedactionPipeline, ProcessResult
    from consented_face_redactor.media.frame_source import OpenCvFrameSource, FakeFrameReader

    # 1. Load config and state
    cfg = _load_config(args.config)
    state_dir = args.state_dir or args.input.parent
    prev_state = _load_track_state(state_dir)

    # 2. Instantiate pipeline
    pipe = RedactionPipeline(cfg)  # no detector attached → stub path
    if prev_state is not None:
        pipe.load_track_state(prev_state)

    # 3. Open video via frame source abstraction
    src: Any = OpenCvFrameSource(args.input)
    fake_src = FakeFrameReader(width=320, height=240, n_frames=60, fps=30.0, ch=3)
    try:
        src.open()
        if src.path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")):
            # Treat as single-frame video
            fake_src = FakeFrameReader(width=src.width or 320, height=src.height or 240, n_frames=1, fps=src.fps if src.fps > 0 else 30.0)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: cannot open input '{args.input.name}': {exc}", file=sys.stderr)
        return 2

    # 4. Prepare output video writer (if writing to file)
    out_path = args.output or (args.input.parent / f"{args.input.stem}_processed.mp4")
    out_writer: Any | None = None
    try:
        import cv2 as _cv2
        if src.height > 0 and src.width > 0:
            fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
            fps_out = max(1.0, src.fps) if src.fps > 0 else 30.0
            out_writer = _cv2.VideoWriter(str(out_path), fourcc, fps_out, (src.width, src.height))
    except Exception as exc:
        print(f"Error: cannot initialize video writer for '{out_path.name}': {exc}", file=sys.stderr)
        return 2

    # 5. Frame loop
    frame_count = 0
    total_frames = src.frame_count if hasattr(src, "frame_count") and src.frame_count >= 0 else -1
    while True:
        success, frame = src.read()
        if not success or frame is None:
            break

        result: ProcessResult | None = None
        try:
            result = pipe.process_frame(
                frame=frame,
                frame_index=src.current_frame_index,
                timestamp=float(src.current_frame_index) / (src.fps if src.fps > 0 else 30.0),
                state=None,
            )
        except (TypeError, ValueError) as exc:
            print(f"Error: pipeline processing failed at frame {frame_count}: {exc}", file=sys.stderr)
            return 2

        # Keep the redacted frame if available; otherwise use original
        processed_frame = result.result_frame if result and hasattr(result, "result_frame") else frame
        if out_writer is not None:
            out_writer.write(processed_frame)

        frame_count += 1
        if total_frames > 0 and frame_count >= total_frames:
            break  # safety exit for known-length sources

    # 6. Cleanup
    if out_writer is not None:
        out_writer.release()
    src.close()

    # 7. Persist track state
    _save_track_state(pipe, state_dir)

    # 8. Summary
    print(f"Processed {frame_count} frame(s) from '{args.input.name}' → '{out_path.name}'")
    return 0


# ------------------------------------------------------------------ #
# gallery enroll command
# ------------------------------------------------------------------ #


def _cmd_gallery_enroll(args: argparse.Namespace) -> int:
    """Enroll a face snapshot into the local gallery database."""
    import numpy as np

    from consented_face_redactor.media.frame_source import OpenCvFrameSource
    from consented_face_redactor.gallery import LocalGallery, EnrollmentValidationError

    # 1. Read the source image to confirm it exists and is readable
    if not args.input.is_file():
        print(f"Error: input snapshot '{args.input.name}' does not exist", file=sys.stderr)
        return 2

    src = OpenCvFrameSource(args.input)
    try:
        src.open()
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: cannot read input image '{args.input.name}': {exc}", file=sys.stderr)
        return 2

    success, frame = src.read()
    if not success or frame is None:
        print(f"Error: failed to decode image '{args.input.name}'", file=sys.stderr)
        return 2

    # Convert BGR → RGB since gallery matcher expects RGB feature vectors.
    try:
        import cv2 as _cv2
        rgb_frame = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB).copy()
    except Exception as exc:
        print(f"Warning: cannot convert to BGR (using raw frame): {exc}", file=sys.stderr)
        rgb_frame = frame

    # 2. Enroll via LocalGallery
    gallery_db_path = Path(args.gallery_db)
    try:
        if gallery_db_path.exists():
            gallery = LocalGallery.from_json_file(gallery_db_path)
        else:
            gallery = LocalGallery()
    except Exception as exc:
        print(f"Error: cannot open gallery '{gallery_db_path.name}': {exc}", file=sys.stderr)
        return 2

    try:
        profile_id, vector = gallery.enroll_face(rgb_frame)
    except EnrollmentValidationError as exc:
        print(f"Enrollment rejected — reason: {exc.reason} — detail: {exc.detail}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: enrollment failed: {exc}", file=sys.stderr)
        return 2

    # 3. Persist the updated gallery
    try:
        gallery.save_to_json_file(gallery_db_path)
    except OSError as exc:
        print(f"Error: cannot write gallery '{gallery_db_path.name}': {exc}", file=sys.stderr)
        return 2

    src.close()

    # 4. Summary
    print(f"Enrolled face into profile '{profile_id}' ({len(vector)}-dim embedding)")
    print(f"Gallery database written: {gallery_db_path}")
    return 0


# ------------------------------------------------------------------ #
# CLI dispatcher (main entry point)
# ------------------------------------------------------------------ #


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    handlers = {
        "inspect-config": _cmd_inspect_config,
        "validate-models": _cmd_validate_models,
        "process-image": _cmd_process_image,
        "process-video": _cmd_process_video,
        "gallery-enroll": _cmd_gallery_enroll,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Error: unknown command '{args.command}'", file=sys.stderr)
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
