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
    cfg_help.add_argument("--strict-config", action="store_true", help="Reject unknown config keys")

    # --------------------------------------------------------------- #
    # validate-models
    # --------------------------------------------------------------- #
    val_help = sub.add_parser(
        "validate-models",
        help="Validate model manifests without loading models",
    )
    val_help.add_argument("--manifest-dir", required=True, help="Directory containing model manifest JSON files")
    val_help.add_argument("--model-dir", default=None, help="Directory containing model binaries; defaults to --manifest-dir")

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
    img_sub.add_argument("--strict-config", action="store_true", help="Reject unknown config keys")
    img_sub.add_argument(
        "--state-dir",
        default=None,
        type=Path,
        help="Directory to persist/load track state JSON (default: same dir as input)",
    )
    _add_runtime_arguments(img_sub)

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
    vid_sub.add_argument("--strict-config", action="store_true", help="Reject unknown config keys")
    vid_sub.add_argument(
        "--state-dir",
        default=None,
        type=Path,
        help="Directory to persist/load track state JSON (default: same dir as input)",
    )
    _add_runtime_arguments(vid_sub)
    vid_sub.add_argument(
        "--tracker", choices=("none", "tapnextpp"), default="none",
        help="Optional temporal point tracker; never grants identity approval",
    )
    vid_sub.add_argument("--tracker-checkpoint", type=Path, default=None, help="Verified TAPNext++ checkpoint; defaults to tracker manifest filename in --model-dir")
    vid_sub.add_argument("--tracker-source-dir", type=Path, default=None, help="Local official google-deepmind/tapnet source checkout")
    vid_sub.add_argument("--tracking-mode", choices=("bidirectional",), default="bidirectional", help="Stored-video reconciliation mode")
    vid_sub.add_argument("--tracker-device", choices=("cuda", "cpu"), default="cuda")
    vid_sub.add_argument("--tracker-max-gap-frames", type=int, default=90, help="Maximum same-profile anchor gap eligible for consensus")
    vid_sub.add_argument("--tracker-minimum-path-iou", type=float, default=0.30, help="Forward/backward bbox agreement gate")
    vid_sub.add_argument("--tracker-minimum-visible-ratio", type=float, default=0.60, help="Minimum visible tracked-point ratio")
    vid_sub.add_argument("--preserve-audio", action="store_true", help="Remux original audio with FFmpeg after successful video processing")
    vid_sub.add_argument("--ffmpeg-path", type=Path, default=None, help="FFmpeg executable required with --preserve-audio")

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
    gal_sub.add_argument("--strict-config", action="store_true", help="Reject unknown config keys")
    gal_sub.add_argument("--model-dir", required=True, type=Path, help="Directory containing verified model binaries")
    gal_sub.add_argument("--manifest-dir", required=True, type=Path, help="Directory containing model manifest JSON files")
    gal_sub.add_argument("--detector-score-threshold", type=float, default=0.5, help="YuNet detection score threshold; does not grant identity approval")
    gal_sub.add_argument("--approval-db", required=True, type=Path, help="Local explicit-approval JSON file")
    gal_sub.add_argument("--approve", action="store_true", help="Record explicit local approval for this enrollment")
    gal_sub.add_argument("--approval-reason", default=None, help="Non-empty reason code required with --approve")

    video_gal_sub = sub.add_parser(
        "gallery-enroll-video",
        help="Enroll diverse face references from a target-only local video",
    )
    video_gal_sub.add_argument("--input", "-i", required=True, type=Path, help="Target-only registration video")
    video_gal_sub.add_argument("--gallery-db", required=True, type=Path, help="Path to local gallery JSON database")
    video_gal_sub.add_argument("--approval-db", required=True, type=Path, help="Path to local explicit-approval JSON")
    video_gal_sub.add_argument("--model-dir", required=True, type=Path, help="Directory containing verified model binaries")
    video_gal_sub.add_argument("--manifest-dir", required=True, type=Path, help="Directory containing model manifest JSON files")
    video_gal_sub.add_argument("--detector-score-threshold", type=float, default=0.5, help="YuNet detection score threshold; does not grant identity approval")
    video_gal_sub.add_argument("--sample-every-n-frames", type=int, default=6, help="Process every Nth frame")
    video_gal_sub.add_argument("--max-references", type=int, default=64, help="Maximum diverse vectors in one profile")
    video_gal_sub.add_argument("--duplicate-similarity", type=float, default=0.995, help="Deduplicate near-identical enrollment vectors")
    video_gal_sub.add_argument("--minimum-cluster-similarity", type=float, default=0.45, help="Connect adjacent target views and quarantine smaller disconnected embedding islands")
    video_gal_sub.add_argument("--profile-id", default=None, help="Existing opaque profile ID to extend")
    video_gal_sub.add_argument("--dry-run", action="store_true", help="Scan and report without changing gallery or approvals")
    video_gal_sub.add_argument("--report-out", type=Path, default=None, help="Optional local enrollment report JSON")
    video_gal_sub.add_argument("--approve", action="store_true", help="Record explicit local approval after successful enrollment")
    video_gal_sub.add_argument("--approval-reason", default=None, help="Non-empty reason code required with --approve")

    return parser


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the all-or-nothing real-model runtime options to a process command."""
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory containing verified model binaries")
    parser.add_argument("--manifest-dir", type=Path, default=None, help="Directory containing model manifest JSON files")
    parser.add_argument("--gallery-db", type=Path, default=None, help="Local embedding gallery JSON file")
    parser.add_argument("--approval-db", type=Path, default=None, help="Local explicit-approval JSON file")
    parser.add_argument("--detector-score-threshold", type=float, default=0.5, help="YuNet detection score threshold; does not grant identity approval")
    parser.add_argument("--evidence-out", type=Path, default=None, help="Optional local JSON evidence destination")
    parser.add_argument("--evaluation-evidence", action="store_true", help="Include face bboxes in explicit local evaluation evidence")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate faces but do not write an output image/video")


# ------------------------------------------------------------------ #
# Existing commands
# ------------------------------------------------------------------ #


def _cmd_inspect_config(args: argparse.Namespace) -> int:
    print("=" * 60)
    print("Config schema inspection")
    print("=" * 60)

    from consented_face_redactor.config import Config

    if args.file is None:
        cfg = Config.default()
    else:
        config_path = Path(args.file)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("configuration must be a JSON object")
            cfg = Config.from_dict(raw, strict=args.strict_config)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Error: invalid config '{config_path.name}': {exc}", file=sys.stderr)
            return 2

    d = cfg.to_dict()

    for key, value in d.items():
        print(f"  {key}: {value!r}")

    if args.verbose:
        print("\n--- Identity authority ---")
        print("  CONFIRMED redaction requires an explicit gallery match.")

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
    model_dir = Path(args.model_dir) if args.model_dir is not None else manifest_dir
    if not model_dir.is_dir():
        print("Error: model directory is unavailable", file=sys.stderr)
        return 2
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
                verify_model_file(entry, model_dir / entry["filename"])
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


def _load_config(config_path: Optional[Path], *, strict: bool = False) -> "Config":
    """Load Config from a JSON file or fall back to defaults."""
    from consented_face_redactor.config import Config

    if config_path is None:
        return Config.default()

    try:
        if not config_path.exists():
            return Config.default()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"Error: cannot read config file '{config_path.name}': {exc}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in '{config_path.name}': {exc}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(raw, dict):
        print(f"Error: invalid config in '{config_path.name}': configuration must be a JSON object", file=sys.stderr)
        sys.exit(2)
    try:
        return Config.from_dict(raw, strict=strict)
    except (ValueError, TypeError) as exc:
        print(f"Error: invalid config in '{config_path.name}': {exc}", file=sys.stderr)
        sys.exit(2)


def _load_verified_model_entries(model_dir: Path, manifest_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return exactly one verified OpenCV detector and embedder manifest entry."""
    from consented_face_redactor.model_manifest import (
        ManifestValidationError,
        load_manifest_from_json,
        verify_model_file,
    )
    if not model_dir.is_dir() or not manifest_dir.is_dir():
        raise ValueError("model and manifest directories must exist")
    manifest_paths = sorted(manifest_dir.glob("*.json"))
    if not manifest_paths:
        raise ValueError("manifest directory contains no JSON files")
    entries: list[dict[str, Any]] = []
    try:
        for manifest_path in manifest_paths:
            entries.extend(load_manifest_from_json(manifest_path))
        detectors = [entry for entry in entries if entry["role"] == "detector" and entry["provider"] == "OpenCV"]
        embedders = [entry for entry in entries if entry["role"] == "embedder" and entry["provider"] == "OpenCV"]
        if len(detectors) != 1 or len(embedders) != 1:
            raise ValueError("manifests must contain exactly one OpenCV detector and one OpenCV embedder")
        detector_entry, embedder_entry = detectors[0], embedders[0]
        verify_model_file(detector_entry, model_dir / detector_entry["filename"])
        verify_model_file(embedder_entry, model_dir / embedder_entry["filename"])
    except ManifestValidationError as exc:
        raise ValueError(f"model manifest validation failed: {exc}") from exc
    return detector_entry, embedder_entry


def _load_verified_tracker_entry(model_dir: Path, manifest_dir: Path) -> dict[str, Any]:
    """Return the one checksum-verified PyTorch TAPNext++ tracker entry."""
    from consented_face_redactor.model_manifest import (
        ManifestValidationError,
        load_manifest_from_json,
        verify_model_file,
    )

    try:
        entries = [
            entry
            for manifest_path in sorted(manifest_dir.glob("*.json"))
            for entry in load_manifest_from_json(manifest_path)
        ]
        trackers = [
            entry
            for entry in entries
            if entry["role"] == "tracker"
            and entry["provider"] == "PyTorch"
            and "tapnextpp" in entry["model_id"].lower()
        ]
        if len(trackers) != 1:
            raise ValueError("manifests must contain exactly one PyTorch TAPNext++ tracker")
        verify_model_file(trackers[0], model_dir / trackers[0]["filename"])
        return trackers[0]
    except ManifestValidationError as exc:
        raise ValueError(f"tracker manifest validation failed: {exc}") from exc


def _build_tracker_runtime(args: argparse.Namespace) -> tuple[Any, dict[str, object]]:
    """Build an explicitly selected verified tracker runtime."""
    if args.tracker != "tapnextpp":
        raise ValueError("unsupported tracker selection")
    if args.tracker_source_dir is None:
        raise ValueError("--tracker-source-dir is required with --tracker tapnextpp")
    if args.tracker_max_gap_frames < 2:
        raise ValueError("--tracker-max-gap-frames must be at least 2")
    if not 0.0 < args.tracker_minimum_path_iou <= 1.0:
        raise ValueError("--tracker-minimum-path-iou must be in (0, 1]")
    if not 0.0 < args.tracker_minimum_visible_ratio <= 1.0:
        raise ValueError("--tracker-minimum-visible-ratio must be in (0, 1]")

    from consented_face_redactor.tracking import TapNextPlusPlusAdapter

    model_dir, manifest_dir = Path(args.model_dir), Path(args.manifest_dir)
    entry = _load_verified_tracker_entry(model_dir, manifest_dir)
    checkpoint = Path(args.tracker_checkpoint) if args.tracker_checkpoint else model_dir / entry["filename"]
    if checkpoint.name != entry["filename"]:
        raise ValueError("--tracker-checkpoint filename must match the verified manifest")
    tracker = TapNextPlusPlusAdapter(
        checkpoint_path=checkpoint,
        vendor_source_dir=Path(args.tracker_source_dir),
        manifest_entry=entry,
        device=args.tracker_device,
        input_resolution=512,
    )
    return tracker, {
        "tracker_model_id": entry["model_id"],
        "tracker_provider": entry["provider"],
        "tracking_mode": args.tracking_mode,
        "tracker_device": args.tracker_device,
        "tracker_max_gap_frames": args.tracker_max_gap_frames,
        "tracker_minimum_path_iou": args.tracker_minimum_path_iou,
        "tracker_minimum_visible_ratio": args.tracker_minimum_visible_ratio,
    }


def _build_enrollment_runtime(args: argparse.Namespace) -> tuple[Any, Any]:
    """Create verified YuNet/SFace adapters shared by image and video enrollment."""
    from consented_face_redactor.adapters.opencv import OpenCvSFaceEmbedder, OpenCvYuNetDetector

    detector_entry, embedder_entry = _load_verified_model_entries(
        Path(args.model_dir), Path(args.manifest_dir)
    )
    return (
        OpenCvYuNetDetector(
            Path(args.model_dir) / detector_entry["filename"],
            model_id=detector_entry["model_id"],
            score_threshold=args.detector_score_threshold,
        ),
        OpenCvSFaceEmbedder(
            Path(args.model_dir) / embedder_entry["filename"],
            model_id=embedder_entry["model_id"],
            preprocessing_revision=embedder_entry["preprocessing_revision"],
        ),
    )


def _load_gallery_and_approval_stores(gallery_path: Path, approval_path: Path) -> tuple[Any, Any]:
    """Load existing stores or construct empty local stores after path checks."""
    from consented_face_redactor.approval_store import ApprovalStore
    from consented_face_redactor.gallery import LocalGallery

    if not gallery_path.parent.is_dir() or not approval_path.parent.is_dir():
        raise ValueError("gallery and approval directories must already exist")
    gallery = LocalGallery.from_json_file(gallery_path) if gallery_path.exists() else LocalGallery()
    approvals = ApprovalStore.load(approval_path) if approval_path.exists() else ApprovalStore()
    return gallery, approvals


def _write_enrollment_report(path: Path | None, report: Any) -> None:
    """Write local-only aggregate enrollment evidence when explicitly requested."""
    if path is None:
        return
    if not path.parent.is_dir():
        raise ValueError("enrollment report destination directory is unavailable")
    payload = {"schema_version": 1, "report": report.to_dict()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reject_output_overwrite(input_path: Path, output_path: Path) -> None:
    """Reject an output that resolves to its input before any write is opened."""
    if input_path.resolve(strict=False) == output_path.resolve(strict=False):
        raise ValueError("output path must differ from input path")


def _load_runtime_components(args: argparse.Namespace) -> tuple[Any | None, Any | None, dict[str, object]]:
    """Build verified detector/gallery dependencies, or the safe stub runtime.

    Runtime flags are deliberately all-or-nothing. A partial configuration is
    an operator error, not a reason to silently run without authorization.
    """
    runtime_names = ("model_dir", "manifest_dir", "gallery_db", "approval_db")
    values = [getattr(args, name, None) for name in runtime_names]
    if not any(values):
        return None, None, {"runtime_mode": "stub_no_redaction"}
    if not all(values):
        raise ValueError(
            "--model-dir, --manifest-dir, --gallery-db, and --approval-db must be supplied together"
        )

    from consented_face_redactor.adapters.opencv import OpenCvSFaceEmbedder, OpenCvYuNetDetector
    from consented_face_redactor.approval_store import ApprovalStore
    from consented_face_redactor.approved_gallery import ApprovedLocalGalleryAdapter
    from consented_face_redactor.gallery import LocalGallery

    model_dir = Path(args.model_dir)
    manifest_dir = Path(args.manifest_dir)
    gallery_path = Path(args.gallery_db)
    approval_path = Path(args.approval_db)
    detector_entry, embedder_entry = _load_verified_model_entries(model_dir, manifest_dir)

    if not gallery_path.is_file() or not approval_path.is_file():
        raise ValueError("gallery and approval files must already exist; enroll a consenting profile first")
    gallery = LocalGallery()
    gallery.load(gallery_path)
    approvals = ApprovalStore.load(approval_path)
    detector = OpenCvYuNetDetector(
        model_dir / detector_entry["filename"],
        model_id=detector_entry["model_id"],
        score_threshold=args.detector_score_threshold,
    )
    embedder = OpenCvSFaceEmbedder(
        model_dir / embedder_entry["filename"],
        model_id=embedder_entry["model_id"],
        preprocessing_revision=embedder_entry["preprocessing_revision"],
    )
    return detector, ApprovedLocalGalleryAdapter(
        embedder=embedder, gallery=gallery, approvals=approvals
    ), {
        "runtime_mode": "verified_opencv_models",
        "detector_model_id": detector_entry["model_id"],
        "embedder_model_id": embedder_entry["model_id"],
        "gallery_revision": approvals.gallery_revision,
    }


def _evidence_row(
    frame_index: int,
    result: Any,
    pipeline: Any,
    *,
    duration_ms: float | None = None,
    include_bboxes: bool = False,
) -> dict[str, object]:
    """Return privacy-minimal frame evidence without pixels or embeddings."""
    approvals = getattr(pipeline, "last_frame_approvals", ())
    row: dict[str, object] = {
        "frame_index": frame_index,
        "track_state": result.track_state.value,
        "is_redacted": bool(result.is_redacted),
        "review_required": bool(result.review_required),
        "approval_reason_codes": [item.reason_code for item in approvals],
        "approved_face_count": sum(item.approved is True for item in approvals),
    }
    if duration_ms is not None:
        row["duration_ms"] = duration_ms
    if include_bboxes:
        decisions = getattr(pipeline, "last_frame_decisions", ())
        row["face_decisions"] = [
            {"bbox": list(decision.bbox), "reason_code": decision.approval.reason_code,
             "approved": decision.approval.approved}
            for decision in decisions
        ]
    return row


def _build_processing_summary(
    rows: list[dict[str, object]],
    *,
    frame_shape: tuple[int, int, int] | None,
    elapsed_seconds: float,
    source_fps: float,
) -> dict[str, object]:
    """Build measurement metadata; no performance value is a pass/fail gate."""
    import os
    import platform
    import sys
    import numpy as np

    durations = [float(row["duration_ms"]) for row in rows if "duration_ms" in row]
    reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in row["approval_reason_codes"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "frame_count": len(rows),
        "frame_shape": list(frame_shape) if frame_shape is not None else None,
        # ``frame_shape`` is (height, width, channels); accept both landscape
        # and portrait Full HD rather than treating orientation as quality.
        "is_fhd_or_higher": bool(
            frame_shape
            and min(frame_shape[:2]) >= 1080
            and max(frame_shape[:2]) >= 1920
        ),
        "source_fps": source_fps if source_fps > 0 else None,
        "elapsed_seconds": elapsed_seconds,
        "measured_fps": len(rows) / elapsed_seconds if elapsed_seconds > 0 else None,
        "p50_latency_ms": float(np.percentile(durations, 50)) if durations else None,
        "p95_latency_ms": float(np.percentile(durations, 95)) if durations else None,
        "approved_redaction_frames": sum(row["is_redacted"] is True for row in rows),
        "review_required_frames": sum(row["review_required"] is True for row in rows),
        "approval_reason_counts": reason_counts,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
        },
    }


def _write_evidence(path: Path | None, *, input_path: Path, runtime: dict[str, object], rows: list[dict[str, object]], summary: dict[str, object] | None = None) -> None:
    """Write optional local evidence only when the operator explicitly asks."""
    if path is None:
        return
    from datetime import UTC, datetime

    if not path.parent.is_dir():
        raise ValueError("evidence destination directory is unavailable")
    payload = {
        "schema_version": 1,
        "run_timestamp": datetime.now(UTC).isoformat(),
        "input_filename": input_path.name,
        "runtime": runtime,
        "summary": summary or {
            "frame_count": len(rows),
            "approved_redaction_frames": sum(row["is_redacted"] is True for row in rows),
            "review_required_frames": sum(row["review_required"] is True for row in rows),
        },
        "frames": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    from time import perf_counter

    from consented_face_redactor.pipeline import RedactionPipeline, ProcessResult
    from consented_face_redactor.media.frame_source import OpenCvFrameSource

    # 1. Load config and state
    cfg = _load_config(args.config, strict=args.strict_config)
    state_dir = args.state_dir or args.input.parent
    prev_state = _load_track_state(state_dir)

    # 2. Instantiate only a fully verified runtime, or the explicit safe stub.
    try:
        detector, gallery, runtime = _load_runtime_components(args)
    except (ValueError, OSError) as exc:
        print(f"Error: runtime initialization failed: {exc}", file=sys.stderr)
        return 2
    pipe = RedactionPipeline(cfg, detector=detector, gallery=gallery)
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
        started = perf_counter()
        result = pipe.process_frame(
            frame=frame,
            frame_index=src.current_frame_index,
            timestamp=float(src.current_frame_index),
            state=None,  # already loaded above
        )
        duration_ms = (perf_counter() - started) * 1_000.0
    except (TypeError, ValueError) as exc:
        print(f"Error: pipeline processing failed: {exc}", file=sys.stderr)
        return 2

    if result is None:
        print("Error: pipeline returned no result", file=sys.stderr)
        return 2

    evidence_rows = [_evidence_row(
        src.current_frame_index, result, pipe, duration_ms=duration_ms,
        include_bboxes=args.evaluation_evidence,
    )]

    # 5. Write output (default path = input name with _redacted suffix)
    out_path = args.output or (args.input.parent / f"{args.input.stem}_redacted{args.input.suffix}")
    if not args.dry_run:
        try:
            _reject_output_overwrite(args.input, out_path)
            import cv2 as _cv2
            if not _cv2.imwrite(str(out_path), result.result_frame):
                raise OSError("OpenCV rejected the output path")
        except Exception as exc:
            print(f"Error: failed to write output '{out_path.name}': {exc}", file=sys.stderr)
            return 2

    # 6. Persist track state
    if not args.dry_run:
        _save_track_state(pipe, state_dir)
    try:
        _write_evidence(
            args.evidence_out, input_path=args.input, runtime=runtime, rows=evidence_rows,
            summary=_build_processing_summary(
                evidence_rows, frame_shape=tuple(frame.shape), elapsed_seconds=duration_ms / 1_000.0,
                source_fps=src.fps,
            ),
        )
    except ValueError as exc:
        print(f"Error: failed to write evidence: {exc}", file=sys.stderr)
        return 2

    # 7. Summary
    destination = "dry-run (no image written)" if args.dry_run else f"'{out_path.name}'"
    print(f"Processed '{args.input.name}' → {destination}")
    print(f"  Track state: {result.track_state.value}")
    print(f"  Is redacted: {result.is_redacted}")
    print(f"  Review required: {result.review_required}")
    return 0


# ------------------------------------------------------------------ #
# process-video command
# ------------------------------------------------------------------ #


def _cmd_process_video_temporal(
    args: argparse.Namespace,
    cfg: Any,
    detector: Any,
    gallery: Any,
    runtime: dict[str, object],
) -> int:
    """Run explicit two-pass temporal analysis and rendering."""
    from time import perf_counter

    from consented_face_redactor.temporal_video_processor import TemporalVideoProcessor
    from consented_face_redactor.tracking.bidirectional import ReconciliationPolicy

    if detector is None or gallery is None:
        print("Error: temporal tracking requires the fully verified face runtime", file=sys.stderr)
        return 2
    try:
        tracker, tracker_runtime = _build_tracker_runtime(args)
        runtime.update(tracker_runtime)
        processor = TemporalVideoProcessor(
            config=cfg,
            detector=detector,
            gallery=gallery,
            tracker=tracker,
            policy=ReconciliationPolicy(
                maximum_anchor_gap_frames=args.tracker_max_gap_frames,
                minimum_path_iou=args.tracker_minimum_path_iou,
                minimum_visible_point_ratio=args.tracker_minimum_visible_ratio,
            ),
        )
        started = perf_counter()
        analysis = processor.analyze(args.input)
        analysis_seconds = perf_counter() - started
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Error: temporal analysis failed: {exc}", file=sys.stderr)
        return 2

    output_path = args.output or (args.input.parent / f"{args.input.stem}_tracked_processed.mp4")
    render_evidence = None
    video_only_path = output_path
    if args.preserve_audio:
        video_only_path = output_path.with_name(f".{output_path.stem}.video-only{output_path.suffix}")
    if not args.dry_run:
        try:
            _reject_output_overwrite(args.input, output_path)
            if output_path.exists() or video_only_path.exists():
                raise FileExistsError("temporal output or its video-only intermediate already exists")
            render_evidence = processor.render(args.input, video_only_path, analysis.plan)
            if args.preserve_audio:
                from consented_face_redactor.media.remux import remux_original_audio

                remux_original_audio(
                    original_video=args.input,
                    processed_video=video_only_path,
                    destination=output_path,
                    ffmpeg_path=args.ffmpeg_path,
                )
                video_only_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"Error: temporal rendering failed: {exc}", file=sys.stderr)
            return 2

    plan_by_frame: dict[int, list[Any]] = {}
    for decision in analysis.plan.decisions:
        plan_by_frame.setdefault(decision.frame_index, []).append(decision)
    rows: list[dict[str, object]] = []
    for row_value in analysis.frame_evidence:
        row = dict(row_value)
        authorized = int(row.get("temporally_authorized_face_count", 0))
        row["duration_ms"] = row["analysis_duration_ms"]
        row["is_redacted"] = authorized > 0
        row["review_required"] = bool(row["detected_face_count"] and authorized == 0)
        row["track_state"] = "confirmed" if authorized else "candidate"
        if args.evaluation_evidence:
            row["temporal_decisions"] = [
                {
                    "profile_id": item.profile_id,
                    "bbox": list(item.bbox) if item.bbox is not None else None,
                    "authorized": item.authorized,
                    "bbox_source": item.bbox_source,
                    "reason_code": item.reason_code,
                    "visible_point_ratio": item.visible_point_ratio,
                }
                for item in plan_by_frame.get(int(row["frame_index"]), [])
            ]
        rows.append(row)
    summary = _build_processing_summary(
        rows,
        frame_shape=analysis.frame_shape,
        elapsed_seconds=analysis_seconds,
        source_fps=analysis.fps,
    )
    summary.update(
        {
            "explicit_anchor_count": sum(
                item.reason_code == "explicit_gallery_anchor" for item in analysis.plan.decisions
            ),
            "bidirectional_propagated_count": sum(
                item.reason_code == "bidirectional_anchor_consensus" for item in analysis.plan.decisions
            ),
            "ambiguous_ranges": [
                {
                    "first_frame": item.first_frame,
                    "last_frame": item.last_frame,
                    "reason_code": item.reason_code,
                }
                for item in analysis.plan.ambiguous_ranges
            ],
            "render": (
                {
                    "input_frame_count": render_evidence.input_frame_count,
                    "output_frame_count": render_evidence.output_frame_count,
                    "redacted_frame_count": render_evidence.redacted_frame_count,
                }
                if render_evidence is not None
                else None
            ),
        }
    )
    try:
        _write_evidence(
            args.evidence_out,
            input_path=args.input,
            runtime=runtime,
            rows=rows,
            summary=summary,
        )
    except ValueError as exc:
        print(f"Error: failed to write evidence: {exc}", file=sys.stderr)
        return 2
    destination = "dry-run (no video written)" if args.dry_run else f"'{output_path.name}'"
    print(
        f"Temporally processed {analysis.plan.input_frame_count} frame(s) from "
        f"'{args.input.name}' → {destination}"
    )
    return 0


def _cmd_process_video(args: argparse.Namespace) -> int:
    """Process every frame of a video through the redaction pipeline."""
    from time import perf_counter

    from consented_face_redactor.pipeline import RedactionPipeline, ProcessResult
    from consented_face_redactor.media.frame_source import OpenCvFrameSource

    try:
        from consented_face_redactor.media.frame_source import FakeFrameReader
    except ImportError:
        FakeFrameReader = None  # type: ignore[misc,assignment]

    # 1. Load config and state
    cfg = _load_config(args.config, strict=args.strict_config)
    if args.preserve_audio and args.dry_run:
        print("Error: --preserve-audio cannot be used with --dry-run", file=sys.stderr)
        return 2
    if args.preserve_audio and args.ffmpeg_path is None:
        print("Error: --preserve-audio requires --ffmpeg-path", file=sys.stderr)
        return 2
    state_dir = args.state_dir or args.input.parent
    prev_state = _load_track_state(state_dir)

    # 2. Instantiate only a fully verified runtime, or the explicit safe stub.
    try:
        detector, gallery, runtime = _load_runtime_components(args)
    except (ValueError, OSError) as exc:
        print(f"Error: runtime initialization failed: {exc}", file=sys.stderr)
        return 2
    if args.tracker != "none":
        return _cmd_process_video_temporal(args, cfg, detector, gallery, runtime)
    pipe = RedactionPipeline(cfg, detector=detector, gallery=gallery)
    if prev_state is not None:
        pipe.load_track_state(prev_state)

    # 3. Open video via frame source abstraction
    src: Any = OpenCvFrameSource(args.input)
    try:
        src.open()
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: cannot open input '{args.input.name}': {exc}", file=sys.stderr)
        return 2

    # 4. Prepare output video writer only when an output is requested.
    out_path = args.output or (args.input.parent / f"{args.input.stem}_processed.mp4")
    video_only_path = out_path
    if args.preserve_audio:
        video_only_path = out_path.with_name(f".{out_path.stem}.video-only{out_path.suffix}")
    out_writer: Any | None = None
    if not args.dry_run:
        try:
            _reject_output_overwrite(args.input, out_path)
            if args.preserve_audio and (out_path.exists() or video_only_path.exists()):
                raise ValueError("audio-preserving output or its temporary file already exists")
            import cv2 as _cv2
            if src.height > 0 and src.width > 0:
                fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
                fps_out = max(1.0, src.fps) if src.fps > 0 else 30.0
                out_writer = _cv2.VideoWriter(str(video_only_path), fourcc, fps_out, (src.width, src.height))
                if not out_writer.isOpened():
                    raise OSError("OpenCV could not open the output writer")
        except Exception as exc:
            print(f"Error: cannot initialize video writer for '{out_path.name}': {exc}", file=sys.stderr)
            return 2

    # 5. Frame loop
    frame_count = 0
    evidence_rows: list[dict[str, object]] = []
    first_frame_shape: tuple[int, int, int] | None = None
    processing_started = perf_counter()
    total_frames = src.frame_count if hasattr(src, "frame_count") and src.frame_count >= 0 else -1
    while True:
        success, frame = src.read()
        if not success or frame is None:
            break

        result: ProcessResult | None = None
        try:
            started = perf_counter()
            result = pipe.process_frame(
                frame=frame,
                frame_index=src.current_frame_index,
                timestamp=float(src.current_frame_index) / (src.fps if src.fps > 0 else 30.0),
                state=None,
            )
            duration_ms = (perf_counter() - started) * 1_000.0
        except (TypeError, ValueError) as exc:
            print(f"Error: pipeline processing failed at frame {frame_count}: {exc}", file=sys.stderr)
            return 2

        # Keep the redacted frame if available; otherwise use original
        processed_frame = result.result_frame if result and hasattr(result, "result_frame") else frame
        if out_writer is not None:
            out_writer.write(processed_frame)

        first_frame_shape = tuple(frame.shape)
        evidence_rows.append(_evidence_row(
            src.current_frame_index, result, pipe, duration_ms=duration_ms,
            include_bboxes=args.evaluation_evidence,
        ))

        frame_count += 1
        if total_frames > 0 and frame_count >= total_frames:
            break  # safety exit for known-length sources

    # 6. Cleanup
    source_fps = src.fps
    if out_writer is not None:
        out_writer.release()
    src.close()

    if args.preserve_audio:
        from consented_face_redactor.media.remux import AudioRemuxError, remux_original_audio

        try:
            remux_original_audio(
                original_video=args.input,
                processed_video=video_only_path,
                destination=out_path,
                ffmpeg_path=args.ffmpeg_path,
            )
        except AudioRemuxError as exc:
            print(f"Error: audio remux failed: {exc}", file=sys.stderr)
            return 2

    # 7. Persist track state
    if not args.dry_run:
        _save_track_state(pipe, state_dir)
    try:
        _write_evidence(
            args.evidence_out, input_path=args.input, runtime=runtime, rows=evidence_rows,
            summary=_build_processing_summary(
                evidence_rows, frame_shape=first_frame_shape,
                elapsed_seconds=perf_counter() - processing_started,
                source_fps=source_fps,
            ),
        )
    except ValueError as exc:
        print(f"Error: failed to write evidence: {exc}", file=sys.stderr)
        return 2

    # 8. Summary
    destination = "dry-run (no video written)" if args.dry_run else f"'{out_path.name}'"
    print(f"Processed {frame_count} frame(s) from '{args.input.name}' → {destination}")
    return 0


# ------------------------------------------------------------------ #
# gallery enroll command
# ------------------------------------------------------------------ #


def _cmd_gallery_enroll(args: argparse.Namespace) -> int:
    """Enroll exactly one detected face with an optional explicit approval."""
    from consented_face_redactor.approval_store import ApprovalRecord
    from consented_face_redactor.gallery import EnrollmentValidationError
    from consented_face_redactor.media.frame_source import OpenCvFrameSource

    # Validate the documented config option even though enrollment has no
    # effect settings of its own. This keeps --strict-config meaningful.
    _load_config(args.config, strict=args.strict_config)
    if args.approve and (not isinstance(args.approval_reason, str) or not args.approval_reason.strip()):
        print("Error: --approve requires a non-empty --approval-reason", file=sys.stderr)
        return 2

    # 1. Validate real models before reading or mutating any gallery file.
    try:
        detector, embedder = _build_enrollment_runtime(args)
    except (ValueError, OSError) as exc:
        print(f"Error: runtime initialization failed: {exc}", file=sys.stderr)
        return 2

    # 2. Read a single enrollment image.
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

    try:
        detections = detector.detect(frame)
    except Exception as exc:
        print(f"Error: face detection failed: {exc}", file=sys.stderr)
        return 2
    if len(detections) != 1:
        print(
            f"Enrollment rejected — expected exactly one detectable face, found {len(detections)}",
            file=sys.stderr,
        )
        return 2
    try:
        embedding, _revision = embedder.embed(frame, detections[0])
    except Exception as exc:
        print(f"Enrollment rejected — embedding failed: {exc}", file=sys.stderr)
        return 2

    # 3. Enroll the real, normalized face embedding.
    gallery_db_path = Path(args.gallery_db)
    approval_path = Path(args.approval_db)
    try:
        gallery, approvals = _load_gallery_and_approval_stores(gallery_db_path, approval_path)
    except Exception as exc:
        print(f"Error: cannot open gallery or approval store: {exc}", file=sys.stderr)
        return 2

    try:
        profile_id = gallery.enroll(embedding)
    except EnrollmentValidationError as exc:
        print(f"Enrollment rejected — reason: {exc.reason} — detail: {exc.detail}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: enrollment failed: {exc}", file=sys.stderr)
        return 2

    # 4. Persist vector and separately record the explicit approval decision.
    try:
        gallery.save(gallery_db_path)
        reason = args.approval_reason.strip() if args.approve else "enrolled_pending_approval"
        approvals.set(profile_id, ApprovalRecord(bool(args.approve), reason))
        approvals.save(approval_path)
    except (OSError, ValueError) as exc:
        print(f"Error: cannot write gallery '{gallery_db_path.name}': {exc}", file=sys.stderr)
        return 2

    src.close()

    # 5. Summary
    print(f"Enrolled face into profile '{profile_id}'")
    print(f"Gallery database written: {gallery_db_path}")
    print(f"Explicit approval: {bool(args.approve)}")
    return 0


def _cmd_gallery_enroll_video(args: argparse.Namespace) -> int:
    """Enroll diverse references from a user-designated target-only video."""
    from consented_face_redactor.approval_store import ApprovalRecord
    from consented_face_redactor.media.frame_source import OpenCvFrameSource
    from consented_face_redactor.video_enrollment import VideoEnrollmentOptions, VideoEnrollmentService

    if args.approve and (not isinstance(args.approval_reason, str) or not args.approval_reason.strip()):
        print("Error: --approve requires a non-empty --approval-reason", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"Error: registration video '{args.input.name}' does not exist", file=sys.stderr)
        return 2
    try:
        detector, embedder = _build_enrollment_runtime(args)
        options = VideoEnrollmentOptions(
            sample_every_n_frames=args.sample_every_n_frames,
            max_references=args.max_references,
            duplicate_similarity=args.duplicate_similarity,
            minimum_cluster_similarity=args.minimum_cluster_similarity,
        )
    except (TypeError, ValueError, OSError) as exc:
        print(f"Error: enrollment runtime initialization failed: {exc}", file=sys.stderr)
        return 2

    source = OpenCvFrameSource(args.input)
    try:
        source.open()
        service = VideoEnrollmentService(
            detector=detector, embedder=embedder, options=options
        )
        candidates, base_report = service.collect(source)
        selected, report = service.select(candidates, base_report)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: enrollment video processing failed: {exc}", file=sys.stderr)
        return 2
    finally:
        source.close()

    try:
        _write_enrollment_report(args.report_out, report)
    except ValueError as exc:
        print(f"Error: cannot write enrollment report: {exc}", file=sys.stderr)
        return 2
    if not selected:
        print("Enrollment rejected — no diverse valid references were selected", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"Dry-run selected {len(selected)} diverse reference(s); no gallery or approval files changed.")
        return 0

    gallery_path = Path(args.gallery_db)
    approval_path = Path(args.approval_db)
    try:
        gallery, approvals = _load_gallery_and_approval_stores(gallery_path, approval_path)
        extending_existing = args.profile_id is not None
        profile_id = gallery.enroll_many(
            [candidate.embedding.copy() for candidate in selected], profile_id=args.profile_id
        )
        if args.approve:
            approvals.set(profile_id, ApprovalRecord(True, args.approval_reason.strip()))
        elif not extending_existing:
            approvals.set(profile_id, ApprovalRecord(False, "enrolled_pending_approval"))
        gallery.save(gallery_path)
        approvals.save(approval_path)
    except Exception as exc:
        print(f"Error: video enrollment could not be persisted: {exc}", file=sys.stderr)
        return 2

    print(f"Enrolled {len(selected)} diverse reference(s) into profile '{profile_id}'")
    print(f"Gallery database written: {gallery_path}")
    print(f"Explicit approval changed: {bool(args.approve)}")
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
        "gallery-enroll-video": _cmd_gallery_enroll_video,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Error: unknown command '{args.command}'", file=sys.stderr)
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
