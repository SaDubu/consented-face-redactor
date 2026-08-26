"""Synthetic, local-only benchmark runner for the public benchmark API."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from consented_face_redactor.benchmark.fake_detector import FakeDetector
from consented_face_redactor.benchmark.fake_gallery import FakeGallery
from consented_face_redactor.config import Config
from consented_face_redactor.gallery_approval import GalleryApproval
from consented_face_redactor.pipeline import RedactionPipeline, TrackState
from consented_face_redactor.effects.mosaic import ellipse_bounds, expand_bbox

RUNNER_VERSION = "1.2.0"
_BBOX = (10.0, 10.0, 70.0, 70.0)


def _make_frame(height: int = 96, width: int = 96) -> np.ndarray:
    """Return deterministic non-uniform synthetic pixels for effect checks."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class RunnerResult:
    """One observed scenario result; failures are data, not omitted rows."""

    category: str
    scenario: str
    passed: bool
    duration_ms: float
    error: str | None = None
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "scenario": self.scenario,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metrics": self.metrics,
        }


def _run_scenario(
    category: str,
    scenario: str,
    action: Callable[[], tuple[bool, dict[str, Any] | None]],
) -> RunnerResult:
    """Execute exactly one scenario and preserve failures as results."""
    started = time.perf_counter()
    try:
        passed, metrics = action()
        error = None
    except Exception as exc:  # benchmark failures must remain observable
        passed, metrics, error = False, None, f"{type(exc).__name__}: {exc}"
    duration_ms = (time.perf_counter() - started) * 1_000.0
    return RunnerResult(category, scenario, passed, duration_ms, error, metrics)


def _confirmed_pipeline(effect_mode: str, *, sticker_png_bytes: bytes | None = None) -> RedactionPipeline:
    """Build a fresh, explicitly approved synthetic pipeline."""
    config = Config.from_dict({
        "effect_mode": effect_mode,
        "recheck_interval_frames": 1,
        "track_lost_ttl_frames": 2,
    })
    if sticker_png_bytes is not None:
        # The production renderer accepts an in-memory PNG asset; no file is used.
        config.sticker_png_bytes = sticker_png_bytes
    return RedactionPipeline(
        config,
        detector=FakeDetector(bboxes=[_BBOX], confs=[0.99]),
        gallery=FakeGallery(GalleryApproval(
            True, "synthetic-approved-profile", 0.82, "explicit_approval", "synthetic-v1"
        )),
    )


def _confirm(pipe: RedactionPipeline, frame: np.ndarray) -> Any:
    """Advance UNSEEN -> CANDIDATE -> CONFIRMED with an explicit gallery match."""
    first = pipe.process_frame(frame, 0, 0.0, None)
    assert first.track_state is TrackState.CANDIDATE
    assert not first.is_redacted
    return pipe.process_frame(frame, 1, 1.0, pipe.save_track_state())


def _approval_metrics(pipe: RedactionPipeline, result: Any) -> dict[str, Any]:
    """Expose the observed authority decision without deriving authority from it."""
    approval = pipe.last_gallery_approval
    return {
        "final_state": result.track_state.value,
        "is_redacted": result.is_redacted,
        "approval_reason": approval.reason_code,
        "gallery_revision": approval.gallery_revision,
        "gallery_recheck_count": pipe.telemetry_snapshot["gallery_recheck_count"],
    }


def run_benchmark(*, category: str = "A") -> dict[str, Any]:
    """Run one benchmark category (``A`` through ``E``) using synthetic data only."""
    runners: dict[str, Callable[[], list[RunnerResult]]] = {
        "A": _run_category_identity_safety,
        "B": _run_category_redaction_accuracy,
        "C": _run_category_track_transitions,
        "D": _run_category_performance,
        "E": _run_category_config_validation,
    }
    try:
        scenarios = runners[category]()
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark category: {category!r}") from exc

    passed_count = sum(result.passed for result in scenarios)
    total_count = len(scenarios)
    return {
        "category": category,
        "results": [result.to_dict() for result in scenarios],
        "passed_count": passed_count,
        "total_count": total_count,
        "pass_rate_pct": passed_count / total_count * 100 if total_count else 0.0,
    }


def _run_category_identity_safety() -> list[RunnerResult]:
    """Prove that detection confidence never replaces gallery approval."""
    def confidence_only() -> tuple[bool, dict[str, Any]]:
        frame = _make_frame()
        pipe = RedactionPipeline(
            Config.from_dict({"recheck_interval_frames": 1}),
            detector=FakeDetector(bboxes=[_BBOX], confs=[1.0]),
        )
        pipe.process_frame(frame, 0, 0.0, None)
        result = pipe.process_frame(frame, 1, 1.0, pipe.save_track_state())
        return (
            result.track_state is TrackState.CANDIDATE and not result.is_redacted,
            _approval_metrics(pipe, result),
        )

    def gallery_approval() -> tuple[bool, dict[str, Any]]:
        frame = _make_frame()
        pipe = _confirmed_pipeline("mosaic")
        result = _confirm(pipe, frame)
        return (
            result.track_state is TrackState.CONFIRMED and result.is_redacted,
            _approval_metrics(pipe, result),
        )

    def empty_gallery() -> tuple[bool, dict[str, Any]]:
        frame = _make_frame()
        pipe = RedactionPipeline(
            Config.from_dict({"recheck_interval_frames": 1}),
            detector=FakeDetector(bboxes=[_BBOX], confs=[0.99]),
            gallery=FakeGallery([]),
        )
        pipe.process_frame(frame, 0, 0.0, None)
        result = pipe.process_frame(frame, 1, 1.0, pipe.save_track_state())
        return (
            result.track_state is TrackState.CANDIDATE and not result.is_redacted,
            _approval_metrics(pipe, result),
        )

    def no_gallery_match() -> tuple[bool, dict[str, Any]]:
        frame = _make_frame()
        pipe = RedactionPipeline(
            Config.from_dict({"recheck_interval_frames": 1}),
            detector=FakeDetector(bboxes=[_BBOX], confs=[0.01]),
            gallery=FakeGallery([]),
        )
        pipe.process_frame(frame, 0, 0.0, None)
        result = pipe.process_frame(frame, 1, 1.0, pipe.save_track_state())
        return (
            result.track_state is TrackState.CANDIDATE and not result.is_redacted,
            _approval_metrics(pipe, result),
        )

    def no_detection() -> tuple[bool, dict[str, Any]]:
        frame = _make_frame()
        result = RedactionPipeline(
            Config.default(), detector=FakeDetector()
        ).process_frame(frame, 0, 0.0, None)
        return (
            result.track_state is TrackState.UNSEEN and not result.is_redacted,
            {"final_state": result.track_state.value, "is_redacted": result.is_redacted},
        )

    def gallery_failure(name: str, gallery: FakeGallery, expected_reason: str) -> RunnerResult:
        def action() -> tuple[bool, dict[str, Any]]:
            frame = _make_frame()
            pipe = RedactionPipeline(
                Config.from_dict({"recheck_interval_frames": 1}),
                detector=FakeDetector(bboxes=[_BBOX], confs=[0.99]),
                gallery=gallery,
            )
            pipe.process_frame(frame, 0, 0.0, None)
            result = pipe.process_frame(frame, 1, 1.0, pipe.save_track_state())
            metrics = _approval_metrics(pipe, result)
            return (
                result.track_state is TrackState.CANDIDATE
                and not result.is_redacted
                and metrics["approval_reason"] == expected_reason,
                metrics,
            )
        return _run_scenario("A", name, action)

    return [
        _run_scenario("A", "A1-confidence-only-no-gallery", confidence_only),
        _run_scenario("A", "A2-explicit-gallery-approval", gallery_approval),
        _run_scenario("A", "A3-empty-gallery", empty_gallery),
        _run_scenario("A", "A4-gallery-no-match", no_gallery_match),
        _run_scenario("A", "A5-no-detection", no_detection),
        gallery_failure("A6-embedding-error", FakeGallery(embed_error=RuntimeError("synthetic")), "embedding_error"),
        gallery_failure("A7-match-error", FakeGallery(match_error=RuntimeError("synthetic")), "gallery_match_error"),
        gallery_failure("A8-malformed-approval", FakeGallery(malformed_result=True), "malformed_approval"),
        gallery_failure(
            "A9-stale-profile",
            FakeGallery(GalleryApproval.denied("stale_profile", gallery_revision="synthetic-v1")),
            "stale_profile",
        ),
    ]


def _outside_mask(shape: tuple[int, int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
    """Return a mask for every pixel outside ``(x1, y1, x2, y2)``."""
    x1, y1, x2, y2 = roi
    mask = np.ones(shape[:2], dtype=bool)
    mask[y1:y2, x1:x2] = False
    return mask


def assert_effect_is_local(
    source: np.ndarray,
    result: np.ndarray,
    affected_regions: tuple[tuple[int, int, int, int], ...],
) -> dict[str, Any]:
    """Assert that every effect region changes and all other pixels are identical."""
    mask = np.ones(source.shape[:2], dtype=bool)
    region_changes: list[bool] = []
    for x1, y1, x2, y2 in affected_regions:
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, source.shape[1]), min(y2, source.shape[0])
        mask[y1:y2, x1:x2] = False
        region_changes.append(not np.array_equal(result[y1:y2, x1:x2], source[y1:y2, x1:x2]))
    outside_preserved = np.array_equal(result[mask], source[mask])
    assert all(region_changes), "one or more effect regions were unchanged"
    assert outside_preserved, "pixels outside effect regions changed"
    return {
        "effect_rois": affected_regions,
        "roi_changed": all(region_changes),
        "outside_preserved": outside_preserved,
    }


def _mosaic_effect_regions(
    source: np.ndarray,
    regions: tuple[tuple[float, float, float, float], ...],
) -> tuple[tuple[int, int, int, int], ...]:
    """Return the actual padded renderer contract for benchmark assertions."""
    config = Config.default()
    if config.mosaic_shape == "ellipse":
        return tuple(
            ellipse_bounds(
                region,
                frame_shape=source.shape,
                horizontal_scale=config.mosaic_ellipse_horizontal_scale,
                vertical_scale=config.mosaic_ellipse_vertical_scale,
            )
            for region in regions
        )
    return tuple(
        expand_bbox(
            region,
            frame_shape=source.shape,
            padding_ratio=config.mosaic_padding_ratio,
        )
        for region in regions
    )


def _run_category_redaction_accuracy() -> list[RunnerResult]:
    """Measure real effects with a common fresh-input scenario wrapper."""
    def mosaic() -> tuple[bool, dict[str, Any]]:
        source = _make_frame()
        result = _confirm(_confirmed_pipeline("mosaic"), source.copy())
        metrics = assert_effect_is_local(
            source,
            result.result_frame,
            _mosaic_effect_regions(source, (_BBOX,)),
        )
        return result.is_redacted, metrics

    def sticker() -> tuple[bool, dict[str, Any]]:
        source = _make_frame()
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba[:, :, :3] = (100, 150, 200)
        rgba[:, :, 3] = 255
        encoded_ok, encoded = cv2.imencode(".png", rgba)
        if not encoded_ok:
            raise RuntimeError("synthetic sticker PNG encoding failed")
        png_bytes = encoded.tobytes()
        assert isinstance(png_bytes, bytes)
        result = _confirm(_confirmed_pipeline("sticker", sticker_png_bytes=png_bytes), source.copy())
        metrics = assert_effect_is_local(source, result.result_frame, ((32, 32, 48, 48),))
        metrics["png_byte_count"] = len(png_bytes)
        return result.is_redacted, metrics

    def no_detection() -> tuple[bool, dict[str, Any]]:
        source = _make_frame()
        result = RedactionPipeline(
            Config.default(), detector=FakeDetector()
        ).process_frame(source.copy(), 0, 0.0, None)
        preserved = np.array_equal(result.result_frame, source)
        return not result.is_redacted and preserved, {"outside_preserved": preserved}

    def multiple_faces() -> tuple[bool, dict[str, Any]]:
        source = _make_frame()
        regions = ((10.0, 10.0, 30.0, 30.0), (55.0, 45.0, 80.0, 75.0))
        pipe = RedactionPipeline(
            Config.from_dict({"effect_mode": "mosaic", "recheck_interval_frames": 1}),
            detector=FakeDetector(bboxes=list(regions), confs=[0.99, 0.99]),
            gallery=FakeGallery(GalleryApproval(True, "synthetic-approved-profile", 0.82, "explicit_approval", "synthetic-v1")),
        )
        result = _confirm(pipe, source.copy())
        metrics = assert_effect_is_local(
            source,
            result.result_frame,
            _mosaic_effect_regions(source, regions),
        )
        return result.is_redacted, metrics

    def clipped_edge() -> tuple[bool, dict[str, Any]]:
        source = _make_frame()
        pipe = RedactionPipeline(
            Config.from_dict({"effect_mode": "mosaic", "recheck_interval_frames": 1}),
            detector=FakeDetector(bboxes=[(-10.0, -5.0, 20.0, 30.0)], confs=[0.99]),
            gallery=FakeGallery(GalleryApproval(True, "synthetic-approved-profile", 0.82, "explicit_approval", "synthetic-v1")),
        )
        result = _confirm(pipe, source.copy())
        metrics = assert_effect_is_local(
            source,
            result.result_frame,
            _mosaic_effect_regions(source, ((-10.0, -5.0, 20.0, 30.0),)),
        )
        return result.is_redacted, metrics

    return [
        _run_scenario("B", "B1-mosaic-effect-roi", mosaic),
        _run_scenario("B", "B2-sticker-effect-roi", sticker),
        _run_scenario("B", "B3-no-detection", no_detection),
        _run_scenario("B", "B4-mosaic-multiple-faces", multiple_faces),
        _run_scenario("B", "B5-mosaic-clipped-edge", clipped_edge),
    ]


class _SequenceDetector:
    """Synthetic detector with one prescribed observation per frame."""

    def __init__(self, detections: list[bool]) -> None:
        self._detections = iter(detections)

    def detect(self, frame: np.ndarray) -> list[Any]:
        return FakeDetector(bboxes=[_BBOX], confs=[0.99]).detect(frame) if next(self._detections) else []


def _run_category_track_transitions() -> list[RunnerResult]:
    """Observe UNSEEN -> CANDIDATE -> CONFIRMED -> LOST -> EXPIRED live."""
    frame = _make_frame()
    pipe = RedactionPipeline(
        Config.from_dict({"recheck_interval_frames": 1, "track_lost_ttl_frames": 2}),
        detector=_SequenceDetector([True, True, False, False, True]),
        gallery=FakeGallery([("synthetic-approved-profile", 0.82)]),
    )
    steps = [
        ("C1-unseen-to-candidate", TrackState.CANDIDATE),
        ("C2-candidate-to-confirmed", TrackState.CONFIRMED),
        ("C3-confirmed-to-lost", TrackState.LOST),
        ("C4-lost-to-expired", TrackState.EXPIRED),
        ("C5-expired-to-candidate", TrackState.CANDIDATE),
    ]
    results: list[RunnerResult] = []
    frame_indices = (0, 1, 2, 4, 5)
    for index, (scenario, expected) in zip(frame_indices, steps):
        def action(index: int = index, expected: TrackState = expected) -> tuple[bool, dict[str, Any]]:
            result = pipe.process_frame(frame.copy(), index, float(index), pipe.save_track_state() if index else None)
            return result.track_state is expected, {
                "observed_state": result.track_state.value,
                "expected_state": expected.value,
                "is_redacted": result.is_redacted,
            }
        results.append(_run_scenario("C", scenario, action))
    return results


def _run_category_performance() -> list[RunnerResult]:
    """Record measured latency and FPS; no host-dependent target is enforced."""
    def latency_measurement() -> tuple[bool, dict[str, Any]]:
        frame = _make_frame()
        pipe = _confirmed_pipeline("mosaic")
        _confirm(pipe, frame)
        for index in range(2, 12):
            pipe.process_frame(frame, index, float(index), pipe.save_track_state())
        samples_ms: list[float] = []
        for index in range(12, 112):
            started = time.perf_counter()
            result = pipe.process_frame(frame, index, float(index), pipe.save_track_state())
            samples_ms.append((time.perf_counter() - started) * 1_000.0)
            if not result.is_redacted:
                raise AssertionError("confirmed synthetic track stopped redacting")
        median_ms = statistics.median(samples_ms)
        telemetry = pipe.telemetry_snapshot
        return median_ms > 0.0, {
            "warmup_frames": 10,
            "sample_frames": len(samples_ms),
            "median_latency_ms": median_ms,
            "p95_latency_ms": float(np.percentile(samples_ms, 95)),
            "fps": 1_000.0 / median_ms,
            "frame_shape": frame.shape,
            "effect_mode": "mosaic",
            "candidate_confidence_median": statistics.median(telemetry["candidate_confidences"]),
            "candidate_confidence_p95": float(np.percentile(telemetry["candidate_confidences"], 95)),
            "gallery_recheck_count": telemetry["gallery_recheck_count"],
            "approval_reason": telemetry["approval_reason"],
            "cpu": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
        }

    def fhd_latency_measurement() -> tuple[bool, dict[str, Any]]:
        """Measure FHD frame processing without turning host speed into a gate."""
        frame = _make_frame(1080, 1920)
        pipe = _confirmed_pipeline("mosaic")
        _confirm(pipe, frame)
        for index in range(2, 5):
            pipe.process_frame(frame, index, float(index), pipe.save_track_state())
        samples_ms: list[float] = []
        for index in range(5, 13):
            started = time.perf_counter()
            result = pipe.process_frame(frame, index, float(index), pipe.save_track_state())
            samples_ms.append((time.perf_counter() - started) * 1_000.0)
            if not result.is_redacted:
                raise AssertionError("confirmed FHD synthetic track stopped redacting")
        median_ms = statistics.median(samples_ms)
        return median_ms > 0.0, {
            "warmup_frames": 3,
            "sample_frames": len(samples_ms),
            "median_latency_ms": median_ms,
            "p95_latency_ms": float(np.percentile(samples_ms, 95)),
            "fps": 1_000.0 / median_ms,
            "frame_shape": frame.shape,
            "is_fhd_or_higher": True,
            "effect_mode": "mosaic",
            "cpu": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
        }

    return [
        _run_scenario("D", "D1-median-synthetic-frame-latency", latency_measurement),
        _run_scenario("D", "D2-fhd-synthetic-frame-latency", fhd_latency_measurement),
    ]


def _run_category_config_validation() -> list[RunnerResult]:
    """Validate the Config behavior that is actually implemented."""
    def defaults_round_trip() -> tuple[bool, dict[str, Any]]:
        original = Config.default()
        serialized = original.to_dict()
        restored = Config.from_dict(serialized)
        return restored.to_dict() == serialized, {"field_count": len(serialized)}

    def partial_and_unknown_keys() -> tuple[bool, dict[str, Any]]:
        config = Config.from_dict({"effect_mode": "sticker", "unused_key": "ignored"})
        return (
            config.effect_mode == "sticker" and config.t_confirm == Config.default().t_confirm,
            {"unknown_key_policy": "ignored", "effect_mode": config.effect_mode},
        )

    def strict_and_legacy_migration() -> tuple[bool, dict[str, Any]]:
        legacy = Config.from_dict({"effect_mode": "sticker"})
        try:
            Config.from_dict({"unused_key": "rejected"}, strict=True)
        except ValueError:
            strict_rejected = True
        else:
            strict_rejected = False
        return strict_rejected and legacy.schema_version == Config.SCHEMA_VERSION, {
            "strict_unknown_key_rejected": strict_rejected,
            "migrated_schema_version": legacy.schema_version,
        }

    return [
        _run_scenario("E", "E1-default-round-trip", defaults_round_trip),
        _run_scenario("E", "E2-partial-and-unknown-key", partial_and_unknown_keys),
        _run_scenario("E", "E3-strict-and-legacy-migration", strict_and_legacy_migration),
    ]


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generate_aggregate_report() -> str:
    """Return a JSON summary computed solely from actual category results."""
    summaries = [run_benchmark(category=category) for category in "ABCDE"]
    total_passed = sum(summary["passed_count"] for summary in summaries)
    total_all = sum(summary["total_count"] for summary in summaries)
    report = {
        "version": RUNNER_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "categories": [
            {
                "name": f"category_{summary['category']}",
                "passed_count": summary["passed_count"],
                "total_count": summary["total_count"],
                "pass_rate_pct": summary["pass_rate_pct"],
                "results": summary["results"],
            }
            for summary in summaries
        ],
        "total_passed": total_passed,
        "total_all": total_all,
        "overall_pass_rate_pct": total_passed / total_all * 100 if total_all else 0.0,
    }
    return json.dumps(report, indent=2)
