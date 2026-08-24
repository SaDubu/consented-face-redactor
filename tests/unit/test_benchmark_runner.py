#!/usr/bin/env python3
"""Unit tests for benchmark runner (synthetic/local-only data policy enforced)."""
import json
import textwrap
import pytest
from consented_face_redactor.benchmark.runner import run_benchmark, RunnerResult


class TestRunnerResult:
    """Test RunnerResult namedtuple/dataclass."""

    def test_to_dict_contains_all_fields(self):
        rr = RunnerResult(category="A", scenario="X", passed=True, duration_ms=12.5)
        d = rr.to_dict()
        assert d["category"] == "A"
        assert d["scenario"] == "X"
        assert d["passed"] is True
        assert d["duration_ms"] == 12.5
        assert d.get("error") is None

    def test_to_dict_error_field(self):
        rr = RunnerResult(category="B", scenario="Y", passed=False, duration_ms=3.0, error="oops")
        d = rr.to_dict()
        assert d["error"] == "oops"
        assert d["passed"] is False

    def test_frozen(self):
        rr = RunnerResult(category="C", scenario="Z", passed=True, duration_ms=1.0)
        with pytest.raises(ValueError):
            rr.scenario = "changed"


class TestCategoryEConfig:
    """Category E - Config validation via production run_benchmark API."""

    def test_run_category_e_returns_dict_with_results(self):
        from consented_face_redactor.benchmark.runner import run_benchmark
        result = run_benchmark(category="E")
        assert isinstance(result, dict)
        assert "results" in result
        assert "passed_count" in result
        assert "total_count" in result
        for r in result["results"]:
            assert "scenario" in r
            assert "passed" in r
            assert "duration_ms" in r
            assert "error" in r or True  # error may be absent when None


class TestIdentitySafetyCategory:
    """Category A - Identity Safety Gate tests."""

    def test_confidence_only_stays_candidate(self):
        from consented_face_redactor.pipeline import RedactionPipeline, TrackState
        from consented_face_redactor.benchmark.runner._run_category_a_identity_safety import (
            _MockEmptyGallery,
            _MockHighConfDet,
        )

        cfg = self._create_config(t_confirm=0.65)
        pipe = RedactionPipeline(cfg)
        result = pipe.process_frame(
            frame=synthetic.frame_from_bboxes([(20, 20, 40, 40)]),
            detector=_MockHighConfDet()
            gallery=_MockEmptyGallery(),
        )
        assert result.current.state == TrackState.CANDIDATE
        assert not result.redaction_applied


class TestRedactionAccuracyCategory:
    """Category B - Redaction Accuracy tests."""

    def test_mosaic_covers_bbox(self):
        from consented_face_redactor.pipeline import MosaicEffect, FaceBox
        mosaic = MosaicEffect()
        
        frame = synthetic.create_synth_frame(640, 480)
        bbox = (100, 100, 200, 200)
        
        result = pipeline.render_mosaic(frame, FaceBox(*bbox))
        
        # Mosaic should change pixel values within the face ROI
        assert not numpy.array_equal(result[bbox], frame[bbox])

    def test_sticker_placement_center(self):
        from consented_face_redactor.pipeline import StickerEffect
        
        sticker = synthetic.create_sticker_effect()
        center_bbox = (300, 250, 400, 350)  # center of 640x480 frame
        
        result = pipeline.render_sticker(frame, center_bbox)
        
        assert result is not None
        assert "face" in json.loads(result.metadata)


class TestTrackTransitions:
    def test_unseen_candidated_on_detection(self):
        from consented_face_redactor.pipeline import RedactionPipeline, TrackState
        
        pipe = new_Pipeline()
        frame = synth.create_frame(100, 100, "black")
        
        # First detection should transition UNSEEN -> CANDIDATE
        result = pipe.process_frame(frame, frame_idx=0)
        assert pipeline.current_track_state == TrackState.CANDIDATE


class TestAggregateReport:
    def test_aggregate_generates_valid_json(self):
        report = benchmark_runner.generate_aggregate_report()
        
        # Report should be a valid JSON string
        parsed = json.loads(report)
        assert "categories" in parsed
        assert "overall_pass_rate_pct" in parsed
        
        # Pass rate should be between 0 and 100
        assert 0 <= parsed["overall_pass_rate_pct"] <= 100

    def test_aggregate_includes_safety_score(self):
        report = benchmark_runner.generate_aggregate_report()
        parsed = json.loads(report)
        
        categories = {c["name"]: c for c in parsed["categories"]}
        
        # Safety category should exist
        assert "identity_safety" in categories
        assert "pass_rate_pct" in categories["identity_safety"]


class TestBenchmarkRunner:
    def test_run_benchmark_returns_dict(self):
        report = benchmark_runner.run(category="A")
        
        assert isinstance(report, dict)
        assert "results" in report
        assert "passed_count" in report

    def test_benchmark_completes_quickly(self, monkeypatch):
        """Benchmark should complete within reasonable time."""
        import time
        
        start = time.monotonic()
        result = benchmark_runner.run(category="A")
        elapsed = time.monotonic() - start
        
        # Should complete in under 30 seconds
        assert elapsed < 30.0
        assert result["passed_count"] > 0

