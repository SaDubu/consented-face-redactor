#!/usr/bin/env python3
"""Unit tests for benchmark runner (synthetic/local-only data policy enforced)."""
import json
import pytest
from consented_face_redactor.benchmark import generate_aggregate_report, run_benchmark
from consented_face_redactor.benchmark.runner import RunnerResult


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
        with pytest.raises(AttributeError):
            rr.scenario = "changed"


class TestBenchmarkCategories:
    @pytest.mark.parametrize("category", list("ABCDE"))
    def test_category_results_are_observed(self, category: str) -> None:
        report = run_benchmark(category=category)

        assert report["total_count"] == len(report["results"])
        assert all(result["duration_ms"] >= 0 for result in report["results"])
        assert all("error" in result for result in report["results"])

    def test_confidence_only_never_confirms_or_redacts(self) -> None:
        result = run_benchmark(category="A")["results"]
        confidence_only = next(row for row in result if row["scenario"].startswith("A1"))

        assert confidence_only["passed"] is True
        assert confidence_only["metrics"]["final_state"] == "candidate"
        assert confidence_only["metrics"]["is_redacted"] is False
        assert confidence_only["metrics"]["approval_reason"] == "gallery_unavailable"

    def test_effect_scenarios_preserve_every_pixel_outside_effect_roi(self) -> None:
        result = run_benchmark(category="B")["results"]
        effects = [row for row in result if row["scenario"].startswith(("B1", "B2"))]

        assert all(row["metrics"]["roi_changed"] for row in effects)
        assert all(row["metrics"]["outside_preserved"] for row in effects)

    def test_gallery_failures_are_fail_closed_with_distinct_reasons(self) -> None:
        rows = run_benchmark(category="A")["results"]
        failures = {row["scenario"]: row for row in rows if row["scenario"].startswith(("A6", "A7", "A8", "A9"))}

        assert {row["metrics"]["approval_reason"] for row in failures.values()} == {
            "embedding_error", "gallery_match_error", "malformed_approval", "stale_profile",
        }
        assert all(row["passed"] is True for row in failures.values())

    def test_performance_reports_measurement_and_environment_metadata(self) -> None:
        row = run_benchmark(category="D")["results"][0]

        assert row["metrics"]["median_latency_ms"] > 0
        assert row["metrics"]["p95_latency_ms"] > 0
        assert row["metrics"]["gallery_recheck_count"] >= 1
        assert row["metrics"]["opencv_version"]


class TestFailureIsolation:
    def test_sticker_encoding_failure_is_reported_and_b3_is_retained(self, monkeypatch) -> None:
        import consented_face_redactor.benchmark.runner as runner

        monkeypatch.setattr(runner.cv2, "imencode", lambda *_args: (False, None))
        result = runner.run_benchmark(category="B")["results"]

        b2 = next(row for row in result if row["scenario"].startswith("B2"))
        b3 = next(row for row in result if row["scenario"].startswith("B3"))
        assert b2["passed"] is False
        assert b2["error"] is not None
        assert b3["passed"] is True


class TestAggregateReport:
    def test_aggregate_is_json_derived_from_all_categories(self) -> None:
        parsed = json.loads(generate_aggregate_report())

        assert {entry["name"] for entry in parsed["categories"]} == {
            "category_A", "category_B", "category_C", "category_D", "category_E",
        }
        assert parsed["total_all"] == sum(entry["total_count"] for entry in parsed["categories"])
        assert parsed["total_passed"] == sum(entry["passed_count"] for entry in parsed["categories"])
        assert "environment" in parsed
        assert parsed["generated_at_utc"]
        assert all("results" in entry for entry in parsed["categories"])
