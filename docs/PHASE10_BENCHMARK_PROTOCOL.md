# Phase 10 Benchmark Protocol

## Purpose
Define the synthetic benchmark suite for evaluating face redaction accuracy, identity safety gate, and performance of the consented-face-redactor pipeline without using real biometric data.

## Constraints
- **Synthetic/local-only** — no model weights, no face images from people, no external downloads
- All media generated programmatically (solid color frames, synthetic noise, geometric patterns)
- Benchmarks use `src/consented_face_redactor/` internal modules only
- Never commit real biometric data, embeddings, or credentials

## Test Categories

### Category A — Identity Safety Gate (`test_bench_identity_safety.py`)
Verify the safety gate: CONFIDENTED state requires explicit gallery identity match, never triggered by confidence alone.

| # | Scenario | Config | Gallery Status | Expected Result |
|---|----------|--------|----------------|-----------------|
| A1 | High conf no gallery match | `t_confirm=0.65` | Empty gallery | Track → EXPIRED (no CONFIDENTED) |
| A2 | Low conf no gallery match | `t_confirm=0.95` | Empty gallery | Track → UNSEEN/CANDIDATE |
| A3 | Gallery match present | `t_confirm=0.65` | Valid gallery entries | Track → CONFIDENTED with match |
| A4 | Match below threshold | `t_confirm=0.65, t_keep=0.55` | All embeddings < 0.3 sim | Track → CANDIDATE (no CONFIDENTED) |
| A5 | Confidence high but no gallery | `effect_mode="mosaic"` | No gallery loaded in pipeline | ProcessResult.is_redacted=False

### Category B — Redaction Accuracy (`test_bench_redaction_accuracy.py`)
Verify mosaic/sticker effects are applied correctly at the bbox level.

| # | Scenario | Expected Output |
|---|----------|-----------------|
| B1 | CONFIDENTED track → mosaic | Every frame has mosaic blocks covering detected face bboxes |
| B2 | CONFIDENTED track → sticker | Every frame has sticker overlay at bbox center |
| B3 | No detections → no redaction | Output frames equal to input frames |

### Category C — Track Transitions (`test_bench_track_transitions.py`)
Verify CANDIDATE/CANDIDATE/CONFIDENTED/LOST/EXPIRED state transitions.

| # | Scenario | Expected Transition |
|---|----------|---------------------|
| C1 | First detection | UNSEEN → CANDIDATE |
| C2 | Gallery match during Candiate | CANDIDATE → CONFIDENTED (has_gallery_match=True) |
| C3 | Face disappears < ttl frames | CONFIDERD → LOST |
| C4 | No detections ≥ t_tl | LOST → EXPIRED |
| C5 | Re-appearance in EXPIRED | EXPIRED → CANDIDATE

### Category D — Performance (`test_bench_performance.py`)
Measure throughput with synthetic data.

| # | Metric | Method |
|---|--------|--------|
| D1 | fps (frames/sec) | process_frame() latency over 100 synthetic frames |
| D2 | memory overhead | RSS delta via `tracemalloc` or `psutil` |

### Category E — Config Validation (`test_bench_config.py`)
Verify all Config constructor fields and defaults.

| # | Scenario | Expected |
|---|----------|----------|
| E1 | Default Config | All default values match docs/CONFIG_SCHEMA.md |
| E2 | Invalid t_confirm > 1.0 | ValueError raised |
| E3 | from_dict() with partial keys | Only provided fields set, others use defaults |

### Category F — Aggregate Report (`test_bench_aggregate_report.py`)
Generate a JSON report matching the template in Section 2 below.

## Benchmark Runner API

```python
from src.consented_face_redactor.benchmark.runner import run_benchmark

report = run_benchmark(
    categories=["A", "B", "C"],        # Which test categories to run
    output_path="benchmarks/report.json",  # Where to write the report
)
```

## Evaluation Criteria (Thresholds for Pass/Fail)

| Category | Pass Threshold |
|----------|----------------|
| A (Safety) | All ≥ 10 sub-tests pass |
| B (Accuracy) | ≥ 95% of face bboxes covered by mosaic/sticker |
| C (Transitions) | All state transitions match expected exactly |
| D (Performance) | ≥ 24 fps on synthetic data at 640x480 |

## Human Approval Required
- Before running benchmarks with any real biometric media
- Before adjusting `t_keep`, `t_confirm`, or `recheck_interval_frames` thresholds
- Before publishing benchmark results to shared channels
