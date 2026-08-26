# Phase 10 Benchmark Protocol

## Scope

The suite uses only generated frames, generated PNG bytes, local code, and
controlled fake detector/gallery adapters. It never downloads models or uses
real biometric media.

Its public API is exactly:

```python
from consented_face_redactor.benchmark import (
    generate_aggregate_report,
    run_benchmark,
)

category_report = run_benchmark(category="A")
aggregate_json = generate_aggregate_report()
```

`run_benchmark(category)` accepts one of `A`, `B`, `C`, `D`, or `E` and returns
that category's observed scenario rows. `generate_aggregate_report()` runs the
same five categories and returns their aggregate as a JSON string. The category
keyword is the only benchmark-runner parameter.

## Scenarios

| Category | Scenarios | Observation |
|---|---|---|
| A | A1–A9 | High confidence, explicit approval, empty/no-match gallery, gallery exceptions, malformed decisions, stale profile, and no detection. Only `GalleryApproval.approved=True` may confirm/redact. |
| B | B1–B5 | Mosaic/sticker, multiple faces, and edge clipping all change their actual effect ROI while preserving every outside pixel. |
| C | C1–C5 | A live sequence observes `UNSEEN → CANDIDATE → CONFIRMED → LOST → EXPIRED → CANDIDATE`. |
| D | D1 | Warm-up plus 100 measured synthetic frames; median/p95 latency, FPS, candidate telemetry, and runtime metadata are reported. FPS has no environment-dependent pass threshold. |
| E | E1–E3 | Default round-trip, compatibility-mode unknown-key behavior, and strict/v1 migration behavior. |

Every scenario has fresh input data. Category B wraps every scenario in
exception capture, so a failed sticker encoding is recorded as B2
`passed=False` and cannot suppress B3.

## Result contract

Each `results` row contains `category`, `scenario`, `passed`, `duration_ms`,
`error`, and optional observed `metrics`. An exception is a retained failure
row, never a missing result. Aggregate JSON is derived from actual category
results and includes timestamp, Git revision when available, runtime metadata,
and the original category rows as an immutable evidence payload.

## Policy boundary

`t_confirm` and `t_keep` are presently non-authorizing quality/calibration
settings. They are not security thresholds. Identity approval is only the
gallery's structured `GalleryApproval` with `approved=True`. `None`, malformed
results, and gallery exceptions are fail-closed denials. Changing a policy threshold or using
real media requires separate human approval.
