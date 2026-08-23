# Consented Face Redactor: Implementation Plan

## 1. Project charter

### 1.1 Objective

Process a user-supplied image or video locally. When a face is verified as a user-enrolled, consented target, apply either mosaic or a supplied transparent sticker to that face and create a separate output file.

### 1.2 Non-goals

- fine-tuning a person-specific recognition model
- detecting identity when the face is absent through full-body re-identification
- surveillance, streaming, GUI, mobile app, cloud inference, or web service
- face replacement, deepfake generation, demographic inference, or identity synthesis
- automatic model/asset download or original media overwrite

### 1.3 Safety and privacy requirements

- The operator is responsible for consent and lawful use before enrollment.
- Reference images, embeddings, original media, crops, debug frames, output media, model weights, credentials, and machine-specific paths stay outside Git.
- An uncertain match must never transfer a redaction effect to a different face.
- Logs contain only opaque profile IDs, aggregate counters, redacted errors, and model manifest IDs.
- Input and output paths must be distinct; output is written atomically when the media backend supports it.

## 2. Research-backed baseline

The system uses detection, normalized face embedding, matching, tracking, and rendering as separate replaceable layers.

- FaceNet establishes the general approach of representing faces as comparable embeddings rather than training one model per target. [Paper](https://arxiv.org/abs/1503.03832)
- ArcFace provides the geometric rationale for discriminative angular embedding comparisons. The MVP uses a pre-trained compatible recognizer and calibrates its threshold on consented evaluation material instead of fine-tuning. [Paper](https://arxiv.org/abs/1801.07698)
- ByteTrack shows that tracking can reduce fragmented trajectories by associating detections. This project adopts its lesson cautiously: motion association may preserve an already-confirmed track briefly, but may not establish or transfer identity. [Paper](https://arxiv.org/abs/2110.06864)
- OpenCV's YuNet and SFace provide a practical initial detector/recognizer pair with five landmarks, alignment, and embedding comparison support. [OpenCV tutorial](https://docs.opencv.org/5.0/tutorials/dnn/dnn_face/dnn_face.html)

Initial technology choices:

| Area | Baseline | Change gate |
| --- | --- | --- |
| Language | Python 3.12 | move a measured bottleneck only after benchmark evidence |
| Detection | YuNet adapter | compare a different detector only when recall evidence demands it |
| Recognition | SFace adapter | compare ArcFace-family alternatives only with license and benchmark Task |
| Video | PyAV or FFmpeg adapter behind an interface | use OpenCV-only write path only when audio preservation is not required |
| Tracking | IoU + Kalman baseline | evaluate ByteTrack adapter after identity-safety tests |
| Tests | pytest and synthetic NumPy frames | local consented benchmark remains outside Git |

## 3. Target architecture

```text
input image/video
  -> decode frame and timestamp
  -> detect all faces and five landmarks
  -> quality gate and alignment
  -> normalized embedding
  -> target-gallery matching
  -> identity-aware temporal track state
  -> smooth/clamp redaction ROI
  -> mosaic or sticker renderer
  -> encoded output and media integrity report
```

Suggested source layout:

```text
src/consented_face_redactor/
  domain/types.py
  config.py
  model_manifest.py
  adapters/detector_yunet.py
  adapters/embedder_sface.py
  gallery.py
  matcher.py
  tracker.py
  effects/mosaic.py
  effects/sticker.py
  pipeline.py
  media/image_io.py
  media/video_io.py
  media/integrity.py
  cli.py
```

The pipeline must depend on internal interfaces, never raw vendor output. `process_frame(frame, frame_index, timestamp, state)` returns a new frame result and new state; it must not mutate the input frame by default.

## 4. Identity and tracking policy

### 4.1 Enrollment

1. Accept several reference images for one opaque local profile ID.
2. Detect exactly one sufficiently large face per accepted image.
3. Reject blur, extreme pose, bad landmark geometry, duplicate vectors, non-finite vectors, and zero-norm vectors.
4. Align, embed, L2-normalize, and retain versioned vectors plus a deterministic centroid.
5. Store the gallery locally outside Git; do not retain source-image paths in its serialized representation.

### 4.2 Match decision

- Compare each candidate embedding against both individual vectors and the centroid using cosine similarity.
- Keep detector confidence, match confidence, and temporal confirmation as different signals.
- Use a calibrated `T_confirm` threshold to start redaction and a lower bounded `T_keep` only for a recently confirmed track.
- Apply a strict preset by default: ambiguous candidates are not redacted rather than treated as the target.

### 4.3 Track state

```text
UNSEEN -> CANDIDATE -> CONFIRMED -> LOST -> EXPIRED
```

- `CANDIDATE` requires consecutive match confirmations.
- `CONFIRMED` is periodically re-verified by embedding; motion alone cannot extend it indefinitely.
- `LOST` may use short TTL motion prediction only to avoid flicker; it cannot jump to another face.
- A frame-index reversal, association collision, or incompatible embedding expires the track.

## 5. Delivery phases

Every phase is a separate immutable Task Contract with one isolated worktree, one Hermes worker lease, narrow write scope, deterministic checks, evidence review, and a Human approval gate.

### Phase 0: Hermes capability gate

**Goal:** prove the exact Hermes agent can use the selected local runner without a wrapper that converts natural language into filesystem or shell actions.

- Fix the Hermes distribution, model tag, runner version, loopback endpoint, context size, and worker Git identity.
- Run read-only tool probe, then one-file write canary, then bounded verification/commit/permitted-push canary.
- Deny arbitrary shell, web access, package installation, external directories, force push, main mutation, reset, rebase, and merge.
- **Pass:** independently verifiable model-authored file, commit ancestry, scope match, remote SHA, and clean security scan.

### Phase 1: Package and configuration baseline

**Scope:** `pyproject.toml`, package initializer, config schema, baseline tests.

- Create Python src layout and command entrypoint without side effects.
- Define strict configuration parsing for effect mode, thresholds, TTL, padding, and input/output paths.
- Reject unknown keys, URLs, token-like fields, NaN/Inf, invalid ranges, and equal input/output paths.
- **Pass:** import, pytest discovery, compile, deterministic config serialization, no created files.

### Phase 2: Model manifest and adapters

**Scope:** manifest example, manifest validator, YuNet/SFace interface contracts, tests.

- Require model ID, role, source, filename, SHA-256, license, input shape, preprocessing revision, and supported provider.
- Implement detector output as `BoundingBox`, five landmarks, and confidence.
- Implement embedder output as a normalized vector with an explicit model revision.
- **Pass:** checksum/license failures are fail-closed; no download code exists.

### Phase 3: Image vertical slice

**Scope:** image loader, fake detector/embedder, pipeline skeleton, mosaic baseline, tests.

- Process one synthetic image through fake adapters before any real model integration.
- Keep effect ROI within frame bounds; preserve pixels outside the ROI exactly.
- Write output to a new path only.
- **Pass:** deterministic byte-level test for fixed synthetic input and config.

### Phase 4: Local gallery and matcher

**Scope:** gallery storage, enrollment validation, cosine matcher, tests.

- Use opaque profile IDs and local gallery versioning.
- Exclude original image path, human name, raw crop, and debug pixel data from saved gallery metadata.
- Produce decision reasons as controlled enums, not model/raw error strings.
- **Pass:** deterministic ordering, malformed input rejection, calibrated threshold fixture support.

### Phase 5: Renderers

**Scope:** mosaic renderer, sticker renderer, asset metadata, tests.

- Mosaic: downscale/upscale ROI, face-size-based block policy, padding, clamp, degenerate ROI handling.
- Sticker: alpha blending, eye-angle rotation, box-anchor fallback, scale policy, frame-edge clipping, source-asset immutability.
- **Pass:** transparent pixels preserve originals; output shape/dtype remain valid; no renderer mutates input arrays.

### Phase 6: Real detector and embedder integration

**Scope:** YuNet and SFace adapters, model fixture protocol, integration tests with synthetic/fake boundary coverage.

- Set detector input size per frame; normalize vendor coordinates at adapter boundary.
- Align with five landmarks before embedding.
- Record only manifest ID and aggregate counts in logs.
- **Pass:** deterministic adapter error handling and no model-path leakage.

### Phase 7: Temporal tracking

**Scope:** tracker state, association policy, smoothing, tests.

- Start from IoU plus Kalman prediction and enforce periodic identity re-check.
- Define `T_confirm`, `T_keep`, recheck interval, maximum lost TTL, smoothing factor, and collision behavior in config.
- Test crossing faces, occlusion, re-entry, low-confidence detection, and frame skipping.
- **Pass:** no effect transfer across two synthetic identities and bounded state memory.

### Phase 8: Video I/O and integrity

**Scope:** decode/encode adapter, progress records, media report, tests.

- Preserve frame order, timestamps where supported, FPS, resolution, frame count, duration, and audio stream where required.
- Use temporary output and atomic rename on success; delete partial output on explicit failure.
- **Pass:** output reopens successfully and report identifies any media mismatch.

### Phase 9: CLI and operational UX

**Commands:** `validate-models`, `enroll`, `inspect-config`, `process-image`, `process-video`.

- Require explicit input, output, profile ID, config, and local model-manifest paths.
- Add `--dry-run` that validates without model load or output creation.
- Suppress internal stack traces and absolute paths by default.
- **Pass:** failure messages are actionable but do not disclose private data.

### Phase 10: Benchmark and calibration

**Scope:** local-only benchmark runner, protocol document, redacted aggregate report template.

- Measure target recall, non-target false-redaction rate, precision/recall by threshold, effect coverage, identity switches, track fragmentation, processing time, and peak memory.
- Measure 720p and 1080p separately; distinguish decode, detection, embedding, rendering, and encode time.
- Keep evaluation faces/videos outside repository and publish only aggregate, consent-safe results.
- **Pass:** Human Owner approves operating threshold and preset behavior based on evidence.

### Phase 11: Release candidate

- Re-run unit, integration, compile, diff, secret/biometric scan, model manifest validation, and benchmark report.
- Verify no model weight, gallery, raw media, output media, token, personal path, or worker reasoning text exists in Git history.
- Human Owner names the exact candidate SHA before merge.

## 6. Verification matrix

| Category | Required evidence |
| --- | --- |
| Identity | target/non-target decision matrix and threshold calibration |
| Privacy | Git scan, logs review, no raw biometric artifact |
| Rendering | ROI bounds, alpha correctness, input immutability |
| Tracking | crossing/occlusion/re-entry negative tests |
| Video | output reopen, FPS/resolution/frame-count/duration/audio comparison |
| Performance | per-stage timing and peak memory at defined resolutions |
| Supply chain | model manifest digest and license check |
| Worker governance | exact task, base SHA, scope, worker commit, remote SHA |

## 7. Roles and stop conditions

| Role | Responsibility |
| --- | --- |
| Human Owner | consent responsibility, stage start, threshold approval, exact merge approval |
| Hermes | directly writes only the active scoped Task, tests, commits, and permitted-pushes; stops after report |
| Codex | writes contracts and documentation, prepares isolated worktrees, verifies evidence and diff; does not author implementation patches |

Stop immediately when scope changes, a model file/license is missing, a face cannot be confidently matched, an effect risks moving to a non-target face, output integrity fails, a forbidden path changes, or a worker cannot perform an allowed action natively.

## 8. First Human decision needed

Before Phase 0, record the exact Hermes agent distribution or repository, model tag, intended local runner, and whether the new repository remains public. Do not install or invoke Hermes until those values are approved.
