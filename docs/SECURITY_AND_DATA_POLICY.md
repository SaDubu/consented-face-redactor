# Security and Data Policy

## 1. Repository classification

This repository is public. It may contain source code, tests built from synthetic arrays, documentation, aggregate benchmark results, model manifests without binaries, and explicitly redistributable sticker assets with license metadata.

It must never contain:

- reference or evaluation face images
- original, intermediate, or output video
- aligned face crops, debug frames, or thumbnails
- face embeddings, gallery exports, or identity labels
- model weights or converted inference engines
- credentials, tokens, private endpoints, personal paths, host identifiers, or agent reasoning output

## 2. Local data classes

| Class | Examples | Storage rule |
| --- | --- | --- |
| Biometric source | reference faces, evaluation faces | private data root; never logged or committed |
| Biometric derived | embeddings, gallery centroid | private data root; owner-only access |
| Media | input, temporary frames, output | private data root; separate input/output paths |
| Model binary | ONNX, engine, weights | local model directory; manifest digest required |
| Public artifact | source, synthetic tests, aggregate metrics | eligible for Git after review |

## 3. Local data root

- The operator chooses one explicit data root outside the repository.
- Windows deployments restrict the directory to the current user through an owner-only ACL. POSIX deployments use an owner-only directory mode such as `0700`.
- Configuration may refer to paths at runtime, but serialized project reports and default logs use logical IDs or repository-independent basenames.
- Temporary files stay below the private data root and are removed after success or explicit failure handling.

## 4. Retention and deletion

- Reference sources are retained only when the operator explicitly chooses to keep them.
- Gallery deletion removes all vectors and metadata for the opaque profile ID.
- Partial output is not promoted to the requested output path.
- Deletion is best-effort filesystem deletion, not a claim of forensic secure erasure; storage encryption and operating-system backup policy remain operator responsibilities.

## 5. Logs and reports

Allowed fields include opaque profile ID, frame/time counters, controlled decision enum, aggregate similarity statistics, model manifest ID, command result, and media integrity summary.

Disallowed fields include pixels, crops, vectors, raw prompts, model reasoning, credentials, complete private paths, and arbitrary exception serialization. Review-required output identifies only frame index or timestamp ranges and reason enums.

## 6. Fail-closed conditions

Stop without producing a clean-complete result when model digest or license validation fails, input equals output, gallery schema is incompatible, identity association collides, a required audio/video property is lost, a forbidden artifact appears in the repository, or review-required ranges remain unacknowledged under the selected policy.
