# Consented Face Redactor

동의받은 특정 인물의 참조 이미지를 로컬에서 등록한 뒤, 이미지 또는 영상에서 그 인물의 얼굴에 모자이크 또는 투명 스티커를 적용하는 개인용 Vision AI 도구입니다.

## Status

`PHASE_4 IMPLEMENTED / INTERMEDIATE REVIEWED` - 구성 검증, 모델 manifest, OpenCV YuNet/SFace adapter, 기본 media I/O, 로컬 gallery와 matcher까지 구현되어 있습니다. Renderer, tracker, end-to-end CLI와 실제 모델 기반 평가는 아직 구현되지 않았습니다.

모델 weight, 참조 얼굴, 영상, embedding, 출력 파일은 저장소에 포함하지 않습니다. 이 저장소는 공개 상태이므로 문서와 재배포가 명시적으로 허용된 asset만 추적합니다.

## Product boundary

- local-only batch processing for consented subjects
- image and video input with separate output files
- face detection, embedding-based identity verification, temporal tracking, mosaic, and sticker rendering
- no cloud face API, live surveillance, face swap, identity synthesis, automatic model download, or original-file overwrite

## Delivery principles

- Human Owner approves every stage and exact integration commit.
- Hermes is an unresolved worker candidate, not an active agent. Its exact distribution and runner must be fixed before an isolated runtime canary.
- A worker directly authors its scoped patch, tests, commit, and permitted push. Codex owns contracts and verifies evidence only.
- A completed stage reports `STAGE_COMPLETE` and `WAITING_FOR_HUMAN`; it never starts the next stage itself.

## Documents

- [Detailed implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Security and data policy](docs/SECURITY_AND_DATA_POLICY.md)
- [Hermes runtime gate](docs/HERMES_RUNTIME_GATE.md)

## Data safety

Never commit face images, video, crops, embeddings, model binaries, generated output, credentials, machine-specific paths, or model reasoning output. A redistributable sticker may be tracked only with source and license metadata.

## Usage

### Quick-start (mosaic mode)

```bash
# 1. Clone this repo and install dependencies
$ pip install -e .

# 2. Create a minimal config file (JSON)
$ cat > run.json <<'EOF'
{
    "effect_mode": "mosaic",
    "input_path": "./samples/input.mp4",
    "output_path": "./outputs/redacted.mp4"
}
EOF

# 3. Run the pipeline
$ redactor run run.json
```

### Configuration fields reference

Configuration schema is defined in [docs/CONFIG_SCHEMA.md](docs/CONFIG_SCHEMA.md). Below are the verified production defaults:

| Field                   | Default     | Notes                                                                                             |
|-------------------------|-------------|---------------------------------------------------------------------------------------------------|
| `effect_mode`           | `"mosaic"`  | Plain string — `"mosaic"` or `"sticker"`. No EffectMode enum.                                    |
| `t_confirm`             | 0.65        | Minimum confidence to transition CANDIDATE → CONFIRMED (only after gallery match)                |
| `t_keep`                | 0.55        | Confidence floor to continue an already CONFIRMED track across brief occlusions                    |
| `track_lost_ttl_frames` | 10          | Frames to keep a track alive after last detection before EXPIRED                                   |
| `recheck_interval_frames` | 30        | How often (in frames) to invoke the gallery matcher while tracking CANDIDATE faces                  |

### Configuration examples

**Sticker mode with custom scale:**

```python
from consented_face_redactor.config import Config

cfg = Config(
    effect_mode="sticker",           # plain string — NOT EffectMode.STICKER
    input_path="./samples/input.mp4",
    output_path="./outputs/redacted_stickered.mp4",
    t_confirm=0.80,                  # higher bar for confirmed identity
    recheck_interval_frames=15,      # re-check gallery more frequently
)
```

**Mosaic mode with tighter tracking:**

```python
cfg = Config(
    effect_mode="mosaic",
    input_path="./samples/images/photo.jpg",
    output_path="./outputs/redacted.jpg",
    t_confirm=0.75,                  # higher confirmation threshold
    track_lost_ttl_frames=5,         # expire faster when face is lost
)
```

### Important — Confidence vs. Identity

Detector confidence alone **never** authorizes CONFIRMED redaction. An explicit gallery identity match is required for the CANDIDATE → CONFIRMED transition. The safety gate always fails closed when the gallery is unavailable or returns no match.
