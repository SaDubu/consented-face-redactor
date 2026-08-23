# Consented Face Redactor

동의받은 특정 인물의 참조 이미지를 로컬에서 등록한 뒤, 이미지 또는 영상에서 그 인물의 얼굴에 모자이크 또는 투명 스티커를 적용하는 개인용 Vision AI 도구입니다.

## Status

`PLANNING` - 구현 코드, 모델 weight, 참조 얼굴, 영상, embedding, 출력 파일은 아직 저장소에 포함하지 않습니다.

## Product boundary

- local-only batch processing for consented subjects
- image and video input with separate output files
- face detection, embedding-based identity verification, temporal tracking, mosaic, and sticker rendering
- no cloud face API, live surveillance, face swap, identity synthesis, automatic model download, or original-file overwrite

## Delivery principles

- Human Owner approves every stage and exact integration commit.
- Hermes is a new worker candidate. It receives no inherited authority and must pass an isolated runtime canary before source work.
- A worker directly authors its scoped patch, tests, commit, and permitted push. Codex owns contracts and verifies evidence only.
- A completed stage reports `STAGE_COMPLETE` and `WAITING_FOR_HUMAN`; it never starts the next stage itself.

## Documents

- [Detailed implementation plan](docs/IMPLEMENTATION_PLAN.md)

## Data safety

Never commit face images, video, crops, embeddings, model binaries, generated output, credentials, machine-specific paths, or model reasoning output.
