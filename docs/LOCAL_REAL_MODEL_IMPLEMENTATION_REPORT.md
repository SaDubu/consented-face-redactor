# 로컬 실제 모델 실행 경로 구현 보고서

> 이 문서는 tracker 도입 전 milestone 기록이다. 실제 모델·영상까지 포함한 최신 상태와 321개 테스트/456 frame 결과는 [TAPNext++ 시간축 구현 보고서](TEMPORAL_TRACKING_IMPLEMENTATION_REPORT.md)가 대체한다.

## 목적과 작업 범위

이 문서는 local clone에서 수행한 “동의된 테스트 영상을 실제 모델로 처리할 수 있는 경로” 구현을 기록한다. 변경은 로컬에만 남아 있으며 commit이나 push는 수행하지 않았다. 모델 파일, 얼굴 이미지, gallery, approval JSON, 입력·출력 영상도 저장소에 포함하지 않는다.

이 milestone 작성 당시에는 실제 YuNet/SFace binary와 사용자 영상 smoke test 전이었다. 이후 공식 모델과 `learn.mp4`/`test.mp4`가 local_data 경로에서 검증됐고 TAPNext++까지 연결됐다.

## 기존 상태에서 확인한 문제

기존에는 YuNet detector와 SFace embedder 클래스가 있었지만 기본 CLI는 `RedactionPipeline(config)`만 생성했다. 즉 detector와 gallery가 주입되지 않아 영상 파일을 읽고 출력할 수는 있어도 실제 얼굴 탐지·가림은 일어나지 않았다.

추가로 다음을 수정했다.

1. enrollment가 실제 얼굴 embedding 대신 이미지 RGB 평균 vector를 저장하던 placeholder였다.
2. `LocalGallery.match()`의 similarity 결과와 pipeline이 요구하는 `GalleryApproval` 사이에 명시 approval 계층이 없었다.
3. 한 얼굴이 승인되면 동일 frame의 모든 얼굴 ROI를 가릴 수 있는 경로가 있었다.
4. BGR 입력을 사용하는 YuNet 앞에서 BGR→RGB 변환이 발생할 수 있었다.
5. 실제 CLI 실행에는 승인 사유를 남기는 evidence 출력이 없었다.

## 구현한 구성요소

| 파일 | 구현 내용 |
| --- | --- |
| `src/consented_face_redactor/approval_store.py` | profile vector와 분리된 `ApprovalRecord`, 엄격한 JSON `ApprovalStore`, atomic local save |
| `src/consented_face_redactor/approved_gallery.py` | SFace embedding → LocalGallery match → explicit approval을 연결하는 runtime adapter |
| `src/consented_face_redactor/pipeline.py` | BGR 보존, detection별 승인, 승인된 ROI만 effect 처리, 최신 frame의 approval 관측 |
| `src/consented_face_redactor/cli.py` | model runtime, 실제 enrollment, dry-run, optional evidence, model-dir validator |
| `src/consented_face_redactor/gallery_approval.py` | 거절 결과에도 profile/similarity를 기록할 수 있도록 확장 |
| `tests/test_approved_gallery.py` | approval 없음·유효 approval·만료 approval·store round-trip 검사 |
| `tests/test_pipeline_face_by_face.py` | 승인된 얼굴만 바뀌고 같은 frame의 거절 얼굴 ROI는 보존되는지 검사 |
| `tests/test_cli_runtime.py` | partial runtime 거절과 evidence opt-in 검사 |

## 실제 처리 흐름

```text
BGR frame
  → OpenCvYuNetDetector.detect(frame)
  → FaceDetection별 bbox + 5 landmarks + confidence
  → ApprovedLocalGalleryAdapter.evaluate(frame, detection)
       → OpenCvSFaceEmbedder.embed(): alignCrop + normalized vector
       → LocalGallery.match(): high similarity 후보
       → ApprovalStore.get(profile_id): explicit current approval 확인
  → GalleryApproval
       → approved=True: 해당 ROI만 mosaic/sticker 적용
       → approved=False: 해당 ROI 원본 보존 + reason code 기록
  → ProcessResult + 선택적 evidence JSON
```

가림을 위해서는 다음이 모두 필요하다.

1. YuNet detection 성공
2. SFace embedding 성공
3. LocalGallery의 high similarity match
4. match된 profile의 ApprovalStore record 존재
5. record가 `approved: true`이고 만료되지 않음

다음은 권한이 아니다: detector confidence, `t_confirm`, `t_keep`, similarity 숫자 단독, gallery 등록 사실, 과거 frame의 승인 결과.

## 다인 frame의 안전성

runtime adapter는 한 frame의 얼굴을 각각 다시 평가한다. A가 승인되고 B가 승인되지 않았다면 A ROI만 가리고 B ROI는 byte-preserved 상태로 남는다. 이 경우 결과는 `is_redacted=True`, `review_required=True`다.

이 절은 `--tracker none` 경로 설명이다. opt-in tracker 경로는 explicit anchor에서만 권한을 시작하고 별도 기하/association gate로 연속성을 제한한다.

## 모델과 runtime option 계약

실제 처리에는 다음 네 option을 모두 제공해야 한다.

```text
--model-dir
--manifest-dir
--gallery-db
--approval-db
```

전부 생략하면 backward-compatible safe stub runtime이 실행되며 redaction은 발생하지 않는다. 일부만 제공하면 오류로 중단한다. manifest directory에서는 정확히 하나의 OpenCV detector와 embedder entry를 찾고, 역할·provider·파일명·SHA-256·preprocessing revision을 검증한다. hash가 맞지 않으면 모델을 load하지 않는다.

`validate-models --manifest-dir ... --model-dir ...` 명령도 같은 hash 검증을 수행한다. `--model-dir`을 생략하면 manifest directory에서 model file을 찾는 기존 호환 동작을 사용한다.

## gallery와 approval 파일

`profiles.json`은 LocalGallery가 소유한다. opaque profile ID, normalized embedding vectors, centroid만 저장하고 이름·원본 이미지·crop은 저장하지 않는다.

`approvals.json`은 별도의 권한 저장소다.

```json
{
  "schema_version": 1,
  "gallery_revision": "local-v1",
  "profiles": {
    "prof-00000000": {
      "approved": true,
      "reason_code": "consented_local_test",
      "expires_at": null
    }
  }
}
```

`approved=false`, 만료된 `expires_at`, 누락된 profile, 손상된 JSON은 모두 no-redaction 또는 runtime 초기화 실패로 처리된다.

## enrollment 동작

`gallery-enroll`은 이제 다음을 수행한다.

1. YuNet이 enrollment image에서 정확히 한 얼굴을 탐지하는지 확인한다.
2. SFace가 landmark 기반 alignment와 L2-normalized embedding을 생성한다.
3. LocalGallery가 dimension, duplicate vector, profile collision을 검증해 저장한다.
4. `--approve --approval-reason`이 있을 때만 approval record를 true로 저장한다.
5. approve option이 없으면 `enrolled_pending_approval` 사유의 false record를 저장한다.

등록과 가림 승인은 분리되어 있다.

## process-image / process-video 및 evidence

verified runtime에서 `process-image`와 `process-video`는 face-by-face approval을 실행하고 output을 만든다. `--dry-run`은 inference와 approval 판단은 수행하지만 결과 이미지·영상은 만들지 않는다.

`--evidence-out`은 명시 opt-in이며 자동 저장하지 않는다. JSON에는 `frame_index`, state, `is_redacted`, `review_required`, approval reason code, 승인 얼굴 수만 기록한다. pixel, crop, embedding vector, 사람 이름은 기록하지 않는다.

## 실제 사용자 테스트 순서

1. 사용자가 확보·검토한 YuNet/SFace ONNX와 정확한 SHA-256 manifest를 local-assets에 둔다.
2. `validate-models`로 manifest와 model hash를 검증한다.
3. 동의된 enrollment image로 `gallery-enroll`을 실행한다.
4. 동의된 test video로 `process-video --dry-run --evidence-out ...`을 먼저 실행한다.
5. evidence의 reason code와 승인 frame 수를 사람이 검토한다.
6. output path를 지정해 실제 결과 영상을 생성한다.

실제 명령과 asset directory 예시는 [REAL_VIDEO_TEST_GUIDE.md](REAL_VIDEO_TEST_GUIDE.md)에 있다.

## 검증 결과

구현 완료 시점에 수행한 검증 결과는 다음과 같다.

```text
pytest: 270 passed
Benchmark A: 9/9
Benchmark B: 5/5
Benchmark C: 5/5
Benchmark D: 1/1
Benchmark E: 3/3
aggregate report JSON parsing: passed
diff whitespace check: passed
```

## 남은 실제 운영 검증

사용자 모델·영상으로 다음은 별도로 확인해야 한다.

- 확보한 ONNX가 설치된 OpenCV version에서 load되는지
- 실제 조명·각도·얼굴 크기에서 detection/embedding 품질이 충분한지
- 여러 얼굴, frame edge, 승인 만료, corrupt gallery에서 dry-run evidence가 기대대로 나오는지
- output codec, FPS, audio 보존 요구 사항

현재 video writer는 처리된 video frame만 MP4로 작성한다. 이후 optional `media.remux` helper와 `--preserve-audio --ffmpeg-path`가 구현됐지만, 최신 실영상 실행 PC의 PATH에는 FFmpeg가 없어 v3 결과에는 사용하지 않았다.

## 관련 문서

- [실제 영상 로컬 테스트 가이드](REAL_VIDEO_TEST_GUIDE.md)
- [코드베이스·함수 참조](CODEBASE_FUNCTION_REFERENCE.md)
- [Config schema](CONFIG_SCHEMA.md)
- [Benchmark protocol](PHASE10_BENCHMARK_PROTOCOL.md)
- [등록 영상 기반 구현 명세](VIDEO_REFERENCE_ENROLLMENT_IMPLEMENTATION_SPEC.md)

## 후속 구현 완료: 등록 영상 coverage 경로

이 보고서 작성 뒤 `gallery-enroll-video`, `VideoEnrollmentService`, `LocalGallery.enroll_many()`, FHD benchmark, output overwrite 보호, optional remux, clean dominant-component enrollment, TAPNext++ 2-pass 처리를 추가했다. 실제 ONNX/체크포인트와 동의된 세로 FHD 영상 smoke test도 완료했다.
