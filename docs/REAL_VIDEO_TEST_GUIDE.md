# 실제 영상 로컬 테스트 가이드

이 절차는 처리 권한이 있는 영상과 명시적으로 동의한 사람의 enrollment 이미지만을 대상으로 한다. 모델, 원본 영상, gallery JSON, approval JSON, 출력 영상, evidence JSON은 저장소에 commit하거나 외부에 업로드하지 않는다.

## 1. 사전 준비

`pip install -e ".[dev]"`로 OpenCV와 CLI 의존성을 설치한다. 자동 모델 다운로드는 하지 않는다. 사용자가 라이선스와 출처를 확인해 확보한 YuNet detector ONNX와 SFace embedder ONNX를 로컬에 둔다.

```text
local-assets/
  models/
    face_detection_yunet.onnx
    face_recognition_sface.onnx
  manifests/
    face-models.json
  gallery/
    profiles.json             # enrollment 뒤 생성
    approvals.json            # enrollment 뒤 생성
  input/
    consented-test.mp4
```

manifest의 `filename`은 model directory 안의 plain filename이어야 한다. 각 entry에는 역할(`detector`/`embedder`), `OpenCV` provider, 실제 SHA-256, input shape, preprocessing revision을 기록한다. placeholder hash를 사용하면 CLI가 fail-closed로 중단한다.

모델이 올바르게 배치되었는지 먼저 검증한다.

```powershell
consented-face-redactor validate-models `
  --manifest-dir .\local-assets\manifests `
  --model-dir .\local-assets\models
```

`--model-dir`을 생략하면 기존 호환성을 위해 manifest directory에서 model file을 찾는다. 실제 처리 전에는 manifest의 SHA-256이 `--model-dir`의 모델 파일과 일치해야 한다.

## 2. 등록 영상으로 다양한 reference를 만들고 명시 승인하기

대상 인물만 포함된 등록 영상이 있다면, 단일 이미지 대신 아래 명령을 사용합니다. 이 명령은 각도를 직접 분류하지 않습니다. sampled frame에서 얻은 embedding의 중복을 줄이고 coverage가 넓은 reference를 하나의 profile로 저장합니다. 다중 얼굴 frame은 자동 등록하지 않습니다.

```powershell
consented-face-redactor gallery-enroll-video `
  --input .\local-assets\input\approved-person-angles.mp4 `
  --gallery-db .\local-assets\gallery\profiles.json `
  --approval-db .\local-assets\gallery\approvals.json `
  --model-dir .\local-assets\models `
  --manifest-dir .\local-assets\manifests `
  --sample-every-n-frames 6 `
  --max-references 64 `
  --approve `
  --approval-reason "consented_local_test" `
  --report-out .\local-assets\gallery\enrollment-report.json
```

먼저 gallery를 바꾸지 않는 검토를 하려면 `--dry-run`을 추가합니다. report에는 frame/crop/vector 대신 후보·중복·다중 얼굴·review frame 통계만 남습니다.

## 3. 한 사람을 단일 이미지로 등록하고 명시 승인하기

enrollment 이미지는 detectable face가 정확히 하나여야 한다. `--approve`는 사용자가 지금 이 profile을 로컬 테스트 대상으로 승인한다는 명시 동작이며, `--approval-reason` 없이는 거절된다.

```powershell
consented-face-redactor gallery-enroll `
  --input .\local-assets\input\approved-person.jpg `
  --gallery-db .\local-assets\gallery\profiles.json `
  --approval-db .\local-assets\gallery\approvals.json `
  --model-dir .\local-assets\models `
  --manifest-dir .\local-assets\manifests `
  --approve `
  --approval-reason "consented_local_test"
```

등록만 하고 승인을 보류하려면 `--approve`와 `--approval-reason`을 생략한다. 이 경우 approval record는 `approved=false`로 저장되며 영상에서 절대 가려지지 않는다.

## 4. 먼저 dry-run 실행

dry-run은 detector, embedding, gallery matching, explicit approval을 모두 실행하지만 출력 영상을 만들지 않는다. 대신 명시한 evidence 파일만 저장한다.

```powershell
consented-face-redactor process-video `
  --input .\local-assets\input\consented-test.mp4 `
  --model-dir .\local-assets\models `
  --manifest-dir .\local-assets\manifests `
  --gallery-db .\local-assets\gallery\profiles.json `
  --approval-db .\local-assets\gallery\approvals.json `
  --dry-run `
  --evidence-out .\local-assets\evidence-dry-run.json
```

evidence에는 frame index, state, redaction 여부, review 여부, reason code, 승인된 얼굴 수만 있다. 프레임 pixel, 얼굴 crop, embedding vector, 이름은 넣지 않는다.

## 5. 결과 영상 생성

dry-run evidence에서 `approved_redaction_frames`와 reason code를 검토한 뒤에만 output을 만든다.

```powershell
consented-face-redactor process-video `
  --input .\local-assets\input\consented-test.mp4 `
  --output .\local-assets\output\consented-test-redacted.mp4 `
  --model-dir .\local-assets\models `
  --manifest-dir .\local-assets\manifests `
  --gallery-db .\local-assets\gallery\profiles.json `
  --approval-db .\local-assets\gallery\approvals.json `
  --evidence-out .\local-assets\output\consented-test-evidence.json
```

`--model-dir`, `--manifest-dir`, `--gallery-db`, `--approval-db`는 모두 함께 제공해야 한다. 일부만 제공하면 CLI는 stub로 조용히 전환하지 않고 오류로 종료한다. 아무 것도 제공하지 않았을 때만 이전 호환성을 위한 safe stub runtime으로 실행되며, 이 경우 가림은 발생하지 않는다.

원본 audio를 결과 MP4에 보존해야 하면, FFmpeg executable을 명시한 opt-in을 사용한다. 이 경로는 먼저 hidden video-only temporary file을 만들고 성공한 remux 뒤에만 destination을 만든다.

```powershell
consented-face-redactor process-video `
  --input .\local-assets\input\consented-test.mp4 `
  --output .\local-assets\output\consented-test-redacted-with-audio.mp4 `
  --model-dir .\local-assets\models `
  --manifest-dir .\local-assets\manifests `
  --gallery-db .\local-assets\gallery\profiles.json `
  --approval-db .\local-assets\gallery\approvals.json `
  --preserve-audio `
  --ffmpeg-path C:\Tools\ffmpeg.exe
```

`--preserve-audio`는 `--dry-run`과 함께 쓸 수 없으며, output path가 input path와 같거나 기존 destination이 있으면 실패한다.

## 6. 기대 결과와 중단 조건

- 명시 승인된 profile과 high similarity가 모두 있을 때만 해당 얼굴 ROI가 가려진다.
- 같은 프레임에 다른 사람이 있어도 그 사람의 approval이 없으면 해당 ROI는 변경하지 않는다.
- empty gallery, low similarity, profile 승인 없음, 승인 만료, gallery/embedding 오류는 모두 no-redaction이다.
- `gallery_evaluation_error`, `similarity_insufficient`, `profile_not_approved`, `approval_expired`는 실패를 숨기는 성공 상태가 아니라 evidence의 reason code다.
- model hash, manifest, gallery, approval file 중 하나가 유효하지 않으면 처리 전에 중단한다.

`--tracker none`은 얼굴별 승인을 매 프레임 다시 평가하고 과거 승인을 재사용하지 않는다. `--tracker tapnextpp`는 별도 2-pass opt-in이며, explicit gallery anchor에서 시작한 위치만 양방향 기하/association 합의가 유지되는 구간에 전파한다. 실제 명령은 [시간축 구현 보고서](TEMPORAL_TRACKING_IMPLEMENTATION_REPORT.md)를 따른다.
