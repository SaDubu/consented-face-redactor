# TAPNext++ 시간축 추적·타원형 모자이크 구현 및 실영상 검증 보고서

## 1. 문서 목적과 현재 상태

이 문서는 `TEMPORAL_TRACKING_AND_STRONG_MOSAIC_WORK_ORDER.md`를 기준으로 local clone에서 실제로 구현한 코드, 함수별 책임, 보안 경계, 테스트, `learn.mp4`/`test.mp4` 실행 결과를 재현 가능하게 기록한다.

- 작업 디렉터리: `<repository-root>`
- 작업 원칙: local clone만 변경
- 원본 `learn.mp4`, `test.mp4`: 이동·변경·덮어쓰기 없음
- 기존 D: 작업본: 접근하거나 변경하지 않음
- Git commit/push: 수행하지 않음
- 과거 `_p10bootstrap.py`, `_p10bootstrap_v2.py`, `_p10test_v3.py`: 사용자의 명시 요청에 따라 local clone에서 삭제 상태로 유지
- 민감 asset/model/output: `local_data/` 또는 `*.mp4` ignore 규칙으로 Git 비추적

현재 결론은 “제공된 test 영상에서 실행 가능한 end-to-end 경로가 완성됐다”이다. clean gallery와 TAPNext++ 시간축 경로를 사용한 최종 관측값은 456/456 frame redaction이다. v4에서는 요청에 따라 강도를 완화하고, bbox를 외접하는 완만한 세로 타원 안에만 mosaic를 합성했다. 이 결과는 해당 영상에 대한 관측이며, 임의의 사람·카메라·가림·교차 장면에서도 100%라는 일반 보장은 아니다.

## 2. 최종 프로그램 동작

~~~text
learn.mp4 (사용자가 target-only라고 지정)
  → YuNet sampled detection
  → SFace embedding
  → 중복 제거
  → embedding graph의 dominant connected component 선택
  → 분리된 작은 component는 자동 등록하지 않고 review 후보로 격리
  → clean LocalGallery + 별도 ApprovalStore 생성

test.mp4
  → Pass 1: 모든 frame의 YuNet detection + SFace gallery evaluation
  → GalleryApproval.approved=True인 얼굴만 identity anchor로 수집
  → 같은 profile anchor 사이의 약한 구간을 TAPNext++로 정방향/역방향 추적
  → 두 bbox 경로가 합의한 frame만 temporal authorization
  → 첫 anchor 이전/마지막 anchor 이후는 한 방향 tracker + 단일 detection association으로 제한 연장
  → Pass 2: 확정된 plan을 다시 읽은 원본 frame에 적용
  → bbox 외접 세로 타원 + 적응형 완화 mosaic
  → sibling temporary MP4 완성 후 atomic replace로 새 결과 publish
  → 원본은 그대로 보존
~~~

tracker가 profile ID를 만들지는 않는다. profile 권한의 시작점은 항상 현재 frame에서 구조적으로 완전한 `GalleryApproval(approved=True, profile_id, gallery_revision, ...)`이다. TAPNext++는 이 anchor의 픽셀 위치를 이어 주고, 기하·visibility·association gate가 실패하면 전파를 중단한다.

## 3. 실제 문제 분석과 수정 경로

### 3.1 최초 실제 결과

기존 frame별 gallery 경로의 관측값은 다음과 같았다.

| 관측 | frame 수 |
| --- | ---: |
| 직접 승인/모자이크 | 209 |
| detection 없음 | 2 |
| detection은 있으나 승인 안 됨 | 245 |
| identity-weak 최장 연속 구간 | 65 |

detector 누락보다 SFace의 frame별 similarity 변동이 끊김의 주원인이었다.

### 3.2 v1: 강한 mosaic + 양방향 TAPNext++

| 항목 | 관측 |
| --- | ---: |
| 직접 anchor | 209 |
| 양방향 보강 | 183 |
| 전체 redaction | 392/456 |
| 미처리 run | 2–28, 419–455 |

2–28은 양방향 경로 불일치로 fail-closed 됐다. 419–455는 마지막 anchor 뒤라 미래 anchor가 없었다.

### 3.3 v2: 마지막 anchor 이후 검증된 한 방향 연장

한 방향 TAPNext++가 유효하고 현재 YuNet detection과 1:1 association이 유지될 때만 연장했다.

| 항목 | 관측 |
| --- | ---: |
| 전체 redaction | 408/456 |
| 후미 연장 | 419–434 |
| 남은 미처리 | 2–28, 435–455 |

frame 435에서 YuNet이 동일 얼굴을 겹치는 두 bbox로 나눠 검출해 기존 “eligible detection이 둘이면 즉시 ambiguity” 규칙이 중단됐다. 상세 점검에서 최적 association과 두 번째 후보의 cost 차이가 충분했고, 이후 모든 frame에서도 tracker visibility 0.95 이상과 주 얼굴 detection association이 유지됐다. 이에 최저 cost가 충분한 margin으로 유일할 때만 duplicate detection을 하나로 선택하도록 수정했다.

### 3.4 등록 오염 발견

contact sheet 검토에서 초기 direct approval bbox가 실제 얼굴 전체가 아니라 귀 쪽의 작은 false crop임을 발견했다. 수치 비교는 다음과 같았다.

| frame 0 후보 | best reference | centroid | 결과 |
| --- | ---: | ---: | --- |
| 귀/부분 false crop | 0.8516 | 0.2835 | 기존에는 승인 |
| 실제 전체 측면 얼굴 | 0.7056 | 0.6266 | 기존에는 거절 |

false crop은 등록 reference #18 하나와만 강하게 일치했다. 기존 gallery의 47개 vector를 graph로 검사하면 similarity 0.45에서 dominant component 36개, 별도 component 9개, singleton 2개로 분리됐다. 기존 enrollment의 `nearest_similarity_min=0.2796`도 오염 가능성을 이미 보여 주고 있었지만, 이전 코드는 `-0.5` 이하만 review로 보내 거의 모든 outlier를 등록했다.

### 3.5 v3 clean enrollment

등록 후보를 similarity graph로 연결하고 가장 큰 component만 자동 등록했다. 중간 각도의 chain으로 연결된 극단 pose는 보존하면서, 작은 분리 component는 review 대상으로 격리한다.

| 항목 | 값 |
| --- | ---: |
| sampled frame | 89 |
| 기존 candidate | 47 |
| clean selected reference | 36 |
| review로 격리 | 11 |
| 직접 anchor | 210 |
| 양방향 합의 frame | 184 |
| 한 방향/edge 연속성 frame | 62 |
| 최종 redaction | 456/456 |
| ambiguous range | 0 |

초기 잘못된 ear anchor가 clean gallery에서 사라졌고, 첫 신뢰 anchor에서 역방향 추적한 위치가 실제 전체 측면 얼굴 detection과 결합됐다.

## 4. 함수별 구현 설명

### 4.1 `config.py`

#### `Config.__init__()`

Config schema v4를 만든다. 기존 effect/state 값 외에 다음 타원형 mosaic 계약을 검증한다.

- `mosaic_grid_cells=12`: 얼굴 짧은 변을 약 12개 block으로 표현해 v3보다 강도를 완화
- `mosaic_min_block_px=10`: 작은 얼굴의 최소 block 크기를 완화
- `mosaic_shape="ellipse"`: 기본 합성 영역을 사각형에서 타원으로 변경
- `mosaic_ellipse_horizontal_scale=1.40`: bbox 반폭 대비 가로 반축
- `mosaic_ellipse_vertical_scale=1.50`: bbox 반높이 대비 세로 반축
- `mosaic_padding_ratio=0.18`: rectangle 호환 모드에서만 사용

#### `Config.from_dict(data, strict=False)`

v1/v2/v3/v4 payload를 읽고 v4 in-memory 객체로 migration한다. 기본은 unknown key를 무시하는 호환 모드이며 `strict=True`만 unknown key를 거절한다.

#### `Config.to_dict()`

schema version과 실제 생성자 field를 모두 직렬화한다.

### 4.2 `effects/mosaic.py`

#### `mosaic_block_size(face_width, face_height, grid_cells, min_block_px)`

얼굴 짧은 변을 `grid_cells`로 나눈 값과 `min_block_px` 중 큰 값을 block 크기로 사용한다. FHD의 큰 얼굴에서 고정 8px보다 훨씬 큰 block을 만든다.

#### `expand_bbox(bbox, frame_shape, padding_ratio)`

bbox를 폭/높이 비율만큼 네 방향으로 확장하고 frame 경계에서 clip한다. 머리카락·턱·회전 얼굴 가장자리 노출을 줄인다.

이 함수는 현재 `mosaic_shape="rectangle"` 호환 경로에서만 사용한다.

#### `ellipse_bounds(bbox, frame_shape, horizontal_scale, vertical_scale)`

raw bbox 중심과 반축 배율로 타원의 bounding rectangle을 계산하고 frame 경계에서 clip한다. 유한 좌표, 세로 우세 비율, bbox 모서리 외접 조건을 검증한다.

#### `ellipse_mask(bbox, frame_shape, horizontal_scale, vertical_scale)`

OpenCV filled ellipse를 사용해 full-frame boolean mask를 만든다. mask 밖은 합성 대상이 아니므로 원본 pixel을 그대로 유지할 수 있다.

#### `MosaicEffect.render(frame, roi)`

효과 영역을 작은 grid로 `INTER_AREA` downscale한 뒤 `INTER_NEAREST`로 원래 크기로 확대한다. rectangle은 전체 ROI를 바꾸고, ellipse는 타원 mask 내부 pixel만 mosaic로 교체한다. 입력 frame은 변경하지 않고 새 배열을 반환한다.

### 4.3 `tracking/protocol.py`

#### `PointTracker`

vendor 독립 protocol이다.

- `initialize(frame, frame_index, query_points)`: 승인 anchor의 query point로 recurrent state 시작
- `update(frame, frame_index)`: 다음 연속 frame의 point/visibility 반환
- `reset()`: video/segment state 제거
- `model_id`: evidence용 모델 식별자

이 protocol에는 profile ID나 승인 함수가 없다.

### 4.4 `tracking/types.py`

- `PointTrackResult`: read-only float32 point와 visibility
- `SimilarityTransform`: scale/rotation/translation과 inlier 수
- `TrackedFaceBox`: tracker가 예측한 bbox와 위치 근거
- `TrackAuthorization`: gallery에서 유래한 profile lease
- `TrackFrameDecision`: frame별 최종 bbox/권한/reason
- `BboxValidation`: 기하 gate 결과

배열은 생성 시 복사하고 write-protect해 후속 코드가 evidence를 몰래 바꾸지 못하게 한다.

### 4.5 `tracking/geometry.py`

#### `seed_face_points()`

YuNet 5 landmark와 bbox 안쪽 4×4 grid를 결합해 기본 21 point를 만든다. 경계점만 사용하지 않아 표정과 edge clipping에 덜 민감하다.

#### `estimate_similarity_transform()`

visibility가 낮은 point를 제거하고 SVD similarity transform을 맞춘다. residual median/MAD 기준으로 outlier를 한 번 제거해 한 점이 bbox를 끌고 가는 현상을 줄인다.

#### `transform_bbox()`

이전 bbox 네 모서리에 scale/rotation/translation을 적용한 뒤 axis-aligned enclosure를 반환한다.

#### `validate_tracked_bbox()`

visible ratio, frame당 scale 변화, center 이동, frame 밖 이탈을 검사한다. 실패는 reason code를 가진다.

### 4.6 `tracking/tapnextpp.py`

#### `_load_tapnextpp_factory()`

공식 local TAPNet source가 완전한지 확인하고 optional torch/torchvision을 lazy import한다. 패키지 import만으로 2.35GB 모델을 load하지 않는다.

#### `TapNextPlusPlusAdapter.__init__()`

manifest가 `role=tracker`, `provider=PyTorch`인지 확인하고 checkpoint filename/SHA-256을 검증한 후에만 공식 `TAPNextPP.from_checkpoint()`를 호출한다.

#### `initialize()` / `update()`

BGR uint8 frame, frame 범위 안의 query point, 연속 frame index를 검증한다. 공식 wrapper가 반환한 위치/visibility를 `PointTrackResult`로 변환한다. frame index가 하나라도 건너뛰면 state 의미가 달라지므로 거절한다.

실제 checkpoint:

| 항목 | 값 |
| --- | --- |
| model | Google DeepMind TAPNext++ VOTSp2026 512 |
| file | `tapnextpp_512.ckpt` |
| SHA-256 | `6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f` |
| license | Apache-2.0 |
| runtime | torch 2.11.0+cu128, torchvision 0.26.0+cu128, einops 0.8.2 |

실제 RTX 5070 Ti smoke test:

- checkpoint load: 약 3.96초
- 첫 FHD frame: 약 0.40초
- 다음 frame: 약 0.074초
- GPU peak allocation: 약 2.43GiB

### 4.7 `tracking/association.py`

#### `association_cost()`

tracker bbox와 YuNet detection의 IoU, frame diagonal 대비 center 거리, scale ratio를 계산한다. hard gate 밖 pair는 assignment 후보가 아니다.

#### `associate_tracks_to_detections()`

최대 match 수를 우선하고 그 안에서 total cost가 최소인 one-to-one assignment를 deterministic하게 계산한다. 한 detection을 두 track이 소유하지 못한다.

#### `detect_crossing_ambiguity()`

겹치는 track의 좌우 순서가 이전 assignment와 뒤집히면 ambiguity를 기록한다.

### 4.8 `tracking/authorization.py`

- `create_authorized_track()`: 현재 `approved=True`에서만 lease 생성
- `refresh_authorized_track()`: 같은 profile의 새 explicit approval만 갱신
- `may_propagate_authorization()`: 기존 lease의 연속성만 판단
- `revoke_track_authorization()`: ambiguity/loss/시간 초과 뒤 비가역 취소

tracker-only 입력으로 profile ID를 만드는 함수가 없다.

### 4.9 `tracking/bidirectional.py`

#### `collect_identity_anchors()`

직접 승인, profile ID, gallery revision이 모두 있는 얼굴만 anchor로 만든다.

#### `split_anchor_segments()`

같은 profile/revision의 연속 anchor 중 설정된 최대 gap 안의 약한 구간만 만든다. 이미 연속 승인된 frame은 불필요하게 추적하지 않는다.

#### `track_segment_forward()` / `track_segment_backward()`

anchor landmark/grid point를 시작으로 실제 frame 순서 또는 역순 frame을 TAPNext++에 공급한다. 각 frame에서 robust bbox를 복원하고 기하 gate를 적용한다.

#### `reconcile_bidirectional_paths()`

동일 profile의 정/역방향 bbox IoU가 정책 이상일 때만 두 bbox 평균을 최종 위치로 사용한다. 한쪽 실패나 불일치는 자동 성공으로 바꾸지 않는다.

#### `build_redaction_track_plan()`

직접 anchor와 합의 decision을 정렬해 불변 plan으로 만든다. 영상 frame 수, profile 집합, ambiguous range를 함께 보존한다.

### 4.10 `temporal_video_processor.py`

#### `TemporalVideoProcessor.analyze(source_path)`

첫 pass에서 detector/gallery를 모든 frame에 적용하고 pixel/crop/embedding을 evidence에 저장하지 않는다. anchor 구간별로 필요한 frame만 다시 읽어 TAPNext++ 정/역방향 경로를 만든다.

첫/마지막 anchor 밖은 최대 90 frame까지 한 방향으로 검토한다. tracker-only streak는 최대 12 frame이며, detection이 있으면 정확히 하나의 association 후보 또는 충분한 cost margin의 최적 duplicate 후보가 있어야 한다. 다른 approved profile과 충돌하면 즉시 중단한다.

#### `TemporalVideoProcessor.render(source_path, destination, plan)`

원본을 처음부터 다시 읽고 plan의 authorized bbox만 mosaic한다. 입력과 출력 경로 동일, 기존 destination, frame 수 불일치를 writer publish 전에 거절한다. sibling temporary 파일에 모든 frame을 성공적으로 기록한 후 `os.replace()`로 최종 결과를 공개한다.

### 4.11 `video_enrollment.py`

#### `extract_enrollment_candidate()`

sampled frame에 face detection이 정확히 하나일 때만 embedding 후보를 만든다. 0개, 다중 검출, 너무 작은 얼굴, embedding 오류는 skip reason으로 남긴다.

#### `select_diverse_references()`

인접 중복 vector를 줄인 뒤 embedding similarity graph를 구성한다. `minimum_cluster_similarity=0.45` 이상 edge로 이어진 가장 큰 component를 target view trajectory로 선택하고 작은 분리 component를 review로 보낸다. dominant component가 max reference보다 크면 farthest-point coverage로 축약한다.

#### `VideoEnrollmentService.collect/select/enroll()`

영상 scan, 후보 선택, gallery batch 저장을 분리한다. dry-run은 report만 만들며 gallery/approval을 변경하지 않는다.

### 4.12 `cli.py`

#### `_load_verified_tracker_entry()`

manifest 전체에서 정확히 하나의 PyTorch TAPNext++ entry를 찾고 checkpoint checksum을 검증한다.

#### `_build_tracker_runtime()`

명시된 source/checkpoint/device/policy를 검증하고 adapter를 만든다.

#### `_cmd_process_video_temporal()`

`--tracker tapnextpp`에서만 2-pass 경로를 선택한다. tracker가 `none`이면 기존 frame-by-frame 경로를 유지한다. 분석/렌더 요약과 frame reason을 opt-in evidence에 기록한다. `--evaluation-evidence`가 있을 때만 bbox/profile/visibility를 추가한다.

## 5. 실제 재현 명령

### 5.1 모델 검증

~~~powershell
python -m consented_face_redactor.cli validate-models `
  --manifest-dir local_data/manifests `
  --model-dir local_data/models
~~~

실측: 1개 manifest의 3개 모델(YuNet, SFace, TAPNext++) 검증 성공.

### 5.2 clean gallery dry-run

~~~powershell
python -m consented_face_redactor.cli gallery-enroll-video `
  --input learn.mp4 `
  --gallery-db local_data/gallery/gallery_clean_v2.json `
  --approval-db local_data/gallery/approvals_clean_v2.json `
  --model-dir local_data/models `
  --manifest-dir local_data/manifests `
  --sample-every-n-frames 6 `
  --minimum-cluster-similarity 0.45 `
  --dry-run `
  --report-out local_data/evidence/learn_enrollment_clean_v2_dry_run.json
~~~

실측: 36개 선택, 파일 변경 없음.

### 5.3 clean gallery 생성

~~~powershell
python -m consented_face_redactor.cli gallery-enroll-video `
  --input learn.mp4 `
  --gallery-db local_data/gallery/gallery_clean_v2.json `
  --approval-db local_data/gallery/approvals_clean_v2.json `
  --model-dir local_data/models `
  --manifest-dir local_data/manifests `
  --sample-every-n-frames 6 `
  --minimum-cluster-similarity 0.45 `
  --approve `
  --approval-reason operator_approved_clean_learn_mp4_20260826 `
  --report-out local_data/evidence/learn_enrollment_clean_v2.json
~~~

### 5.4 최종 영상 처리

~~~powershell
python -m consented_face_redactor.cli process-video `
  --input test.mp4 `
  --output local_data/output/test_target_mosaic_tapnextpp_v3_clean.mp4 `
  --model-dir local_data/models `
  --manifest-dir local_data/manifests `
  --gallery-db local_data/gallery/gallery_clean_v2.json `
  --approval-db local_data/gallery/approvals_clean_v2.json `
  --detector-score-threshold 0.5 `
  --tracker tapnextpp `
  --tracker-source-dir local_data/vendor/tapnet `
  --tracker-device cuda `
  --tracking-mode bidirectional `
  --tracker-max-gap-frames 90 `
  --tracker-minimum-path-iou 0.30 `
  --tracker-minimum-visible-ratio 0.60 `
  --evidence-out local_data/evidence/test_tapnextpp_v3_clean.json
~~~

## 6. 검증 결과

### 6.1 자동 검증

~~~text
compileall: passed
pytest: 321 passed
git diff --check: passed
Benchmark A: 9/9
Benchmark B: 5/5
Benchmark C: 5/5
Benchmark D: 2/2
Benchmark E: 3/3
Aggregate JSON: 24/24, 5 categories, parsing passed
~~~

Category B는 mosaic padding 도입 후 raw detection ROI만 effect 범위라고 가정해 처음 실패했다. benchmark assertion을 실제 padded renderer contract로 바꿨고, ROI 밖 모든 pixel 보존 검사가 다시 통과했다. 실패를 삭제하거나 성공으로 바꾸지 않았다.

### 6.2 실제 영상

| 항목 | 입력 | 최종 출력 |
| --- | ---: | ---: |
| frame | 456 | 456 |
| FPS | 30 | 30 |
| width | 1080 | 1080 |
| height | 1920 | 1920 |
| FHD 이상 | yes | yes |

최종 evidence:

- `local_data/evidence/test_tapnextpp_v3_clean.json`
- direct anchor: 210
- bidirectional consensus: 184
- total redacted frame: 456
- review-required frame: 0
- ambiguous range: 0
- analysis elapsed: 약 61.33초
- measured analysis throughput: 약 7.43fps
- p50 face analysis latency: 약 31.50ms
- p95 face analysis latency: 약 38.50ms

출력:

- `local_data/output/test_target_mosaic_tapnextpp_v3_clean.mp4`
- `local_data/evidence/tapnextpp_v3_clean_contact_sheet.jpg`

contact sheet에서 frame 0, 1, 15, 28, 29, 418, 440, 455를 원본과 비교했고 측면/정면 얼굴 전체가 강한 mosaic로 덮였다.

### 6.3 v4 타원형·완화 mosaic 재실행

사용자 피드백에 따라 identity/tracking plan은 그대로 유지하고 renderer의 공간·강도 계약만 변경했다.

| 항목 | v3 | v4 기본 |
| --- | ---: | ---: |
| shape | padded rectangle | circumscribed vertical ellipse |
| grid cells | 8 | 12 |
| minimum block | 16px | 10px |
| horizontal half-axis scale | 해당 없음 | 1.40 |
| vertical half-axis scale | 해당 없음 | 1.50 |

타원은 단순히 bbox 안에 내접하지 않는다. `1 / 1.40² + 1 / 1.50² < 1`을 만족하므로 bbox 네 모서리가 타원 안에 들어가고, 얼굴 bbox 전체를 감싼다. 세로 반축은 가로 반축보다 약 7% 길어 원형보다 세로로 길지만 과도한 캡슐 모양은 피한다. 실제 합성은 boolean mask 안에서만 일어나므로 타원 밖 pixel은 renderer 입력과 동일하다.

재실행 명령의 핵심 차이는 새 출력·evidence 이름과 `--evaluation-evidence`다. mosaic 설정은 Config v4 기본값을 사용했다.

~~~powershell
python -m consented_face_redactor.cli process-video `
  --input test.mp4 `
  --output local_data/output/test_target_mosaic_tapnextpp_v4_ellipse.mp4 `
  --model-dir local_data/models `
  --manifest-dir local_data/manifests `
  --gallery-db local_data/gallery/gallery_clean_v2.json `
  --approval-db local_data/gallery/approvals_clean_v2.json `
  --detector-score-threshold 0.5 `
  --tracker tapnextpp `
  --tracker-source-dir local_data/vendor/tapnet `
  --tracker-device cuda `
  --tracking-mode bidirectional `
  --tracker-max-gap-frames 90 `
  --tracker-minimum-path-iou 0.30 `
  --tracker-minimum-visible-ratio 0.60 `
  --evaluation-evidence `
  --evidence-out local_data/evidence/test_tapnextpp_v4_ellipse.json
~~~

| v4 관측 | 값 |
| --- | ---: |
| frame | 456/456 |
| 직접 gallery anchor | 210 |
| 양방향 전파 | 184 |
| redacted frame | 456 |
| review-required frame | 0 |
| ambiguous range | 0 |
| 입력/출력 해상도 | 1080×1920 |
| 입력/출력 FPS | 30 |
| 분석 경과 시간 | 약 61.36초 |
| 분석 처리율 | 약 7.43fps |
| p50 / p95 분석 latency | 약 32.38 / 39.40ms |

최종 산출물:

- `local_data/output/test_target_mosaic_tapnextpp_v4_ellipse.mp4`
- `local_data/evidence/test_tapnextpp_v4_ellipse.json`
- `local_data/evidence/tapnextpp_v4_ellipse_comparison.jpg`

비교 sheet의 frame 0, 15, 29, 218, 418, 440, 455를 원본/v3/v4 세 열로 검사했다. v4는 측면·정면 모두 얼굴을 타원으로 감싸고, v3의 큰 사각형보다 배경을 덜 가리며, 더 작은 block으로 강도가 완화됐다.

## 7. 안전 경계와 아직 남은 검증

1. tracker confidence나 detection confidence는 identity 권한이 아니다.
2. clean enrollment component 선택은 false crop 오염 방어이지 사람 승인 대체가 아니다. review 후보 11개를 사람이 확인하면 더 좋다.
3. 456/456은 대상자가 주로 혼자 등장하는 제공 영상 결과다. 실제 두 사람이 겹치거나 교차하는 consented 영상으로 identity transfer 회귀를 추가해야 한다.
4. TAPNext++는 무거운 모델이다. 기존 요구의 “tracker 없음”과 현재 “끊김 개선을 위해 SOTA tracker 사용”은 서로 다른 운영 mode로 CLI에서 분리했다. `--tracker none`은 기존 가벼운 경로다.
5. 분석은 약 7.43fps로 실시간 30fps보다 느리지만 저장 영상 batch 처리 요구에는 동작한다. 성능 수치는 pass/fail threshold가 아니다.
6. FFmpeg executable을 현재 PATH에서 찾지 못해 최종 v3/v4는 video stream만 썼다. 원본에 audio가 있고 보존이 필요하면 `--preserve-audio --ffmpeg-path ...`를 명시해야 한다.
7. checkpoint와 공식 source는 자동 다운로드하지 않는다. source/license/checksum을 사용자가 통제한다.
8. 결과 destination은 기존 파일을 덮어쓰지 않는다. v1/v2/v3/v4를 별도 이름으로 남겨 비교 가능하다.
9. local evidence에 bbox를 넣으려면 `--evaluation-evidence`를 명시한다. 기본 evidence는 pixel/crop/embedding을 저장하지 않는다.
10. model/gallery/output은 Git에 포함하지 않는다. 변경 코드는 아직 commit/push되지 않았다.

## 8. 검토에서 도출된 12개 핵심 인사이트

1. frame coverage만 보면 209→392 개선이 성공처럼 보이지만 contact sheet가 ear-only bbox 오류를 발견했다. 정량과 시각 검토는 반드시 함께 있어야 한다.
2. “등록 영상에 target만 등장”과 “모든 detector crop이 target 얼굴”은 다르다. 한 사람의 귀·배경·부분 형상이 false face로 검출될 수 있다.
3. reference 수가 많다고 gallery가 강해지지 않는다. 오염된 한 reference는 best-reference max 정책에서 강한 false anchor를 만들 수 있다.
4. centroid 하나만 쓰면 극단 pose recall이 떨어지고 best reference 하나만 쓰면 오염에 취약하다. view graph의 연결성이 둘 사이의 실용적인 중간 증거다.
5. 양방향 tracker 합의는 약한 similarity gap을 채우면서도 한 방향 drift를 발견한다. 저장 영상이므로 미래 frame을 활용하는 것이 합리적이다.
6. direct identity authority와 bbox localization 품질은 별개다. 승인 결과가 맞는 profile을 뜻하더라도 detector bbox가 얼굴 전체를 덮는지는 별도 검증이 필요하다.
7. YuNet의 중복 bbox는 곧바로 “두 사람”을 뜻하지 않는다. one-to-one 최적 cost와 second-best margin을 함께 봐야 한다.
8. 반대로 가장 가까운 bbox 하나만 고르면 사람 교차에서 identity transfer 위험이 있다. hard gate와 ambiguity margin을 모두 유지해야 한다.
9. 강한 mosaic padding은 effect locality contract도 바꾼다. benchmark가 raw bbox 밖 변경을 실패로 보고한 것은 테스트가 실제 renderer 계약을 따라가야 한다는 증거다.
10. tracker를 library import 시 즉시 load하면 기본 사용자도 거대한 CUDA 의존성을 떠안는다. lazy optional adapter가 기능 격리에 중요하다.
11. 분석 plan과 렌더링을 분리하면 identity 판단이 output writer 실패나 codec 상태에 영향을 받지 않는다. 결과 publish도 atomic하게 만들 수 있다.
12. 100% frame coverage는 끝이 아니라 새 검증 시작점이다. 다음 품질 gate는 real multi-person crossing, 장시간 완전 가림, 재등장, 조명 급변에서 승인 profile이 다른 사람에게 넘어가지 않는지다.
