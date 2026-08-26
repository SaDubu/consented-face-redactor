# 코드베이스·함수 참조

이 문서는 현재 로컬 작업 트리의 `src/consented_face_redactor`와 공개 CLI를 코드 기준으로 검토한 결과다. 설계 의도만 설명하지 않고, **현재 연결되어 실제로 실행되는 동작**과 아직 연결되지 않은 부분을 구분한다. 이 문서 작성 시점의 변경은 모두 로컬에만 있으며 commit/push하지 않았다.

> 2026-08-26 보강: 기존 frame-by-frame 경로에 더해 `--tracker tapnextpp` opt-in 2-pass 경로가 구현됐다. 시간축 함수의 최신 실측·계약은 [TAPNext++ 구현 보고서](TEMPORAL_TRACKING_IMPLEMENTATION_REPORT.md)를 함께 본다.

## 1. 프로그램이 하는 일

`consented-face-redactor`는 동영상 또는 이미지 프레임에서 얼굴을 탐지하고, 갤러리가 **명시적으로 승인한** 대상만 가리는 것을 목표로 한 Python 패키지다. 핵심 안전 규칙은 다음 한 문장으로 요약된다.

> detector confidence나 similarity 숫자만으로는 얼굴을 가리지 않으며, `GalleryApproval.approved is True`만이 `CONFIRMED`와 redaction을 허용한다.

현재 구현은 안전한 상태 전이, 로컬 갤러리 저장·검증, mosaic/sticker 효과, OpenCV 입출력, benchmark를 제공한다. `--model-dir`, `--manifest-dir`, `--gallery-db`, `--approval-db`를 모두 제공하면 manifest hash를 확인한 YuNet/SFace와 명시 승인 adapter를 CLI에 주입한다. 옵션을 전혀 제공하지 않은 이전 호환 경로는 safe no-redaction stub이며, 일부만 제공하면 오류로 중단한다.

### 전체 처리 흐름

```text
파일/카메라 프레임 (BGR ndarray)
        │
        ▼
OpenCvFrameSource ──► RedactionPipeline.process_frame()
                           │
             detector adapter가 있을 때만 탐지
                           │
                           ▼
                  CANDIDATE + telemetry
                           │
              gallery.embed()/match() (선택적)
                           │
                 GalleryApproval.approved == True?
                     │ yes                    │ no/오류/형식오류
                     ▼                        ▼
             CONFIRMED + effect          CANDIDATE, review 필요
                     │
                     ▼
           MosaicEffect 또는 StickerEffect
                     │
                     ▼
          ProcessResult(result_frame, is_redacted, state, review_required)
```

`CONFIRMED` 상태에서 탐지가 사라지면 `LOST`가 되고, `track_lost_ttl_frames`가 지난 뒤 `EXPIRED`가 된다. `LOST`와 `EXPIRED`에서는 가림을 계속 유지하지 않는다.

## 2. 패키지와 공개 API

### `consented_face_redactor.__init__`

패키지 최상위에서 다음을 재노출한다.

| 이름 | 역할 |
| --- | --- |
| `ApprovalRecord`, `ApprovalStore` | profile vector와 분리된 명시 approval JSON 계약 |
| `ApprovedLocalGalleryAdapter` | SFace embedding, LocalGallery similarity, ApprovalStore를 순서대로 결합하는 실제 runtime adapter |
| `GalleryApproval` | 갤러리가 내린 불변 승인/거절 결과 |
| `GalleryApprovalProtocol` | gallery adapter가 제공해야 하는 `embed`, `match` 형태 |

벤치마크 공개 API에는 `categories`, `output_path`, `benchmark_runner.run()`이 없다. 결과 저장은 호출자가 반환 JSON을 명시적으로 저장할 때만 발생한다.

## 3. 설정: `config.py`

### `Config`

불변 dataclass가 아니라 일반 class이며, 기본값과 JSON 형태의 설정 계약을 담당한다. 현재 `SCHEMA_VERSION = 4`다.

주요 필드는 effect 설정, `recheck_interval_frames`, `track_lost_ttl_frames`, 그리고 telemetry calibration 값인 `t_confirm`, `t_keep`이다. 마지막 두 값은 현재 권한 임계값으로 사용되지 않는다.

| 함수/메서드 | 입력과 반환 | 실제 기능 |
| --- | --- | --- |
| `Config.default()` | 없음 → `Config` | 모든 기본값으로 새 설정을 만든다. |
| `Config.from_dict(data, strict=False)` | dict → `Config` | v1(버전 필드 없음), v2, v3, v4 payload를 읽는다. 기본 모드에서는 알 수 없는 key를 무시한다. `strict=True`에서는 unknown key를 `ValueError`로 거절한다. schema version이 지원 범위 밖이거나 data가 dict가 아니어도 거절한다. |
| `Config.to_dict()` | 없음 → `dict` | 모든 Config 필드와 `schema_version`을 포함한 독립된 직렬화 dict를 만든다. |

### 설정 해석의 안전 경계

- `t_confirm`, `t_keep`은 candidate confidence의 관측/향후 calibration 값이다. `RedactionPipeline`은 이 숫자로 identity 승인을 하지 않는다.
- `from_dict(..., strict=False)`의 unknown key 무시는 기존 JSON과의 호환성을 위한 의도된 기본값이다. 오타 검출이 필요하면 CLI의 `--strict-config` 또는 직접 `strict=True`를 사용한다.
- v4 mosaic 필드의 type/range와 타원 기하를 검증한다. `mosaic_grid_cells`는 2–64, rectangle padding은 0–0.5, 최소 block은 1–256, 타원 반축 배율은 1.0–3.0이다. 기본은 grid 12, 최소 block 10, 가로/세로 반축 1.40/1.50의 외접 세로 타원이다. 기존 telemetry/state 필드의 의미는 바꾸지 않았다.

## 4. 승인 계약: `gallery_approval.py`

### `GalleryApproval`

`@dataclass(frozen=True, slots=True)` 결과 객체다. 불변으로 만들어 pipeline이 결과를 중간에 바꾸지 못하게 한다.

| 필드 | 의미 |
| --- | --- |
| `approved: bool` | 유일한 권한 신호. 정확히 `True`일 때만 confirmed/redaction 가능 |
| `profile_id: str | None` | 갤러리 내부의 불투명 profile 식별자 |
| `similarity: float | None` | 기록/분석용 유사도이며 pipeline의 권한 판단에 사용하지 않음 |
| `reason_code: str` | `approved`, `empty_gallery`, `similarity_insufficient`, `gallery_match_error` 같은 판정 사유 |
| `gallery_revision: str | None` | 갤러리 데이터/adapter revision의 추적용 값 |

| 함수/메서드 | 입력과 반환 | 실제 기능 |
| --- | --- | --- |
| `GalleryApproval.denied(reason_code, ...)` | 사유 및 선택 metadata → `GalleryApproval` | `approved=False`인 표준 거절 객체를 만든다. gallery 오류·형식 오류를 일관된 관측 결과로 바꿀 때 사용한다. |

### `GalleryApprovalProtocol`

구현체가 제공해야 할 구조적 타입 계약이다.

| 메서드 | 계약 |
| --- | --- |
| `embed(frame)` | 프레임에서 gallery query용 embedding을 만들거나 `None`을 반환한다. |
| `match(embedding)` | 반드시 `GalleryApproval`을 반환한다. tuple/list/숫자는 유효한 승인 신호가 아니다. |

## 5. 핵심 상태 머신: `pipeline.py`

### 데이터 타입

| 타입 | 필드 | 용도 |
| --- | --- | --- |
| `DetectionResult` | `bboxes`, `landmarks`, `confidences` | detector 결과를 vendor 형식과 분리한 내부 표준 형태 |
| `EmbeddingResult` | `vectors`, `model_revision` | embedding 출력의 표준 형태. 현재 pipeline의 주 경로에서는 직접 사용하지 않음 |
| `MatchDecision` | `is_target`, `confidence`, `reason_code`, `profile_id` | 과거/보조 표현. 권한은 `GalleryApproval`으로 통일되어야 함 |
| `TrackState` | `UNSEEN`, `CANDIDATE`, `CONFIRMED`, `LOST`, `EXPIRED` | 단일 track의 상태 열거형 |
| `ProcessResult` | `result_frame`, `is_redacted`, `track_state`, `review_required` | 한 프레임 처리 후 반환값 |

### module helper

| 함수 | 실제 기능 |
| --- | --- |
| `_import_effects()` | import 비용과 순환 의존을 줄이기 위해 `StickerEffect`, `MosaicConfig`, `FaceBox`를 처리 시점에 불러온다. |
| `_build_sticker_effect(config, scale_factor, anchor)` | config의 PNG bytes, 배율, anchor로 `StickerEffect`를 생성한다. |
| `_apply_effect_to_bbox(frame, bbox, mode, config, effect_proxy)` | 좌표를 프레임 안으로 clip한 뒤 mosaic/sticker/none을 적용한다. 입력을 바꾸지 않고 새 배열을 반환한다. 알 수 없는 mode도 효과 없이 복사본을 반환한다. |

### `RedactionPipeline`

생성자는 `config`와 선택적인 `detector`, `gallery`를 받는다. dict config는 `Config.from_dict`로 변환한다. detector가 없으면 빈 탐지 결과가 기본값이다.

| 속성/메서드 | 입력과 반환 | 실제 기능 |
| --- | --- | --- |
| `current_track_state` | property → `TrackState` | 현재 내부 상태를 읽는다. |
| `detector_requires_bgr_input` | property → bool | 연결된 detector의 `model_id == "yunet"` 여부로 저장된 색상 처리 플래그를 읽는다. |
| `has_detector` | property → bool | detector가 주입되었는지 알려 준다. |
| `last_gallery_approval` | property → `GalleryApproval` | 마지막 gallery 결과를 노출한다. 이 property를 읽는 행위는 권한을 부여하지 않는다. |
| `last_frame_approvals` | property → tuple | 최신 프레임의 얼굴별 approval 목록을 노출한다. evidence에는 reason code와 승인 수만 기록한다. |
| `telemetry_snapshot` | property → dict | candidate confidence 목록, gallery 재검사 수, 마지막 reason/revision을 관측용으로 반환한다. |
| `_run_detector(frame)` | ndarray → list | detector가 없으면 빈 list, 있으면 BGR frame 그대로 `detect`에 전달한다. YuNet adapter와 FrameSource의 색상 계약은 BGR다. |
| `_evaluate_detection(frame, detection)` | → `GalleryApproval` | production adapter의 `evaluate(frame, detection)`을 호출한다. adapter 예외 또는 잘못된 반환은 거절로 바꾼다. |
| `_process_face_by_face_approvals(...)` | → `ProcessResult` | 검출된 각 얼굴을 독립 승인하고, `approved=True`인 ROI만 effect 처리한다. 한 사람의 승인이 다른 얼굴에 전파되지 않는다. |
| `process_frame(frame, frame_index, timestamp, state)` | frame → `ProcessResult` | 아래 상태 전이를 수행하는 공개 중심 함수다. `timestamp`와 외부 `state`는 현재 호환성 인자이며 내부 frame index/state가 권위다. |
| `save_track_state()` | 없음 → dict | schema v2 snapshot: 상태, frame index, `lost_frame_index`, `confirmed_profile_id`를 만든다. |
| `load_track_state(snapshot)` | dict → 없음 | v1/v2 snapshot을 검증해 복원한다. v1의 `CONFIRMED`/`LOST`는 loss 시점·profile 근거가 없으므로 `CANDIDATE`로 degrade한다. fail-open 복원은 하지 않는다. |

### `process_frame`의 상태 전이

1. detector 결과가 없으면:
   - `CONFIRMED → LOST`이며 결과는 **가리지 않고** review-required다.
   - `LOST`는 TTL 전까지 유지되며 TTL 이상이면 `EXPIRED`가 된다.
   - `CANDIDATE → LOST`, `UNSEEN`은 그대로 no-redaction이다.
2. 탐지가 있으면 confidence는 telemetry에 기록되고 상태는 candidate 평가로 들어간다.
3. production `ApprovedLocalGalleryAdapter`가 있으면 각 detection을 매 프레임 SFace embedding → LocalGallery match → explicit approval 순으로 평가한다. 이 규칙은 `RedactionPipeline` 단독/`--tracker none` 경로다. tracker opt-in video는 별도 `TemporalVideoProcessor`가 explicit anchor 사이의 위치 연속성만 전파한다.

## 시간축 추적 추가 모듈

| 파일/함수 | 실제 기능 |
| --- | --- |
| `tracking.protocol.PointTracker` | profile 권한을 포함하지 않는 point tracker 계약 |
| `tracking.tapnextpp.TapNextPlusPlusAdapter` | manifest/hash 검증 뒤 공식 TAPNext++를 lazy load하고 연속 BGR frame을 추적 |
| `tracking.geometry.seed_face_points()` | 5 landmark와 inset grid를 query point로 생성 |
| `tracking.geometry.estimate_similarity_transform()` | visible point의 robust scale/rotation/translation 계산 |
| `tracking.geometry.validate_tracked_bbox()` | visibility, scale, 이동, frame 경계 fail-closed gate |
| `tracking.association.associate_tracks_to_detections()` | hard gate 뒤 deterministic one-to-one bbox association |
| `tracking.authorization.create_authorized_track()` | 현재 explicit GalleryApproval에서만 profile lease 생성 |
| `tracking.bidirectional.reconcile_bidirectional_paths()` | 같은 profile의 정/역방향 bbox가 합의할 때만 중간 frame 승인 |
| `TemporalVideoProcessor.analyze()` | detection/gallery metadata와 tracker를 사용해 불변 plan 생성 |
| `TemporalVideoProcessor.render()` | plan만 소비해 임시 MP4를 완성한 뒤 atomic publish |
4. `approval.approved is True`인 경우에만 그 **해당 bounding box**에 effect를 적용한다.
5. 그 밖의 모든 경우(승인 없음, empty gallery, no match, exception, malformed return)는 해당 ROI를 보존한다. 하나라도 거절된 얼굴이 있으면 `review_required=True`다.

이 분리는 detector confidence가 높아도 허가되지 않은 사람을 가리지 않는다는 안전 계약이다.

### snapshot 호환성

v2 snapshot은 `lost_frame_index`와 confirmed profile reference를 저장해 process restart 뒤 LOST TTL을 계산할 수 있다. legacy v1은 시간 근거가 없으므로 confirmed 권한을 유지하지 않는다. snapshot이 깨졌거나 state/schema가 유효하지 않으면 `ValueError`를 내는 것이 호출자가 안전하게 새 상태로 시작하도록 하는 경계다.

## 6. domain 및 효과

### `domain/types.py`

| 타입/함수 | 실제 기능 |
| --- | --- |
| `FaceBox(x1, y1, x2, y2)` | 정수 얼굴 ROI. `width`, `height`, `area`, `as_slice()` property로 크기와 NumPy slice를 제공한다. 생성 시 좌표 순서/음수 여부를 validation한다. |
| `FiveLandmarks` | 좌·우 눈, 코, 입 좌·우 좌표를 저장한다. `eye_angle`은 sticker 회전 계산용 각도다. |
| `MosaicConfig` | mosaic block 크기, rectangle/ellipse shape, 타원 반축 배율을 보관한다. 저수준 직접 생성의 shape 기본은 호환성을 위해 rectangle이다. |
| `StickerConfig` | sticker byte/scale/anchor 정책을 보관한다. |
| `BlurConfig` | blur 설정 데이터 형태이나 현재 redaction pipeline의 effect 분기에는 연결되지 않았다. |

### `effects/mosaic.py`의 `MosaicEffect`

| 함수/메서드 | 실제 기능 |
| --- | --- |
| `__init__(config)` | `MosaicConfig`를 보관한다. |
| `mosaic_block_size(face_width, face_height, ...)` | 얼굴 짧은 변, grid 수, 최소 block으로 적응형 block 크기를 계산한다. |
| `expand_bbox(bbox, ...)` | rectangle 호환 mode용 padding bounds를 계산하고 frame 경계로 clip한다. |
| `ellipse_bounds(bbox, ...)` | bbox를 외접하는 타원의 bounding rectangle을 계산하고 기하 조건을 검증한다. |
| `ellipse_mask(bbox, ...)` | 타원 내부만 `True`인 full-frame boolean mask를 만든다. |
| `render(frame, face)` | 효과 영역을 축소 후 nearest-neighbor로 확대해 pixel mosaic를 만든다. ellipse mode에서는 mask 내부만 바꾸고 외부는 byte-identical하게 보존한다. 입력 frame은 복사하며 비정상/빈 ROI에는 복사본을 반환한다. |

### `effects/sticker.py`의 `StickerEffect`

| 함수/메서드 | 실제 기능 |
| --- | --- |
| `__init__(png_bytes, scale_factor, anchor, eye_rotation)` | PNG bytes를 OpenCV로 decode하고, 배율·anchor·눈 기울기 회전 여부를 검증/저장한다. invalid PNG에는 예외가 난다. |
| `_place(frame, sticker, x, y)` | alpha 채널을 사용해 sticker의 프레임 내 교집합만 blend한다. frame edge를 벗어나도 clip한다. |
| `render(frame, face, landmarks)` | 얼굴 크기에서 sticker를 resize하고 선택적으로 `landmarks.eye_angle`만큼 회전한 뒤 center/eyes 등의 anchor 위치에 배치한다. 입력 대신 copy를 돌려 준다. |

효과의 공간적 계약은 benchmark helper `assert_effect_is_local`이 검사한다. 즉 영향 ROI 안에는 실제 변화가 있고 ROI 밖 픽셀은 원본과 byte-identical이어야 한다.

## 7. detector adapter: `adapters/`

### `detection_iface.py`

| 타입/함수 | 실제 기능 |
| --- | --- |
| `FaceDetection` | `bbox`, `landmarks`, `confidence`를 가진 표준 detection 레코드다. |
| `DetectorAdapter` | `model_id`, `detect(frame)`를 요구하는 Protocol이다. pipeline은 이 최소 계약만 의존한다. |
| `validate_detection(...)` | bbox 좌표·landmark shape·confidence 범위를 검사하고 올바른 `FaceDetection`을 만든다. |

### `opencv.py`

| 클래스/함수 | 실제 기능 |
| --- | --- |
| `OpenCvYuNetDetector.__init__(model_path, ...)` | OpenCV `FaceDetectorYN`을 생성하고 모델 경로·입력 크기·threshold를 설정한다. |
| `OpenCvYuNetDetector.detect(frame)` | 프레임 크기를 detector에 알리고 OpenCV row를 `FaceDetection`으로 정규화한다. model inference 또는 OpenCV 오류는 호출자에게 전달될 수 있다. |
| `OpenCvSFaceEmbedder.__init__(model_path, ...)` | OpenCV `FaceRecognizerSF` 모델을 lazy-load하도록 준비한다. |
| `OpenCvSFaceEmbedder.embed(frame, detection)` | YuNet 15-value row로 face alignment를 하고 finite한 L2-normalized embedding과 preprocessing revision을 반환한다. |

## 8. media 입출력: `media/frame_source.py`

| 클래스/메서드 | 실제 기능 |
| --- | --- |
| `FrameSource` | `open`, `read`, `close`, width/height/fps/frame count 계약을 정의하는 Protocol/추상 인터페이스다. |
| `OpenCvFrameSource.__init__(path)` | 이미지/비디오 path와 OpenCV capture 상태를 준비한다. |
| `open()` | path를 검증하고 `cv2.VideoCapture`를 연다. 실패 시 예외다. |
| `read()` | 다음 프레임을 BGR ndarray로 읽고 `(success, frame)`을 반환한다. frame index를 전진시킨다. |
| `close()` | capture resource를 release한다. |
| `width`, `height`, `fps`, `frame_count`, `current_frame_index` | source metadata와 현재 읽기 위치 property다. |
| `FakeFrameReader` | 테스트용 in-memory frame 시퀀스 reader다. `open/read/close`와 metadata property로 production reader와 같은 모양을 제공한다. |

## 9. 로컬 갤러리: `gallery.py`

`LocalGallery`는 실제 vector 비교와 JSON persistence를 제공한다. profile에는 이름·원본 이미지 같은 PII를 넣지 않고, profile ID와 정규화된 vectors/centroid만 저장한다.

| 타입/함수 | 실제 기능 |
| --- | --- |
| `_cosine_similarity(left, right)` | 두 L2 normalized vector의 dot product similarity를 계산한다. |
| `_json_object_without_duplicate_keys(pairs)` | JSON parse 중 중복 key를 발견하면 `LocalGalleryError`로 거절한다. |
| `MatchResult` | `profile_id`, `confidence`, `score_category`를 담는 매칭 결과. `is_match` property는 profile ID 존재 여부를 알려 준다. |
| `LocalGalleryError` | gallery input, persistence, validation 오류용 예외다. |
| `LocalGallery.__init__(...)` | high/medium/충돌/중복 threshold를 검증하고 빈 갤러리를 만든다. |
| `_threshold(category)` | `high`/`medium` category에 대응하는 threshold를 찾는다. |
| `_normalize_enrollment(vector)` | 수치형·finite·1차원인 enrollment vector를 float32 L2-normalized 형태로 바꾼다. |
| `_assert_dimension(vector)` | 첫 vector로 embedding dimension을 고정하고 이후 vector 차원이 같음을 보장한다. |
| `_normalized_centroid(vectors)` | profile vectors의 평균을 다시 L2 normalize하여 centroid를 만든다. |
| `_profile_vectors(profile)` | 직렬화된 profile vectors를 ndarray 배열로 돌린다. |
| `_profile_similarity(vector, profile)` | query vector와 profile의 reference vectors/centroid를 비교해 최고 similarity를 계산한다. |
| `enroll(vector, profile_id=None)` | vector를 기존 profile에 추가하거나 새 profile을 만든다. duplicate vector와 다른 profile 충돌을 거절한다. |
| `add_reference(profile_id, vector)` | 지정 profile에 추가 reference를 등록하는 `enroll` 편의 경로다. |
| `match(vector, top_k=1, confidence_threshold=None)` | similarity 내림차순으로 profile 후보를 돌린다. `MatchResult.is_match`는 calibrated high category일 때만 True다. |
| `save(path)` | 완전히 validated된 `to_dict()` payload를 임시 파일+`os.replace` 방식으로 atomic 저장하고 권한을 가능한 범위에서 0600으로 제한한다. |
| `load(path)` | 파일 전체를 엄격히 parse/validate한 후보 gallery를 만든 뒤에만 현재 state를 교체한다. 중간 실패로 기존 gallery가 부분 변경되지 않는다. |
| `to_dict()` | PII 없는 독립 JSON serialization payload를 만든다. |
| `_saved_vector(value, dimension, name)` | 디스크에서 읽은 vector의 길이·수치성·finite·정규화를 검증한다. |
| `from_dict(data)` | root key, version, thresholds, profile IDs, vectors, centroids, collision/counter를 모두 엄격하게 검증한 gallery를 생성한다. |
| `profile_count`, `profile_ids`, `embedding_dimension` | gallery 관측용 read-only property다. |
| `save_to_json_file(path)` | 호환 목적으로 JSON 저장을 제공하는 별도 wrapper다. 신규 사용은 더 엄격한 `save`를 우선한다. |
| `from_json_file(path)` | JSON을 읽어 `from_dict`로 검증하는 classmethod wrapper다. |

### 실제 승인 adapter: `approval_store.py`, `approved_gallery.py`

| 타입/함수 | 실제 기능 |
| --- | --- |
| `ApprovalRecord` | `approved`, `reason_code`, 선택적인 timezone 포함 `expires_at`을 가진 불변 local permission record다. `is_current()`는 만료 여부를 검사한다. |
| `ApprovalStore` | profile ID별 record와 `gallery_revision`을 별도 JSON에 보관한다. `from_dict/load`는 exact schema만 읽고, `save`는 atomic write를 쓴다. |
| `ApprovedLocalGalleryAdapter.evaluate(frame, detection)` | SFace embedding을 만들고 LocalGallery high match를 찾은 뒤, ApprovalStore의 current `approved=True` record가 있을 때만 `GalleryApproval(True, ...)`을 반환한다. |

`LocalGallery.match()` 결과 자체는 권한이 아니다. 이 adapter를 거치지 않고 `LocalGallery`를 pipeline의 gallery로 직접 주입하면 legacy 계약 오류로 fail-closed 된다.

## 10. 기본 gallery adapter: `gallery_matcher.py`

| 클래스/메서드 | 실제 기능 |
| --- | --- |
| `GalleryMatcher.__init__(...)` | DB path 등의 adapter 설정을 보관한다. |
| `embed(frame)` | 현재 stub으로 `None`을 반환한다. 실제 얼굴 embedding inference는 없다. |
| `match(embedding)` | 현재 similarity를 계산하지 않고, non-empty database에서도 `GalleryApproval.denied("similarity_not_evaluated")`를 반환한다. |

따라서 이 클래스는 안전 계약을 시연하기 위한 placeholder다. 실제 승인 시스템으로 오해하면 안 된다.

## 11. model manifest: `model_manifest.py`

모델 파일을 로드하기 전에 manifest와 hash를 검증하기 위한 모듈이다.

| 함수/클래스 | 실제 기능 |
| --- | --- |
| `ManifestValidationError` | manifest schema/파일 검증 실패 예외다. |
| `_json_object_without_duplicate_keys(pairs)` | JSON manifest의 duplicate key를 거절한다. |
| `validate_manifest(entry)` | 한 model entry의 필수 key, id, filename, SHA-256, provider 등을 엄격히 확인한다. |
| `load_manifest_from_json(path)` | JSON object/list를 읽고 각 entry를 validate한다. |
| `sha256_file(path)` | 파일을 chunk 단위로 읽어 SHA-256 hex digest를 계산한다. |
| `verify_model_file(entry, path)` | regular file 여부, bytes 크기, SHA-256이 manifest와 일치하는지 확인한다. |

## 12. CLI: `cli.py`

명령은 `python -m consented_face_redactor.cli <subcommand>` 형태다.

| 함수 | 실제 기능 |
| --- | --- |
| `_build_parser()` | `inspect-config`, `validate-models`, `process-image`, `process-video`, `gallery-enroll`와 각 option을 정의한다. |
| `_cmd_inspect_config(args)` | default 또는 JSON config를 읽고 `to_dict()`의 모든 field를 출력한다. `--strict-config`이면 unknown key를 거절한다. |
| `_cmd_validate_models(args)` | manifest directory의 JSON을 전부 읽고 중복 model ID/filename, 모델 size/hash를 확인한다. |
| `_load_config(path, strict=False)` | config file을 읽거나 없으면 defaults를 반환한다. parse/schema 오류에는 stderr 출력 후 process exit 2다. |
| `_load_track_state(state_dir)` | `track_state.json`이 있으면 raw JSON dict를 읽고, 읽기 실패면 `None`을 반환한다. |
| `_save_track_state(pipeline, state_dir)` | pipeline v2 snapshot을 `track_state.json`에 쓴다. |
| `_load_verified_model_entries(model_dir, manifest_dir)` | 정확히 하나의 OpenCV detector와 embedder manifest entry를 찾고 model hash를 검증한다. |
| `_load_runtime_components(args)` | runtime option 네 개가 전부 없으면 no-redaction stub을, 전부 있으면 YuNet/SFace/ApprovedLocalGalleryAdapter를 만든다. 일부만 있으면 오류다. |
| `_evidence_row(...)`, `_write_evidence(...)` | pixel/vector 없이 frame별 decision summary를 만들고, `--evidence-out`이 명시된 경우에만 local JSON으로 쓴다. |
| `_cmd_process_image(args)` | verified runtime이면 실제 얼굴별 승인 처리 후 image를 쓰고, `--dry-run`이면 output 없이 evidence만 남긴다. |
| `_cmd_process_video(args)` | verified runtime으로 각 frame을 처리해 선택적으로 MP4와 evidence를 쓴다. |
| `_cmd_gallery_enroll(args)` | 정확히 한 얼굴을 YuNet으로 검출하고 SFace embedding을 LocalGallery에 저장한다. `--approve --approval-reason`일 때만 별도 ApprovalStore record를 True로 만든다. |
| `main(argv=None)` | parser를 만들고 subcommand 함수로 dispatch하며 적절한 exit code를 반환한다. |

### CLI 사용 시 주의

- `gallery-enroll`은 Config schema를 검증하지만 effect config를 사용하지 않는다. 이는 해당 command가 enrollment-only이기 때문이다.
- 실제 redaction은 model/manifest/gallery/approval option 네 개를 모두 제공한 경우에만 수행한다.
- 상태 파일은 input 옆 또는 `--state-dir`에 작성되므로, CLI 실행은 input 외에 이 로컬 파일을 변경한다.

## 13. benchmark: `benchmark/`

### `benchmark/fake_gallery.py`

| 클래스/메서드 | 실제 기능 |
| --- | --- |
| `FakeGallery` | benchmark/test에서 approval 경로를 결정적으로 제어한다. legacy match list 입력도 호환성상 구조화된 approval으로 변환한다. |
| `embed(frame)` | 설정된 embedding을 반환하거나 지정된 예외를 발생시킨다. |
| `match(embedding)` | 설정된 `GalleryApproval`, malformed value 또는 예외를 반환/발생시켜 fail-closed 경로를 검증한다. |

### `benchmark/runner.py`

| 타입/함수 | 실제 기능 |
| --- | --- |
| `BenchmarkResult` | scenario 이름, pass 여부, duration, metrics, failure detail을 담는 결과 레코드다. |
| `_noise_frame(...)` | ROI 변화 검증이 무의미해지지 않도록 deterministic noise 이미지 fixture를 만든다. |
| `assert_effect_is_local(source, result, affected_regions)` | affected ROI 안의 변화와 ROI 밖 전체 pixel 보존을 함께 assertion한다. |
| `_scenario(...)` | fresh input/pipeline을 만들고 실제 시간 측정, 예외 포착, 실패 결과 append를 수행하는 공통 wrapper다. |
| `run_category_a()` | confidence-only, explicit approval, empty/no match, detector 없음, embed/match exception, malformed approval, stale profile을 검사한다. |
| `run_category_b()` | mosaic/sticker/no detection/multi-face/edge clipping의 effect ROI 및 pixel 보존을 검사한다. |
| `run_category_c()` | 실제 sequence로 `UNSEEN → CANDIDATE → CONFIRMED → LOST → EXPIRED`를 관측한다. |
| `run_category_d()` | warm-up 후 synthetic frame 반복 처리의 median/p95 latency, FPS와 환경 정보를 측정한다. FPS는 pass/fail threshold가 아니다. |
| `run_category_e()` | Config default, strict unknown key, v1 migration을 실제 assertion으로 검사한다. |
| `run_benchmark(category)` | `A`~`E` 하나를 실행한다. 잘못된 category는 오류다. |
| `generate_aggregate_report()` | 모든 category 결과, UTC 시각, Git revision(가능할 때), runner version, Python/platform/CPU/OpenCV/numpy metadata를 JSON 문자열로 만든다. 파일 저장은 하지 않는다. |

## 14. 테스트의 역할

`tests/`는 import/export, Config round-trip/strict, gallery 저장과 승인 결과, pipeline fail-closed 상태 전이, CLI contract, effect ROI, video path, benchmark A–E를 나눈다.

특히 다음 테스트 성질이 중요하다.

- gallery approval이 없거나 오류여도 `CONFIRMED`/`is_redacted=True`가 되지 않는다.
- effect 테스트는 ROI 밖 이미지 변경을 허용하지 않는다.
- benchmark scenario가 예외를 삼키거나 누락하지 않고 `passed=False` 결과를 남긴다.
- CLI `inspect-config`의 출력 field 집합은 `Config.to_dict()`와 일치한다.

## 15. 코드 리뷰 결론과 운영 전 필수 통합

이 프로그램의 가장 좋은 성질은 “확인하지 못한 사람을 자동으로 가리지 않는다”는 점이다. 다만 실제 현장 사용 전에는 다음 연결을 구현·검증해야 한다.

1. 사용자 제공 모델과 동의된 실제 영상으로 YuNet/SFace smoke test를 수행한다.
2. enrollment 품질 기준(blur, pose, 최소 face size)과 다중 reference enrollment을 별도 승인 절차로 추가한다.
3. 장기 track와 성능 최적화가 필요하면 identity 승인을 재사용하지 않는 안전한 multi-track association 설계를 별도 review한다.
4. real-model 영상, 여러 얼굴, frame edge, process restart, gallery 장애에서 privacy/security review와 사람 승인을 수행한다.

그 전까지 benchmark의 통과는 이 저장소의 **현재 안전 계약과 회귀 방지**를 뜻하며, 실제 인식 정확도나 운영 보안 승인을 뜻하지 않는다.

## 16. 등록 영상과 FHD 실행 경로

### `video_enrollment.py`

| 타입/함수 | 실제 기능 |
| --- | --- |
| `VideoEnrollmentOptions` | sample 간격, 최대 reference 수, duplicate similarity, 최소 face 크기처럼 enrollment coverage를 제어하는 비권한 옵션이다. |
| `EnrollmentCandidate` | target-only 등록 영상에서 추출된 immutable normalized embedding과 frame 관측값이다. 원본 image/crop을 보관하지 않는다. |
| `EnrollmentSkip` | `no_face`, `multiple_faces`, `face_too_small`, `embedding_error` 등 자동 등록하지 않은 sampled frame의 사유다. |
| `EnrollmentReport` | 후보·중복·다중 얼굴·review frame·reference similarity 통계를 JSON-safe 형태로 제공한다. |
| `iter_sampled_frames()` | 이미 open된 source를 한 번만 순차 read하고 선택 frame만 yield한다. tracker나 frame cache가 없다. |
| `extract_enrollment_candidate()` | 얼굴 하나만 탐지된 frame에서 SFace embedding을 얻거나 skip reason을 반환한다. |
| `select_diverse_references()` | near-duplicate vector를 줄이고 farthest-point coverage로 최대 reference 수를 제한한다. extreme outlier는 자동 등록 대신 review 목록에 넣는다. |
| `VideoEnrollmentService.collect/select/enroll` | frame 수집, diversity 선택, atomic `LocalGallery.enroll_many()`를 분리한다. dry-run은 collect/select만 수행한다. |

### 추가 runtime 관측 및 보호

- `LocalGallery.enroll_many()`는 detached staging gallery에서 모든 vector를 검증해 batch 오류 시 원본 gallery를 바꾸지 않는다.
- `LocalGallery.profile_similarity_details()`는 최고 reference index/score와 centroid score를 권한이 아닌 diagnostics로 제공한다.
- `ApprovedLocalGalleryAdapter.last_observation`은 마지막 match 진단을 제공하지만, pipeline의 권한 신호는 계속 `GalleryApproval.approved`뿐이다.
- `RedactionPipeline.last_frame_decisions`는 bbox와 approval을 연결한다. 기본 evidence에는 bbox를 쓰지 않고 CLI `--evaluation-evidence`에서만 opt-in한다.
- `gallery-enroll-video`는 target-only 등록 영상의 multi-reference profile command다. `--dry-run`은 gallery/approval을 수정하지 않는다.
- `process-image`/`process-video`는 input/output resolved path가 같으면 write 전에 거절한다. evidence summary에는 FHD 여부, latency percentile, 실측 FPS, reason count를 기록한다.
- `media.remux.remux_original_audio()`는 선택적 FFmpeg audio remux helper이며, 두 input을 덮어쓰거나 기존 destination을 바꾸지 않는다. CLI에서는 `process-video --preserve-audio --ffmpeg-path ...`로 명시 opt-in한다.
