# Consented Face Redactor 시스템 아키텍처 가이드

## 1. 이 문서의 목적

이 문서는 Consented Face Redactor의 구성요소, 데이터 계약, 처리 단계, 모델별 책임과 안전 경계를 설명하는 기술 참조다.

- 이 프로그램은 정확히 무엇을 하는가?
- 일반 얼굴 모자이크 프로그램과 무엇이 다른가?
- 왜 detector, gallery, approval, tracker를 분리했는가?
- 실제 영상에서 어떤 오류가 발생했고 어떻게 해결했는가?
- 각 모델과 라이브러리는 어디에 쓰였는가?
- 456/456이라는 결과를 어떻게 검증했는가?
- 현재 한계와 다음 개선 방향은 무엇인가?

상세 개발 연대기는 [포트폴리오 사례 연구](../portfolio/PROJECT_PORTFOLIO_CASE_STUDY.md), 그림 중심 설명은 [시각적 흐름도](PROJECT_FLOWCHART.md)를 함께 참고한다.

## 2. 문제와 핵심 설계

### 2.1 문제

일반적인 `detect → similarity threshold → mosaic` 구조는 얼굴 각도가 변할 때 similarity가 흔들려 모자이크가 끊긴다. threshold를 낮추면 다른 사람을 잘못 가릴 위험이 커진다.

### 2.2 계층 분리

신호를 네 계층으로 분리했다.

1. Detection: 얼굴 후보 위치
2. Identity approval: 그 얼굴을 가릴 명시 권한
3. Temporal continuity: 승인된 얼굴 위치의 시간축 연속성
4. Presentation: 승인된 위치에 적용할 타원형 모자이크

### 2.3 실제 데이터에서 확인한 오류

초기 영상에서는 detector가 놓친 frame이 2개뿐인데 245개 frame이 similarity 부족으로 승인되지 않았다. tracker를 도입한 뒤에도 영상 양끝 공백과 중복 bbox 문제가 남았다. 더 중요한 문제는 등록 gallery 안에 귀 부분 false crop이 들어가 높은 similarity를 만들었다는 점이었다.

### 2.4 적용한 해결책

- 같은 profile anchor 사이를 TAPNext++로 양방향 추적
- forward/backward bbox IoU 합의
- detection association cost와 second-best margin
- 등록 embedding graph의 dominant component만 자동 선택
- 승인 plan과 renderer를 분리한 2-pass 처리

### 2.5 결과

초기 209/456에서 v1 392/456, v2 408/456, clean enrollment v3 456/456으로 개선했다. v4에서는 identity/tracking을 유지하면서 사용자 피드백에 따라 강도를 낮춘 외접 세로 타원 mosaic로 변경했다. 전체 테스트 321개와 benchmark 24/24가 통과했다.

## 3. 시스템 경계

### 입력

- 대상자만 등장하는 동의된 등록 영상
- 처리할 원본 영상
- 사용자가 직접 확보한 YuNet/SFace/TAPNext++ model file
- model 출처, license, SHA-256을 담은 manifest
- 명시적인 gallery approval

### 출력

- 원본과 다른 경로의 처리된 MP4
- frame count, 승인 reason, tracking 결과, 환경 정보를 담은 JSON evidence
- 등록 후보·선택·review 통계

### 의도적으로 하지 않는 일

- cloud face API 호출
- 자동 model download
- 원본 영상 overwrite
- detector confidence만으로 redaction
- tracker만으로 profile ID 생성
- 무단 신원 확인이나 감시
- face swap 또는 얼굴 합성

## 4. 구성요소별 역할

| 구성요소 | 사용 기술 | 입력 | 출력 | 권한 생성 가능 여부 |
| --- | --- | --- | --- | --- |
| Face detector | OpenCV YuNet | BGR frame | bbox, 5 landmarks, confidence | 불가 |
| Face embedder | OpenCV SFace | 정렬된 얼굴 crop | embedding vector | 불가 |
| LocalGallery | NumPy cosine similarity | embedding | 후보 profile과 similarity | 불가 |
| ApprovalStore | versioned JSON | profile ID | 명시 승인 상태 | 승인 근거 보관 |
| Gallery adapter | Python dataclass/Protocol | detection과 gallery 결과 | `GalleryApproval` | 가능, `approved=True`일 때만 |
| Point tracker | PyTorch TAPNext++ | frame과 21개 point | point 위치와 visibility | 불가 |
| Geometry | NumPy/SVD | point trajectory | similarity transform과 bbox | 불가 |
| Association | IoU·거리·scale cost | tracked bbox와 detections | one-to-one match | 불가 |
| Temporal planner | 2-pass Python pipeline | anchors와 tracking evidence | immutable frame plan | 기존 권한의 제한적 전파 |
| Renderer | OpenCV/NumPy | frame과 승인 bbox | 타원 mosaic frame | 불가 |
| Media writer | OpenCV, optional FFmpeg | 처리 frame/audio | 새 MP4 | 해당 없음 |

## 5. 기술 스택과 선택 이유

### Python 3.12

OpenCV, NumPy, PyTorch 생태계와 빠른 실험·테스트 작성에 적합하다. type hint, dataclass, Protocol로 모델 경계를 명시했다.

### OpenCV YuNet

얼굴 bbox와 5개 landmark를 함께 제공한다. SFace 정렬과 tracker point seed에 같은 detection 정보를 사용할 수 있다. detector confidence는 위치 후보 품질일 뿐 identity 권한으로 사용하지 않는다.

### OpenCV SFace

얼굴을 고정 길이 embedding으로 변환한다. 다양한 등록 각도를 하나의 centroid로 압축하지 않고 여러 reference와 비교해 pose recall을 보존한다.

### NumPy

embedding 정규화와 cosine similarity, deterministic noise fixture, mask 비교, robust geometry 계산에 사용한다.

### PyTorch와 CUDA

TAPNext++ 공식 checkpoint 실행에 사용한다. 무거운 의존성이므로 tracker가 선택될 때만 lazy import/load한다.

### TAPNext++

얼굴 내부의 여러 point를 frame 사이에서 추적한다. 일반 MOT처럼 사람 전신 ID를 다시 예측하지 않고, 이미 승인된 얼굴 내부 위치의 연속성을 계산하는 데 사용한다.

### pytest

Config migration, approval fail-closed, multi-face 분리, tracker geometry, 양방향 합의, renderer locality, CLI, media output을 자동 검증한다.

### JSON evidence

실행 결과를 문서의 선언이 아니라 파싱 가능한 artifact로 남긴다. 기본 evidence에는 frame pixel, face crop, embedding을 저장하지 않는다.

## 6. 등록 단계 상세 설계

### 6.1 Sampling

`iter_sampled_frames()`가 일정 간격의 frame만 선택한다. 연속 frame은 거의 같은 표정과 각도를 가지므로 전체 frame을 모두 reference로 저장하면 중복과 저장 비용만 늘어난다.

### 6.2 Candidate extraction

`extract_enrollment_candidate()`는 다음 조건을 적용한다.

- detection이 정확히 하나
- 최소 얼굴 크기 통과
- landmark와 embedding 생성 성공
- embedding shape와 값이 유효

0개 또는 여러 얼굴이 검출되면 자동 등록하지 않고 skip reason을 기록한다.

### 6.3 Deduplication

시간상 인접하고 similarity가 매우 높은 embedding은 같은 view의 중복으로 간주해 줄인다.

### 6.4 Similarity graph

남은 embedding을 node로 두고 similarity 0.45 이상인 pair를 edge로 연결한다. 가장 큰 connected component는 정면, 반측면, 측면이 중간 pose로 연결된 target view trajectory로 해석한다.

### 6.5 Quarantine

dominant component에 속하지 않는 작은 island와 singleton은 자동 등록하지 않는다. 실제 데이터에서는 47개 후보 중 36개를 선택하고 11개를 review로 격리했다.

### 6.6 Explicit approval

gallery에 vector가 저장됐다는 사실만으로 사용할 수 없다. `--approve --approval-reason ...`가 있어야 ApprovalStore에 명시 승인 기록이 생긴다.

## 7. 처리 단계 상세 설계

### 7.1 Pass 1: 직접 승인 anchor 수집

모든 frame에서 얼굴별로 YuNet과 SFace를 실행한다. gallery adapter가 다음 필드를 가진 `GalleryApproval`을 반환한다.

```text
approved
profile_id
similarity
reason_code
gallery_revision
```

`approved=True`, profile ID, gallery revision이 모두 있는 detection만 identity anchor가 된다.

### 7.2 Point seed

각 anchor에서 YuNet 5 landmarks와 bbox 내부 4×4 grid를 결합해 기본 21개 point를 만든다. landmark만 사용하면 표정·가림에 취약하고 bbox corner만 사용하면 배경을 따라갈 수 있기 때문에 얼굴 내부에 분산한다.

### 7.3 Robust bbox reconstruction

TAPNext++가 반환한 visible point로 scale, rotation, translation을 포함하는 similarity transform을 추정한다. residual median과 MAD를 사용해 outlier point를 한 번 제거한다.

### 7.4 Bidirectional consensus

같은 profile의 anchor A와 B 사이 gap을 A에서 정방향, B에서 역방향으로 추적한다. 같은 frame의 두 bbox가 정책 IoU 이상일 때만 평균 bbox를 승인 위치로 사용한다.

한쪽 경로만 성공하거나 두 경로가 다른 곳을 가리키면 자동 성공으로 바꾸지 않는다.

### 7.5 Edge extension

첫 anchor 이전과 마지막 anchor 이후에는 양방향 anchor 쌍이 없다. 이 구간은 제한된 단방향 tracker와 현재 detector association이 함께 유지될 때만 연장한다.

- 최대 gap 90 frame
- tracker-only 연속 최대 12 frame
- visibility 기준
- bbox scale·중심 이동 기준
- detection association
- 다른 approved profile 충돌 금지

### 7.6 Immutable plan

분석이 끝나면 frame index별 profile, bbox, source reason을 가진 redaction plan을 만든다. renderer는 신원을 다시 판단하지 않고 이 plan만 소비한다.

### 7.7 Pass 2: rendering

원본 영상을 처음부터 다시 읽는다. plan에 승인 bbox가 있는 frame만 mosaic한다. 모든 frame을 sibling temporary MP4에 기록한 뒤 frame 수와 writer 성공을 확인하고 `os.replace()`로 destination을 만든다.

## 8. 타원형 모자이크 설계

### 강도

```text
block_size = max(round(shorter_face_side / 12), 10)
```

얼굴 크기에 비례하므로 FHD의 큰 얼굴과 작은 얼굴에서 상대적으로 일정한 강도를 만든다.

### 형태

- 가로 반축: bbox 반폭 × 1.40
- 세로 반축: bbox 반높이 × 1.50
- 기본 shape: ellipse

축은 다음 외접 조건을 만족해야 한다.

```text
1 / horizontal_scale² + 1 / vertical_scale² <= 1
```

이 조건 때문에 bbox 네 모서리가 타원 내부에 들어간다. 타원 bounding rectangle에서 mosaic를 계산하지만 boolean mask 안의 pixel만 합성하므로 타원 밖은 입력과 동일하다.

## 9. 상태와 데이터 계약

### `GalleryApproval`

불변 dataclass다. similarity와 approval을 분리하고 오류 원인을 reason code로 보존한다.

### `TrackAuthorization`

명시 승인에서 시작한 profile lease다. tracker는 lease를 새로 만들 수 없고 기존 lease의 위치만 제한적으로 전파한다.

### `RedactionTrackPlan`

Pass 1의 결과다. renderer가 사용할 frame별 승인 위치를 고정해 분석 결과가 writer 상태에 따라 바뀌지 않게 한다.

### `Config`

현재 schema v4다. v1/v2/v3를 읽어 v4로 migration한다. 기본 loader는 호환성을 위해 unknown key를 무시하고 `strict=True`에서만 거절한다.

### Track snapshot

snapshot schema v2는 state, frame index, lost frame index, confirmed profile reference를 저장한다. 시간 정보가 없는 legacy confirmed/lost snapshot은 fail-open하지 않고 candidate로 낮춘다.

## 10. 주요 함수 설명

| 파일 | 함수 | 역할 |
| --- | --- | --- |
| `video_enrollment.py` | `iter_sampled_frames()` | 등록 영상 sampling |
| `video_enrollment.py` | `extract_enrollment_candidate()` | 유효한 단일 얼굴 embedding 후보 생성 |
| `video_enrollment.py` | `select_diverse_references()` | dominant component와 다양한 reference 선택 |
| `approved_gallery.py` | `evaluate()` | embedding match와 명시 approval 결합 |
| `pipeline.py` | `_process_face_by_face_approvals()` | 여러 얼굴을 독립 승인·렌더링 |
| `pipeline.py` | `process_frame()` | 경량 mode 상태 머신 |
| `tracking/geometry.py` | `seed_face_points()` | 21개 face point 구성 |
| `tracking/geometry.py` | `estimate_similarity_transform()` | robust point transform 계산 |
| `tracking/association.py` | `association_cost()` | IoU·거리·scale 기반 cost |
| `tracking/bidirectional.py` | `reconcile_bidirectional_paths()` | forward/backward bbox 합의 |
| `tracking/authorization.py` | `may_propagate_authorization()` | 기존 profile lease 전파 가능성 판단 |
| `temporal_video_processor.py` | `analyze()` | Pass 1 anchor·tracking plan 생성 |
| `temporal_video_processor.py` | `render()` | Pass 2 atomic output 생성 |
| `effects/mosaic.py` | `ellipse_bounds()` | 외접 타원 계산과 검증 |
| `effects/mosaic.py` | `ellipse_mask()` | 타원 boolean mask 생성 |
| `effects/mosaic.py` | `MosaicEffect.render()` | 타원 내부 adaptive mosaic 합성 |

전체 함수는 [코드베이스·함수 참조](../CODEBASE_FUNCTION_REFERENCE.md)에 더 자세히 정리돼 있다.

## 11. 주요 설계 의사결정 근거

### 11.1 Threshold 완화 대신 시간축 연속성을 사용한 이유

초기 미처리 247개 중 detector miss는 2개뿐이고 245개는 similarity 변동이었다. threshold를 낮추면 recall과 함께 다른 사람의 false approval 위험도 올라간다. 그래서 명시 승인 frame을 anchor로 두고 위치 continuity만 tracker로 보강했다.

### 11.2 양방향 추적을 사용한 이유

저장 영상은 미래 frame을 사용할 수 있다. 정방향 하나가 drift해도 역방향이 다른 위치를 가리키면 불일치를 검출할 수 있다.

### 11.3 Gallery reference를 전부 자동 등록하지 않은 이유

대상자 한 명만 나오는 등록 영상에서도 detector가 귀나 부분 얼굴을 false crop으로 만들 수 있다. 실제 오염 reference 하나가 높은 max similarity를 만들었기 때문에 dominant component 밖 11개를 격리했다.

### 11.4 Rectangle 대신 ellipse를 기본으로 사용한 이유

강한 사각형은 배경을 과도하게 가렸다. 내접 타원은 얼굴 모서리를 노출할 수 있어 bbox를 외접하는 1.40/1.50 타원을 사용했다. 이는 identity 계층이 아니라 renderer 계층의 변경이다.

### 11.5 456/456 결과의 해석 범위

제공된 단일 대상 영상에서는 그렇지만 다중 인물 crossing과 장시간 완전 가림은 별도 검증이 필요하다. 정확도를 일반화하지 않는다.

## 12. 검증 구조

### Synthetic unit test

민감 데이터 없이 승인 실패, 상태 전이, geometry, mask locality를 결정적으로 재현한다.

### Integration test

Config–CLI, detector–gallery, multi-face, tracker plan, media output 경계를 검사한다.

### Benchmark A–E

- A: identity safety
- B: effect locality와 예외 보존
- C: state transition
- D: latency와 환경 metadata
- E: Config contract

### Real-video evidence

실제 FHD 영상의 frame 수, FPS, anchor, propagation, review, reason을 기록한다.

### Visual inspection

contact sheet로 원본과 결과를 비교한다. 이 단계가 정량 수치만으로 찾지 못한 ear false crop을 발견했다.

## 13. 현재 한계와 후속 작업

- 다중 인물 crossing에서 ID switch 검증
- 장시간 완전 가림 후 gallery 재승인
- 격리된 등록 reference의 사람 검토 UI
- FFmpeg audio remux 실제 환경 검증
- hardware profile별 성능 baseline
- optional segmentation mask 비교
- 여러 consented profile 동시 처리 평가
