# 강한 Mosaic 및 SOTA 기반 시간축 얼굴 추적 작업지시서

> 문서 상태: local 구현·실영상 검증 완료, commit/push 전 사용자 확인 대기
> 작성 기준일: 2026-08-26
> 적용 저장소: consented-face-redactor-phase10-local
> 적용 원칙: local clone에서만 작업하며 commit/push하지 않는다. 원본 영상과 기존 결과물은 덮어쓰지 않는다.

> 실행 결과 요약: Config v3 강한 적응형 mosaic, TAPNext++ adapter, detection association, anchor authorization, 양방향 합의, 2-pass atomic renderer, clean enrollment component filter를 구현했다. 이후 사용자 피드백에 따라 Config v4 기본을 강도가 완화된 외접 세로 타원 mosaic로 바꿨다. 전체 pytest 321개와 benchmark A–E 24/24가 통과했다. clean gallery를 사용한 `test.mp4` 실측은 v4에서도 456/456 frame redaction, 입력/출력 1080×1920·30fps·456 frame 일치였다. 상세 evidence와 한계는 `TEMPORAL_TRACKING_IMPLEMENTATION_REPORT.md`에 기록한다.

## 1. 작업 목적

이 작업의 목적은 실제 처리 결과에서 확인된 다음 두 문제를 해결하는 것이다.

1. 얼굴 mosaic 블록이 너무 작아 가림 강도가 약하다.
2. 동일한 대상 인물이 계속 등장하는데도 frame별 SFace 판정 변동으로 mosaic가 중간중간 끊어진다.

최종 목표는 사용자가 승인한 대상 profile의 얼굴만 시간축으로 안정적으로 따라가면서, 얼굴의 식별 단서가 충분히 감소하도록 더 강한 mosaic를 적용하는 것이다.

이번 변경은 기존의 등록 영상 기반 multi-reference gallery와 명시 승인 구조를 폐기하지 않는다. 기존 구조는 target track을 시작하거나 다시 획득할 때의 신원 권한으로 유지하고, tracker는 이미 승인된 얼굴의 위치 연속성을 계산하는 보조 계층으로만 사용한다.

## 2. 현재 결과에서 확인된 기준선

실제 입력:

| 구분 | 파일 | 해상도 | FPS | frame 수 |
| --- | --- | ---: | ---: | ---: |
| 등록 영상 | learn.mp4 | 1080 × 1920 | 약 30 | 530 |
| 처리 영상 | test.mp4 | 1080 × 1920 | 30 | 456 |

현재 등록 결과:

- profile ID: prof-00000000
- 선택된 reference: 47개
- approval reason: operator_approved_learn_mp4_20260826

현재 처리 evidence 분석:

| 관측 | frame 수 | 의미 |
| --- | ---: | --- |
| 승인된 대상 얼굴 mosaic | 209 | 해당 frame에서 GalleryApproval.approved=True |
| 얼굴 검출 자체가 없음 | 2 | detector gap |
| 얼굴은 검출됐으나 승인되지 않음 | 245 | 대부분 frame별 SFace similarity 변동 |
| detected-but-not-approved 최장 연속 구간 | 65 | 약 2.17초 동안 같은 인물의 temporal identity를 사용하지 못함 |

따라서 현재 깜빡임의 주원인은 detector가 얼굴을 놓치는 문제가 아니다. Production 경로가 매 frame의 얼굴을 독립적으로 embedding하고, 그 frame에서 gallery 승인을 통과하지 못하면 직전 frame에서 같은 사람으로 확인되었더라도 즉시 mosaic를 중단하는 것이 핵심 원인이다.

기준 evidence:

- local_data/evidence/learn_enrollment.json
- local_data/evidence/test_processing_verified.json

이 두 파일은 local_data 아래의 민감한 로컬 증거이며 Git에 추가하지 않는다.

## 3. 직접 확인된 코드 원인

### 3.1 약한 mosaic

현재 pipeline의 mosaic 분기는 얼굴 크기와 관계없이 다음과 같이 8 pixel block을 강제한다.

~~~python
mosaic_cfg = MosaicConfig(force_block_size=8)
~~~

큰 얼굴에서 8 pixel block은 수십 개의 작은 mosaic cell을 만들기 때문에 윤곽과 세부 단서가 많이 남는다. domain의 MosaicConfig에 자동 크기 계산 필드가 존재하지만 pipeline의 강제값 때문에 실제 production 경로에서는 사용되지 않는다.

수정 대상:

- src/consented_face_redactor/pipeline.py
- src/consented_face_redactor/effects/mosaic.py
- src/consented_face_redactor/domain/types.py
- src/consented_face_redactor/config.py

### 3.2 시간축 identity가 없는 production 경로

현재 ApprovedLocalGalleryAdapter가 연결된 경로는 검출된 얼굴마다 매 frame evaluate()를 호출하고, 그 frame에서 approved=True인 bbox만 effect 처리한다.

현재 TrackState는 UNSEEN, CANDIDATE, CONFIRMED, LOST, EXPIRED 상태를 나타내지만 실제 bounding box나 동일 객체를 frame 사이에서 추적하는 tracker는 아니다.

수정 대상:

- src/consented_face_redactor/pipeline.py
- src/consented_face_redactor/approved_gallery.py
- src/consented_face_redactor/cli.py

신규 temporal 계층은 pipeline의 GalleryApproval 의미를 변경하지 않고 별도 모듈로 구현한다.

## 4. tracker 기술 선정

### 4.1 SOTA의 의미

SOTA는 모든 종류의 tracking을 통틀어 하나의 모델을 의미하지 않는다.

- MOT benchmark의 사람 전신 ID association
- single-object bounding-box tracking
- video object segmentation
- arbitrary point tracking

은 서로 다른 문제다.

이 프로젝트는 이미 gallery로 승인된 얼굴 내부의 위치를 frame 사이에서 이어야 한다. 따라서 person detector 기반 MOT보다 얼굴 내부의 여러 점을 장기 추적하는 Tracking Any Point 계열이 적합하다.

### 4.2 1차 선택: TAPNext++

TAPNext++를 1차 구현 대상으로 한다.

선정 이유:

- 2026년 공개된 최신 SOTA급 point tracker다.
- Google DeepMind 공식 tapnet 저장소에 PyTorch 구현이 있다.
- frame-by-frame wrapper와 tracking state를 제공한다.
- 얼굴 landmark와 얼굴 내부 grid point를 query로 초기화할 수 있다.
- tracker 입력을 축소하고 결과 좌표를 FHD 공간으로 복원할 수 있다.
- 코드가 Apache-2.0으로 제공되어 provenance 관리가 명확하다.

공식 자료:

- 논문: https://arxiv.org/abs/2604.10582
- 구현: https://github.com/google-deepmind/tapnet

### 4.3 비교 후보

#### Track-On-R

Real-world motion과 online memory에 강한 2026년 후보지만, 공식 환경이 CUDA 12.1/MMCV 중심이고 DINOv3 backbone 접근이 필요하다. RTX 5070 Ti 환경에서 extension 호환성 검증 비용이 크므로 1차 기본값으로 사용하지 않는다.

- https://github.com/gorkaydemir/track_on

#### CoTracker3 Online

구현과 checkpoint가 성숙한 fallback이다. TAPNext++가 현재 환경에서 안정적으로 실행되지 않거나 정확도/속도 기준을 충족하지 못할 때 비교한다.

- https://github.com/facebookresearch/co-tracker
- https://arxiv.org/abs/2410.11831

### 4.4 사용하지 않을 기본 후보

| 후보 | 기본값에서 제외하는 이유 |
| --- | --- |
| ByteTrack/DeepSORT | detector box association 중심이며 최신 point tracking 요구에 맞지 않음 |
| MOTIP | 최신 MOT이지만 얼굴 ROI보다 사람 전신 detector와 ID prediction에 맞춰짐 |
| SAM 2.1 | 정밀 mask가 필요하지 않은 mosaic 작업에는 계산량과 통합 복잡도가 과도함 |
| OpenCV legacy trackers | 간단하지만 장기 가림, 빠른 회전, appearance 변화 대응이 약함 |

### 4.5 최종 선택 gate

TAPNext++를 바로 production 확정하지 않는다. test.mp4의 복사본/로컬 결과로 다음을 측정한다.

- 승인 anchor 사이의 gap 복원율
- 1~2 frame flicker 제거율
- 얼굴 교차 시 ID switch
- visibility 저하 후 잘못된 bbox 전파
- p50/p95 latency
- 전체 처리 FPS
- 최대 VRAM
- checkpoint 및 dependency 재현성

TAPNext++가 CUDA/runtime 호환성 또는 정확도 기준을 통과하지 못하면 CoTracker3 Online으로 같은 contract를 실행한다. tracker 구현체가 바뀌어도 authorization과 renderer 코드는 바뀌지 않아야 한다.

## 5. 변경 후 전체 구조

~~~text
[1차 분석 pass]
test.mp4
  → YuNet: 모든 얼굴 검출
  → SFace: 얼굴별 embedding
  → LocalGallery + ApprovalStore
  → 명시 승인 anchor frame 수집

[시간축 추적 pass]
승인 anchor의 bbox + landmarks
  → 얼굴 내부 query point 생성
  → TAPNext++ 정방향 tracking
  → TAPNext++ 역방향 tracking
  → detection과 point path association
  → 양방향 경로 합의
  → frame별 TargetTrackPlan 생성

[렌더링 pass]
test.mp4 재읽기
  → TargetTrackPlan의 승인된 bbox
  → padding 적용
  → 강한 adaptive mosaic
  → 별도 결과 영상
  → local evidence JSON
~~~

## 6. 변경 불가 안전 계약

다음 규칙은 구현 편의를 위해 완화해서는 안 된다.

1. tracker는 신원 승인을 생성할 수 없다.
2. target track의 시작은 반드시 current GalleryApproval.approved=True anchor에서만 가능하다.
3. tracker confidence, point visibility, IoU는 위치 연속성 근거일 뿐 gallery approval을 대체하지 않는다.
4. 승인된 track의 profile_id가 다른 track이나 detection으로 자동 복사되어서는 안 된다.
5. 완전히 끊기거나 모호해진 track의 재획득은 gallery 재승인이 필요하다.
6. 여러 얼굴이 교차하면 단순 최근접 bbox만으로 target ID를 넘기지 않는다.
7. tracker 예외나 malformed output은 성공으로 처리하지 않는다.
8. tracker 없이 실행하는 기존 호환 경로를 유지한다.
9. 원본 영상, embedding vector, 얼굴 crop은 evidence에 저장하지 않는다.
10. 모델 checkpoint는 manifest hash 검증 후에만 로드한다.
11. 모델을 runtime에서 자동 다운로드하지 않는다.
12. 결과 영상과 evidence는 기존 파일을 덮어쓰지 않는다.

## 7. Phase 0 — 기준선 보존과 평가 구간 정의

### 7.1 작업

- 현재 Git status와 diff를 기록한다.
- 기존 test_target_mosaic_verified.mp4를 기준 결과로 보존한다.
- test_processing_verified.json을 읽어 frame을 다음 세 그룹으로 분류한다.
  - gallery-approved anchor
  - detection-present but identity-weak
  - no-detection
- 연속 gap 구간을 자동 계산한다.
- 1 frame, 2 frame, 3~15 frame, 16~90 frame gap 통계를 만든다.

### 7.2 신규 함수

~~~python
def classify_identity_frames(evidence: dict) -> IdentityFrameSummary:
    """Classify frames without reading or storing face pixels."""

def group_contiguous_ranges(frame_indices: Sequence[int]) -> tuple[FrameRange, ...]:
    """Convert frame indices into deterministic contiguous ranges."""

def build_baseline_continuity_report(evidence: dict) -> ContinuityReport:
    """Measure flicker and unapproved runs before tracker integration."""
~~~

### 7.3 신규 타입

~~~python
@dataclass(frozen=True, slots=True)
class FrameRange:
    start_frame: int
    end_frame: int
    length: int

@dataclass(frozen=True, slots=True)
class ContinuityReport:
    frame_count: int
    approved_frame_count: int
    no_detection_ranges: tuple[FrameRange, ...]
    identity_weak_ranges: tuple[FrameRange, ...]
    single_frame_hole_count: int
    longest_identity_weak_range: int
~~~

### 7.4 통과 기준

- 기존 결과 파일을 변경하지 않는다.
- 기준선 report가 456 frame을 모두 분류한다.
- 서로 다른 그룹에 같은 frame이 중복되지 않는다.
- 통계는 local evidence만 사용하고 얼굴 이미지를 생성하지 않는다.

## 8. Phase 1 — Config schema v3와 강한 mosaic

### 8.1 신규 Config 필드

~~~python
mosaic_grid_cells: int = 8
mosaic_padding_ratio: float = 0.18
mosaic_min_block_px: int = 16
~~~

필드 의미:

| 필드 | 의미 | 강도 관계 |
| --- | --- | --- |
| mosaic_grid_cells | 얼굴의 짧은 변을 나눌 mosaic cell 수 | 작을수록 강함 |
| mosaic_padding_ratio | detection/track bbox를 사방으로 확장할 비율 | 클수록 넓게 가림 |
| mosaic_min_block_px | 작은 얼굴에 적용할 최소 pixel block 크기 | 클수록 강함 |

초기 권장값:

| profile | grid cells | padding | min block |
| --- | ---: | ---: | ---: |
| strong 기본 | 8 | 0.18 | 16 |
| maximum 비교용 | 5 | 0.25 | 24 |

profile 이름을 Config의 source of truth로 만들지는 않는다. 실제 저장값은 세 숫자이며, CLI 또는 문서 예제에서만 strong/maximum 조합을 설명한다.

### 8.2 수정 함수

- Config.__init__()
- Config.from_dict()
- Config.to_dict()
- MosaicEffect.render()
- _apply_effect_to_bbox()

### 8.3 신규 함수

~~~python
def expand_bbox(
    bbox: tuple[float, float, float, float],
    *,
    frame_shape: tuple[int, int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Expand and clip one bbox without changing its center identity."""

def mosaic_block_size(
    face_width: int,
    face_height: int,
    *,
    grid_cells: int,
    min_block_px: int,
) -> int:
    """Calculate adaptive block size from the shorter face side."""
~~~

### 8.4 renderer 계산

~~~text
short_side = min(face_width, face_height)
block_size = max(round(short_side / mosaic_grid_cells), mosaic_min_block_px)
small_width  = max(face_width  // block_size, 1)
small_height = max(face_height // block_size, 1)
~~~

pipeline의 force_block_size=8은 제거한다.

### 8.5 validation

- mosaic_grid_cells: bool이 아닌 2 이상 64 이하 정수
- mosaic_padding_ratio: finite float, 0.0 이상 0.5 이하
- mosaic_min_block_px: bool이 아닌 1 이상 256 이하 정수
- schema v1/v2는 기존 기본값으로 v3 memory representation에 migration
- strict=False의 unknown-key 호환 정책 유지
- strict=True의 unknown-key 거절 유지

### 8.6 통과 기준

- strong 설정의 큰 얼굴 ROI가 대략 8×aspect cell 수준으로 축소된다.
- ROI 내부의 세부 색상 수가 현재 force 8px 결과보다 명확히 감소한다.
- expanded ROI 밖 pixel은 원본과 byte-identical하다.
- edge clipping으로 음수/초과 slice가 생기지 않는다.
- 입력 frame을 mutate하지 않는다.

## 9. Phase 2 — Tracker protocol과 공통 타입

### 9.1 신규 파일

~~~text
src/consented_face_redactor/tracking/__init__.py
src/consented_face_redactor/tracking/protocol.py
src/consented_face_redactor/tracking/types.py
src/consented_face_redactor/tracking/geometry.py
~~~

### 9.2 Protocol

~~~python
class PointTracker(Protocol):
    @property
    def model_id(self) -> str: ...

    def reset(self) -> None: ...

    def initialize(
        self,
        frame: np.ndarray,
        *,
        frame_index: int,
        query_points: np.ndarray,
    ) -> PointTrackResult: ...

    def update(
        self,
        frame: np.ndarray,
        *,
        frame_index: int,
    ) -> PointTrackResult: ...
~~~

### 9.3 공통 타입

~~~python
@dataclass(frozen=True, slots=True)
class PointTrackResult:
    frame_index: int
    points_xy: np.ndarray
    visibility: np.ndarray
    model_revision: str

@dataclass(frozen=True, slots=True)
class TrackedFaceBox:
    bbox: tuple[float, float, float, float]
    visible_point_ratio: float
    inlier_point_count: int
    source: str

@dataclass(frozen=True, slots=True)
class TrackAuthorization:
    track_id: str
    profile_id: str
    gallery_revision: str
    origin_frame_index: int
    last_gallery_approval_frame: int

@dataclass(frozen=True, slots=True)
class TrackFrameDecision:
    frame_index: int
    track_id: str | None
    profile_id: str | None
    bbox: tuple[float, float, float, float] | None
    authorized: bool
    bbox_source: str
    visible_point_ratio: float | None
    review_required: bool
    reason_code: str
~~~

### 9.4 불변성

- NumPy array는 생성 시 copy하고 writeable=False로 설정한다.
- bbox는 frame 범위에 clip하기 전 원본 model output을 권한 신호로 사용하지 않는다.
- source와 reason_code는 통제된 문자열 집합을 사용한다.
- point tracker output의 길이와 finite 값을 검증한다.

### 9.5 통과 기준

- fake tracker와 TAPNext++ adapter가 동일 protocol test를 통과한다.
- malformed shape, NaN, Inf, visibility 길이 불일치는 명시적 예외다.
- tracker 결과에 approved 또는 GalleryApproval 필드를 넣지 않는다.

## 10. Phase 3 — 얼굴 query point와 bbox 복원

### 10.1 신규 함수

~~~python
def seed_face_points(
    bbox: tuple[float, float, float, float],
    landmarks: np.ndarray,
    *,
    grid_side: int = 4,
) -> np.ndarray:
    """Combine five landmarks and an inset face grid."""

def estimate_similarity_transform(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    visibility: np.ndarray,
) -> SimilarityTransform:
    """Estimate robust translation, scale, and rotation."""

def transform_bbox(
    bbox: tuple[float, float, float, float],
    transform: SimilarityTransform,
) -> tuple[float, float, float, float]:
    """Move the approved bbox using a robust point transform."""

def validate_tracked_bbox(
    previous_bbox,
    current_bbox,
    *,
    frame_shape,
    visible_point_ratio,
    limits,
) -> BboxValidation:
    """Reject impossible motion, scale, aspect, and visibility."""
~~~

### 10.2 point 배치

- YuNet의 5 landmarks를 포함한다.
- bbox 경계 바로 위가 아니라 10~15% 안쪽에 grid를 배치한다.
- 얼굴 표정 변화에 덜 민감한 눈·코 주변 point를 우선한다.
- 턱·입 point만으로 bbox가 계산되지 않도록 최소 inlier 수를 둔다.

### 10.3 bbox 복원

단순 min/max point box를 사용하지 않는다. 하나의 잘못된 point가 bbox 전체를 끌고 갈 수 있기 때문이다.

권장 순서:

1. visibility 기준으로 point 제거
2. 이전 point와 현재 point의 robust similarity transform 계산
3. median translation과 scale sanity check
4. 이전 approved/fused bbox에 transform 적용
5. 현재 YuNet detection과 일치하면 detection을 포함해 fuse
6. 렌더링 직전에 mosaic padding 적용

### 10.4 통과 기준

- 일부 point가 outlier여도 bbox 중심이 급격히 튀지 않는다.
- 회전된 얼굴에서 bbox가 완전히 붕괴하지 않는다.
- point visibility가 기준보다 낮으면 authorized bbox를 만들지 않는다.
- frame 밖으로 나간 bbox는 안전하게 clip된다.

## 11. Phase 4 — TAPNext++ adapter

### 11.1 신규 파일

src/consented_face_redactor/tracking/tapnextpp.py

### 11.2 신규 클래스

~~~python
class TapNextPlusPlusAdapter(PointTracker):
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        model_id: str,
        device: str,
        input_resolution: int = 256,
    ) -> None: ...
~~~

### 11.3 책임

- checkpoint 존재 확인
- manifest 검증 완료 경로만 허용
- lazy model initialization
- BGR uint8 입력 validation
- BGR→RGB 변환
- FHD 좌표↔model 좌표 변환
- torch.inference_mode 사용
- CUDA autocast 적용 가능성 측정
- frame index 단조 증가 검사
- video별 reset
- point와 visibility를 CPU float32 read-only array로 반환

### 11.4 신규 예외

~~~python
class TrackerInitializationError(RuntimeError): ...
class TrackerInferenceError(RuntimeError): ...
class TrackerContractError(ValueError): ...
~~~

### 11.5 model manifest

기존 manifest schema의 role=tracker를 사용한다.

필수 기록:

- model_id
- role=tracker
- source
- filename
- SHA-256
- license
- input_shape
- preprocessing_revision
- provider

자동 다운로드 코드는 만들지 않는다. checkpoint 획득은 별도 명령과 사용자 승인으로 수행하고, 획득 후 digest를 manifest에 고정한다.

### 11.6 환경 gate

현재 확인된 GPU:

- NVIDIA GeForce RTX 5070 Ti
- VRAM 약 16GB

검증 순서:

1. nvidia-smi
2. PyTorch CUDA availability
3. GPU architecture 지원
4. 16 point warm-up
5. 64 frame synthetic sequence
6. learn.mp4 일부 구간
7. test.mp4 일부 구간

CUDA OOM, unsupported kernel, checkpoint mismatch가 나오면 production 처리로 넘어가지 않는다.

## 12. Phase 5 — Detection과 track association

### 12.1 신규 파일

src/consented_face_redactor/tracking/association.py

### 12.2 신규 함수

~~~python
def association_cost(
    predicted_track: TrackedFaceBox,
    detection: FaceDetection,
    *,
    frame_shape,
) -> AssociationCost:
    """Calculate non-authorizing geometric continuity evidence."""

def associate_tracks_to_detections(
    tracks: Sequence[TrackedFaceBox],
    detections: Sequence[FaceDetection],
    *,
    policy: AssociationPolicy,
) -> AssociationResult:
    """Return one-to-one assignments plus unmatched items."""

def detect_crossing_ambiguity(
    assignments: AssociationResult,
    *,
    previous_assignments,
) -> tuple[Ambiguity, ...]:
    """Detect target transfer risk during overlap and crossing."""
~~~

### 12.3 association 신호

사용 가능:

- predicted bbox와 detection bbox IoU
- center distance / frame diagonal
- scale 및 aspect 변화
- tracked point가 detection 안에 남은 비율
- landmark geometry residual
- 직전 velocity와의 일관성

사용 금지:

- tracker 점수만으로 target profile 결정
- detection confidence만으로 target profile 결정
- 단순 bbox 최근접만으로 profile 승계

여러 얼굴이 있을 때 one-to-one assignment를 보장한다. 필요하면 Hungarian algorithm을 사용하되, hard gate를 통과하지 못한 pair는 assignment 대상에서 제외한다.

### 12.4 통과 기준

- 한 detection을 두 track이 동시에 소유하지 않는다.
- 한 track이 두 detection으로 분기하지 않는다.
- 교차 구간에서 target profile이 상대 얼굴로 이동하지 않는다.
- ambiguity는 누락하지 않고 review-required evidence가 된다.

## 13. Phase 6 — Anchor 기반 authorization

### 13.1 신규 파일

src/consented_face_redactor/tracking/authorization.py

### 13.2 핵심 용어

- anchor: 해당 frame에서 current GalleryApproval.approved=True가 직접 관측된 얼굴
- continuity evidence: tracker와 detection association이 같은 얼굴의 연속성을 지지하는 관측
- authorization propagation: anchor에서 시작한 profile 권한을 연속된 동일 track 구간에만 적용
- reacquisition: 끊어진 뒤 다시 등장한 얼굴을 gallery가 새로 승인하는 과정

### 13.3 신규 함수

~~~python
def create_authorized_track(
    approval: GalleryApproval,
    detection: FaceDetection,
    *,
    frame_index: int,
) -> AuthorizedTrack:
    """Create a track only from an explicit current approval."""

def refresh_authorized_track(
    track: AuthorizedTrack,
    approval: GalleryApproval,
    *,
    frame_index: int,
) -> AuthorizedTrack:
    """Refresh gallery evidence without changing profile identity."""

def may_propagate_authorization(
    track: AuthorizedTrack,
    observation: TrackObservation,
    *,
    policy: ContinuityPolicy,
) -> AuthorizationDecision:
    """Decide continuity; never create a new profile approval."""

def revoke_track_authorization(
    track: AuthorizedTrack,
    *,
    frame_index: int,
    reason_code: str,
) -> AuthorizedTrack:
    """End propagation after ambiguity, loss, or contradiction."""
~~~

### 13.4 초기 continuity 정책

| 설정 | 초기 관측값 | 의미 |
| --- | ---: | --- |
| tracker_only_max_frames | 12 | detection이 전혀 없을 때 tracker-only 유지 상한 |
| identity_refresh_max_frames | 90 | detection association은 유지되나 gallery anchor가 없는 구간 상한 |
| minimum_visible_point_ratio | 0.60 | bbox 복원에 필요한 visible point 비율 |
| max_scale_ratio_per_frame | 1.25 | 비정상 box 폭증 방지 |
| minimum_detection_iou | 관측 후 결정 | detection-track 결합 gate |

이 값은 security threshold라고 문서화하지 않는다. test.mp4와 별도 교차 시나리오에서 관측한 뒤 사람이 승인한다.

### 13.5 즉시 revoke 조건

- 다른 approved profile과 충돌
- track/detection association ambiguity
- visible point ratio 미달
- bbox area 또는 scale 급변
- frame 경계를 벗어나 유효 bbox가 사라짐
- model output malformed
- tracker reset 또는 frame 순서 역행
- gallery revision이 허용되지 않은 방식으로 변경

### 13.6 통과 기준

- tracker-only 입력으로 profile_id를 만들 수 없다.
- revoked track이 과거 approval을 재사용하지 않는다.
- 재등장한 얼굴은 gallery anchor 전까지 mosaic되지 않는다.
- propagated decision은 GalleryApproval과 구분되는 reason_code를 가진다.

권장 reason code:

- explicit_gallery_anchor
- tracked_from_explicit_approval
- bidirectional_anchor_consensus
- tracker_visibility_insufficient
- track_detection_ambiguous
- tracker_only_limit_exceeded
- identity_refresh_required
- conflicting_profile

## 14. Phase 7 — 양방향 batch tracking

### 14.1 선택 이유

입력이 저장된 영상이므로 미래 frame을 사용할 수 있다. 한쪽 anchor에서 끝까지 밀어붙이는 것보다 gap 뒤의 승인 anchor에서 역방향으로 추적해 두 경로가 같은 얼굴을 가리키는지 비교할 수 있다.

이 방식은 65 frame identity-weak 구간을 무조건 승인하는 것이 아니라, 구간 양쪽의 gallery anchor가 동일 profile이고 두 tracker 경로가 합의할 때만 채울 수 있게 한다.

### 14.2 신규 파일

src/consented_face_redactor/tracking/bidirectional.py

### 14.3 신규 타입

~~~python
@dataclass(frozen=True, slots=True)
class IdentityAnchor:
    frame_index: int
    profile_id: str
    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    gallery_revision: str

@dataclass(frozen=True, slots=True)
class TrackPath:
    direction: str
    profile_id: str
    decisions: tuple[TrackFrameDecision, ...]

@dataclass(frozen=True, slots=True)
class RedactionTrackPlan:
    input_frame_count: int
    profile_ids: tuple[str, ...]
    decisions: tuple[TrackFrameDecision, ...]
    ambiguous_ranges: tuple[FrameRange, ...]
~~~

### 14.4 신규 함수

~~~python
def collect_identity_anchors(
    frame_analysis: Sequence[FrameAnalysis],
) -> tuple[IdentityAnchor, ...]:
    """Collect only direct GalleryApproval anchors."""

def split_anchor_segments(
    anchors: Sequence[IdentityAnchor],
    *,
    max_gap_frames: int,
) -> tuple[AnchorSegment, ...]:
    """Create same-profile intervals eligible for reconciliation."""

def track_segment_forward(...) -> TrackPath:
    """Track from the left explicit anchor."""

def track_segment_backward(...) -> TrackPath:
    """Track reversed frames from the right explicit anchor."""

def reconcile_bidirectional_paths(
    forward: TrackPath,
    backward: TrackPath,
    detections: Sequence[FrameAnalysis],
    *,
    policy: ReconciliationPolicy,
) -> tuple[TrackFrameDecision, ...]:
    """Authorize only frames whose two paths and detections agree."""

def build_redaction_track_plan(...) -> RedactionTrackPlan:
    """Produce one deterministic decision per input frame."""
~~~

### 14.5 합의 규칙

중간 frame을 채우려면 다음을 모두 확인한다.

1. 양쪽 anchor의 profile_id가 같다.
2. 양쪽 anchor의 gallery revision이 허용 가능한 관계다.
3. forward/backward bbox가 충분히 겹치거나 중심/scale이 일관된다.
4. 해당 frame detection과의 association이 모순되지 않는다.
5. visible point ratio가 기준 이상이다.
6. 다른 얼굴과의 assignment가 모호하지 않다.

한 방향만 유효한 경우:

- 짧은 tracker-only gap에서는 제한적으로 사용할 수 있다.
- 긴 identity-weak gap에서는 자동 승인하지 않고 review-required로 남긴다.

### 14.6 통과 기준

- 동일 입력과 동일 모델에서 TrackPlan이 deterministic하다.
- frame당 target profile별 decision이 하나만 존재한다.
- plan의 frame 수가 input frame 수와 정확히 같다.
- 양방향 불일치가 성공으로 변환되지 않는다.

## 15. Phase 8 — 영상 분석과 렌더링 분리

### 15.1 신규 파일

src/consented_face_redactor/temporal_video_processor.py

### 15.2 신규 클래스

~~~python
class TemporalVideoProcessor:
    def analyze(
        self,
        source: FrameSource,
    ) -> RedactionTrackPlan: ...

    def render(
        self,
        source: FrameSource,
        destination: Path,
        plan: RedactionTrackPlan,
    ) -> ProcessingEvidence: ...
~~~

### 15.3 analyze pass

- detector와 gallery evaluation 수행
- raw frame/crop/embedding을 저장하지 않음
- bbox, landmarks, reason code, latency만 memory 또는 local evidence 구조에 유지
- anchor와 detection metadata 생성
- tracker forward/backward 실행
- 최종 plan 생성

### 15.4 render pass

- 입력 영상을 처음부터 다시 순차 읽기
- plan의 authorized bbox에만 mosaic 적용
- padding은 renderer 직전에 적용
- output은 sibling temporary video에 작성
- 모든 frame 성공 후 최종 destination으로 atomic move
- 실패하면 최종 결과 파일을 만들지 않음

### 15.5 기존 process_frame과의 관계

- image 처리와 tracker 비활성 video 경로는 RedactionPipeline.process_frame() 유지
- tracker 활성 batch video는 TemporalVideoProcessor 사용
- RedactionPipeline에 미래 frame이나 전체 video state를 억지로 주입하지 않음
- public benchmark API run_benchmark(category)와 generate_aggregate_report()는 유지

### 15.6 통과 기준

- 분석 pass는 output video를 생성하지 않는다.
- render pass는 identity 판단을 다시 바꾸지 않는다.
- plan과 output frame 수가 다르면 실패한다.
- 원본 경로와 output 경로가 같으면 writer 생성 전 거절한다.
- 기존 destination이 있으면 명시적으로 거절한다.

## 16. Phase 9 — CLI와 Config 계약

### 16.1 CLI 추가 옵션

~~~text
--tracker none|tapnextpp|cotracker3
--tracker-checkpoint PATH
--tracking-mode forward|bidirectional
--tracker-only-max-frames N
--identity-refresh-max-frames N
~~~

모자이크 강도는 config source of truth로 유지한다.

### 16.2 runtime 조립 함수

수정:

- _add_runtime_arguments()
- _load_verified_model_entries()
- _load_runtime_components()
- _cmd_process_video()

신규:

~~~python
def _load_verified_tracker_entry(
    model_dir: Path,
    manifest_dir: Path,
    *,
    tracker_name: str,
) -> dict[str, object]:
    """Return one hash-verified tracker entry."""

def _build_tracker_runtime(
    args: argparse.Namespace,
    manifest_entry: dict[str, object],
) -> PointTracker:
    """Construct the requested tracker without downloading assets."""

def _build_temporal_processor(
    args: argparse.Namespace,
    runtime: RuntimeComponents,
) -> TemporalVideoProcessor:
    """Compose identity, tracker, policy, and renderer dependencies."""
~~~

### 16.3 CLI validation

- tracker=none이면 tracker checkpoint가 없어야 한다.
- tracker가 활성화되면 checkpoint와 manifest가 모두 필요하다.
- partial tracker configuration은 오류다.
- bidirectional은 seek 가능한 video source에서만 허용한다.
- image command에는 bidirectional tracking을 노출하지 않는다.
- dry-run은 TrackPlan/evidence만 만들고 output video를 쓰지 않는다.

### 16.4 Config schema

schema_version을 3으로 올린다.

v1/v2 입력:

- 읽기 가능
- 새 mosaic 필드는 v3 기본값 사용
- 다음 to_dict()에서 v3로 직렬화

strict mode:

- v3 known key만 허용
- tracker model path는 Config에 넣지 않음

## 17. Phase 10 — Evidence 확장

### 17.1 frame별 필드

~~~json
{
  "frame_index": 120,
  "track_id": "track-0001",
  "profile_id": "prof-00000000",
  "authorized": true,
  "bbox_source": "bidirectional_anchor_consensus",
  "visible_point_ratio": 0.875,
  "inlier_point_count": 14,
  "last_gallery_approval_frame": 82,
  "continuity_age": 38,
  "review_required": false,
  "reason_code": "tracked_from_explicit_approval"
}
~~~

bbox 좌표는 기존처럼 evaluation-evidence opt-in일 때만 저장한다.

### 17.2 summary 필드

- direct_anchor_frames
- propagated_redaction_frames
- tracker_only_frames
- identity_weak_frames_recovered
- ambiguous_frames
- rejected_track_transfers
- longest_propagated_range
- tracker_model_id
- tracker_model_revision
- tracking_mode
- tracker_p50_latency_ms
- tracker_p95_latency_ms
- maximum_vram_mb
- detector/gallery/tracker reason counts
- mosaic_grid_cells
- mosaic_padding_ratio

### 17.3 개인정보 조건

저장 금지:

- frame image
- face crop
- embedding vector
- tracker feature tensor
- 실제 사람 이름

허용:

- opaque profile ID
- frame index
- bbox opt-in
- 통제된 reason code
- aggregate metric

## 18. Phase 11 — Unit 및 contract test

### 18.1 Mosaic test

- adaptive block size
- strong/maximum parameter 차이
- ROI unique-color 감소
- padding
- frame edge clipping
- multiple ROI locality
- input immutability

### 18.2 Protocol test

- fake tracker contract
- TAPNext++ adapter output shape
- read-only arrays
- NaN/Inf rejection
- frame ordering
- reset
- initialization before update

### 18.3 Authorization test

- no gallery anchor, no authorized track
- explicit anchor starts one track
- tracker confidence alone cannot authorize
- same-profile refresh
- conflicting profile revokes
- expired approval cannot initialize
- gallery revision mismatch
- revoked track cannot revive itself

### 18.4 Temporal scenario test

1. 승인 → identity-weak 1 frame → 승인
2. 승인 → identity-weak 65 frames → 같은 profile 승인
3. 승인 → detector missing 2 frames → 승인
4. 승인 대상과 비승인 얼굴 교차
5. target 완전 퇴장 후 다른 얼굴 등장
6. target 퇴장 후 재등장과 gallery reacquisition
7. forward/backward 합의
8. forward/backward 불일치
9. tracker exception
10. malformed point output
11. FHD portrait
12. FHD landscape

### 18.5 CLI test

- tracker flags all-or-nothing
- manifest role=tracker
- checkpoint hash mismatch
- tracker=none compatibility
- bidirectional dry-run
- output existing rejection
- evidence field contract

## 19. Phase 12 — Benchmark 확장

기존 A~E category를 깨지 않는다.

권장 확장:

- Category B: strong mosaic strength/locality 추가
- Category C: tracker temporal continuity 추가
- Category D: tracker off/on FHD 성능 비교
- 신규 Category F가 필요하면 public contract 문서와 테스트를 같은 patch에서 변경

측정:

- synthetic 1080×1920
- 실제 test.mp4
- tracker warm-up 제외 p50/p95
- VRAM
- analysis pass FPS
- render pass FPS
- 전체 elapsed

성능 수치는 환경 측정값으로 보고하고 승인 전 pass/fail threshold로 고정하지 않는다.

## 20. Phase 13 — 실제 영상 비교 실행

### 20.1 결과 파일

기존 결과 보존:

- local_data/output/test_target_mosaic_verified.mp4

신규 결과:

- local_data/output/test_target_mosaic_tracker_strong.mp4
- local_data/output/test_target_mosaic_tracker_maximum.mp4
- local_data/evidence/test_tracking_comparison.json

### 20.2 비교 항목

| 항목 | 기존 | 신규 목표 |
| --- | ---: | --- |
| 전체 frame | 456 | 456 |
| direct approved frames | 209 | 기준선으로 유지 |
| detector 없는 frames | 2 | tracker-only 조건부 복원 |
| detected identity-weak frames | 245 | anchor 합의 구간만 복원 |
| 1~2 frame flicker | 기준선 계산 | 0 목표 |
| wrong-person mosaic | 시각 검토 | 0 필수 |
| output 크기/FPS | 1080×1920 / 30 | 동일 |

### 20.3 사람 검토가 필요한 구간

자동 metric만으로 대상자 여부를 완전히 증명하지 않는다.

다음을 별도 timestamp 목록으로 제공한다.

- 가장 긴 65 frame gap
- 두 얼굴이 겹치거나 교차한 구간
- 양방향 결과가 불일치한 구간
- tracker-only bbox가 사용된 구간
- strong/maximum 강도 비교 구간

### 20.4 통과 기준

- target으로 확인된 구간에서 눈에 보이는 mosaic 깜빡임이 없어야 한다.
- 다른 사람에게 mosaic가 이동한 frame이 없어야 한다.
- 얼굴 전체가 strong mosaic ROI 안에 포함되어야 한다.
- 원본 test.mp4 SHA-256이 실행 전후 동일해야 한다.
- 결과 영상이 OpenCV로 열리고 456 frame을 모두 decode할 수 있어야 한다.
- 전체 pytest 통과
- git diff --check 통과
- commit/push 없음

## 21. 실패 처리와 중단 조건

다음 상황에서는 결과를 성공으로 보고하지 않는다.

- tracker checkpoint hash 불일치
- PyTorch/CUDA unsupported kernel
- tracker output NaN/Inf
- frame 수 불일치
- target track이 다른 얼굴로 이동한 관측
- 양방향 경로가 반복적으로 불일치
- output writer가 중간 실패
- 기존 결과 파일 overwrite 시도
- tracker dependency가 기존 YuNet/SFace runtime을 깨뜨림

실패 시:

1. 최종 destination을 만들지 않는다.
2. 임시 video를 정리한다.
3. failure reason을 local evidence에 기록한다.
4. 기존 tracker 없는 결과를 보존한다.
5. fallback 후보로 넘어갈지는 별도 검증 결과 후 결정한다.

## 22. 구현 순서와 patch 경계

변경을 다음 순서의 작은 patch로 나눈다.

1. 기준선 report helper와 test
2. Config v3 및 mosaic 강도
3. tracking protocol/types/geometry
4. fake tracker와 protocol test
5. TAPNext++ adapter와 manifest gate
6. detection-track association
7. authorization propagation
8. forward tracking
9. backward tracking과 reconciliation
10. TemporalVideoProcessor
11. CLI wiring
12. evidence
13. benchmark
14. 실제 test.mp4 실행
15. 문서와 handoff

각 patch 뒤에 targeted pytest와 git diff --check를 실행한다. 전체 pytest는 Phase 1, 5, 9, 12, 14 완료 후 실행한다.

## 23. 수정 파일 목록

### 기존 파일

- .gitignore
- README.md
- docs/CONFIG_SCHEMA.md
- docs/CODEBASE_FUNCTION_REFERENCE.md
- docs/PHASE10_BENCHMARK_PROTOCOL.md
- docs/REAL_VIDEO_TEST_GUIDE.md
- src/consented_face_redactor/config.py
- src/consented_face_redactor/domain/types.py
- src/consented_face_redactor/effects/mosaic.py
- src/consented_face_redactor/model_manifest.py
- src/consented_face_redactor/pipeline.py
- src/consented_face_redactor/cli.py
- src/consented_face_redactor/benchmark/runner.py

### 신규 파일

- src/consented_face_redactor/tracking/__init__.py
- src/consented_face_redactor/tracking/protocol.py
- src/consented_face_redactor/tracking/types.py
- src/consented_face_redactor/tracking/geometry.py
- src/consented_face_redactor/tracking/tapnextpp.py
- src/consented_face_redactor/tracking/association.py
- src/consented_face_redactor/tracking/authorization.py
- src/consented_face_redactor/tracking/bidirectional.py
- src/consented_face_redactor/temporal_video_processor.py
- tests/test_mosaic_strength.py
- tests/test_tracking_protocol.py
- tests/test_tracking_geometry.py
- tests/test_tracking_association.py
- tests/test_tracking_authorization.py
- tests/test_bidirectional_tracking.py
- tests/test_temporal_video_processor.py
- tests/test_cli_tracking.py

## 24. 완료 산출물

구현 완료 시 다음을 제공한다.

1. strong adaptive mosaic 구현
2. hash-verified TAPNext++ tracker adapter
3. tracker와 gallery 권한이 분리된 temporal track manager
4. forward/backward anchor reconciliation
5. tracker 활성/비활성 CLI
6. tracker evidence와 continuity report
7. 기존/strong/maximum 결과 비교
8. 실제 test.mp4 최종 결과 영상
9. 전체 pytest 결과
10. 변경 파일 및 미해결 위험 목록

## 25. 최종 승인 체크리스트

- [ ] force_block_size=8 하드코딩 제거
- [ ] Config v3 migration 및 strict contract 통과
- [ ] strong mosaic ROI 강도 확인
- [ ] tracker가 GalleryApproval을 생성하지 않음
- [ ] explicit anchor 없이는 target track 시작 불가
- [ ] profile ID 자동 전파/교체 금지
- [ ] crossing ambiguity test 통과
- [ ] 65 frame identity-weak 구간 검증
- [ ] bidirectional disagreement가 review-required로 남음
- [ ] target 재등장 시 gallery 재승인
- [ ] 원본 영상 hash 보존
- [ ] output frame/FPS/해상도 보존
- [ ] local_data Git ignore 유지
- [ ] 전체 pytest 통과
- [ ] git diff --check 통과
- [ ] commit/push하지 않음

## 26. 최종 설계 결론

이번 문제는 단순 detector threshold 조정으로 해결할 수 없다. 실제로 얼굴 검출이 없는 frame은 2개뿐이며, 대부분의 끊김은 매 frame 신원을 새로 판단하면서 이전 frame의 동일 객체 정보를 버리는 구조에서 발생한다.

따라서 구현의 중심은 다음과 같아야 한다.

~~~text
명시 승인된 얼굴 anchor
  + TAPNext++ 얼굴 내부 point tracking
  + detection association
  + forward/backward 경로 합의
  + tracker와 identity 권한의 분리
  + 얼굴 크기 기반 strong adaptive mosaic
~~~

이 구조는 대상 얼굴의 각도와 표정 때문에 SFace 점수가 일시적으로 낮아지는 구간을 시간축 연속성으로 복원하면서도, tracker가 다른 사람에게 승인 권한을 넘기지 못하도록 제한한다.

이 문서의 모든 단계는 구현 계획이며, 체크박스는 코드·테스트·실제 영상 결과가 확인되기 전까지 완료로 표시하지 않는다.

## 27. 후속 작업 — 완화된 외접 타원 mosaic

사용자가 최종 결과를 검토한 뒤 “강도를 조금 낮추고, bbox 사각형 대신 사각형에 외접하는 과하지 않은 세로 타원”을 요청했다. 이 후속 변경은 tracker와 gallery 권한을 수정하지 않고 effect renderer에만 제한했다.

### 27.1 구현 계약

- Config schema를 v4로 올리고 v1/v2/v3 읽기 호환을 유지한다.
- production 기본은 `mosaic_shape="ellipse"`다.
- 가로/세로 반축 배율은 1.40/1.50이며 bbox 네 모서리를 포함해야 한다.
- v3의 grid 8/minimum 16px을 grid 12/minimum 10px로 바꿔 강도를 완화한다.
- 타원 밖 pixel은 입력과 동일하게 보존한다.
- `rectangle` mode와 18% padding은 이전 동작 재현을 위한 호환 옵션으로 유지한다.
- 저수준 `MosaicConfig()`의 기본 shape은 기존 직접 호출자를 깨지 않도록 rectangle을 유지하되, `Config.default()`를 받는 production pipeline 기본은 ellipse다.

### 27.2 검증 결과

- 타원 bounds, mask 외부 보존, bbox 모서리 포함, frame-edge clipping을 unit test로 확인했다.
- 기존 mosaic/sticker/temporal/config targeted test 61개가 통과했다.
- 전체 pytest 321개가 통과했다.
- 실제 `test.mp4`를 TAPNext++ 양방향 mode로 다시 처리해 456/456 frame을 redaction했다.
- 입력/출력은 1080×1920, 30fps, 456 frame으로 일치했다.
- 원본, v3 rectangle, v4 ellipse의 7개 대표 frame 비교 sheet를 생성해 시각 점검했다.
- 산출물은 `local_data/output/test_target_mosaic_tapnextpp_v4_ellipse.mp4`이며 원본이나 v3 결과를 덮어쓰지 않았다.
- 모든 변경은 local clone에만 있으며 commit/push하지 않았다.
