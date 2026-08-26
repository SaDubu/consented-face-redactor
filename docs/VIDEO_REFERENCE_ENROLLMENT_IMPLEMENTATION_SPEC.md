# 등록 영상 기반 대상 얼굴 Mosaic 구현 명세

> 구현 상태: 2026-08-25 기준 Phase 0–8의 코드·synthetic test 경로를 local clone에 구현했다. 실제 model binary와 동의된 FHD 영상으로 하는 smoke/quality 평가는 asset 제공 후 별도로 실행한다.

## 1. 목적

이 명세의 목표는 사용자가 제공한 **대상 인물만 포함된 등록 영상**에서 다양한 외형 상태의 얼굴 embedding을 하나의 profile에 저장하고, 별도 입력 영상의 매 frame에서 그 profile과 명시 approval이 모두 확인된 얼굴에만 mosaic를 적용하는 것이다.

목표는 정면·고품질 사진을 고르는 것이 아니다. 등록 영상에 포함된 좌우 회전, 상하 각도, 거리, 표정, 조명, 안경·모자 등 다양한 모습이 embedding reference 집합에 충분히 반영되도록 하는 것이다.

다음은 범위에 포함한다.

- YuNet detector와 SFace embedder 사용
- target-only 등록 영상의 다중-reference enrollment
- 대상 profile의 explicit approval 분리 저장
- 입력 영상의 frame별·얼굴별 독립 판별
- 승인된 얼굴 ROI만 mosaic/sticker 처리
- FHD 이상 입력의 실제 측정과 evidence 기록
- 원본 비디오를 덮어쓰지 않는 별도 결과 생성

다음은 이번 설계에서 의도적으로 제외한다.

- object tracker, tracker model, optical flow, 장기 identity association
- cloud face API, 자동 모델 다운로드, 외부 업로드
- detector confidence 또는 similarity만으로 redaction 승인
- 원본 영상·crop·embedding을 기본 evidence에 저장

## 2. 변경 전제와 고정 안전 규칙

현재 runtime은 `GalleryApproval.approved is True`일 때만 mosaic를 허용한다. 이 권한 경계는 모든 phase에서 유지한다.

```text
detector confidence             권한 아님
embedding similarity            권한 아님
gallery profile 존재            권한 아님
과거 frame의 승인              권한 아님
ApprovalStore의 approved=True  유일한 권한 신호
```

`t_confirm`, `t_keep`은 계속 telemetry/calibration 값으로만 남긴다. 이 기능을 구현하면서 security threshold라고 재정의하지 않는다.

## 3. 최종 처리 구조

```text
[등록 영상: target-only]
  OpenCvFrameSource
    → YuNet: face 1개인지 확인
    → SFace: embedding 생성
    → 중복 제거 및 diversity coverage 선택
    → LocalGallery profile에 다수 reference 저장
    → ApprovalStore에 명시 승인 저장

[처리 영상]
  frame마다 YuNet으로 모든 얼굴 탐지
    → 얼굴마다 SFace embedding
    → LocalGallery의 다수 reference와 비교
    → ApprovalStore의 current explicit approval 확인
    → approved=True인 bbox만 mosaic
    → 별도 output video + 선택적 evidence JSON
```

## 4. Phase 0 — 공통 타입과 정책 경계

### 4.1 신규 파일

`src/consented_face_redactor/video_enrollment.py`

### 4.2 신규 타입

```python
@dataclass(frozen=True, slots=True)
class VideoEnrollmentOptions:
    sample_every_n_frames: int
    max_references: int
    duplicate_similarity: float
    min_face_width: int
    min_face_height: int
    max_review_candidates: int

@dataclass(frozen=True, slots=True)
class EnrollmentCandidate:
    frame_index: int
    timestamp_s: float
    embedding: np.ndarray
    detector_confidence: float

@dataclass(frozen=True, slots=True)
class EnrollmentSkip:
    frame_index: int
    timestamp_s: float
    reason_code: str

@dataclass(frozen=True, slots=True)
class EnrollmentReport:
    input_frame_count: int | None
    sampled_frame_count: int
    candidate_count: int
    selected_reference_count: int
    no_face_count: int
    multiple_face_count: int
    face_too_small_count: int
    embedding_error_count: int
    duplicate_count: int
    review_frame_indices: tuple[int, ...]
    nearest_similarity_min: float | None
    nearest_similarity_median: float | None
    nearest_similarity_p95: float | None
```

### 4.3 각 필드의 의미

| 필드 | 역할 | 권한 여부 |
| --- | --- | --- |
| `sample_every_n_frames` | 영상 전체를 메모리에 쌓지 않기 위한 sampling 간격 | 권한 아님 |
| `max_references` | 하나의 profile에 저장할 최대 diverse reference 수 | 권한 아님 |
| `duplicate_similarity` | 거의 동일한 외형 vector의 반복 저장을 막는 deduplication 기준 | 권한 아님 |
| `min_face_width/height` | embedding이 무의미해질 정도로 작은 detection을 거르는 입력 유효성 기준 | 권한 아님 |
| `review_frame_indices` | 자동 등록을 보류한 frame 번호. 원본 등록 영상에서 사람이 확인할 수 있음 | 권한 아님 |

`EnrollmentCandidate`에는 원본 frame, crop, 이름을 넣지 않는다. 등록 영상은 사용자가 보관하며, 보고서는 frame 번호와 통계만 남긴다.

### 4.4 통과 기준

- 모든 타입은 immutable dataclass다.
- 입력 frame 또는 embedding array를 외부에서 변경할 수 없도록 필요 시 copy/read-only 처리한다.
- `duplicate_similarity`는 `[0, 1]` finite float만 받는다.
- `max_references`, sampling 간격, 최소 face size는 양의 정수만 받는다.

## 5. Phase 1 — 등록 영상 frame 순회와 후보 추출

### 5.1 수정 대상

- [media/frame_source.py](../src/consented_face_redactor/media/frame_source.py): 수정 불필요. 기존 `OpenCvFrameSource` 재사용.
- [adapters/opencv.py](../src/consented_face_redactor/adapters/opencv.py): 수정 불필요. 기존 YuNet/SFace 재사용.
- 신규 [video_enrollment.py](../src/consented_face_redactor/video_enrollment.py): 후보 추출 구현.

### 5.2 신규 함수

```python
def iter_sampled_frames(
    source: FrameSource,
    *,
    sample_every_n_frames: int,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield sampled BGR frames without retaining prior frames."""
```

동작:

1. `source.open()`은 caller가 수행한다.
2. `source.read()`로 끝까지 순차 읽기 한다.
3. frame index가 sampling 간격에 맞을 때만 yield한다.
4. timestamp는 `frame_index / fps`로 만들되 fps가 없으면 frame index를 관측값으로 둔다.
5. frame을 저장하거나 crop file을 생성하지 않는다.

```python
def extract_enrollment_candidate(
    frame: np.ndarray,
    *,
    frame_index: int,
    timestamp_s: float,
    detector: DetectorAdapter,
    embedder: EmbedderAdapter,
    options: VideoEnrollmentOptions,
) -> EnrollmentCandidate | EnrollmentSkip:
    """Return one target-only enrollment candidate or an auditable skip."""
```

동작:

1. `detector.detect(frame)`를 한 번 호출한다.
2. detection이 없으면 `EnrollmentSkip(..., "no_face")`.
3. detection이 둘 이상이면 `EnrollmentSkip(..., "multiple_faces")`.
4. 한 얼굴이 최소 폭/높이보다 작으면 `EnrollmentSkip(..., "face_too_small")`.
5. `embedder.embed(frame, detection)`이 실패하면 `EnrollmentSkip(..., "embedding_error")`.
6. embedding의 shape, finite 값, L2 norm을 검증한다.
7. 통과한 후보만 `EnrollmentCandidate`로 반환한다.

### 5.3 중요한 정책

이 phase는 “얼굴이 예쁜지”나 “정면인지”로 후보를 거르지 않는다. 얼굴 하나가 있는 target-only 영상이라는 전제를 확인하고, 모델 입력으로 사용할 수 있는 embedding인지 만 검증한다.

등록 영상에 다른 사람이 등장하면 자동으로 누구를 target이라고 추론하지 않는다. 해당 frame은 `multiple_faces`로 report에 남기고 제외한다.

### 5.4 테스트

신규 `tests/test_video_enrollment.py`에 다음을 추가한다.

- 빈 영상/프레임에서 `no_face`
- 두 detection에서 `multiple_faces`
- 작은 bbox에서 `face_too_small`
- embedder 예외에서 `embedding_error`
- valid detection에서 normalized candidate 반환
- source가 순차 read되고 모든 frame이 메모리에 누적되지 않음

## 6. Phase 2 — 관점 coverage를 위한 reference 선택

### 6.1 신규 함수

```python
def nearest_reference_similarity(
    candidate: np.ndarray,
    references: Sequence[np.ndarray],
) -> float | None:
    """Return the highest cosine similarity to selected references."""

def select_diverse_references(
    candidates: Sequence[EnrollmentCandidate],
    *,
    options: VideoEnrollmentOptions,
) -> tuple[list[EnrollmentCandidate], list[EnrollmentCandidate], int]:
    """Return selected references, review candidates, and duplicate count."""
```

### 6.2 선택 알고리즘

1. 첫 valid candidate는 reference로 선택한다.
2. 새 candidate의 nearest similarity가 `duplicate_similarity` 이상이면 거의 같은 모습으로 보고 duplicate count만 늘린다.
3. 그보다 낮으면 새 외형 상태로 보고 selection 후보에 둔다.
4. `max_references` 전까지는 새로운 후보를 추가한다.
5. 최대 reference 수를 넘으면 farthest-point coverage 전략을 쓴다. 즉 현재 reference들과 가장 덜 비슷한 candidate가 coverage를 넓히므로 우선 남긴다.
6. 기존 selected set과 지나치게 다른 candidate는 자동 폐기하지 않고 review 후보로 넣는다. target-only 영상이라면 극단적인 옆얼굴·가림일 수 있으므로 즉시 오염으로 단정하지 않는다.

### 6.3 review 처리

초기 구현에서는 review candidate의 원본 crop을 저장하지 않는다. `frame_index`와 reason만 report에 남긴다. 사용자는 등록 원본 영상을 열어 해당 frame을 확인할 수 있다.

후속 개선에서만 명시 opt-in `--review-export-dir`를 추가할 수 있다. 이 경우에도 원본 얼굴 crop은 기본 저장하지 않고, 사용자가 지정한 로컬 directory에만 생성한다.

### 6.4 통과 기준

- 연속한 동일 얼굴 frame이 profile reference 수를 무한정 늘리지 않는다.
- 시간상 떨어진 옆얼굴/표정 변화 embedding이 reference에 남는다.
- target-only 전제가 깨질 수 있는 다중 얼굴 frame은 profile을 오염시키지 않는다.
- `duplicate_similarity`는 redaction 권한으로 사용되지 않는다.

## 7. Phase 3 — LocalGallery batch 등록과 match diagnostics

### 7.1 수정 대상

[gallery.py](../src/consented_face_redactor/gallery.py)

### 7.2 신규 타입과 함수

```python
@dataclass(frozen=True, slots=True)
class ProfileSimilarity:
    best_similarity: float
    best_reference_index: int | None
    centroid_similarity: float

def _profile_similarity_details(
    self,
    vector: np.ndarray,
    data: dict[str, Any],
) -> ProfileSimilarity:
    ...

def enroll_many(
    self,
    embeddings: Sequence[np.ndarray],
    *,
    profile_id: str | None = None,
) -> str:
    ...
```

### 7.3 기존 함수 수정 방식

| 기존 함수 | 수정 내용 |
| --- | --- |
| `_profile_similarity()` | 삭제하지 않는다. `_profile_similarity_details(...).best_similarity`를 반환하는 compatibility wrapper로 바꾼다. |
| `match()` | 현재 반환 타입 `list[MatchResult]`를 유지한다. 내부에서 상세 similarity를 계산해 best score를 사용한다. public API는 깨지지 않는다. |
| `enroll()` | 단일 image enrollment 용도로 그대로 유지한다. batch 등록은 `enroll_many()`가 담당한다. |
| `to_dict()/from_dict()` | profile JSON schema는 기존 vector list 표현을 유지한다. reference별 metadata를 넣지 않아 PII·schema 복잡도를 늘리지 않는다. |

### 7.4 `enroll_many()`의 atomic 규칙

1. 현재 gallery를 `to_dict()` → `from_dict()`로 staging clone한다.
2. 모든 embedding을 staging clone에만 등록한다.
3. dimension 불일치, duplicate, 다른 profile collision, 비정상 vector가 하나라도 있으면 원본 gallery는 변경하지 않는다.
4. 모두 성공했을 때만 원본 internal state를 staging state로 교체한다.
5. caller는 성공한 `profile_id` 하나만 받는다.

### 7.5 테스트

신규 `tests/test_gallery_similarity_diagnostics.py`:

- best reference와 centroid similarity가 분리 계산되는지
- `enroll_many()` 중간 오류 시 원본 gallery가 그대로인지
- 여러 reference가 하나의 profile에 저장되는지
- 다른 profile과 collision하면 전체 batch가 rollback되는지

## 8. Phase 4 — VideoEnrollmentService

### 8.1 신규 클래스

`video_enrollment.py`에 다음 class를 추가한다.

```python
class VideoEnrollmentService:
    def __init__(
        self,
        *,
        detector: DetectorAdapter,
        embedder: EmbedderAdapter,
        options: VideoEnrollmentOptions,
    ) -> None:
        ...

    def collect(
        self,
        source: FrameSource,
    ) -> tuple[list[EnrollmentCandidate], EnrollmentReport]:
        ...

    def select(
        self,
        candidates: Sequence[EnrollmentCandidate],
    ) -> tuple[list[EnrollmentCandidate], EnrollmentReport]:
        ...

    def enroll(
        self,
        source: FrameSource,
        gallery: LocalGallery,
        *,
        profile_id: str | None = None,
    ) -> tuple[str, EnrollmentReport]:
        ...
```

### 8.2 책임 분리

| 메서드 | 책임 | state 변경 |
| --- | --- | --- |
| `collect()` | frame 순회, candidate/skip 통계 생성 | 없음 |
| `select()` | embedding diversity 기준 selected/review 계산 | 없음 |
| `enroll()` | selected embedding을 `gallery.enroll_many()`에 전달 | gallery만, 성공 시 한 번 |

이 분리로 CLI가 `--dry-run`을 제공할 수 있다. dry-run은 collect/select/report만 실행하며 gallery·approval JSON을 수정하지 않는다.

## 9. Phase 5 — CLI command와 approval 저장

### 9.1 수정 대상

[cli.py](../src/consented_face_redactor/cli.py)

### 9.2 `_build_parser()` 수정

새 subcommand `gallery-enroll-video`를 추가한다.

```text
gallery-enroll-video
  --input PATH                 등록 영상
  --gallery-db PATH            LocalGallery JSON
  --approval-db PATH           ApprovalStore JSON
  --model-dir PATH             검증할 model binary directory
  --manifest-dir PATH          model manifest directory
  --sample-every-n-frames INT
  --max-references INT
  --profile-id ID              기존 profile에 reference를 추가할 때만
  --dry-run
  --report-out PATH
  --approve
  --approval-reason TEXT
```

### 9.3 신규 helper

```python
def _build_enrollment_runtime(
    args: argparse.Namespace,
) -> tuple[OpenCvYuNetDetector, OpenCvSFaceEmbedder]:
    ...

def _load_gallery_and_approval_stores(
    gallery_path: Path,
    approval_path: Path,
) -> tuple[LocalGallery, ApprovalStore]:
    ...

def _write_enrollment_report(
    path: Path | None,
    report: EnrollmentReport,
) -> None:
    ...

def _cmd_gallery_enroll_video(args: argparse.Namespace) -> int:
    ...
```

`_build_enrollment_runtime()`은 이미 존재하는 `_load_verified_model_entries()`를 재사용한다. YuNet/SFace model hash 검증과 adapter 생성 코드를 image/video enrollment 사이에 중복하지 않는다.

### 9.4 `_cmd_gallery_enroll()` 리팩터링

기존 단일 이미지 enrollment는 삭제하지 않는다. 다만 실제 모델 검증과 gallery/approval load 코드를 새 helper로 이동한다.

```text
_cmd_gallery_enroll
  → _build_enrollment_runtime
  → 단일 frame의 extract_enrollment_candidate
  → gallery.enroll
  → ApprovalStore.set
```

### 9.5 approval 규칙

- `--approve` 없이 등록하면 `ApprovalRecord(False, "enrolled_pending_approval")`.
- `--approve`가 있으면 non-empty `--approval-reason`을 필수로 요구.
- 등록 영상 처리와 approval 행위는 구분.
- gallery 저장 또는 approval 저장이 실패하면 성공 메시지를 내지 않음.

## 10. Phase 6 — 실제 영상 처리의 diagnostics와 안전성

### 10.1 수정 대상

- [approved_gallery.py](../src/consented_face_redactor/approved_gallery.py)
- [pipeline.py](../src/consented_face_redactor/pipeline.py)
- [cli.py](../src/consented_face_redactor/cli.py)

### 10.2 `ApprovedLocalGalleryAdapter.evaluate()`

기존 함수는 그대로 public runtime 진입점으로 유지한다.

```python
def evaluate(
    self,
    frame: np.ndarray,
    detection: FaceDetection,
) -> GalleryApproval:
    ...
```

수정 내용:

- `LocalGallery`의 `ProfileSimilarity` 진단을 내부 관측값으로 보관.
- `approved=True`를 반환하는 조건은 바꾸지 않음.
- low similarity는 `similarity_insufficient`.
- profile은 있으나 approval record가 없으면 `profile_not_explicitly_approved`.
- false record는 `profile_not_approved`.
- 만료 record는 `approval_expired`.

새 관측 타입:

```python
@dataclass(frozen=True, slots=True)
class MatchObservation:
    profile_id: str | None
    best_similarity: float | None
    best_reference_index: int | None
    centroid_similarity: float | None
    reason_code: str
```

이 타입은 report/evaluation에만 사용한다. pipeline은 계속 `GalleryApproval`만 권한 신호로 사용한다.

### 10.3 `RedactionPipeline` 확장

현재 `_process_face_by_face_approvals()`는 approved ROI만 effect 처리한다. 이 규칙은 유지한다.

추가 타입:

```python
@dataclass(frozen=True, slots=True)
class FaceDecision:
    bbox: tuple[float, float, float, float]
    approval: GalleryApproval
```

추가 property:

```python
@property
def last_frame_decisions(self) -> tuple[FaceDecision, ...]:
    ...
```

수정 함수:

```python
def _process_face_by_face_approvals(...):
    # bboxes와 approvals를 zip하여 FaceDecision 저장
    # approved=True bbox만 _apply_effect_to_bbox로 전달
```

기본 evidence에는 bbox를 저장하지 않는다. bbox가 필요할 때만 `--evaluation-evidence`처럼 별도 opt-in을 추가한다.

## 11. Phase 7 — FHD 성능, output 보호, audio 선택 기능

### 11.1 `cli.py`의 output overwrite 보호

새 helper:

```python
def _reject_output_overwrite(input_path: Path, output_path: Path) -> None:
    """Reject resolved input/output path equality before opening VideoWriter."""
```

수정 함수:

- `_cmd_process_image()`
- `_cmd_process_video()`

두 command는 output path가 input path와 같으면 exit code 2로 실패해야 한다. 원본 overwrite 금지는 옵션이 아니라 고정 동작이다.

### 11.2 evidence summary 확장

새 helper:

```python
def _build_processing_summary(
    rows: Sequence[dict[str, object]],
    *,
    frame_shape: tuple[int, int, int],
    elapsed_seconds: float,
    source_fps: float,
) -> dict[str, object]:
```

기록 항목:

- FHD 이상 여부와 frame shape
- 처리 frame 수
- p50/p95 latency, 실측 FPS
- 얼굴 수와 승인 얼굴 수
- reason code별 count
- OpenCV/Numpy/Python/platform/CPU 정보

성능 수치는 보고값이다. 사용자가 하드웨어별 목표를 승인하기 전에는 pass/fail security 기준으로 사용하지 않는다.

### 11.3 audio 보존은 별도 phase

현재 OpenCV `VideoWriter`는 audio를 복사하지 않는다. audio가 필요하면 신규 `src/consented_face_redactor/media/remux.py`에 다음만 추가한다.

```python
def remux_original_audio(
    *,
    original_video: Path,
    processed_video: Path,
    destination: Path,
    ffmpeg_path: Path,
) -> None:
```

이 함수는 optional이고, ffmpeg 실행 실패 시 원본과 기존 결과를 덮어쓰지 않는다. mosaic 판별 기능과 분리하여 사용자 승인 후 구현한다.

## 12. Phase 8 — benchmark, 테스트, 문서

### 12.1 수정 대상

- [benchmark/runner.py](../src/consented_face_redactor/benchmark/runner.py)
- `tests/test_video_enrollment.py`
- `tests/test_gallery_similarity_diagnostics.py`
- `tests/test_cli_video_enrollment.py`
- `tests/test_pipeline_face_decisions.py`
- `tests/test_fhd_processing.py`
- 선택적 `tests/test_media_remux.py`

### 12.2 benchmark 추가

공개 benchmark category API A–E는 유지한다.

| Category | 추가 scenario | 검증 |
| --- | --- | --- |
| A | 두 얼굴 중 한 얼굴만 explicit approved | approved ROI만 변하고 다른 ROI는 보존 |
| D | FHD synthetic mosaic processing | latency/FPS/environment 측정만 수행 |
| E | enrollment report/options schema | strict validation과 report serialization |

### 12.3 필수 테스트

- target-only 등록 영상의 여러 embedding이 한 profile에 들어감
- 다중 얼굴 frame은 자동 등록하지 않음
- duplicate frame은 reference count를 불필요하게 늘리지 않음
- batch enrollment 중 오류가 나면 gallery 전체가 rollback됨
- 승인 target만 mosaic되고 non-target ROI/ROI 밖 pixel이 동일함
- low similarity, gallery 오류, expired approval은 no-redaction
- FHD synthetic frame의 shape가 유지됨
- input/output 동일 path는 거절됨
- `--dry-run`은 gallery, approval, output video를 쓰지 않음

### 12.4 문서 갱신

갱신 대상:

- `README.md`: 실제 등록 영상 기반 workflow 요약
- `docs/REAL_VIDEO_TEST_GUIDE.md`: `gallery-enroll-video` 명령 추가
- `docs/CODEBASE_FUNCTION_REFERENCE.md`: 새 타입/함수 설명
- `docs/LOCAL_REAL_MODEL_IMPLEMENTATION_REPORT.md`: 구현 결과와 실제 측정값 추가

## 13. 단계별 수용 기준

| Phase | 수용 기준 |
| --- | --- |
| 0 | option/type validation과 report schema test 통과 |
| 1 | target-only 영상에서 valid candidate/skip reason이 실제 frame 관측으로 생성 |
| 2 | 중복은 줄이고 embedding diversity는 유지하며, 다중 얼굴 frame은 등록하지 않음 |
| 3 | batch 등록은 atomic이고 기존 gallery API가 깨지지 않음 |
| 4 | dry-run은 저장 없이 report만 반환하며 실제 실행은 하나의 profile만 갱신 |
| 5 | CLI가 검증된 model/manifest 없이 enrollment하지 않음 |
| 6 | 어떤 approval에서도 해당 approval의 bbox 외 얼굴을 가리지 않음 |
| 7 | 원본 overwrite 없음, FHD 측정값과 환경 metadata 기록 |
| 8 | targeted pytest, 전체 pytest, benchmark A–E, aggregate JSON parse, diff check 통과 |

## 14. 구현 순서

1. `video_enrollment.py` 타입과 candidate extraction
2. coverage selection 및 `EnrollmentReport`
3. `LocalGallery.enroll_many()`과 similarity diagnostics
4. `VideoEnrollmentService`
5. CLI `gallery-enroll-video`와 dry-run/report
6. single-image enrollment 공통 helper 리팩터링
7. adapter/pipeline diagnostics
8. output overwrite 보호와 FHD evidence
9. benchmark/test/documentation
10. 사용자 등록 영상 dry-run → 실제 profile 생성 → FHD 처리 영상 dry-run → 결과 영상 생성

실제 대상 인식률 목표와 FHD FPS 목표는 마지막 단계의 동의된 평가 영상 결과를 사용자와 검토한 뒤 결정한다. 반면 승인 없는 얼굴을 가리지 않는 fail-closed 규칙은 모든 단계에서 고정이다.
