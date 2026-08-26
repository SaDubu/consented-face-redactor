# 동의 기반 특정 인물 얼굴 비식별화 시스템 — 프로젝트 포트폴리오 사례 연구

## 1. 프로젝트 한 줄 소개

등록 영상에서 사용자가 동의한 한 사람의 얼굴 표현을 로컬 gallery로 구성하고, 별도의 대상 영상에서 그 사람에게만 얼굴 모자이크를 적용하는 FHD 배치 처리 시스템을 설계·구현·검증했다.

이 프로젝트의 기술적 핵심은 얼굴 검출, 신원 승인, 시간축 위치 추적, 시각 효과를 하나의 점수나 모델에 섞지 않고 각각 독립된 계약으로 분리한 것이다. 최종 시스템에서 detector confidence는 얼굴 후보 위치를 제안할 뿐이며, `GalleryApproval.approved=True`만 신원 권한을 생성한다. TAPNext++ tracker는 이 권한이 시작된 위치를 시간축으로 이어 주지만 profile ID를 만들거나 다른 사람에게 권한을 넘길 수 없다.

### 수행 역할

- 기존 코드·테스트·문서의 계약 감사와 기준선 복원
- 얼굴 승인 보안 경계와 상태 머신 설계
- 실제 YuNet/SFace 모델 adapter 및 manifest 검증 연결
- 등록 영상 기반 multi-reference gallery와 오염 방어 알고리즘 구현
- TAPNext++ 기반 양방향 시간축 추적 설계·통합
- FHD 2-pass 분석/렌더링과 atomic output 구현
- adaptive rectangle 및 외접 ellipse mosaic renderer 구현
- synthetic benchmark, unit/integration/e2e test, 실제 영상 evidence 설계
- 사용자 피드백을 반영한 반복적 품질 개선과 기술 문서화

### 기술 스택

| 영역 | 기술 |
| --- | --- |
| Language | Python 3.12 |
| Vision | OpenCV, YuNet, SFace |
| Numerical processing | NumPy |
| Temporal tracking | PyTorch, CUDA, Google DeepMind TAPNext++ |
| Test | pytest, synthetic deterministic fixture, real-video evidence |
| Data contract | versioned JSON, frozen dataclass, Protocol, SHA-256 manifest |
| Media pipeline | OpenCV VideoCapture/VideoWriter, optional FFmpeg remux |
| Collaboration boundary | local-only Git worktree, 민감 asset Git 비추적, no commit/push |

---

## 2. 프로젝트 배경과 실제 요구사항

사용자의 원래 목표는 다음과 같았다.

1. 목표 인물만 등장하는 다양한 각도의 영상을 얼굴 등록 자료로 사용한다.
2. 목표 인물이 등장하는 별도의 영상을 입력한다.
3. 영상을 frame 단위로 분석해 목표 인물의 얼굴에만 모자이크를 적용한다.
4. 원본 파일은 변경하지 않고 새로운 결과 파일을 생성한다.
5. FHD 이상의 영상을 처리한다.
6. 처음에는 무거운 객체 추적을 사용하지 않는 구조를 선호한다.
7. 실제 결과에서 모자이크가 중간에 끊기는 문제가 확인된 뒤에는 최신 tracker를 제한적으로 도입한다.
8. 최종 시각 효과는 지나치게 강하지 않은 모자이크와 얼굴 bbox를 외접하는 세로 타원 형태여야 한다.

요구사항은 작업 중 구체화됐다. 처음에는 “무거운 추적을 피한다”가 중요했지만, 실제 영상에서 frame별 얼굴 유사도가 크게 흔들리며 가림이 끊기는 것이 확인됐다. 따라서 최종 설계는 두 실행 모드를 모두 유지한다.

- `--tracker none`: 기존의 가벼운 frame-by-frame 처리
- `--tracker tapnextpp`: 저장 영상에 적합한 2-pass 시간축 처리

이 선택은 기존 요구를 폐기한 것이 아니라, 성능 비용과 연속성 품질을 사용자가 선택할 수 있도록 분리한 것이다.

---

## 3. 결과 요약

실제 입력 자료는 다음과 같았다.

| 구분 | 파일 | 역할 | 해상도 | frame |
| --- | --- | --- | ---: | ---: |
| 등록 영상 | `learn.mp4` | 목표 인물의 다양한 얼굴 각도 등록 | 1080×1920 | 530 |
| 처리 영상 | `test.mp4` | 목표 인물 검출 및 모자이크 | 1080×1920 | 456 |

최종 v4 결과:

| 지표 | 결과 |
| --- | ---: |
| 전체 입력 frame | 456 |
| 전체 출력 frame | 456 |
| 모자이크 적용 frame | 456 |
| 직접 gallery 승인 anchor | 210 |
| 양방향 tracker 보강 frame | 184 |
| review-required frame | 0 |
| ambiguous range | 0 |
| 입력/출력 FPS | 30 / 30 |
| 입력/출력 해상도 | 1080×1920 / 1080×1920 |
| 분석 처리율 | 약 7.43fps |
| p50 분석 latency | 약 32.38ms |
| p95 분석 latency | 약 39.40ms |
| 전체 자동 테스트 | 321 passed |
| benchmark | A–E, 24/24 passed |

이 수치는 제공된 한 영상에서 측정된 결과다. 임의의 인물, 다중 인물 교차, 완전 가림, 다른 카메라 조건까지 100% 정확하다는 일반화 주장은 하지 않는다.

최종 로컬 산출물:

- 영상: `local_data/output/test_target_mosaic_tapnextpp_v4_ellipse.mp4`
- 실행 증거: `local_data/evidence/test_tapnextpp_v4_ellipse.json`
- 원본/v3/v4 비교: `local_data/evidence/tapnextpp_v4_ellipse_comparison.jpg`

모든 영상, 얼굴 embedding, 모델 weight, gallery, evidence는 Git 비추적 `local_data/`에 유지했다. 원본 `learn.mp4`와 `test.mp4`는 덮어쓰지 않았으며 commit과 push도 수행하지 않았다.

---

## 4. 문제를 다시 정의한 과정

처음에는 이 문제를 다음처럼 단순하게 볼 수 있었다.

~~~text
얼굴 검출 → 얼굴 embedding → similarity threshold → mosaic
~~~

하지만 이 구조에는 세 가지 서로 다른 질문이 섞여 있다.

1. 이 위치에 얼굴이 있는가?
2. 이 얼굴이 사용자가 승인한 사람인가?
3. 앞뒤 frame에서 같은 얼굴 위치가 어디로 이동했는가?

detector confidence는 첫 번째 질문에만 답한다. gallery similarity는 두 번째 질문을 위한 관측값이지만, similarity 숫자만으로 승인 정책 전체를 표현할 수 없다. tracker는 세 번째 질문을 다루지만 신원 자체를 증명하지 않는다.

따라서 최종 문제를 다음 네 계층으로 재정의했다.

~~~text
Detection     : 얼굴 후보 위치 생성
Identity      : GalleryApproval을 통한 명시적 신원 승인
Continuity    : 승인 anchor 위치의 제한적 시간축 전파
Presentation  : 승인된 위치에만 mosaic/sticker 합성
~~~

이 분리는 단순한 코드 정리가 아니라 안전성의 핵심이다. tracker가 흔들려 다른 얼굴 쪽으로 이동하더라도 tracker confidence가 곧 identity approval이 되지 않는다. 반대로 detector confidence가 높아도 gallery 승인이 없다면 모자이크하지 않는다.

---

## 5. 전체 아키텍처

### 5.1 등록 경로

~~~text
learn.mp4
  │
  ├─ sampled frame 추출
  ├─ YuNet 얼굴 검출
  ├─ 단일 얼굴·최소 크기 검증
  ├─ SFace embedding
  ├─ 인접 중복 제거
  ├─ embedding similarity graph 구성
  ├─ dominant connected component 선택
  ├─ 작은 분리 component는 review 후보로 격리
  └─ LocalGallery + ApprovalStore 저장
~~~

### 5.2 처리 경로

~~~text
test.mp4
  │
  ├─ Pass 1: YuNet detection + SFace gallery evaluation
  │       │
  │       └─ GalleryApproval.approved=True인 bbox만 identity anchor
  │
  ├─ anchor 사이 gap을 TAPNext++ 정방향/역방향 추적
  ├─ 두 경로의 bbox 합의 및 detection association 검증
  ├─ 불일치·visibility 저하·profile 충돌은 fail-closed
  ├─ 불변 redaction plan 생성
  │
  └─ Pass 2: 원본을 다시 읽어 승인된 bbox에만 타원 mosaic
          ├─ sibling temporary MP4 기록
          ├─ frame 수·writer 성공 검증
          └─ os.replace()로 결과 파일 공개
~~~

2-pass 구조를 선택한 이유는 저장 영상에서는 미래 frame을 사용할 수 있기 때문이다. online tracker 하나의 예측을 즉시 렌더링하면 drift가 발생한 frame을 나중에 되돌리기 어렵다. 분석과 렌더링을 분리하면 forward/backward 합의를 완료한 뒤 확정된 plan만 원본에 적용할 수 있다.

---

## 6. 단계별 개발 과정과 문제 해결

## 6.1 Phase 0 — 기준선과 공개 계약 복원

### 발견한 문제

기준 커밋 `f151680`의 문서, benchmark, 테스트에는 현재 코드에 존재하지 않는 API 설명이 섞여 있었다.

- `categories`
- `output_path`
- `benchmark_runner.run()`
- 존재하지 않는 enum과 private module import
- 실제 assertion 없이 성공으로 처리하는 benchmark 결과
- `duration_ms=float(passed)`처럼 성공 여부를 시간으로 위장한 값

### 원인

문서와 테스트가 구현보다 앞서 작성되거나, 과거 설계 초안이 실제 public API와 함께 갱신되지 않았다. 이 상태에서는 테스트가 통과하더라도 사용자가 README의 예제로 프로그램을 실행할 수 없다.

### 해결

public benchmark API를 다음 두 개로 고정했다.

~~~python
run_benchmark(category="A")
generate_aggregate_report()
~~~

`run_benchmark`는 한 category의 실제 scenario 결과를 요약한 dict를 반환하고, aggregate 함수는 A–E 실제 결과를 모아 JSON 문자열을 반환한다. 저장 위치는 함수가 결정하지 않으며 호출자가 명시적으로 저장해야 한다.

### 설계 이유

벤치마크 함수가 자동으로 파일을 쓰면 테스트 실행만으로 로컬 상태가 변한다. 반환값 중심 API는 library와 CLI의 책임을 나누고, 로컬 전용·명시적 저장 원칙을 지킬 수 있다.

---

## 6.2 Phase 1 — tuple gallery 결과를 명시적 승인 계약으로 변경

### 발견한 문제

기존 gallery matcher는 `(profile_id, similarity)` 같은 tuple 또는 유사한 비구조 결과를 반환할 수 있었다. pipeline이 tuple 위치와 similarity 값을 다시 해석하면 다음 위험이 생긴다.

- `0.8`이 단순 관측값인지 승인 결과인지 불명확하다.
- tuple 순서가 바뀌어도 type checker와 runtime이 놓칠 수 있다.
- empty gallery, similarity 부족, adapter 오류를 같은 `None`으로 표현한다.
- fake gallery와 production gallery의 의미가 달라질 수 있다.

### 해결

`@dataclass(frozen=True, slots=True)`인 `GalleryApproval`을 도입했다.

~~~python
GalleryApproval(
    approved: bool,
    profile_id: str | None,
    similarity: float | None,
    reason_code: str,
    gallery_revision: str | None,
)
~~~

pipeline은 `approval.approved is True`만 권한으로 소비한다. similarity, detector confidence, `t_confirm`, `t_keep`은 telemetry일 뿐 독립 승인 신호가 아니다.

### 오류 처리

- gallery 없음
- empty gallery
- `embed()` 예외
- `match()` 예외
- malformed return
- stale profile
- similarity 부족

은 모두 `CANDIDATE` 또는 review-required 상태로 fail-closed된다. 예외 때문에 scenario 결과가 누락되지 않도록 benchmark도 `passed=False`와 reason code를 남긴다.

### 설계 이유

신원 확인은 보안 경계다. 단순 숫자나 위치 기반 tuple보다 불변 결과 객체가 감사 가능하고, 생산 adapter와 테스트 fake가 동일한 의미를 공유할 수 있다.

---

## 6.3 Phase 2 — Config와 CLI 계약 정리

### 발견한 문제

문서에는 validation이 있다고 쓰여 있지만 구현은 unknown key를 조용히 무시했고, CLI는 Config에 존재하지 않는 enum이나 slot을 참조하는 부분이 있었다. unknown key를 즉시 전면 거절하면 기존 사용자 설정이 깨지고, 계속 무시하면 오타가 숨겨지는 양면 문제가 있었다.

### 해결

호환 모드와 엄격 모드를 분리했다.

~~~python
Config.from_dict(payload)               # unknown key 무시
Config.from_dict(payload, strict=True)  # unknown key 거절
~~~

schema version을 도입하고 현재 v4까지 migration 경로를 유지했다. `inspect-config` 출력의 key 집합은 `Config.to_dict()`와 동일해야 한다는 contract test도 추가했다.

### 설계 이유

library API 기본값은 기존 payload 호환성을 유지하고, 오타 검출이 필요한 CLI나 운영 도구만 strict mode를 opt-in할 수 있다. schema version을 먼저 두면 이후 필드 의미가 바뀌어도 명시적 migration을 작성할 수 있다.

---

## 6.4 Phase 3 — 성공을 만드는 benchmark에서 실패를 드러내는 benchmark로

### 발견한 문제

기존 benchmark 일부는 실제 pipeline을 실행하지 않거나 `passed=True`를 하드코딩했다. 예외가 나면 결과 자체가 사라지는 경로도 있었다. 이런 benchmark는 회귀를 발견하지 못하고 완료 표시만 만든다.

### 해결

Category A–E를 실제 관측 기반으로 재구성했다.

| Category | 검증 대상 |
| --- | --- |
| A | gallery 승인 없이는 어떤 confidence에서도 redaction되지 않는지 |
| B | mosaic/sticker 효과 범위와 예외 보존 |
| C | `UNSEEN → CANDIDATE → CONFIRMED → LOST → EXPIRED` 실제 전이 |
| D | warm-up 후 median/p95 latency와 환경 metadata |
| E | Config round-trip, strict mode, schema migration |

각 scenario는 fresh frame, pipeline, gallery, state를 사용한다. 공통 wrapper가 실제 시간을 측정하고 예외도 실패 결과로 append한다.

### 실제로 발견한 회귀

강한 mosaic에서 bbox padding을 추가한 뒤 Category B의 locality test가 실패했다. 테스트는 raw detection bbox만 효과 범위라고 가정했지만 renderer의 새 계약은 padded bbox였다.

실패를 무시하거나 assertion을 약하게 만들지 않았다. effect가 실제로 영향을 미칠 수 있는 영역을 helper가 계산하게 하고, 그 바깥의 모든 pixel이 원본과 동일한지 검사하도록 수정했다. 이후 ellipse mode에서는 같은 helper가 ellipse bounds와 mask 외부 보존을 확인하도록 확장됐다.

---

## 6.5 Phase 4 — 실제 모델 연결과 등록 영상 처리

### 구현

- OpenCV YuNet: 얼굴 bbox와 5개 landmark 검출
- OpenCV SFace: 얼굴 embedding 생성
- model manifest: 파일명, 역할, provider, SHA-256 검증
- `gallery-enroll-video`: sampled frame을 이용한 multi-reference 등록
- LocalGallery와 ApprovalStore 분리

### 왜 manifest 검증을 넣었는가

모델 파일 이름만 같다고 동일 모델이라고 볼 수 없다. 잘못된 weight, 불완전한 다운로드, 다른 revision을 사용하면 같은 코드에서도 결과가 바뀐다. 따라서 runtime 진입 전에 model role, provider, 파일명, checksum을 확인한다.

### 왜 Gallery와 ApprovalStore를 분리했는가

embedding vector가 존재한다는 사실과 사용자가 그 profile 사용을 승인했다는 사실은 다르다. LocalGallery는 특징 벡터를 저장하고 ApprovalStore는 별도의 명시 승인 기록을 보관한다. 이 분리로 데이터를 등록했지만 아직 사용할 수 없는 상태를 표현할 수 있다.

---

## 6.6 Phase 5 — 첫 실제 영상 실행에서 드러난 끊김

### 관측

초기 frame-by-frame 실행 결과는 다음과 같았다.

| 관측 | frame |
| --- | ---: |
| 직접 승인 및 mosaic | 209 |
| detector가 얼굴을 놓침 | 2 |
| 얼굴은 있지만 승인되지 않음 | 245 |
| 가장 긴 연속 미승인 구간 | 65 |

### 잘못된 초기 가정

처음에는 모자이크가 끊기는 이유를 detector 누락이라고 예상하기 쉬웠다. 실제 evidence를 보니 detection 없는 frame은 2개뿐이었다. 대부분은 얼굴이 검출됐지만 각도, 표정, 흐림에 따라 SFace similarity가 frame마다 흔들려 승인되지 않은 경우였다.

### 해결 방향

detector threshold를 무작정 낮추거나 gallery threshold를 완화하지 않았다. 그렇게 하면 recall은 오를 수 있지만 다른 사람까지 승인할 위험이 커진다. 대신 명시 승인된 frame을 identity anchor로 사용하고 그 사이 위치만 tracker로 연결하는 구조를 선택했다.

---

## 6.7 Phase 6 — tracker 선정과 권한 분리

### 후보 비교

- ByteTrack/DeepSORT: 사람 전신 detector와 MOT association에 적합하지만 얼굴 내부 위치 연속성에는 과도하거나 부정확할 수 있다.
- SAM 2.1: 정밀 segmentation에는 유리하지만 단순 mosaic bbox 갱신에는 계산량이 크다.
- OpenCV legacy tracker: 가볍지만 긴 회전·가림·appearance 변화에 취약하다.
- CoTracker3: 성숙한 fallback 후보다.
- TAPNext++: 얼굴 내부 여러 점을 장기 추적하고 frame-by-frame wrapper를 제공하는 최신 point tracker다.

### 선택

공식 Google DeepMind TAPNext++를 opt-in 기본 tracker로 선택했다. 얼굴 landmark 5개와 bbox 내부 grid 16개, 총 21개 point를 seed로 사용한다.

### 중요한 제약

tracker protocol에는 profile ID나 승인 함수가 없다. 입력은 frame과 point이고 출력은 point 위치와 visibility다. 이를 통해 타입과 모듈 경계 차원에서 tracker가 신원 권한을 생성하지 못하도록 했다.

### lazy loading 이유

TAPNext++ checkpoint는 크고 CUDA/PyTorch 의존성이 무겁다. 패키지를 import하는 것만으로 모델을 load하면 tracker를 사용하지 않는 사용자도 비용을 부담한다. 그래서 `--tracker tapnextpp`가 선택된 뒤에만 source, torch, checkpoint를 검증하고 load한다.

---

## 6.8 Phase 7 — v1 양방향 추적: 392/456

첫 구현은 같은 profile의 두 명시 anchor 사이를 정방향과 역방향으로 각각 추적했다. 두 경로 bbox의 IoU가 기준 이상일 때만 중간 frame을 승인했다.

결과:

- 직접 anchor: 209
- 양방향 보강: 183
- 전체 처리: 392/456
- 미처리: frame 2–28, 419–455

frame 2–28은 두 경로가 합의하지 않아 fail-closed됐다. 419–455는 마지막 anchor 이후라 미래 anchor가 없었다. 양방향 합의는 안전했지만 영상 양끝을 처리하지 못하는 구조적 한계가 드러났다.

---

## 6.9 Phase 8 — v2 단방향 edge extension: 408/456

첫 anchor 이전과 마지막 anchor 이후에는 양방향 anchor 쌍이 없다. 이 구간을 무조건 tracker로 채우지 않고 다음 조건을 추가했다.

- 최대 90 frame 이내
- tracker visibility 기준 통과
- bbox scale/center 이동 기하 gate 통과
- 현재 YuNet detection과 association
- 다른 approved profile과 충돌하지 않음
- tracker-only 연속 구간은 최대 12 frame

이 결과 419–434가 추가로 처리되어 408/456이 됐다.

### 새 오류: 동일 얼굴의 중복 bbox

frame 435에서 YuNet이 한 얼굴을 겹치는 두 bbox로 검출했다. 초기 규칙은 후보가 둘이면 곧바로 ambiguity로 처리해 전파를 중단했다. 그러나 시각 확인과 cost 분석 결과 두 후보 중 하나가 tracker bbox와 명확히 더 잘 맞았고 second-best와 충분한 margin이 있었다.

### 수정

단순히 “후보 수가 2개면 실패”가 아니라 다음을 계산했다.

- IoU
- frame diagonal 대비 중심 거리
- scale ratio
- best association cost
- second-best와의 margin

최적 후보가 hard gate를 통과하고 두 번째 후보와 충분한 차이가 있을 때만 하나를 선택했다. 이 규칙은 중복 detection을 처리하면서도 실제 두 사람이 가까이 있는 경우를 무조건 하나로 합치지 않는다.

---

## 6.10 Phase 9 — 정량 100%보다 먼저 발견한 등록 오염

### 발견 계기

초기 결과의 contact sheet를 확인하다 frame 0의 승인 bbox가 얼굴 전체가 아니라 귀 주변의 작은 false crop이라는 것을 발견했다. 단순 frame coverage만 봤다면 성공으로 오판할 수 있었다.

### 수치 분석

| frame 0 후보 | best reference similarity | gallery centroid similarity | 기존 판정 |
| --- | ---: | ---: | --- |
| 귀/부분 false crop | 0.8516 | 0.2835 | 승인 |
| 실제 측면 전체 얼굴 | 0.7056 | 0.6266 | 거절 |

false crop은 등록 reference #18 하나와만 매우 강하게 일치했다. gallery가 best-reference max 정책을 사용하므로 오염된 reference 하나가 높은 승인 점수를 만들었다.

### 근본 원인

“등록 영상에는 대상자 한 명만 있다”는 사실을 “검출된 모든 crop이 올바른 대상 얼굴이다”로 잘못 확대 해석했다. 한 사람만 나오는 영상에서도 detector는 귀, 부분 얼굴, 배경을 false face로 검출할 수 있다.

### 해결: dominant embedding component

47개 reference를 similarity graph로 구성했다. similarity 0.45 이상인 embedding 사이에 edge를 만들었을 때 다음 구조가 나왔다.

- dominant component: 36개
- 분리 component: 9개
- singleton: 2개

가장 큰 connected component만 자동 등록하고 나머지 11개는 review 후보로 격리했다.

### centroid 하나를 쓰지 않은 이유

정면과 극단 측면 얼굴은 서로 similarity가 낮을 수 있다. centroid 하나만 대표로 사용하면 다양한 pose recall이 떨어진다. connected component는 정면→반측면→측면처럼 중간 각도로 연결되는 view trajectory를 보존한다. 반면 고립된 false crop은 큰 component와 연결되지 않아 제거할 수 있다.

### 결과

- clean reference: 36개
- review 격리: 11개
- direct anchor: 210개
- bidirectional consensus: 184개
- edge continuity: 62개
- 최종 v3 redaction: 456/456

초기 ear anchor가 사라졌고 첫 신뢰 anchor의 역방향 위치가 실제 측면 얼굴 detection과 연결됐다.

---

## 6.11 Phase 10 — 강한 사각형 mosaic와 사용자 피드백

초기 production pipeline에는 사실상 고정 8px block이 사용돼 큰 FHD 얼굴에서 모자이크가 너무 약했다. 이를 얼굴 짧은 변에 비례하는 적응형 block으로 바꿨다.

~~~python
block_size = max(
    round(min(face_width, face_height) / mosaic_grid_cells),
    mosaic_min_block_px,
)
~~~

v3에서는 grid 8, minimum 16px, bbox 18% padding을 사용해 강한 사각형 mosaic를 만들었다. 식별 단서를 크게 줄였지만 실제 결과 검토 후 두 가지 UX 문제가 확인됐다.

1. 모자이크 강도가 필요 이상으로 컸다.
2. 사각형이 머리 주변 배경까지 넓게 가려 시각적으로 거칠었다.

이 피드백은 identity/tracking 문제가 아니라 presentation layer 문제였다. 따라서 gallery나 tracker를 다시 건드리지 않고 effect 설정과 renderer만 수정했다.

---

## 6.12 Phase 11 — v4 완화된 외접 세로 타원

### 강도 변경

| 설정 | v3 | v4 |
| --- | ---: | ---: |
| grid cells | 8 | 12 |
| minimum block | 16px | 10px |

grid cells가 많아지면 같은 얼굴을 더 많은 block으로 표현하므로 각 block이 작아지고 강도가 완화된다.

### 형태 변경

기본 shape을 `ellipse`로 추가하고 가로/세로 반축 배율을 1.40/1.50으로 정했다.

단순 내접 타원을 쓰면 bbox 모서리에 있는 턱, 관자놀이, 회전 얼굴 일부가 mask 밖으로 노출될 수 있다. 따라서 다음 조건으로 bbox 네 모서리가 타원 내부에 있는지 검증한다.

~~~text
1 / horizontal_scale² + 1 / vertical_scale² <= 1
~~~

1.40과 1.50은 이 조건을 만족한다. 세로가 가로보다 약 7% 길어 원형보다 얼굴형에 가깝지만 과도하게 긴 캡슐 형태는 아니다.

### pixel 보존 계약

`ellipse_bounds()`는 모자이크 계산에 필요한 최소 사각 영역을 구한다. `ellipse_mask()`는 실제 타원 내부만 `True`인 boolean mask를 만든다. `MosaicEffect.render()`는 타원 bounding rectangle의 mosaic를 계산하되 mask 내부 pixel만 결과에 복사한다. 따라서 mask 밖은 입력과 byte-identical하다.

### 결과

v4도 456/456 frame을 처리했고, 원본/v3/v4 3열 contact sheet에서 측면과 정면 대표 frame을 확인했다. v3보다 배경을 덜 가리고 강도는 낮아졌지만 얼굴 식별 영역은 계속 덮였다.

---

## 7. 주요 모듈과 함수별 책임

## 7.1 `config.py`

### `Config.__init__()`

현재 schema v4 설정 객체를 만들고 mosaic 숫자와 타원 기하를 검증한다. 잘못된 shape, 무한대·NaN 배율, 세로보다 긴 가로축, bbox 모서리를 감싸지 못하는 축 조합을 거절한다.

### `Config.from_dict(data, strict=False)`

v1/v2/v3/v4 payload를 v4 객체로 읽는다. 기본 모드는 unknown key를 무시해 호환성을 유지하고 strict mode는 오타를 즉시 거절한다.

### `Config.to_dict()`

현재 생성자 필드와 schema version을 모두 직렬화한다. CLI 설정 출력과 contract test의 source of truth다.

## 7.2 `gallery_approval.py`

### `GalleryApproval`

불변 승인 결과다. `approved`, `profile_id`, `similarity`, `reason_code`, `gallery_revision`을 함께 보존한다.

### `GalleryApproval.denied()`

adapter 오류, empty gallery, similarity 부족 같은 실패를 일관된 거절 결과로 만든다.

### `GalleryApprovalProtocol`

fake와 production gallery가 같은 `embed()`/`match()` 반환 계약을 사용하게 한다.

## 7.3 `approved_gallery.py`

### `ApprovedLocalGalleryAdapter.evaluate()`

한 detection의 얼굴을 정렬·embedding하고 LocalGallery에서 후보를 찾은 뒤 ApprovalStore에서 명시 승인 상태를 확인한다. similarity가 높더라도 profile approval이 없으면 승인하지 않는다.

## 7.4 `video_enrollment.py`

### `iter_sampled_frames()`

등록 영상을 일정 간격으로 순회해 연속 frame 전체를 불필요하게 처리하지 않는다.

### `extract_enrollment_candidate()`

정확히 하나의 유효 얼굴만 있는 frame에서 embedding 후보를 만든다. 검출 없음, 다중 얼굴, 작은 얼굴, embedding 오류는 skip reason으로 기록한다.

### `select_diverse_references()`

중복을 줄이고 embedding graph의 dominant component를 선택한다. 다양성과 오염 방어 사이의 핵심 함수다.

### `VideoEnrollmentService.collect()/select()/enroll()`

영상 읽기, reference 선택, 저장 mutation을 분리한다. dry-run에서는 report만 만들고 gallery를 변경하지 않는다.

## 7.5 `pipeline.py`

### `_apply_effect_to_bbox()`

좌표를 frame 안으로 clip하고 Config에 따라 rectangle 또는 ellipse mosaic, sticker, no-effect를 실행한다. 입력 frame을 직접 변경하지 않는다.

### `_evaluate_detection()`

production gallery의 구조화된 승인 결과를 받는다. adapter 예외와 malformed return은 승인으로 해석하지 않는다.

### `_process_face_by_face_approvals()`

한 frame의 얼굴을 각각 독립 평가한다. 한 얼굴의 승인이 다른 bbox에 전파되는 취약점을 막는다.

### `process_frame()`

가벼운 frame-by-frame 경로의 중심 함수다. detection, gallery approval, effect, `UNSEEN/CANDIDATE/CONFIRMED/LOST/EXPIRED` 상태 전이를 수행한다.

### `save_track_state()/load_track_state()`

snapshot schema v2로 lost frame index와 confirmed profile reference를 저장한다. 시간 근거가 없는 legacy v1의 `CONFIRMED/LOST`는 안전하게 `CANDIDATE`로 낮춘다.

## 7.6 `tracking/geometry.py`

### `seed_face_points()`

5개 landmark와 bbox 내부 4×4 grid를 결합한다. 눈·코·입 landmark만으로는 가림이나 회전에 취약하고, bbox 모서리만으로는 배경을 따라갈 수 있어 얼굴 내부 분산 point를 함께 사용한다.

### `estimate_similarity_transform()`

visible point만 사용해 scale, rotation, translation을 계산하고 median/MAD로 outlier를 제거한다.

### `validate_tracked_bbox()`

visibility, scale 변화, 중심 이동, frame 이탈을 검사해 tracker drift를 reason code가 있는 실패로 바꾼다.

## 7.7 `tracking/association.py`

### `association_cost()`

IoU, 중심 거리, 크기 비율을 하나의 deterministic cost로 계산한다.

### `associate_tracks_to_detections()`

한 detection이 두 track에 배정되지 않는 one-to-one assignment를 만든다.

### `detect_crossing_ambiguity()`

겹치는 두 track의 순서가 바뀌는 상황을 감지해 identity transfer 가능성을 표시한다.

## 7.8 `tracking/authorization.py`

### `create_authorized_track()`

현재 frame의 명시 gallery 승인에서만 profile lease를 만든다.

### `refresh_authorized_track()`

같은 profile의 새 승인만 기존 lease를 갱신한다.

### `may_propagate_authorization()`

시간·visibility·연속성 조건 안에서 기존 권한의 위치만 전파할 수 있는지 판단한다.

### `revoke_track_authorization()`

ambiguity, loss, 충돌 뒤 권한을 제거한다. tracker가 나중에 다시 보이더라도 gallery 재승인 없이 자동 복구하지 않는다.

## 7.9 `tracking/bidirectional.py`

### `collect_identity_anchors()`

`approved=True`, profile ID, gallery revision이 모두 존재하는 얼굴만 anchor로 수집한다.

### `track_segment_forward()/track_segment_backward()`

같은 구간을 양쪽 anchor에서 각각 추적한다.

### `reconcile_bidirectional_paths()`

정방향과 역방향 bbox가 충분히 겹칠 때만 합의 결과를 만든다. 한쪽 성공을 전체 성공으로 바꾸지 않는다.

### `build_redaction_track_plan()`

직접 anchor, 양방향 합의, 제한된 edge extension을 frame별 불변 decision으로 만든다.

## 7.10 `temporal_video_processor.py`

### `analyze()`

첫 pass에서 모든 frame의 detection과 gallery 평가를 수행하고 anchor 및 tracking plan을 만든다. 기본 evidence에는 원본 pixel, crop, embedding을 저장하지 않는다.

### `render()`

원본 영상을 다시 읽어 승인된 plan만 렌더링한다. destination 존재, 동일 input/output, frame 수 오류를 publish 전에 거절한다.

## 7.11 `effects/mosaic.py`

### `mosaic_block_size()`

얼굴 크기에 비례하는 block size를 계산해 FHD의 큰 얼굴과 작은 얼굴에서 상대적으로 일관된 강도를 제공한다.

### `expand_bbox()`

rectangle 호환 모드에서 bbox padding과 frame clipping을 계산한다.

### `ellipse_bounds()`

외접 타원의 bounding rectangle과 축 조건을 검증한다.

### `ellipse_mask()`

OpenCV filled ellipse 기반 boolean mask를 만든다.

### `MosaicEffect.render()`

`INTER_AREA`로 축소하고 `INTER_NEAREST`로 확대해 mosaic를 만든 뒤, rectangle 전체 또는 ellipse mask 내부에 합성한다.

---

## 8. 테스트 전략

### 8.1 단위 테스트

- Config v1–v4 migration
- strict/non-strict unknown key
- ellipse 축 validation
- bbox 모서리의 mask 포함
- 타원 밖 pixel 보존
- frame edge clipping
- GalleryApproval 불변성과 malformed return 거절
- gallery embed/match 예외 fail-closed
- tracking geometry와 association cost
- bidirectional disagreement 거절
- snapshot v1→v2 안전 migration

### 8.2 통합 테스트

- detector confidence만 높은 frame이 redaction되지 않는지
- 승인된 얼굴과 미승인 얼굴을 face-by-face로 분리하는지
- mosaic와 sticker가 각자의 ROI에서만 동작하는지
- sticker encoding 오류 뒤에도 다음 benchmark 결과가 남는지
- CLI config key와 `Config.to_dict()`가 일치하는지
- tracker none/tapnextpp 실행 경로가 분리되는지
- destination overwrite와 partial output을 방지하는지

### 8.3 실제 영상 검증

자동 수치만 사용하지 않고 contact sheet를 만들었다. 이 시각 검토가 ear-only false crop을 발견했고, 이후 원본/v3 rectangle/v4 ellipse를 동일 frame에서 비교하는 근거가 됐다.

### 8.4 현재 검증 결과

~~~text
compileall: passed
pytest: 321 passed
git diff --check: passed
Category A: 9/9
Category B: 5/5
Category C: 5/5
Category D: 2/2
Category E: 3/3
Aggregate JSON: 24/24
~~~

---

## 9. 개발 중 발생한 오류와 배운 점

### 오류 1 — 문서 API와 구현 API가 달랐다

문서의 예제가 존재하지 않는 함수와 인자를 설명했다. 해결은 문서에 맞춘 wrapper를 무작정 늘리는 것이 아니라 실제 필요한 public API 두 개를 먼저 확정하고 README, protocol, test를 동일하게 수정하는 것이었다.

### 오류 2 — benchmark가 실패를 관측하지 않았다

하드코딩 성공값과 의미 없는 duration이 있었다. 실제 pipeline을 실행하고 예외도 결과 row로 남기는 wrapper를 도입했다.

### 오류 3 — similarity를 권한처럼 해석할 가능성이 있었다

tuple matcher를 `GalleryApproval`로 바꾸고 pipeline이 `approved=True`만 소비하게 했다.

### 오류 4 — Config 문서와 validation이 달랐다

schema version, strict opt-in, round-trip contract test로 코드·문서·CLI의 source of truth를 `Config.to_dict()`로 통일했다.

### 오류 5 — 실제 끊김 원인을 detector miss로 오판할 수 있었다

evidence를 frame별로 집계해 detector miss는 2개뿐이고 미승인은 245개임을 확인했다. threshold 완화 대신 temporal continuity로 해결했다.

### 오류 6 — 양방향 추적만으로 영상 양끝을 처리하지 못했다

미래 anchor가 없는 edge 구간에 제한된 단방향 extension과 detection association을 추가했다.

### 오류 7 — 중복 detector bbox를 두 사람으로 오해했다

후보 수만 세지 않고 association cost와 second-best margin을 비교했다.

### 오류 8 — 등록 영상이 한 사람뿐이라는 가정이 gallery 청결성을 보장하지 않았다

ear false crop이 높은 best-reference 점수를 만들었다. embedding graph의 dominant component를 사용해 11개 outlier를 review로 격리했다.

### 오류 9 — frame coverage 100%가 품질 100%를 의미하지 않았다

contact sheet가 잘못된 승인 위치를 발견했다. 정량 coverage, reason evidence, 시각 검토를 함께 사용해야 했다.

### 오류 10 — renderer가 바뀌자 benchmark의 ROI 가정이 틀렸다

padding과 ellipse mask를 실제 effect contract에 포함하도록 reusable locality assertion을 수정했다.

### 오류 11 — 강한 익명화가 항상 좋은 UX는 아니었다

v3는 식별 단서를 강하게 줄였지만 배경을 과도하게 가렸다. identity 계층은 유지하고 presentation 계층만 v4 타원형·완화 mosaic로 변경했다.

### 오류 12 — 최종 검증 스크립트에서도 public API를 잘못 호출했다

처음에는 keyword-only인 `run_benchmark(category=...)`를 positional로 호출했고, 다음에는 반환 dict를 `RunnerResult` 목록으로 오해했다. 함수 정의를 다시 확인해 `passed_count/total_count` 계약으로 검증했다. 이 과정은 README와 실행 예제가 실제 반환형까지 명확히 설명해야 한다는 교훈을 다시 확인시켰다.

### 오류 13 — 오디오 보존 도구가 환경에 없었다

현재 PC의 PATH에서 FFmpeg executable을 찾지 못해 v3/v4 결과는 video stream만 기록했다. 처리 자체를 거짓 완료로 표시하지 않고 이 한계를 보고서에 명시했다. `--preserve-audio --ffmpeg-path ...` 경로는 구현돼 있지만 실제 FFmpeg 설치·지정 검증이 남아 있다.

---

## 10. 왜 이 코드 구성인가

### 10.1 권한과 품질을 분리하기 위해

`GalleryApproval`과 tracking 결과를 다른 타입과 모듈로 둔 이유는 높은 tracker visibility나 detector confidence가 신원 승인으로 승격되는 것을 구조적으로 막기 위해서다.

### 10.2 외부 모델을 교체 가능하게 만들기 위해

detector, gallery, point tracker는 Protocol/adapter 경계 뒤에 있다. YuNet, SFace, TAPNext++ 구현 세부사항이 pipeline 전체로 퍼지지 않으므로 향후 모델 비교가 가능하다.

### 10.3 로컬 개인정보를 코드와 분리하기 위해

영상, crop, embedding, model weight, output은 `local_data/`와 ignore 규칙 아래에 둔다. 코드 저장소가 공개돼도 생체 정보가 같이 올라가지 않게 한다.

### 10.4 실패를 재현 가능하게 만들기 위해

승인 reason, gallery revision, anchor 수, propagation reason, 환경 metadata, frame summary를 evidence에 남긴다. 문서의 “통과” 문장보다 실제 JSON artifact가 우선한다.

### 10.5 부분 출력으로 원본이나 결과를 손상시키지 않기 위해

renderer는 sibling temporary 파일을 완성한 후 atomic replace한다. 입력과 출력 경로가 같거나 destination이 이미 있으면 거절한다.

### 10.6 사용자가 비용과 품질을 선택하게 하기 위해

가벼운 frame-by-frame mode와 무거운 TAPNext++ mode를 함께 유지한다. FHD 저장 영상의 연속성이 중요할 때만 tracker 비용을 지불한다.

### 10.7 시각 효과 변경이 identity에 영향을 주지 않게 하기 위해

v3 rectangle에서 v4 ellipse로 바꿀 때 gallery와 tracker를 수정할 필요가 없었다. 이는 presentation layer가 분리됐다는 실제 증거다.

---

## 11. 정량적 개선 과정

| 버전 | 핵심 변경 | 처리 frame | 남은 문제 |
| --- | --- | ---: | --- |
| 초기 | frame별 gallery 평가 | 209/456 | similarity 변동으로 심한 끊김 |
| v1 | 강한 mosaic + 양방향 TAPNext++ | 392/456 | 영상 양끝, 초기 불일치 |
| v2 | 제한된 단방향 edge extension | 408/456 | duplicate bbox에서 중단 |
| v3 | association margin + clean enrollment | 456/456 | 강한 사각형 UX |
| v4 | 완화된 외접 세로 타원 | 456/456 | 다중 인물 일반화 검증 필요 |

이 표에서 중요한 점은 456이라는 숫자만이 아니다. v3가 456에 도달하기 전에 등록 오염을 발견해 수정했다. 잘못된 얼굴 영역을 456 frame 모두 안정적으로 추적하는 것은 성공이 아니라 더 큰 실패이기 때문이다.

---

## 12. 성능과 자원 사용에 대한 판단

실제 분석 처리율은 약 7.43fps로 source 30fps보다 느리다. 따라서 현재 구현은 실시간 스트리밍보다 로컬 저장 영상 batch processing에 적합하다.

TAPNext++ smoke test에서 checkpoint load 약 3.96초, 첫 FHD frame 약 0.40초, 이후 frame 약 0.074초, GPU peak 약 2.43GiB가 관측됐다. 성능 benchmark는 특정 컴퓨터의 통과 기준으로 사용하지 않고 CPU, platform, Python/OpenCV/numpy, frame shape, warm-up, sample count, median/p95와 함께 측정값으로 기록한다.

추후 하드웨어별 목표를 만들려면 먼저 승인된 environment profile을 정의해야 한다. 환경이 다른 결과를 단순 regression으로 판단하지 않는다.

---

## 13. 보안·개인정보 설계

1. 얼굴 데이터와 결과 영상은 Git에 추가하지 않는다.
2. cloud face API나 자동 모델 다운로드를 사용하지 않는다.
3. 모델 파일은 manifest와 SHA-256으로 검증한다.
4. detector confidence는 identity 권한이 아니다.
5. similarity는 관측값이며 승인 결과 객체를 대체하지 않는다.
6. tracker는 profile ID를 생성하지 않는다.
7. gallery 오류와 malformed return은 fail-closed다.
8. 다중 얼굴은 face-by-face 승인한다.
9. evidence 기본값에는 frame pixel, crop, embedding을 넣지 않는다.
10. 원본 파일은 덮어쓰지 않고 새 결과만 생성한다.

---

## 14. 한계와 정직한 완료 기준

### 현재 확인된 한계

- 456/456은 대상자가 주로 혼자 등장하는 제공 영상의 결과다.
- 실제 두 사람이 교차하거나 얼굴이 겹치는 영상 검증이 추가로 필요하다.
- 장시간 완전 가림 뒤 재등장할 때 gallery 재승인 동작을 더 강하게 시험해야 한다.
- 등록에서 격리된 11개 reference 후보는 사람이 직접 검토하면 더 좋다.
- 분석은 실시간 30fps가 아니다.
- 현재 최종 파일은 FFmpeg 부재로 원본 오디오가 포함되지 않는다.
- ellipse가 모든 pose에서 완전한 실제 얼굴 segmentation을 의미하지는 않는다. bbox 기반 외접 mask다.

### 완료라고 말할 수 있는 범위

- 제공된 `learn.mp4`로 clean multi-reference gallery를 만들 수 있다.
- 제공된 `test.mp4`를 FHD 해상도와 frame 수를 유지해 처리할 수 있다.
- 이 영상에서는 456/456 frame에 대상 얼굴 redaction plan이 존재한다.
- 사용자 피드백에 맞춘 완화된 타원형 mosaic 결과를 생성했다.
- 테스트 321개와 benchmark 24개가 통과한다.
- 모든 변경과 민감 산출물은 로컬에만 있다.

---

## 15. 후속 개선 로드맵

### 15.1 다중 인물 crossing 평가 세트

승인 대상과 비대상이 가까워지고 서로 위치를 바꾸는 consented 영상을 구성한다. ID switch, false redaction, missed redaction을 frame 단위로 기록한다.

### 15.2 등록 reference 사람 검토 UI

dominant component 36개와 격리 후보 11개의 contact sheet를 만들어 사용자가 승인·거절할 수 있게 한다. 자동 clustering은 review 우선순위만 제공하고 사람 승인을 대체하지 않는다.

### 15.3 audio remux 실환경 검증

FFmpeg 실행 파일을 명시하고 원본 audio stream을 새 video stream과 remux한다. duration, stream count, codec, sync를 검증한다.

### 15.4 target profile별 calibration

detector confidence median/p95, gallery similarity 분포, recheck 횟수를 evidence로 누적한다. `t_confirm`과 `t_keep`은 먼저 telemetry calibration으로만 사용하고, 권한 정책 변경은 별도 보안 검토와 사람 승인을 거친다.

### 15.5 속도 최적화

anchor가 충분히 가까운 구간은 tracking을 생략하고, tracker 입력 resize와 segment batching을 비교한다. 성능 개선이 visibility나 bbox 합의를 약화하지 않는지 같은 evidence로 검증한다.

### 15.6 정밀 mask 옵션

현재 ellipse는 계산량과 일관성이 좋은 bbox 기반 효과다. 필요하면 별도 opt-in segmentation mode를 추가하되, identity 권한과 mask 생성은 계속 분리한다.

---

## 16. 이 프로젝트에서 도출한 12가지 핵심 통찰

1. 얼굴이 검출됐다는 사실과 그 사람이 목표 인물이라는 사실은 전혀 다른 계약이다.
2. tracker는 위치 기억 장치이지 신원 증명 장치가 아니다.
3. reference 수가 많을수록 좋은 gallery가 되는 것이 아니라, 오염된 reference 하나가 max similarity 정책 전체를 위험하게 만들 수 있다.
4. centroid 하나는 극단 pose를 잃고 best-reference 하나는 outlier에 취약하다. view graph의 연결성이 두 문제 사이의 실용적인 절충점이다.
5. 저장 영상에서는 미래 anchor를 사용할 수 있으므로 양방향 합의가 online 단방향 예측보다 안전하다.
6. 100% frame coverage는 올바른 위치와 올바른 신원을 함께 확인했을 때만 의미가 있다.
7. duplicate detection과 실제 다중 인물은 후보 개수만으로 구분할 수 없다. 기하 cost와 margin이 필요하다.
8. benchmark의 공간 계약은 renderer가 실제로 바꾸는 범위를 따라야 한다.
9. 예외가 결과에서 사라지면 시스템은 실제보다 좋아 보인다. 실패도 append-only evidence가 돼야 한다.
10. 강한 익명화와 좋은 시각 품질은 동일한 목표가 아니다. 사용 목적에 맞는 presentation 설정이 필요하다.
11. 모델, gallery, approval, tracking, rendering을 분리하면 한 계층의 변경이 다른 계층의 보안 의미를 바꾸지 않는다.
12. 실제 영상과 contact sheet는 synthetic unit test가 찾지 못하는 데이터 오염과 시각적 실패를 발견한다.

---

## 17. 포트폴리오에서 강조할 역량

### 문제 분석

막연히 “인식률이 낮다”고 결론 내리지 않고 detection miss와 identity rejection을 frame evidence로 분리했다.

### 안전한 ML 시스템 설계

모델 점수 하나에 권한을 맡기지 않고 명시 승인, 시간축 continuity, effect를 독립 계층으로 설계했다.

### 실제 데이터 디버깅

contact sheet와 embedding graph를 함께 사용해 ear false crop이라는 등록 오염을 발견하고 수정했다.

### 알고리즘 선택

MOT, segmentation, legacy tracker, point tracking의 목적을 비교한 뒤 얼굴 내부 point continuity에 맞는 TAPNext++를 선택했다.

### 호환 가능한 API 설계

schema version과 strict opt-in을 사용해 기존 설정을 깨지 않으면서 validation을 강화했다.

### 검증 중심 개발

321개 테스트, A–E benchmark 24개, 실제 FHD 영상, JSON evidence, contact sheet를 서로 다른 검증 층으로 사용했다.

### 실패 공개와 범위 관리

처리율이 실시간보다 느린 점, 다중 인물 검증이 남은 점, FFmpeg가 없어 오디오를 보존하지 못한 점을 완료 결과와 함께 명확히 기록했다.

---

## 18. 재현과 상세 문서

- 전체 함수 설명: [CODEBASE_FUNCTION_REFERENCE.md](../CODEBASE_FUNCTION_REFERENCE.md)
- 실제 영상 실행: [REAL_VIDEO_TEST_GUIDE.md](../REAL_VIDEO_TEST_GUIDE.md)
- 등록 영상 명세: [VIDEO_REFERENCE_ENROLLMENT_IMPLEMENTATION_SPEC.md](../VIDEO_REFERENCE_ENROLLMENT_IMPLEMENTATION_SPEC.md)
- tracking 작업지시서: [TEMPORAL_TRACKING_AND_STRONG_MOSAIC_WORK_ORDER.md](../TEMPORAL_TRACKING_AND_STRONG_MOSAIC_WORK_ORDER.md)
- 구현·실측 보고서: [TEMPORAL_TRACKING_IMPLEMENTATION_REPORT.md](../TEMPORAL_TRACKING_IMPLEMENTATION_REPORT.md)
- 설정 계약: [CONFIG_SCHEMA.md](../CONFIG_SCHEMA.md)
- benchmark protocol: [PHASE10_BENCHMARK_PROTOCOL.md](../PHASE10_BENCHMARK_PROTOCOL.md)

---

## 19. 최종 회고

이 프로젝트에서 가장 큰 기술적 전환은 tracker를 추가한 것이 아니다. “모델이 같은 사람 같다고 말한다”와 “시스템이 이 사람을 가려도 된다고 승인한다”를 분리한 것이다.

초기 209/456이라는 결과를 threshold 조정만으로 끌어올렸다면 짧은 시간 안에 더 높은 coverage를 만들 수 있었을 것이다. 그러나 실제 gallery 오염 사례처럼 잘못된 reference 하나가 높은 similarity를 만들 수 있기 때문에 그것은 안전한 해결이 아니었다. 명시 approval에서 시작한 anchor, 양방향 tracker 합의, detection association, dominant embedding component, fail-closed reason을 결합해 456/456을 만들었다.

또한 v3의 강한 사각형이 기술적으로는 잘 가렸더라도 사용자가 원하는 결과의 모양과 강도에는 맞지 않았다. 계층을 분리해 둔 덕분에 identity와 tracking을 훼손하지 않고 renderer만 v4 외접 타원으로 교체할 수 있었다.

결과적으로 이 작업은 얼굴 인식 모델을 호출하는 단일 기능 구현이 아니라, 실제 데이터에서 나타나는 불확실성·오염·시간축 단절·출력 품질을 관측 가능한 계약과 단계적 검증으로 관리한 로컬 Vision AI 시스템 개발 사례다.
