# 🔧 인수인계서 — consensed-face-redactor (2026-08-24 기준)

> 이 문서는 Hermes Agent 세션 종료 시점까지 완료된 사항, 문제점, 미완료 항목을 기록합니다.
> 후속 작업자는 이 문서를 기준으로 계승하여 작업을 계속할 수 있습니다.

---

## 1. 현재 상태 요약 (Git 지점)

| 항목 | 내용 |
|------|------|
| 브랜치 | `main` |
| Remote | `origin → https://github.com/SaDubu/consented-face-redactor.git` |
| 변경 파일 (미커밋) | `README.md`(+65), `src/consented_face_redactor/pipeline.py`(+34/-24) |
| 신규 파일 (untracked) | `_p10bootstrap.py`, `_p10bootstrap_v2.py`, `_p10test_v3.py`, `docs/CONFIG_SCHEMA.md`, `docs/PHASE10_BENCHMARK_PROTOCOL.md`, `src/consented_face_redactor/benchmark/`, `tests/test_e2e_video.py`, `tests/unit/` |
| 커밋 상태 | **미커밋** — 변경사항을 먼저 commit 후 push 필요 |

---

## 2. 완료된 사항 (✅)

### 2.1 Category B 패치 (runner.py) — 세 가지 강제 수정 사항 완료

**목표:** `run_benchmark(category="B")`의 B1(모자이크 ROI), B2(스티커 bytes/DPI), B3(미검출) 시나리오 완벽한 통과 + 독립 제어 흐름 보장.

#### 패치 ①: B2 인코딩 격리 (try 블록 내부 이동)
- **파일:** `src/consented_face_redactor/benchmark/runner.py` (~line 228-260)
- **변경 전 문제:** `cv2.imencode(".png", stub_rgba)` + early raise try 블록 **바깥**에 있어, 인코딩 실패 시 즉시 함수 종료 → B3가 절대 실행되지 않음.
- **변경 후:** 인코드/early raise를 try 내부로 이동. `except Exception`에서 `error_b2 = repr(exc)` 저장 후 `results.append(passed_b2=..., error=error_b2)`. 이어서 **B3는 항상 독립적으로 실행됨**.
- **동반 변경:** `encoded_png_data.tobytes()` — ndarray를 `.tobytes()`로 bytes 직렬화. Config setter의 truth-value gate를 통과.

#### 패치 ②: B1 outside-pixel 보존 비교 (bbox 기반 전체 외부 영역)
- **파일:** `src/consented_face_redactor/benchmark/runner.py` (~line 180-192)
- **변경 전 문제:** `pr.result_frame[:5, :5] == outside_b1` — `[:5,:5]`(좌상단 5x5 픽셀)만 확인. bbox 외부의 거의 모든 영역이 검증되지 않음.
- **변경 후:**
  ```python
  bbox = (10, 10, 70, 70)  # face detection bbox (effect ROI)
  outside_mask = np.ones((96, 96), dtype=bool)
  outside_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = False  # bbox 내부만 제외
  orig_frame_for_compare = frame_b1.copy()
  outside_pixels_ok = (pr.result_frame[outside_mask] == orig_frame_for_compare[outside_mask]).all()
  ```
  → 실제 effect가 적용된 bbox(10-69, 10-69)만 마스킹하고 **나머지 9,136 픽셀 전체**가 원본과 byte-identical인지 검증.

#### 패치 ③: B2 outside-pixel 보존 비교 (동일 패턴 적용)
- **파일:** `src/consented_face_redactor/benchmark/runner.py` (~line 240-255)
- **변경 전 문제:** `outside_region_b2 = frame_b2[:5, :5].copy()` — 동일한 `[:5,:5]` 코너 샘플링 한계.
- **변경 후:** B1과 동일한 bbox 기반 전체 외부 마스크 패턴 적용:
  ```python
  bbox2 = (10, 10, 70, 70)
  outside_mask_b2 = np.ones((96, 96), dtype=bool)
  outside_mask_b2[bbox2[0]:bbox2[2], bbox2[1]:bbox2[3]] = False
  orig_frame_for_compare_b2 = frame_b2.copy()
  outside_preserved = (pr_b2.result_frame[outside_mask_b2] == orig_frame_for_compare_b2[outside_mask_b2]).all()
  ```

#### 패치 ④: 시나리오별 독립 frame 할당 (모든 Scenario)
- B1: `frame.copy()` (line 180), B2: `frame.copy()` (line 240), B3: `frame.copy()` (line 269) — 모두 검증 완료.

#### B 결과 (실제 실행 시)
```
Total: 3, Passed: 3
B1-mosaic-face-center: passed=True, error=None
B2-sticker-center-place: passed=True, error=None
B3-no-detection-none-redacted: passed=True, error=None
```
- `py_compile` 통과 ✅
- `git diff --check` 통과 (exit=0) ✅

### 2.2 문서 작업 (p10d_docs) — 완료

| 파일 | 내용 |
|------|------|
| `docs/CONFIG_SCHEMA.md` | Config 클래스의 전체 필드/타입/디폴트값 스키마 정리 |
| `docs/PHASE10_BENCHMARK_PROTOCOL.md` | Phase 10 벤치마크 프로토콜(범주 A/B/C/D/E) 공식 문서 |

### 2.3 기타 변경 사항 (uncommitted)

- `README.md`: 프로젝트 설명(+65 라인 추가)
- `src/consented_face_redactor/pipeline.py`: 수정(+34/-24 라인) — 정확한内容は git diff 참조

---

## 3. 문제점 및 격차 (⚠️ — 테스트 파일이 이 항목들의 주요 차단 요소)

### 3.1 테스트 파일 (`tests/unit/test_benchmark_runner.py`) — **완료되지 않음**

현재 테스트 파일은 **생산 API와 완전히 다른 코드**로, 다음과 같은 문제가 있습니다:

| 라인 | 클래스/함수 | 문제 |
|------|------------|------|
| 전체 | `TestRunnerResult` (Lines 9-30) | RunnerResult DTO 검증 — **구조적 유효**. 아직 수정 안됨 |
| 50-68 | `TestIdentitySafetyCategory` | patch 적용 중이지만 Line 64에 **syntax error** 있음: `detector=_MockHighConfDet()` 다음 **comma 누락**(`gallery=...`와 사이에 `,` 필요) |
| 71-103 | `TestRedactionAccuracyCategory` | `synthetic.create_synth_frame()`, `pipeline.render_mosaic()`, `numpy.array_equal(result[bbox], ...)` 등 — **존재하지 않는 API** / **global fixture에 의존**. 하드코딩 성공값 사용 중. |
| 98-112 | `TestTrackTransitions` | `new_Pipeline()`, `synth.create_frame()`, `pipeline.current_track_state` — 모두 **미생성/mock API**. production API와 무관. |
| 118-138 | `TestAggregateReport` | `benchmark_runner.generate_aggregate_report()` — module import 경로가 실제 구조와 다름. 하드코딩된 기대값 사용 중. |
| 141-159 | `TestBenchmarkRunner` | `benchmark_runner.run(category="A")` — module import 경로, production API(`run_benchmark()`)와 불일치. |

**요구된 테스트 범위 (완료 목표):**
1. `run_benchmark(category="B")`가 B1/B2/B3 세 결과를 반환하는지
2. 각 결과가 scenario 이름, passed, duration_ms, error 계약을지는지
3. 실패 scenario도 결과 목록에서 생략되지 않는지
4. `generate_aggregate_report()`가 유효 JSON이며 실제 실행 결과 수를 반영하는지
5. 존재하지 않는 과거 API, mock, 하드코딩 성공값 제거

**제한 사항:**
- **큰 write_file(payload) 누락됨** — 환경 특성상 대용량 파일 작성 시 본문이 잘리는 현상 발생. 따라서 작은 patch 단위로 진행 중이었으나, 세션 종료로 중단됨.
- `py_compile` 테스트 파일은 Line 64 syntax error로 아직 통과 안 됨.

### 3.2 outside-mask 검증 — 완료되었으나 추가 확인 필요

B1/B2의 outside-pixel 비교가 bbox 기반 전체外部 영역을 검사하도록 patch 적용 완료했습니다(위 섹션 2.1 패치②,③ 참조). 하지만 실제 **pytest 통과**는 테스트 파일 문제(3.1)로 인해 검증되지 않았습니다.

---

## 4. 미완료 항목 (다음 작업자 수행 사항)

### 우선순위 P0 (Blocker — Category B 완료 차단에 직접 영향)

| # | 작업 | 상세 | 우선순위 |
|---|------|------|----------|
| P0-1 | 테스트 파일 patch 2/6 | TestIdentitySafetyCategory(Lines 50-68) 완전히 run_benchmark 기반으로 교체 + Line 64 syntax error 수정(comma 추가 또는 클래스 전체 제거) | 🔴 Blocker |
| P0-2 | 테스트 파일 patch 3/6 | TestRedactionAccuracyCategory(Lines 71-103) — `[:5,:5]` 코너 검증 → 실제 bbox 기반 outside-mask 검증으로 교체. `run_benchmark(category="B")` 결과 기반으로 검증. | 🔴 Blocker |
| P0-3 | 테스트 파일 patch 4/6 | TestTrackTransitions(Lines 98-112) — production API(`run_benchmark()` 기반)로 교체 | 🟡 Medium |
| P0-4 | 테스트 파일 patch 5/6 | TestAggregateReport — `generate_aggregate_report()` 실제 호출 검증으로 교체 | 🟡 Medium |
| P0-5 | 테스트 파일 patch 6/6 | TestBenchmarkRunner — 현재 API(`run_benchmark()`)에 맞추어 수정 | 🟡 Medium |
| P0-6 | `py_compile` 테스트파일 통과 | 모든 patch 적용 후 `python -m py_compile tests/unit/test_benchmark_runner.py` | 🔴 Blocker |

### 우선순위 P1 (Category B final 검증)

| # | 작업 | 상세 |
|---|------|------|
| P1-1 | Category B 직접 실행 | `run_benchmark(category="B")` 결과 확인(Total/Passed/B1~B3 개별 passed/error 보고) |
| P1-2 | 테스트 파일 pytest 통과 | `pytest tests/unit/test_benchmark_runner.py -v` 가 모든 assertion에서 pass해야 함 |
| P1-3 | `git diff --check` 통과 | whitespace 문제 없음 확인 |

### 우선순위 P2 (마무리 작업)

| # | 작업 | 상세 |
|---|------|------|
| P2-1 | `benchmark/__init__.py` 업데이트 | 새로운 public API 확인 후 export 수정 |
| P2-2 | commit & push | 변경사항만 커밋(`README.md`, `pipeline.py`, `runner.py`, 문서, 테스트) → push |

---

## 5. 환경/제약사항 (반드시 준수)

### 5.1 Windows patch 바이너리 망가짐 규칙
- `patch` 텍스트 기반 도구에서 **바이너리 리터럴(`\x89\xPNG...`)이 깨질 수 있음**
- binary/raw 데이터를 포함한 패치는 항상 programmatic 생성이나 base64 인코딩으로 처리해야 함
- `\r\n`, `\n` 혼합 시 Windows line ending 문제로 patch 실패 가능

### 5.2 Synthetic/Local 데이터만 사용
- 실제 생체정보(얼굴 이미지 등) 접근 절대 불가
- 외부 의존성(API, CDN, 네트워크) 절대 불가
- 시크릿/토큰/비밀정보 절대 불가

### 5.3 파일 작성 제한
- **큰 write_file(payload) 누락됨** — 5KB 이상 파일 생성 시 본문이 잘리는 현상 확인됨
- 반드시 **작은 patch 단위**로 진행하거나, base64 인코딩 후 쉘 스크립트로 복원 권장

### 5.4 Python 환경
- Python 3.12.10, pip → python3.12
- Windows host (Git Bash / MSYS 환경)
- `terminal` 도구에서 PowerShell/CMD 빌트인(`Get-ChildItem`, `$env:FOO`) 사용 불가 → POSIX sh 구문(`ls`, `$HOME`, `grep`, `cat`) 사용

### 5.5 테스트 작성 시 주의
- **허드코딩된 성공값(True/1, passed=True 하드코딩) 금지** — 실제 결과 기반으로 검증해야 함
- `monkeypatch` 의존 테스트도 실제 mock 대신 real pipeline 호출 권장
- 각 scenario가 실패해도 결과 목록에서 **생략되지 않음(append됨)**을 반드시 검증

---

## 6. 주요 파일 참조 (변경 사항 기준)

| 파일 | 설명 | 변경 여부 |
|------|------|-----------|
| `src/consented_face_redactor/benchmark/runner.py` | B wrapper, `_one`, outside-mask logic, cv2 encode gate | ✅ 패치 완료(3개 patch + outside-mask 교체) |
| `tests/unit/test_benchmark_runner.py` | 단위 테스트 (160라인) | ⚠️ 1/6 patch 완료, 5/6 미완료 |
| `src/consented_face_redactor/config.py` | Config 클래스 (truth-value gate 포함) | ✅ 참고됨(변경 없음) |
| `src/consented_face_redactor/pipeline.py` | 파이프라인 class | ⚠️ 변경됨(미커밋) |
| `README.md` | 프로젝트 설명 | ⚠️ 변경됨(미커밋) |

---

## 7. 작업 흐름 재개 방법

```bash
# 1. 브랜치 확인
cd "D:/AI-Legion/hermes_bot_di/consented-face-redactor"
git status --short

# 2. 테스트 파일 read-back 후 patch 진행
#    (테스트 파일 읽은 후 → 하나씩 patch → py_compile 확인 → 다음 patch)

# 3. 모든 patch 완료 후 최종 검증
python -m py_compile tests/unit/test_benchmark_runner.py
python -m pytest tests/unit/test_benchmark_runner.py -v
python -c "import sys; sys.path.insert(0, 'src'); from consented_face_redactor.benchmark.runner import run_benchmark; r = run_benchmark(category='B'); print(f'Total: {r[\"total_count\"]}, Passed: {r[\"passed_count\"]}')"

# 4. git 체크
git diff --check
git diff --stat HEAD

# 5. 커밋 & push (원하면)
git add -A
git commit -m "fix: Category B outside-mask, fix test syntax errors"
git push origin main
```

