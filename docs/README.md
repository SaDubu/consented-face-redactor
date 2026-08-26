# 문서 인덱스

GitHub 첫 화면의 설치·실행 절차는 루트 [README.md](../README.md)에서 확인할 수 있습니다. 이 디렉터리는 프로젝트의 상세 설계, 코드 계약, 검증 결과와 포트폴리오를 목적별로 분리합니다.

## 사용 가이드

- [REAL_VIDEO_TEST_GUIDE.md](REAL_VIDEO_TEST_GUIDE.md): 모델·입력·gallery를 준비하고 실제 영상을 처리하는 절차
- [CONFIG_SCHEMA.md](CONFIG_SCHEMA.md): Config schema v4, strict mode, 타원형 mosaic 설정
- [SECURITY_AND_DATA_POLICY.md](SECURITY_AND_DATA_POLICY.md): 생체 정보·모델·결과물의 로컬 보관 정책

## 코드·테스트 참조

- [CODEBASE_FUNCTION_REFERENCE.md](CODEBASE_FUNCTION_REFERENCE.md): 현재 실행되는 모듈과 함수별 책임
- [PHASE10_BENCHMARK_PROTOCOL.md](PHASE10_BENCHMARK_PROTOCOL.md): Category A–E benchmark 계약
- [VIDEO_REFERENCE_ENROLLMENT_IMPLEMENTATION_SPEC.md](VIDEO_REFERENCE_ENROLLMENT_IMPLEMENTATION_SPEC.md): 등록 영상 기반 multi-reference gallery 명세

## 설계·구현·검증 기록

- [SYSTEM_ARCHITECTURE_GUIDE.md](architecture/SYSTEM_ARCHITECTURE_GUIDE.md): 구성요소, 데이터 계약, 알고리즘과 기술 스택
- [PROJECT_FLOWCHART.md](architecture/PROJECT_FLOWCHART.md): 등록·승인·추적·렌더링과 보안 경계 시각화
- [LOCAL_REAL_MODEL_IMPLEMENTATION_REPORT.md](LOCAL_REAL_MODEL_IMPLEMENTATION_REPORT.md): YuNet/SFace 실제 모델 연결 보고서
- [TEMPORAL_TRACKING_AND_STRONG_MOSAIC_WORK_ORDER.md](TEMPORAL_TRACKING_AND_STRONG_MOSAIC_WORK_ORDER.md): TAPNext++ 및 mosaic 작업지시서
- [TEMPORAL_TRACKING_IMPLEMENTATION_REPORT.md](TEMPORAL_TRACKING_IMPLEMENTATION_REPORT.md): 버전별 실제 영상 결과와 최종 v4 검증

## 포트폴리오

- [PROJECT_PORTFOLIO_CASE_STUDY.md](portfolio/PROJECT_PORTFOLIO_CASE_STUDY.md): 문제 발견, 오류, 원인 분석, 설계 선택, 함수 책임, 검증과 한계를 설명한 상세 사례 연구

포트폴리오는 프로젝트 실행 설명과 분리된 독립 문서입니다. GitHub 방문자는 루트 README에서 빠르게 실행 방법을 확인하고, 기술적 의사결정이 필요한 경우에만 포트폴리오 문서로 이동할 수 있습니다.
