# Consented Face Redactor 시각적 흐름도

이 문서는 프로젝트를 처음 보는 사람이 “어떤 자료가 들어오고, 어떤 판단을 거쳐, 무엇이 결과로 나오는가”를 빠르게 이해하기 위한 시각 자료다.

## 1. 전체 시스템 한눈에 보기

```mermaid
flowchart TD
    subgraph ENROLL[1. 대상자 등록]
        A[대상자만 등장하는 영상] --> B[일정 간격으로 frame 추출]
        B --> C[YuNet 얼굴 검출]
        C --> D{유효한 얼굴이 정확히 하나인가?}
        D -- 아니오 --> E[등록 제외 및 review 사유 기록]
        D -- 예 --> F[SFace embedding 생성]
        F --> G[중복 embedding 제거]
        G --> H[Embedding similarity graph]
        H --> I[가장 큰 연결 component 선택]
        I --> J[Clean LocalGallery 저장]
        J --> K[사용자의 명시 ApprovalStore 기록]
    end

    subgraph ANALYZE[2. 대상 영상 분석 · Pass 1]
        L[처리할 원본 영상] --> M[모든 frame의 YuNet detection]
        M --> N[각 얼굴의 SFace embedding]
        N --> O[LocalGallery match]
        K --> P{명시 승인된 profile인가?}
        O --> P
        P -- 아니오 --> Q[권한 없음 · redaction 금지]
        P -- 예 --> R[GalleryApproval 승인 anchor]
        R --> S[TAPNext++ forward tracking]
        R --> T[TAPNext++ backward tracking]
        S --> U{두 경로가 같은 얼굴 위치에 합의하는가?}
        T --> U
        U -- 아니오 --> V[Fail-closed · review 또는 미처리]
        U -- 예 --> W[Detection association과 기하 검증]
        W --> X{Visibility·IoU·이동 조건 통과?}
        X -- 아니오 --> V
        X -- 예 --> Y[불변 RedactionTrackPlan]
    end

    subgraph RENDER[3. 결과 렌더링 · Pass 2]
        Y --> Z[원본 영상을 처음부터 다시 읽기]
        Z --> AA[승인된 bbox만 선택]
        AA --> AB[외접 세로 타원 mask 생성]
        AB --> AC[타원 내부에 adaptive mosaic]
        AC --> AD[Sibling temporary MP4 기록]
        AD --> AE{전체 frame 기록 성공?}
        AE -- 아니오 --> AF[임시 결과 폐기 · 기존 파일 보존]
        AE -- 예 --> AG[Atomic replace로 새 결과 공개]
    end

    L -. 입력 원본은 수정하지 않음 .-> AG
    Q -. detector·tracker 점수만으로 승인 불가 .-> V
```

## 2. 각 AI 구성요소가 답하는 질문

```mermaid
flowchart LR
    A[YuNet Detector] -->|얼굴 후보 bbox와 landmark| B[어디에 얼굴이 있는가?]
    C[SFace Embedder] -->|얼굴 특징 vector| D[등록 얼굴과 얼마나 유사한가?]
    E[GalleryApproval] -->|approved·profile·reason·revision| F[이 얼굴을 가릴 권한이 있는가?]
    G[TAPNext++] -->|point 위치와 visibility| H[승인된 얼굴 위치가 다음 frame에서 어디인가?]
    I[MosaicEffect] -->|ellipse mask와 pixelation| J[승인된 영역을 어떻게 표현할 것인가?]
```

가장 중요한 구분은 `유사하다`와 `승인됐다`가 같지 않다는 점이다. similarity는 gallery 판단에 사용되는 관측값이고, pipeline이 소비하는 최종 권한은 구조화된 `GalleryApproval.approved=True`뿐이다.

## 3. 사람 구별 결과가 전달되는 방식

```mermaid
flowchart TD
    A[현재 frame의 얼굴 detection] --> B[SFace gallery 평가]
    B --> C{GalleryApproval.approved=True?}
    C -- 아니오 --> D[CANDIDATE 또는 review]
    C -- 예 --> E[Profile ID가 있는 explicit anchor]
    E --> F[TrackAuthorization lease]
    F --> G{Tracker visibility와 bbox gate 통과?}
    G -- 아니오 --> H[권한 revoke 또는 전파 중단]
    G -- 예 --> I{같은 profile anchor의 양방향 합의?}
    I -- 아니오 --> H
    I -- 예 --> J[해당 frame 위치에만 권한 전파]
    J --> K[RedactionTrackPlan에 bbox 기록]

    L[높은 detector confidence] -. 단독 승인 불가 .-> D
    M[높은 tracker visibility] -. 단독 승인 불가 .-> D
    N[높은 similarity 숫자] -. 구조화된 승인 없이는 불가 .-> D
```

## 4. 등록 오염을 막는 흐름

```mermaid
flowchart TD
    A[Sampled face embeddings] --> B[시간상 인접한 중복 제거]
    B --> C[Similarity 0.45 이상인 pair 연결]
    C --> D[Connected components 계산]
    D --> E{가장 큰 component인가?}
    E -- 예 --> F[다양한 target pose trajectory로 간주]
    E -- 아니오 --> G[False crop 가능성 · review 격리]
    F --> H{Reference 수가 최대값보다 큰가?}
    H -- 아니오 --> I[전체 component 등록]
    H -- 예 --> J[Farthest-point coverage로 축약]
    J --> I
```

실제 등록 영상에서는 47개 후보가 36개의 dominant component, 9개의 별도 component, 2개의 singleton으로 분리됐다. 귀 주변 false crop을 포함한 11개 후보를 자동 등록하지 않고 review로 격리했다.

## 5. 상태 머신

frame-by-frame 경량 경로의 상태는 다음과 같다.

```mermaid
stateDiagram-v2
    [*] --> UNSEEN
    UNSEEN --> CANDIDATE: 얼굴 후보 검출
    CANDIDATE --> CONFIRMED: GalleryApproval.approved=True
    CANDIDATE --> LOST: 후보가 사라짐
    CONFIRMED --> LOST: detection 또는 승인 연속성 손실
    LOST --> CANDIDATE: 얼굴 재검출 · 재평가 필요
    LOST --> EXPIRED: TTL 경과
    EXPIRED --> CANDIDATE: 새 얼굴 후보 검출
    EXPIRED --> [*]
```

`LOST` 상태는 이전 승인 사실을 기억할 수 있지만 얼굴을 계속 가리는 권한은 아니다. 재등장한 얼굴은 gallery 또는 temporal plan의 검증된 근거가 필요하다.

## 6. 버전별 개선 흐름

```mermaid
flowchart LR
    A[초기<br/>209/456<br/>frame별 gallery] --> B[v1<br/>392/456<br/>양방향 TAPNext++]
    B --> C[v2<br/>408/456<br/>edge 단방향 연장]
    C --> D[v3<br/>456/456<br/>clean enrollment와 association margin]
    D --> E[v4<br/>456/456<br/>완화된 외접 타원 mosaic]
```

숫자만 개선한 것이 아니다. v3 이전 contact sheet에서 귀 부분 false crop을 발견했고, gallery 오염을 제거한 뒤에야 456/456을 최종 결과로 인정했다.
