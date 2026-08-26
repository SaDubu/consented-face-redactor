# Configuration Schema Reference

`Config` serializes schema version 4. Payloads without `schema_version` are
treated as legacy v1; v1/v2/v3 payloads are migrated in memory when read.

```python
Config(
    effect_mode="mosaic",
    t_confirm=0.65,
    t_keep=0.55,
    track_lost_ttl_frames=10,
    recheck_interval_frames=30,
    mosaic_grid_cells=12,
    mosaic_padding_ratio=0.18,
    mosaic_min_block_px=10,
    mosaic_shape="ellipse",
    mosaic_ellipse_horizontal_scale=1.40,
    mosaic_ellipse_vertical_scale=1.50,
    schema_version=4,
)
```

| Field | Default | Implemented behavior |
|---|---:|---|
| `schema_version` | `4` | Serialization contract version. v1/v2/v3 input remains readable. |
| `effect_mode` | `"mosaic"` | String passed to the renderer. |
| `t_confirm` | `0.65` | Non-authorizing quality/calibration value. |
| `t_keep` | `0.55` | Non-authorizing continuity-quality value. |
| `track_lost_ttl_frames` | `10` | LOST-to-EXPIRED frame interval. |
| `recheck_interval_frames` | `30` | legacy `embed()/match()` gallery adapter의 candidate recheck interval. 실제 `ApprovedLocalGalleryAdapter`는 tracker가 없으므로 얼굴별 승인을 매 프레임 재평가한다. |
| `mosaic_grid_cells` | `12` | 얼굴 짧은 변을 나눌 적응형 block grid 수. 값이 커져 이전 v3보다 block이 작다. |
| `mosaic_padding_ratio` | `0.18` | `mosaic_shape="rectangle"` 호환 모드에서 bbox를 네 방향으로 확장하는 비율. |
| `mosaic_min_block_px` | `10` | 작은 얼굴에도 적용할 최소 mosaic block 크기. |
| `mosaic_shape` | `"ellipse"` | 기본 공간 효과 마스크. `"rectangle"`은 이전 사각형 동작을 유지한다. |
| `mosaic_ellipse_horizontal_scale` | `1.40` | bbox 반폭 대비 타원 가로 반축 배율. |
| `mosaic_ellipse_vertical_scale` | `1.50` | bbox 반높이 대비 타원 세로 반축 배율. |

## Loading policy

```python
Config.from_dict(payload)               # compatibility mode: ignores unknown keys
Config.from_dict(payload, strict=True)  # rejects unknown keys
```

The CLI exposes the same opt-in behavior through `--strict-config`. Current
Config는 v4 mosaic 수치와 타원 기하를 검증한다. 두 반축은 유한한 1.0–3.0 값이어야 하고, 세로 반축은 가로보다 작을 수 없으며, bbox 네 모서리가 타원 안에 들어가야 한다. `effect_mode`와 legacy
telemetry/state 필드의 의미·강한 validation은 기존 호환 계약을 유지한다.

## Mosaic spatial contract

- 기본 `ellipse` 모드는 raw face bbox의 중심을 타원 중심으로 사용한다.
- 1.40/1.50 배율은 bbox 네 모서리를 포함하면서 세로가 가로보다 약 7% 긴 외접 타원을 만든다.
- mosaic 계산 영역은 타원의 bounding rectangle이지만 실제 합성은 boolean ellipse mask 내부에만 적용한다.
- mask 밖 pixel은 입력과 byte-identical하게 보존한다.
- 강도는 `max(round(min(face_width, face_height) / 12), 10)`으로 정한다.
- 저수준 `MosaicConfig()` 단독 기본은 기존 호출자 호환을 위해 `rectangle`을 유지하지만, 공개 `Config.default()`와 `RedactionPipeline`의 production 기본은 `ellipse`다.

## Identity boundary

`t_confirm` and `t_keep` never authorize identity. `CONFIRMED` and redaction
require a structured `GalleryApproval` with `approved=True`.
