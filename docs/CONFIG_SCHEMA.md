# Configuration Schema Reference

`consented_face_redactor.config.Config` — verified against production `config.py`.

## Config constructor (real signature)

```python
cfg = Config(
    effect_mode="mosaic",                    # str: "mosaic" or "sticker"
    t_confirm=0.65,                          # float (candidate → CONFIRMED confidence floor)
    t_keep=0.55,                             # float (confirmation keep-threshold)
    track_lost_ttl_frames=10,                # int (frames before LOST→EXPIRED)
    recheck_interval_frames=30,              # int (gallery recheck interval in CANDIDATE)
)
```

## Fields (production-verified)

| Field                    | Type          | Default | Description                                           |
|--------------------------|---------------|---------|-------------------------------------------------------|
| `effect_mode`            | `str`         | `"mosaic"` | Visual effect. Valid: `"mosaic"`, `"sticker"`.     |
| `t_confirm`              | `float`       | 0.65    | Confidence threshold when CANDIDATE → CONFIRMED.      |
| `t_keep`                 | `float`       | 0.55    | Minimum confirmed-trace confidence for continuity.    |
| `track_lost_ttl_frames`  | `int`         | 10      | Frames to keep a track after last detection.          |
| `recheck_interval_frames`| `int`         | 30      | Gallery re-check cadence while tracking.              |

## Default Configuration (`Config.default()`)

```python
from consented_face_redactor.config import Config
cfg = Config.default()  # All defaults above applied.
```

## Validation Rules (production as-of-date)

- **`t_confirm < 1.0`** — `ValueError` if ≥ 1.0.
- **Path constraints**: Both `input_path` and `output_path` must resolve to absolute local filesystem paths; remote URLs are rejected; empty paths (whitespace-only strings) raise a `"distinct paths"` error when equal.
- **Positive integers**: Any int config field requiring a positive value (`mosaic_block_size_px`, `mosaic_padding_px`) rejects zero or negative with a generic `ValueError`.

## Serialization round-trip

```python
d = cfg.to_dict()        # Returns dict[str, Any].
cfg2 = Config.from_dict(d)  # Equivalent instance (fields subset-safe).
```

