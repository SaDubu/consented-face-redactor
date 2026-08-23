"""Tests for config module — strict validation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# ------------------------------------------------------------------ #
# Defaults
# ------------------------------------------------------------------ #


class TestDefault:
    def test_default_values(self):
        from consented_face_redactor.config import Config, EffectMode, UncertainPolicy

        cfg = Config.default()

        assert cfg.effect_mode == EffectMode.MOSAIC.value
        assert cfg.mosaic_block_size_px == 8
        assert cfg.mosaic_padding_px == 0
        assert cfg.sticker_scale_f == pytest.approx(1.0)
        assert cfg.t_confirm == pytest.approx(0.65)
        assert cfg.t_keep == pytest.approx(0.55)
        assert cfg.uncertain_policy == UncertainPolicy.PRIVACY_SAFE.value
        assert cfg.track_lost_ttl_frames == 3
        assert cfg.recheck_interval_frames == 2
        assert cfg.max_face_area_ratio == pytest.approx(0.85)
        assert cfg.min_face_area_px == 256

    def test_from_dict_returns_default(self):
        from consented_face_redactor.config import Config

        d = {
            "effect_mode": "mosaic",
            "mosaic_block_size_px": 8,
            "mosaic_padding_px": 0,
            "sticker_scale_f": 1.0,
            "t_confirm": 0.65,
            "t_keep": 0.55,
            "uncertain_policy": "privacy_safe",
            "track_lost_ttl_frames": 3,
            "recheck_interval_frames": 2,
        }
        cfg = Config.from_dict(d)
        assert cfg.effect_mode == "mosaic"
        assert cfg.t_confirm == pytest.approx(0.65)


# ------------------------------------------------------------------ #
# Valid inputs
# ------------------------------------------------------------------ #


class TestValidInputs:
    def test_all_enum_values_per_effect_mode(self):
        from consented_face_redactor.config import Config, EffectMode

        for em in EffectMode:
            cfg = Config(effect_mode=em.value)
            assert cfg.effect_mode == em.value

    def test_custom_t_keep_lt_t_confirm(self):
        from consented_face_redactor.config import Config

        cfg = Config(t_confirm=0.7, t_keep=0.4)
        assert cfg.t_confirm > cfg.t_keep

    def test_float_fields_are_canonicalized(self):
        from consented_face_redactor.config import Config

        cfg = Config(sticker_scale_f=1, t_confirm=1, t_keep=0)
        assert isinstance(cfg.sticker_scale_f, float)
        assert isinstance(cfg.t_confirm, float)
        assert isinstance(cfg.t_keep, float)

    def test_input_and_output_paths_valid(self, tmp_path: Path):
        from consented_face_redactor.config import Config

        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        inp.touch()
        out.touch()

        cfg = Config(input_path=inp, output_path=out)
        assert cfg.input_path == inp.resolve()
        assert cfg.output_path == out.resolve()

    def test_to_dict_roundtrip(self):
        from consented_face_redactor.config import Config

        original = Config(
            effect_mode="sticker",
            mosaic_block_size_px=16,
            t_confirm=0.8,
            sticker_scale_f=1.25,
        )
        d = original.to_dict()
        restored = Config.from_dict(d)

        assert restored.effect_mode == "sticker"
        assert restored.mosaic_block_size_px == 16
        assert restored.t_confirm == pytest.approx(original.t_confirm)
        assert restored.sticker_scale_f == pytest.approx(1.25)

    def test_no_side_effects(self, tmp_path: Path):
        """Config creation must never create files."""
        from consented_face_redactor.config import Config

        initial_pwd = Path.cwd()
        # Ensure we check from the current directory
        before = set(str(p) for p in Path(".").rglob("*"))  # type: ignore[assignment]
        Config.default()
        after = set(str(p) for p in Path(".").rglob("*"))  # type: ignore[assignment]

        assert before == after


# ------------------------------------------------------------------ #
# Invalid inputs — rejection
# ------------------------------------------------------------------ #


class TestInvalidInputs:
    @pytest.mark.parametrize("effect_mode", ["blur", "censor", "__unknown__"])
    def test_reject_invalid_effect_mode(self, effect_mode: str):
        from consented_face_redactor.config import Config

        with pytest.raises(ValueError, match="effect_mode must be one of"):
            Config(effect_mode=effect_mode)

    @pytest.mark.parametrize(
        "key,value", [
            ("mosaic_block_size_px", 0),
            ("mosaic_block_size_px", -1),
            ("track_lost_ttl_frames", 0),
            ("track_lost_ttl_frames", -1),
        ],
    )
    def test_reject_non_positive_int(self, key: str, value):
        from consented_face_redactor.config import Config

        with pytest.raises(ValueError):
            Config(**{key: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "key,value", [
            ("t_confirm", 1.50),
            ("t_confirm", -0.01),
            ("t_keep", 1.50),     # out of [0,1] range
        ],
    )
    def test_reject_float_out_of_range(self, key: str, value):
        from consented_face_redactor.config import Config

        extra: dict = {key: value}
        with pytest.raises(ValueError):
            Config(**extra)  # type: ignore[arg-type]

    def test_reject_equal_input_output_paths(self, tmp_path: Path):
        from consented_face_redactor.config import Config

        p = tmp_path / "same.mp4"
        p.touch()

        with pytest.raises(ValueError, match="input_path and output_path must be distinct"):
            Config(input_path=p, output_path=p)

    def test_reject_nan_t_confirm(self):
        from consented_face_redactor.config import Config

        with pytest.raises(ValueError):
            Config(t_confirm=float("nan"))  # type: ignore[arg-type]

    def test_reject_inf_t_keep(self):
        from consented_face_redactor.config import Config

        with pytest.raises(ValueError):
            Config(t_keep=float("inf"), t_confirm=0.9)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mosaic_block_size_px": True},
            {"mosaic_padding_px": False},
            {"t_confirm": True},
            {"track_lost_ttl_frames": True},
        ],
    )
    def test_rejects_boolean_numeric_values(self, kwargs):
        from consented_face_redactor.config import Config

        with pytest.raises(ValueError):
            Config(**kwargs)

    def test_rejects_empty_and_remote_paths(self):
        from consented_face_redactor.config import Config

        with pytest.raises(ValueError, match="empty"):
            Config(input_path="   ")
        with pytest.raises(ValueError, match="local filesystem"):
            Config(input_path="https://example.invalid/input.mp4")

    def test_error_does_not_echo_token_like_effect_value(self):
        from consented_face_redactor.config import Config

        secret = "A" * 48
        with pytest.raises(ValueError) as error:
            Config(effect_mode=secret)
        assert secret not in str(error.value)

    @pytest.mark.parametrize(
        "uncertain_policy", ["aggressive", "auto_reject", "__fake__"],
    )
    def test_reject_invalid_uncertain_policy(self, uncertain_policy: str):
        from consented_face_redactor.config import Config

        with pytest.raises(ValueError, match="uncertain_policy must be one of"):
            Config(uncertain_policy=uncertain_policy)


# ------------------------------------------------------------------ #
# from_dict — rejection
# ------------------------------------------------------------------ #


class TestFromDictRejection:
    def test_reject_non_object_payload(self):
        from consented_face_redactor.config import Config

        with pytest.raises(ValueError, match="object"):
            Config.from_dict([])  # type: ignore[arg-type]

    def test_reject_unknown_key(self):
        from consented_face_redactor.config import Config

        with pytest.raises(ValueError, match="Unknown config keys"):
            Config.from_dict({"effect_mode": "mosaic", "bogus_field": 42})

    def test_reject_url_in_value(self):
        from consented_face_redactor.config import Config

        with pytest.raises(ValueError, match="disallowed value"):
            Config.from_dict({
                "effect_mode": "https://evil.example.com/token?x=y",  # type: ignore[arg-type]
            })

    def test_reject_nan_in_float_value(self):
        import numpy as np

        from consented_face_redactor.config import Config

        with pytest.raises(ValueError, match="must be finite"):
            Config.from_dict({"t_confirm": float("nan")})  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# Serialization & repr / immutability
# ------------------------------------------------------------------ #


class TestSerialization:
    def test_roundtrip_preserves_values(self):
        from consented_face_redactor.config import Config

        original = Config(
            effect_mode="sticker",
            mosaic_block_size_px=12,
            t_confirm=0.75,
            sticker_scale_f=0.9,
            track_lost_ttl_frames=5,
        )
        d = original.to_dict()
        restored = Config.from_dict(d)

        # All fields match
        for key in Config.__slots__:
            o_val = getattr(original, key)
            r_val = getattr(restored, key)
            if isinstance(o_val, float):
                assert r_val == pytest.approx(o_val), f"{key} mismatch"
            else:
                assert r_val == o_val, f"{key} mismatch"

    def test_config_is_immutable_after_validation(self):
        from consented_face_redactor.config import Config

        config = Config.default()
        with pytest.raises(AttributeError, match="immutable"):
            config.t_confirm = 2.0
