"""Tests for the Config contract implemented by the production class."""

from consented_face_redactor.config import Config


class TestConfigDefaults:
    def test_default_serializes_all_constructor_fields(self) -> None:
        config = Config.default()

        assert config.to_dict() == {
            "schema_version": 4,
            "effect_mode": "mosaic",
            "t_confirm": 0.65,
            "t_keep": 0.55,
            "track_lost_ttl_frames": 10,
            "recheck_interval_frames": 30,
            "mosaic_grid_cells": 12,
            "mosaic_padding_ratio": 0.18,
            "mosaic_min_block_px": 10,
            "mosaic_shape": "ellipse",
            "mosaic_ellipse_horizontal_scale": 1.4,
            "mosaic_ellipse_vertical_scale": 1.5,
        }


class TestConfigFromDict:
    def test_round_trip_preserves_all_serialized_values(self) -> None:
        original = Config(
            effect_mode="sticker",
            t_confirm=0.8,
            t_keep=0.4,
            track_lost_ttl_frames=5,
            recheck_interval_frames=2,
        )

        assert Config.from_dict(original.to_dict()).to_dict() == original.to_dict()

    def test_partial_dict_uses_defaults_and_ignores_unknown_keys(self) -> None:
        config = Config.from_dict({"effect_mode": "sticker", "unused_key": "ignored"})

        assert config.effect_mode == "sticker"
        assert config.t_confirm == Config.default().t_confirm
        assert "unused_key" not in config.to_dict()

    def test_strict_mode_rejects_unknown_key(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unknown config keys"):
            Config.from_dict({"unused_key": "rejected"}, strict=True)

    def test_legacy_payload_migrates_to_current_schema(self) -> None:
        migrated = Config.from_dict({"effect_mode": "sticker"})

        assert migrated.schema_version == Config.SCHEMA_VERSION
        assert migrated.to_dict()["schema_version"] == 4

    def test_v2_payload_migrates_with_strong_mosaic_defaults(self) -> None:
        migrated = Config.from_dict({"schema_version": 2, "effect_mode": "mosaic"})

        assert migrated.schema_version == 4
        assert migrated.mosaic_grid_cells == 12
        assert migrated.mosaic_padding_ratio == 0.18
        assert migrated.mosaic_min_block_px == 10
        assert migrated.mosaic_shape == "ellipse"

    def test_mosaic_configuration_validation(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="mosaic_grid_cells"):
            Config(mosaic_grid_cells=1)
        with pytest.raises(ValueError, match="mosaic_padding_ratio"):
            Config(mosaic_padding_ratio=0.51)
        with pytest.raises(ValueError, match="mosaic_min_block_px"):
            Config(mosaic_min_block_px=0)
        with pytest.raises(ValueError, match="mosaic_shape"):
            Config(mosaic_shape="triangle")
        with pytest.raises(ValueError, match="at least as tall"):
            Config(
                mosaic_ellipse_horizontal_scale=1.5,
                mosaic_ellipse_vertical_scale=1.4,
            )
        with pytest.raises(ValueError, match="enclose"):
            Config(
                mosaic_ellipse_horizontal_scale=1.1,
                mosaic_ellipse_vertical_scale=1.1,
            )
