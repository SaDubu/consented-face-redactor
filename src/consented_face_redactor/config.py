"""Configuration for the consented-face-redactor pipeline."""

from __future__ import annotations


class Config:
    """Pipeline configuration with defaults."""

    def __init__(
        self,
        effect_mode: str = "mosaic",
        *,
        t_confirm: float = 0.65,
        t_keep: float = 0.55,
        track_lost_ttl_frames: int = 10,
        recheck_interval_frames: int = 30,
    ) -> None:
        """Initialize with config values.

        Parameters
        ----------
        effect_mode : str
            The redaction effect to apply (e.g., 'mosaic', 'blur').
        t_confirm : float
            Candidate confidence threshold required to transition
            CANDIDATE → CONFIRMED.
        t_keep : float
            Minimum confidence to keep a previously CONFIRMED track
            alive without full re-confirmation.
        track_lost_ttl_frames : int
            Number of consecutive frames with no face detection after which
            an LOST track becomes EXPIRED.
        recheck_interval_frames : int
            Every *N* frames while CANDIDATE, invoke the gallery matcher to
        check for identity confirmation.
        """
        self.effect_mode = effect_mode
        self.t_confirm = t_confirm
        self.t_keep = t_keep
        self.track_lost_ttl_frames = track_lost_ttl_frames
        self.recheck_interval_frames = recheck_interval_frames

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        """Build a Config instance from a plain dict (for baseline test compat)."""
        mapping: dict[str, object] = {}
        for key in ("effect_mode", "t_confirm", "t_keep",
                     "track_lost_ttl_frames", "recheck_interval_frames"):
            if key in d:
                mapping[key] = d[key]
        return cls(**mapping)
