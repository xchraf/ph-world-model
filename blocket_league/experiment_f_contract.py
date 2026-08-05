"""Pure, simulator-free preregistered contract for Experiment F."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal


Variant = Literal[
    "full",
    "no_jacobian",
    "single_horizon",
    "shuffled_lens",
    "skew_only",
    "constant_port",
]

REGISTERED_VARIANTS: tuple[Variant, ...] = (
    "full",
    "no_jacobian",
    "single_horizon",
    "shuffled_lens",
    "skew_only",
    "constant_port",
)

REGISTERED_SYSTEMS: tuple[str, ...] = ("pendulum", "blocket")


@dataclass(frozen=True)
class HiddenExcitationConfig:
    """Public seal configuration; it contains no realized effort trajectory."""

    frames: int = 17
    image_size: int = 64
    hold_min: int = 1
    hold_max: int = 4
    coast_probability: float = 0.20
    blocket_contact_probability: float = 0.60
    blocket_precontact_gap_min: float = 0.025
    blocket_precontact_gap_max: float = 0.085
    blocket_approach_hold_min: int = 4
    blocket_approach_hold_max: int = 7
    blocket_approach_effort_min: float = 0.65
    blocket_approach_effort_max: float = 0.95

    def __post_init__(self) -> None:
        if self.frames < 3:
            raise ValueError("frames must be at least three")
        if self.image_size < 8:
            raise ValueError("image_size must be at least eight")
        if self.hold_min < 1 or self.hold_max < self.hold_min:
            raise ValueError("invalid hidden-effort hold interval")
        if not 0.0 <= self.coast_probability < 1.0:
            raise ValueError("coast_probability must be in [0, 1)")
        if not 0.0 <= self.blocket_contact_probability <= 1.0:
            raise ValueError("blocket_contact_probability must be in [0, 1]")
        if not (
            0.0
            < self.blocket_precontact_gap_min
            <= self.blocket_precontact_gap_max
            < 0.20
        ):
            raise ValueError("invalid Blocket pre-contact gap interval")
        if (
            self.blocket_approach_hold_min < 1
            or self.blocket_approach_hold_max < self.blocket_approach_hold_min
        ):
            raise ValueError("invalid Blocket approach hold interval")
        if not (
            0.0
            < self.blocket_approach_effort_min
            <= self.blocket_approach_effort_max
            <= 1.0
        ):
            raise ValueError("invalid Blocket approach effort interval")


def hidden_excitation_config_sha256(config: HiddenExcitationConfig) -> str:
    encoded = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExperimentFConfig:
    seed: int = 151_910_737
    fit_trajectories: int = 4_096
    validation_trajectories: int = 512
    test_trajectories: int = 512
    history_frames: int = 8
    transitions: int = 8
    cache_frames: int = 24
    image_size: int = 64
    patch_size: int = 4
    backbone_preset: str = "tiny"
    variants: tuple[Variant, ...] = REGISTERED_VARIANTS

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("experiment seed must be an integer")
        for name in (
            "fit_trajectories",
            "validation_trajectories",
            "test_trajectories",
            "history_frames",
            "transitions",
            "cache_frames",
            "image_size",
            "patch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"experiment {name} must be a positive integer")
        if self.cache_frames < self.history_frames + self.transitions:
            raise ValueError("cache_frames must cover history_frames + transitions")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.backbone_preset not in {"nano", "micro", "tiny", "small"}:
            raise ValueError("unknown registered backbone preset")
        if (
            type(self.variants) is not tuple
            or not self.variants
            or len(set(self.variants)) != len(self.variants)
            or any(variant not in REGISTERED_VARIANTS for variant in self.variants)
        ):
            raise ValueError("experiment variants must be a unique registered tuple")


__all__ = [
    "ExperimentFConfig",
    "HiddenExcitationConfig",
    "REGISTERED_SYSTEMS",
    "REGISTERED_VARIANTS",
    "Variant",
    "hidden_excitation_config_sha256",
]
