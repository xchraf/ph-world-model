from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import stat

import numpy as np

from .env import BlocketLeagueEnv, WorldConfig
from .experiment_f_contract import (
    HiddenExcitationConfig,
    hidden_excitation_config_sha256,
)
from .passive_control_systems import PendulumConfig, PendulumEnv


FORBIDDEN_OPTIMIZATION_WORDS = (
    "action",
    "control",
    "force",
    "input",
    "state",
    "torque",
    "event",
)


def experiment_f_blocket_world_config(*, image_size: int = 64) -> WorldConfig:
    """Return the one sealed Blocket arena used throughout Experiment F.

    Disabling goals turns the goal mouth into the ordinary right wall.  Thus a
    trajectory can contain impacts but can never enter a goal pause or replace
    its physical state with a kickoff.  All other physics, notably the default
    player/puck drags, are shared unchanged by video generation and evaluation.
    """

    return WorldConfig(image_size=image_size, goals_enabled=False)


def action_free_environment_config_sha256(system: str, *, image_size: int) -> str:
    """Seal generator physics without exposing it on the trainer mount."""

    if system == "blocket":
        config: object = experiment_f_blocket_world_config(image_size=image_size)
    elif system == "pendulum":
        config = PendulumConfig(image_size=image_size)
    else:
        raise ValueError(f"unknown system {system!r}")
    encoded = json.dumps(
        {
            "type": f"{type(config).__module__}.{type(config).__qualname__}",
            "config": asdict(config),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def private_producer_seed_from_file(path: Path, *, system: str) -> int:
    """Derive a system seed from owner-private 128-bit launch entropy."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("producer seed source must be a nonsymbolic regular file")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077 or not mode & 0o400:
        raise PermissionError("producer seed source must be owner-private and readable")
    encoded = path.read_text(encoding="ascii").strip()
    if len(encoded) != 32 or any(
        character not in "0123456789abcdef" for character in encoded
    ):
        raise ValueError("producer seed source must contain exactly 128-bit lowercase hex")
    digest = hashlib.sha256(
        bytes.fromhex(encoded) + b"\0" + system.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def assert_pixels_only_payload(payload: Mapping[str, object]) -> None:
    """Reject any payload capable of leaking simulator-side supervision."""

    if set(payload) != {"frames"}:
        raise AssertionError(
            "the optimization payload must contain exactly the single key 'frames'"
        )
    for key in payload:
        lowered = key.lower()
        if any(word in lowered for word in FORBIDDEN_OPTIMIZATION_WORDS):
            raise AssertionError(f"forbidden optimization field: {key!r}")
    frames = payload["frames"]
    if not isinstance(frames, np.ndarray):
        raise AssertionError("frames must be a numpy array")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise AssertionError("frames must have shape (time, height, width, 3)")
    if frames.dtype != np.uint8:
        raise AssertionError("frames must be uint8 so no hidden float channel can leak")


def pixels_only_sha256(payload: Mapping[str, object]) -> str:
    assert_pixels_only_payload(payload)
    frames = payload["frames"]
    assert isinstance(frames, np.ndarray)
    digest = hashlib.sha256()
    digest.update(str(frames.shape).encode("ascii"))
    digest.update(str(frames.dtype).encode("ascii"))
    digest.update(frames.tobytes())
    return digest.hexdigest()


def _piecewise_hidden_efforts(
    rng: np.random.Generator,
    *,
    count: int,
    dimensions: int,
    hold_min: int,
    hold_max: int,
    coast_probability: float,
) -> np.ndarray:
    """Create zero-centred, persistently exciting efforts inside the firewall."""

    efforts: list[np.ndarray] = []
    while len(efforts) < count:
        if rng.random() < coast_probability:
            effort = np.zeros(dimensions, dtype=np.float32)
        else:
            effort = rng.normal(size=dimensions).astype(np.float32)
            norm = max(float(np.linalg.norm(effort)), 1e-7)
            effort *= float(rng.uniform(0.25, 1.0)) / norm
        hold = int(rng.integers(hold_min, hold_max + 1))
        efforts.extend([effort.copy() for _ in range(hold)])

        # Antithetic excitation makes the conditional effort distribution
        # symmetric without exposing any sign convention to the learner.
        if len(efforts) < count and bool(rng.integers(0, 2)):
            hold = int(rng.integers(hold_min, hold_max + 1))
            efforts.extend([-effort.copy() for _ in range(hold)])
    return np.stack(efforts[:count])


def make_action_free_pendulum_video(
    seed: int,
    *,
    config: HiddenExcitationConfig = HiddenExcitationConfig(),
) -> dict[str, np.ndarray]:
    """Render a pendulum under unrecorded excitation and return pixels only."""

    rng = np.random.default_rng(seed ^ 0x50A5501)
    environment = PendulumEnv(
        seed=seed,
        config=PendulumConfig(image_size=config.image_size),
    )
    private_efforts = _piecewise_hidden_efforts(
        rng,
        count=config.frames - 1,
        dimensions=1,
        hold_min=config.hold_min,
        hold_max=config.hold_max,
        coast_probability=config.coast_probability,
    )
    frames = [environment.render()]
    for effort in private_efforts:
        environment.step(float(effort[0] * environment.config.max_torque))
        frames.append(environment.render())

    # Do not add anything to this return value.  In particular, private_efforts
    # and environment.state must cease to exist at this boundary.
    payload = {"frames": np.stack(frames)}
    assert_pixels_only_payload(payload)
    return payload


def make_action_free_blocket_video(
    seed: int,
    *,
    config: HiddenExcitationConfig = HiddenExcitationConfig(),
) -> dict[str, np.ndarray]:
    """Render collision-rich Blocket video under unrecorded continuous thrust."""

    rng = np.random.default_rng(seed ^ 0xB10C0E7)
    environment = BlocketLeagueEnv(
        seed=seed,
        config=experiment_f_blocket_world_config(image_size=config.image_size),
    )
    state = environment.state
    state.reset_timer = 0
    state.score = 0
    contact_rich = bool(rng.random() < config.blocket_contact_probability)
    approach_direction: np.ndarray | None = None
    if contact_rich:
        # The orientation is uniform on the circle, so the private approach
        # distribution has no privileged action sign or axis over the corpus.
        # The geometry merely guarantees that action-free *video* contains a
        # useful number of contact transitions.  Neither this branch bit nor
        # any quantity below crosses the pixels-only producer boundary.
        angle = float(rng.uniform(-np.pi, np.pi))
        approach_direction = np.asarray(
            (np.cos(angle), np.sin(angle)), dtype=np.float32
        )
        puck = rng.uniform((0.31, 0.31), (0.69, 0.69)).astype(np.float32)
        gap = float(
            rng.uniform(
                config.blocket_precontact_gap_min,
                config.blocket_precontact_gap_max,
            )
        )
        separation = environment.config.player_radius + environment.config.puck_radius + gap
        player = puck - approach_direction * separation
        state.player_velocity = rng.uniform(-0.04, 0.04, size=2).astype(np.float32)
        state.puck_velocity = rng.uniform(-0.03, 0.03, size=2).astype(np.float32)
    else:
        for _ in range(128):
            player = rng.uniform((0.18, 0.18), (0.48, 0.82)).astype(np.float32)
            puck = rng.uniform((0.48, 0.18), (0.82, 0.82)).astype(np.float32)
            minimum = (
                environment.config.player_radius
                + environment.config.puck_radius
                + 0.06
            )
            if float(np.linalg.norm(player - puck)) > minimum:
                break
        state.player_velocity = rng.uniform(-0.20, 0.20, size=2).astype(np.float32)
        state.puck_velocity = rng.uniform(-0.16, 0.16, size=2).astype(np.float32)
    state.player_position = player.astype(np.float32)
    state.puck_position = puck.astype(np.float32)

    private_efforts = _piecewise_hidden_efforts(
        rng,
        count=config.frames - 1,
        dimensions=2,
        hold_min=config.hold_min,
        hold_max=config.hold_max,
        coast_probability=config.coast_probability,
    )
    if approach_direction is not None:
        approach_hold = int(
            rng.integers(
                config.blocket_approach_hold_min,
                config.blocket_approach_hold_max + 1,
            )
        )
        approach_hold = min(approach_hold, len(private_efforts))
        approach_magnitude = float(
            rng.uniform(
                config.blocket_approach_effort_min,
                config.blocket_approach_effort_max,
            )
        )
        private_efforts[:approach_hold] = approach_direction * approach_magnitude
    frames = [environment.render()]
    for effort in private_efforts:
        environment.step_vector(effort)
        frames.append(environment.render())
        if environment.state.reset_timer != 0 or environment.state.score != 0:
            raise AssertionError("continuous Experiment-F arena entered goal state")
    payload = {"frames": np.stack(frames)}
    assert_pixels_only_payload(payload)
    return payload


def make_action_free_video(
    system: str,
    seed: int,
    *,
    config: HiddenExcitationConfig = HiddenExcitationConfig(),
) -> dict[str, np.ndarray]:
    if system == "pendulum":
        return make_action_free_pendulum_video(seed, config=config)
    if system == "blocket":
        return make_action_free_blocket_video(seed, config=config)
    raise ValueError(f"unknown system {system!r}")
