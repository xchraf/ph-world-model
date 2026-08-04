from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi

import numpy as np

from .env import PALETTE


@dataclass(frozen=True)
class PendulumConfig:
    image_size: int = 64
    fps: int = 20
    substeps: int = 4
    gravity: float = 9.81
    length: float = 0.31
    mass: float = 1.0
    damping: float = 0.22
    max_torque: float = 4.0

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def inertia(self) -> float:
        return self.mass * self.length**2


@dataclass
class PendulumState:
    angle: float = 0.0
    angular_velocity: float = 0.0
    tick: int = 0

    def vector(self) -> np.ndarray:
        return np.asarray((self.angle, self.angular_velocity), dtype=np.float32)


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + pi) % (2.0 * pi) - pi


class PendulumEnv:
    """A rendered damped pendulum with a bounded continuous torque port."""

    def __init__(self, seed: int = 0, config: PendulumConfig | None = None) -> None:
        self.config = config or PendulumConfig()
        self.rng = np.random.default_rng(seed)
        self.state = PendulumState()
        axis = (np.arange(self.config.image_size, dtype=np.float32) + 0.5) / self.config.image_size
        self._grid = np.meshgrid(axis, axis)
        self.reset()

    def reset(self) -> PendulumState:
        self.state = PendulumState(
            angle=float(self.rng.uniform(-pi, pi)),
            angular_velocity=float(self.rng.uniform(-1.6, 1.6)),
        )
        return self.state

    def set_state(self, state: PendulumState) -> None:
        self.state = PendulumState(
            angle=float(state.angle),
            angular_velocity=float(state.angular_velocity),
            tick=int(state.tick),
        )

    def step(self, torque: float = 0.0) -> PendulumState:
        config = self.config
        torque = float(np.clip(torque, -config.max_torque, config.max_torque))
        h = config.dt / config.substeps
        for _ in range(config.substeps):
            angular_acceleration = (
                -config.mass * config.gravity * config.length * np.sin(self.state.angle)
                - config.damping * self.state.angular_velocity
                + torque
            ) / config.inertia
            self.state.angular_velocity += float(angular_acceleration * h)
            self.state.angle = float(wrap_angle(self.state.angle + self.state.angular_velocity * h))
        self.state.tick += 1
        return self.state

    def energy(self) -> float:
        config = self.config
        return float(
            0.5 * config.inertia * self.state.angular_velocity**2
            + config.mass * config.gravity * config.length * (1.0 - np.cos(self.state.angle))
        )

    def render(self) -> np.ndarray:
        x, y = self._grid
        pivot = np.asarray((0.5, 0.43), dtype=np.float32)
        bob = pivot + self.config.length * np.asarray(
            (np.sin(self.state.angle), np.cos(self.state.angle)), dtype=np.float32
        )
        image = np.empty((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
        image[:] = PALETTE["field"]

        segment = bob - pivot
        denominator = max(float(np.dot(segment, segment)), 1e-8)
        projection = ((x - pivot[0]) * segment[0] + (y - pivot[1]) * segment[1]) / denominator
        projection = np.clip(projection, 0.0, 1.0)
        closest_x = pivot[0] + projection * segment[0]
        closest_y = pivot[1] + projection * segment[1]
        rod = (x - closest_x) ** 2 + (y - closest_y) ** 2 <= 0.010**2
        image[rod] = PALETTE["line"]

        pivot_mask = (x - pivot[0]) ** 2 + (y - pivot[1]) ** 2 <= 0.040**2
        pivot_core = (x - pivot[0]) ** 2 + (y - pivot[1]) ** 2 <= 0.017**2
        image[pivot_mask] = PALETTE["player"]
        image[pivot_core] = PALETTE["player_core"]

        bob_mask = (x - bob[0]) ** 2 + (y - bob[1]) ** 2 <= 0.048**2
        bob_core = (x - bob[0]) ** 2 + (y - bob[1]) ** 2 <= 0.020**2
        image[bob_mask] = PALETTE["puck"]
        image[bob_core] = PALETTE["puck_core"]
        return image


def make_passive_pendulum_clip(
    seed: int,
    *,
    context_frames: int = 8,
    future_frames: int = 16,
    image_size: int = 64,
) -> dict[str, np.ndarray]:
    """Return zero-torque pixels only; the payload deliberately has no actions."""

    env = PendulumEnv(seed=seed, config=PendulumConfig(image_size=image_size))
    frames = [env.render()]
    for _ in range(context_frames + future_frames - 1):
        env.step(0.0)
        frames.append(env.render())
    array = np.stack(frames)
    return {
        "frames": array,
        "context": array[:context_frames],
        "target": array[context_frames:],
    }


def pendulum_target_frames(
    angle: float,
    *,
    frames: int,
    image_size: int = 64,
) -> np.ndarray:
    env = PendulumEnv(seed=0, config=PendulumConfig(image_size=image_size))
    env.set_state(PendulumState(angle=float(wrap_angle(angle)), angular_velocity=0.0))
    rendered = env.render()
    return np.repeat(rendered[None], frames, axis=0)

