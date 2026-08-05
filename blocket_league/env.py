from __future__ import annotations

from dataclasses import dataclass, field
from math import exp

import numpy as np

from .pixel_palette import PALETTE


ACTION_VECTORS = np.asarray(
    [
        (0.0, 0.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
    ],
    dtype=np.float32,
)
ACTION_NAMES = (
    "coast",
    "up",
    "up-right",
    "right",
    "down-right",
    "down",
    "down-left",
    "left",
    "up-left",
)

@dataclass(frozen=True)
class WorldConfig:
    image_size: int = 64
    fps: int = 20
    substeps: int = 4
    wall: float = 0.045
    player_radius: float = 0.064
    puck_radius: float = 0.043
    player_mass: float = 1.8
    puck_mass: float = 1.0
    player_acceleration: float = 3.35
    player_drag: float = 1.55
    puck_drag: float = 0.12
    max_player_speed: float = 1.25
    restitution: float = 0.91
    goal_low: float = 0.35
    goal_high: float = 0.65
    goal_pause_steps: int = 7
    goals_enabled: bool = True

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


@dataclass
class WorldState:
    player_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    player_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    puck_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    puck_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    score: int = 0
    tick: int = 0
    reset_timer: int = 0
    last_event: str = "kickoff"

    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                *self.player_position,
                *self.player_velocity,
                *self.puck_position,
                *self.puck_velocity,
                float(self.score),
                float(self.reset_timer),
            ],
            dtype=np.float32,
        )


class BlocketLeagueEnv:
    """Deterministic two-disc physics with a single goal on the right wall."""

    def __init__(self, seed: int = 0, config: WorldConfig | None = None) -> None:
        self.config = config or WorldConfig()
        self.rng = np.random.default_rng(seed)
        self.state = WorldState()
        self._grid = self._make_grid(self.config.image_size)
        self.reset(full=True)

    @staticmethod
    def _make_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
        axis = (np.arange(size, dtype=np.float32) + 0.5) / size
        return np.meshgrid(axis, axis)

    def reset(self, *, full: bool = False) -> WorldState:
        score = 0 if full else self.state.score
        tick = 0 if full else self.state.tick
        self.state = WorldState(
            player_position=np.asarray(
                (0.27 + self.rng.uniform(-0.025, 0.025), 0.5 + self.rng.uniform(-0.16, 0.16)),
                dtype=np.float32,
            ),
            player_velocity=self.rng.uniform(-0.06, 0.06, size=2).astype(np.float32),
            puck_position=np.asarray(
                (0.57 + self.rng.uniform(-0.045, 0.045), 0.5 + self.rng.uniform(-0.19, 0.19)),
                dtype=np.float32,
            ),
            puck_velocity=self.rng.uniform(-0.08, 0.08, size=2).astype(np.float32),
            score=score,
            tick=tick,
            last_event="kickoff",
        )
        return self.state

    def step(self, action: int) -> WorldState:
        if not 0 <= action < len(ACTION_VECTORS):
            raise ValueError(f"Action must be in [0, 8], got {action}")

        direction = ACTION_VECTORS[action].copy()
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            direction /= norm
        return self._step_direction(direction)

    def step_vector(self, direction: np.ndarray) -> WorldState:
        """Advance under a bounded continuous thrust vector.

        The original categorical action API remains unchanged.  This separate
        deployment-only interface lets experiments hide a new linear actuator
        coordinate system without changing the autonomous passive-video data.
        """

        direction = np.asarray(direction, dtype=np.float32)
        if direction.shape != (2,) or not bool(np.isfinite(direction).all()):
            raise ValueError("direction must be a finite two-dimensional vector")
        norm = float(np.linalg.norm(direction))
        if norm > 1.0:
            direction = direction / norm
        return self._step_direction(direction)

    def _step_direction(self, direction: np.ndarray) -> WorldState:
        direction = np.asarray(direction, dtype=np.float32)

        state = self.state
        state.tick += 1
        state.last_event = "coast"
        if state.reset_timer > 0:
            state.reset_timer -= 1
            state.last_event = "goal"
            if state.reset_timer == 0:
                self.reset(full=False)
            return self.state

        norm = float(np.linalg.norm(direction))
        if norm > 0:
            state.last_event = "thrust"

        h = self.config.dt / self.config.substeps
        for _ in range(self.config.substeps):
            state.player_velocity += direction * self.config.player_acceleration * h
            state.player_velocity *= exp(-self.config.player_drag * h)
            state.puck_velocity *= exp(-self.config.puck_drag * h)
            self._cap_player_speed()

            state.player_position += state.player_velocity * h
            state.puck_position += state.puck_velocity * h

            self._collide_discs()
            self._collide_player_walls()
            if self.config.goals_enabled and self._puck_scored():
                state.score += 1
                state.reset_timer = self.config.goal_pause_steps
                state.last_event = "goal"
                break
            self._collide_puck_walls()
        return state

    def _cap_player_speed(self) -> None:
        velocity = self.state.player_velocity
        speed = float(np.linalg.norm(velocity))
        if speed > self.config.max_player_speed:
            velocity *= self.config.max_player_speed / speed

    def _collide_discs(self) -> None:
        state = self.state
        delta = state.puck_position - state.player_position
        distance = float(np.linalg.norm(delta))
        minimum = self.config.player_radius + self.config.puck_radius
        if distance >= minimum:
            return

        if distance < 1e-7:
            normal = np.asarray((1.0, 0.0), dtype=np.float32)
            distance = 1e-7
        else:
            normal = delta / distance

        inverse_player = 1.0 / self.config.player_mass
        inverse_puck = 1.0 / self.config.puck_mass
        overlap = minimum - distance
        correction = normal * overlap / (inverse_player + inverse_puck)
        state.player_position -= correction * inverse_player
        state.puck_position += correction * inverse_puck

        closing_speed = float(np.dot(state.puck_velocity - state.player_velocity, normal))
        if closing_speed < 0:
            impulse = -(1.0 + self.config.restitution) * closing_speed / (
                inverse_player + inverse_puck
            )
            state.player_velocity -= normal * impulse * inverse_player
            state.puck_velocity += normal * impulse * inverse_puck
            state.last_event = "impact"

    def _bounce_axis(self, position: np.ndarray, velocity: np.ndarray, axis: int, radius: float) -> None:
        low = self.config.wall + radius
        high = 1.0 - self.config.wall - radius
        if position[axis] < low:
            position[axis] = low
            velocity[axis] = abs(velocity[axis]) * self.config.restitution
            self.state.last_event = "wall"
        elif position[axis] > high:
            position[axis] = high
            velocity[axis] = -abs(velocity[axis]) * self.config.restitution
            self.state.last_event = "wall"

    def _collide_player_walls(self) -> None:
        self._bounce_axis(
            self.state.player_position,
            self.state.player_velocity,
            0,
            self.config.player_radius,
        )
        self._bounce_axis(
            self.state.player_position,
            self.state.player_velocity,
            1,
            self.config.player_radius,
        )

    def _puck_scored(self) -> bool:
        state = self.state
        inside_mouth = self.config.goal_low <= state.puck_position[1] <= self.config.goal_high
        return bool(
            inside_mouth
            and state.puck_position[0] + self.config.puck_radius >= 1.0 - self.config.wall
            and state.puck_velocity[0] > 0
        )

    def _collide_puck_walls(self) -> None:
        state = self.state
        self._bounce_axis(state.puck_position, state.puck_velocity, 1, self.config.puck_radius)
        low = self.config.wall + self.config.puck_radius
        right = 1.0 - self.config.wall - self.config.puck_radius
        if state.puck_position[0] < low:
            state.puck_position[0] = low
            state.puck_velocity[0] = abs(state.puck_velocity[0]) * self.config.restitution
            state.last_event = "wall"
        elif state.puck_position[0] > right:
            state.puck_position[0] = right
            state.puck_velocity[0] = -abs(state.puck_velocity[0]) * self.config.restitution
            state.last_event = "wall"

    def policy_action(self, exploration: float = 0.12) -> int:
        """A noisy hand-built striker used to produce collision-rich data."""

        state = self.state
        if state.reset_timer > 0:
            return 0
        if self.rng.random() < exploration:
            return int(self.rng.integers(0, len(ACTION_VECTORS)))

        goal = np.asarray((1.0, 0.5), dtype=np.float32)
        puck_to_goal = goal - state.puck_position
        puck_to_goal /= max(float(np.linalg.norm(puck_to_goal)), 1e-6)
        staging = state.puck_position - puck_to_goal * 0.115
        if np.linalg.norm(state.player_position - state.puck_position) < 0.13:
            staging = goal
        delta = staging - state.player_position
        sx = 0 if abs(float(delta[0])) < 0.022 else (1 if delta[0] > 0 else -1)
        sy = 0 if abs(float(delta[1])) < 0.022 else (1 if delta[1] > 0 else -1)
        for index, vector in enumerate(ACTION_VECTORS):
            if int(vector[0]) == sx and int(vector[1]) == sy:
                return index
        return 0

    def render(self) -> np.ndarray:
        """Render an HWC uint8 frame using a deliberately small, exact palette."""

        config = self.config
        xx, yy = self._grid
        image = np.empty((config.image_size, config.image_size, 3), dtype=np.uint8)
        image[:] = PALETTE["outside"]
        inside = (xx >= config.wall) & (xx <= 1 - config.wall) & (yy >= config.wall) & (yy <= 1 - config.wall)
        image[inside] = PALETTE["field"]

        line_width = 0.008
        center_line = inside & (np.abs(xx - 0.5) < line_width)
        center_ring = inside & (np.abs(np.hypot(xx - 0.5, yy - 0.5) - 0.14) < line_width)
        image[center_line | center_ring] = PALETTE["line"]

        wall_mask = (
            (np.abs(xx - config.wall) < 0.012)
            | (np.abs(xx - (1 - config.wall)) < 0.012)
            | (np.abs(yy - config.wall) < 0.012)
            | (np.abs(yy - (1 - config.wall)) < 0.012)
        )
        goal_mouth = (xx > 1 - config.wall - 0.018) & (yy >= config.goal_low) & (yy <= config.goal_high)
        image[wall_mask & ~goal_mouth] = PALETTE["wall"]
        image[goal_mouth] = PALETTE["goal"]

        for index in range(self.state.score % 5):
            cx = 0.455 + index * 0.022
            pip = (xx - cx) ** 2 + (yy - 0.023) ** 2 <= 0.0065**2
            image[pip] = PALETTE["goal"]

        self._draw_disc(image, self.state.player_position, config.player_radius, "player", "player_core")
        self._draw_disc(image, self.state.puck_position, config.puck_radius, "puck", "puck_core")
        return image

    def _draw_disc(
        self,
        image: np.ndarray,
        position: np.ndarray,
        radius: float,
        fill: str,
        core: str,
    ) -> None:
        xx, yy = self._grid
        squared = (xx - position[0]) ** 2 + (yy - position[1]) ** 2
        image[squared <= radius**2] = PALETTE[fill]
        image[squared <= (radius * 0.31) ** 2] = PALETTE[core]
