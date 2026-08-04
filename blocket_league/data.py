from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .env import BlocketLeagueEnv, WorldConfig


STATE_NAMES = (
    "player_x",
    "player_y",
    "player_vx",
    "player_vy",
    "puck_x",
    "puck_y",
    "puck_vx",
    "puck_vy",
    "score",
    "reset_timer",
)


def passive_kickoff_state(score: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic moving kickoff selected by the visible score."""

    phase = score % 5
    angle = 0.35 + phase * (2.0 * np.pi / 5.0)
    player_position = np.asarray((0.27, 0.34 + 0.08 * phase), dtype=np.float32)
    puck_position = np.asarray((0.58, 0.66 - 0.08 * phase), dtype=np.float32)
    player_velocity = 0.48 * np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float32)
    puck_angle = angle + np.pi + 0.42
    puck_velocity = 0.38 * np.asarray((np.cos(puck_angle), np.sin(puck_angle)), dtype=np.float32)
    return player_position, player_velocity, puck_position, puck_velocity


def make_clip(
    seed: int,
    *,
    context_frames: int = 4,
    future_frames: int = 8,
    image_size: int = 64,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    env = BlocketLeagueEnv(seed=seed, config=WorldConfig(image_size=image_size))

    previous_action = 0
    for _ in range(int(rng.integers(10, 80))):
        if rng.random() < 0.34:
            previous_action = env.policy_action(exploration=0.18)
        env.step(previous_action)

    frame_count = context_frames + future_frames
    frames: list[np.ndarray] = [env.render()]
    states: list[np.ndarray] = [env.state.vector()]
    actions: list[int] = []
    events: list[int] = []
    event_ids = {"coast": 0, "thrust": 1, "impact": 2, "wall": 3, "goal": 4, "kickoff": 5}

    for _ in range(frame_count - 1):
        if rng.random() < 0.31:
            previous_action = env.policy_action(exploration=0.16)
        env.step(previous_action)
        actions.append(previous_action)
        frames.append(env.render())
        states.append(env.state.vector())
        events.append(event_ids.get(env.state.last_event, 0))

    frame_array = np.stack(frames)
    state_array = np.stack(states)
    action_array = np.asarray(actions[context_frames - 1 :], dtype=np.int64)
    event_array = np.asarray(events[context_frames - 1 :], dtype=np.int64)
    return {
        "frames": frame_array,
        "context": frame_array[:context_frames],
        "target": frame_array[context_frames:],
        "actions": action_array,
        "state": state_array[context_frames:],
        "events": event_array,
        "all_state": state_array,
        "all_events": np.asarray([0, *events], dtype=np.int64),
        "all_actions": np.asarray([0, *actions], dtype=np.int64),
    }


def make_passive_clip(
    seed: int,
    *,
    context_frames: int = 8,
    future_frames: int = 64,
    image_size: int = 64,
    goal_centered: bool = False,
    puck_angle_center_degrees: float | None = None,
    puck_angle_width_degrees: float = 0.0,
) -> dict[str, np.ndarray]:
    """Render collision-rich physics without an action or control channel.

    Both discs receive randomized initial momentum and then coast.  The larger
    disc uses the same low drag as the puck so its motion remains an autonomous
    part of the physical state rather than an unobserved player input.
    """

    rng = np.random.default_rng(seed)
    config = WorldConfig(
        image_size=image_size,
        player_acceleration=0.0,
        player_drag=0.12,
        puck_drag=0.12,
    )
    env = BlocketLeagueEnv(seed=seed, config=config)
    def random_velocity(low: float = 0.22, high: float = 0.72) -> np.ndarray:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        speed = rng.uniform(low, high)
        return (speed * np.asarray((np.cos(angle), np.sin(angle)))).astype(np.float32)

    def sampled_puck_velocity(low: float = 0.22, high: float = 0.72) -> np.ndarray:
        if puck_angle_center_degrees is None:
            return random_velocity(low, high)
        half_width = np.deg2rad(puck_angle_width_degrees) / 2.0
        angle = np.deg2rad(puck_angle_center_degrees) + rng.uniform(-half_width, half_width)
        speed = rng.uniform(low, high)
        return (speed * np.asarray((np.cos(angle), np.sin(angle)))).astype(np.float32)

    def initialize_round(*, force_goal: bool = False, kickoff: bool = False) -> None:
        state = env.state
        if kickoff:
            (
                state.player_position,
                state.player_velocity,
                state.puck_position,
                state.puck_velocity,
            ) = passive_kickoff_state(state.score)
            return
        if force_goal:
            # Put the puck a few frames from scoring so a short clip contains
            # the complete goal -> pause -> moving kickoff transition. Vary the
            # existing score so repeated browser goals do not create a novel
            # scoreboard pattern.
            state.score = int(rng.integers(0, 5))
            state.player_position = rng.uniform((0.14, 0.14), (0.58, 0.86)).astype(np.float32)
            state.puck_position = np.asarray(
                (
                    rng.uniform(0.80, 0.86),
                    rng.uniform(config.goal_low + 0.055, config.goal_high - 0.055),
                ),
                dtype=np.float32,
            )
            state.player_velocity = random_velocity(0.22, 0.62)
            state.puck_velocity = np.asarray(
                (rng.uniform(0.56, 0.76), rng.uniform(-0.035, 0.035)),
                dtype=np.float32,
            )
            return

        # Rejection sample two separated discs. A substantial fraction are
        # aimed toward one another so collisions are common without controls.
        for _ in range(100):
            player = rng.uniform((0.14, 0.14), (0.72, 0.86)).astype(np.float32)
            puck = rng.uniform((0.28, 0.14), (0.86, 0.86)).astype(np.float32)
            if np.linalg.norm(player - puck) > config.player_radius + config.puck_radius + 0.08:
                break
        state.player_position = player
        state.puck_position = puck
        if rng.random() < 0.6:
            axis = puck - player
            axis /= max(float(np.linalg.norm(axis)), 1e-6)
            tangent = np.asarray((-axis[1], axis[0]), dtype=np.float32)
            state.player_velocity = axis * rng.uniform(0.28, 0.68) + tangent * rng.uniform(-0.12, 0.12)
            state.puck_velocity = -axis * rng.uniform(0.18, 0.58) + tangent * rng.uniform(-0.12, 0.12)
            if puck_angle_center_degrees is not None:
                state.puck_velocity = sampled_puck_velocity(0.18, 0.58)
        else:
            state.player_velocity = random_velocity()
            state.puck_velocity = sampled_puck_velocity()
        state.player_velocity = state.player_velocity.astype(np.float32)
        state.puck_velocity = state.puck_velocity.astype(np.float32)

    initialize_round(force_goal=goal_centered)

    frame_count = context_frames + future_frames
    frames: list[np.ndarray] = [env.render()]
    states: list[np.ndarray] = [env.state.vector()]
    events: list[int] = []
    event_ids = {"coast": 0, "impact": 2, "wall": 3, "goal": 4, "kickoff": 5}
    for _ in range(frame_count - 1):
        env.step(0)
        if env.state.last_event == "kickoff":
            # BlocketLeagueEnv is shared with the action-conditioned world and
            # normally restarts almost stationary. Passive clips must restart
            # from the same moving-state distribution used at clip boundaries.
            initialize_round(kickoff=True)
        frames.append(env.render())
        states.append(env.state.vector())
        events.append(event_ids.get(env.state.last_event, 0))

    frame_array = np.stack(frames)
    state_array = np.stack(states)
    return {
        "frames": frame_array,
        "context": frame_array[:context_frames],
        "target": frame_array[context_frames:],
        "all_state": state_array,
        "all_events": np.asarray([0, *events], dtype=np.int64),
        "state": state_array[context_frames:],
        "events": np.asarray(events[context_frames - 1 :], dtype=np.int64),
    }


class PassiveClipDataset:
    """Deterministic autonomous-physics videos with no action field."""

    def __init__(
        self,
        samples: int,
        *,
        seed: int = 0,
        frames: int = 24,
        image_size: int = 64,
        goal_centered_fraction: float = 0.0,
        excluded_puck_angle_center_degrees: float | None = None,
        excluded_puck_angle_width_degrees: float = 0.0,
        excluded_collision_quadrant: str | None = None,
    ):
        if not 0.0 <= goal_centered_fraction <= 1.0:
            raise ValueError("goal_centered_fraction must be in [0, 1]")
        if excluded_puck_angle_width_degrees < 0.0 or excluded_puck_angle_width_degrees >= 360.0:
            raise ValueError("excluded_puck_angle_width_degrees must be in [0, 360)")
        if excluded_puck_angle_width_degrees > 0.0 and goal_centered_fraction > 0.0:
            raise ValueError("direction-holdout caches cannot include goal-centered clips")
        if excluded_collision_quadrant not in {
            None,
            "upper-left",
            "upper-right",
            "lower-left",
            "lower-right",
        }:
            raise ValueError("unknown excluded_collision_quadrant")
        self.samples = samples
        self.seed = seed
        self.frames = frames
        self.image_size = image_size
        self.goal_centered_fraction = goal_centered_fraction
        self.excluded_puck_angle_center_degrees = excluded_puck_angle_center_degrees
        self.excluded_puck_angle_width_degrees = excluded_puck_angle_width_degrees
        self.excluded_collision_quadrant = excluded_collision_quadrant

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> object:
        import torch

        base_seed = self.seed + index * 9_973
        clip = None
        for attempt in range(128):
            clip_seed = base_seed + attempt * 1_000_003
            curriculum_rng = np.random.default_rng(clip_seed ^ 0x5A17)
            candidate = make_passive_clip(
                clip_seed,
                context_frames=1,
                future_frames=self.frames - 1,
                image_size=self.image_size,
                goal_centered=curriculum_rng.random() < self.goal_centered_fraction,
            )
            accepted = True
            if self.excluded_puck_angle_center_degrees is not None:
                velocities = candidate["all_state"][:, 6:8]
                speed = np.linalg.norm(velocities, axis=1)
                angle = np.rad2deg(np.arctan2(velocities[:, 1], velocities[:, 0]))
                delta = (
                    angle - self.excluded_puck_angle_center_degrees + 180.0
                ) % 360.0 - 180.0
                moving_in_wedge = (speed > 0.05) & (
                    np.abs(delta) < self.excluded_puck_angle_width_degrees / 2.0
                )
                accepted = accepted and not np.any(moving_in_wedge)
            if self.excluded_collision_quadrant is not None:
                states = candidate["all_state"]
                midpoints = (states[:, :2] + states[:, 4:6]) / 2.0
                in_quadrant = points_in_quadrant(
                    midpoints,
                    self.excluded_collision_quadrant,
                )
                excluded_impacts = (candidate["all_events"] == 2) & in_quadrant
                accepted = accepted and not np.any(excluded_impacts)
            if accepted:
                clip = candidate
                break
        if clip is None:
            raise RuntimeError(f"could not sample an accepted passive clip for index {index}")
        video = torch.from_numpy(clip["frames"].copy()).permute(0, 3, 1, 2).float()
        return video.div(127.5).sub(1.0)


def points_in_quadrant(points: np.ndarray, quadrant: str) -> np.ndarray:
    """Return a mask for arena points; rendered y coordinates increase downward."""
    vertical, horizontal = quadrant.split("-")
    x_matches = points[..., 0] < 0.5 if horizontal == "left" else points[..., 0] >= 0.5
    y_matches = points[..., 1] < 0.5 if vertical == "upper" else points[..., 1] >= 0.5
    return x_matches & y_matches


class ClipDataset:
    """A deterministic map-style PyTorch dataset with no stored video corpus."""

    def __init__(
        self,
        samples: int,
        *,
        seed: int = 0,
        context_frames: int = 4,
        future_frames: int = 8,
        image_size: int = 64,
    ) -> None:
        self.samples = samples
        self.seed = seed
        self.context_frames = context_frames
        self.future_frames = future_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        clip = make_clip(
            self.seed + index * 9_973,
            context_frames=self.context_frames,
            future_frames=self.future_frames,
            image_size=self.image_size,
        )

        def video_tensor(value: np.ndarray) -> object:
            tensor = torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float()
            return tensor.div(127.5).sub(1.0)

        return {
            "context": video_tensor(clip["context"]),
            "target": video_tensor(clip["target"]),
            "actions": torch.from_numpy(clip["actions"].copy()).long(),
            "state": torch.from_numpy(clip["state"].copy()).float(),
            "events": torch.from_numpy(clip["events"].copy()).long(),
        }


def export_dataset(
    output: Path,
    *,
    samples: int,
    seed: int,
    context_frames: int,
    future_frames: int,
    image_size: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": np.asarray(1),
        "seed": np.asarray(seed),
        "context_frames": np.asarray(context_frames),
        "future_frames": np.asarray(future_frames),
        "image_size": np.asarray(image_size),
        "state_names": np.asarray(STATE_NAMES),
    }
    np.savez_compressed(output / "manifest.npz", **manifest)
    for index in range(samples):
        clip = make_clip(
            seed + index * 9_973,
            context_frames=context_frames,
            future_frames=future_frames,
            image_size=image_size,
        )
        np.savez_compressed(output / f"clip-{index:06d}.npz", **clip)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic Blocket League clips")
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--future-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()
    export_dataset(
        args.output,
        samples=args.samples,
        seed=args.seed,
        context_frames=args.context_frames,
        future_frames=args.future_frames,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
