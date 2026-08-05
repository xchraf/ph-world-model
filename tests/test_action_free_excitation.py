from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from blocket_league.action_free_excitation import (
    HiddenExcitationConfig,
    assert_pixels_only_payload,
    experiment_f_blocket_world_config,
    make_action_free_video,
    pixels_only_sha256,
)
from blocket_league.direct_physical_evaluation import make_builtin_probe_candidates
from blocket_league.env import BlocketLeagueEnv, PALETTE, WorldConfig


class ActionFreeExcitationTests(unittest.TestCase):
    def test_both_systems_cross_the_firewall_as_pixels_only(self) -> None:
        config = HiddenExcitationConfig(frames=7, image_size=24)
        for system in ("pendulum", "blocket"):
            with self.subTest(system=system):
                payload = make_action_free_video(system, 17, config=config)
                self.assertEqual(set(payload), {"frames"})
                self.assertEqual(payload["frames"].shape, (7, 24, 24, 3))
                self.assertEqual(payload["frames"].dtype, np.uint8)
                assert_pixels_only_payload(payload)

    def test_generation_is_deterministic_and_hashed_after_deletion(self) -> None:
        config = HiddenExcitationConfig(frames=6, image_size=20)
        first = make_action_free_video("pendulum", 91, config=config)
        second = make_action_free_video("pendulum", 91, config=config)
        self.assertEqual(pixels_only_sha256(first), pixels_only_sha256(second))
        np.testing.assert_array_equal(first["frames"], second["frames"])

    def test_blocket_pixels_are_contact_rich_without_exporting_contact_labels(self) -> None:
        config = HiddenExcitationConfig(frames=24, image_size=64)
        player_colours = np.stack((PALETTE["player"], PALETTE["player_core"]))
        puck_colours = np.stack((PALETTE["puck"], PALETTE["puck_core"]))

        def centroid(frame: np.ndarray, colours: np.ndarray) -> np.ndarray:
            mask = (frame[..., None, :] == colours).all(axis=-1).any(axis=-1)
            coordinates = np.argwhere(mask)
            self.assertGreater(len(coordinates), 0)
            return coordinates.mean(axis=0)[::-1] / config.image_size

        visually_contacting = 0
        trajectory_count = 128
        for index in range(trajectory_count):
            payload = make_action_free_video(
                "blocket", 151_910_737 + index * 104_729, config=config
            )
            self.assertEqual(set(payload), {"frames"})
            minimum_distance = min(
                float(
                    np.linalg.norm(
                        centroid(frame, player_colours)
                        - centroid(frame, puck_colours)
                    )
                )
                for frame in payload["frames"]
            )
            # Pixel centroids are quantized; this tolerance is below one pixel
            # beyond the exact sum of the two registered disc radii.
            visually_contacting += int(
                minimum_distance
                <= WorldConfig().player_radius
                + WorldConfig().puck_radius
                + 1.0 / config.image_size
            )
        self.assertGreaterEqual(visually_contacting, 64)

    def test_firewall_rejects_extra_metadata(self) -> None:
        frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            assert_pixels_only_payload({"frames": frames, "state": np.zeros((4, 2))})
        with self.assertRaises(AssertionError):
            assert_pixels_only_payload({"frames": frames, "actions": np.zeros(3)})

    def test_continuous_blocket_arena_cannot_score_pause_or_reset(self) -> None:
        config = experiment_f_blocket_world_config(image_size=24)
        self.assertFalse(config.goals_enabled)
        environment = BlocketLeagueEnv(seed=7, config=config)
        original_state = environment.state
        environment.state.score = 0
        environment.state.reset_timer = 0
        environment.state.player_position[:] = (0.20, 0.20)
        environment.state.player_velocity[:] = 0.0
        environment.state.puck_position[:] = (
            1.0 - config.wall - config.puck_radius - 1e-4,
            0.5,
        )
        environment.state.puck_velocity[:] = (1.0, 0.0)
        for expected_tick in range(1, 13):
            environment.step_vector(np.zeros(2, dtype=np.float32))
            self.assertIs(environment.state, original_state)
            self.assertEqual(environment.state.tick, expected_tick)
            self.assertEqual(environment.state.score, 0)
            self.assertEqual(environment.state.reset_timer, 0)
            self.assertNotIn(environment.state.last_event, {"goal", "kickoff"})

    def test_generation_and_physical_evaluation_share_exact_blocket_config(self) -> None:
        generated_configs = []

        def capture_config(*, image_size: int):
            result = experiment_f_blocket_world_config(image_size=image_size)
            generated_configs.append(result)
            return result

        with patch(
            "blocket_league.action_free_excitation.experiment_f_blocket_world_config",
            side_effect=capture_config,
        ):
            make_action_free_video(
                "blocket", 23, config=HiddenExcitationConfig(frames=4, image_size=64)
            )
        self.assertEqual(len(generated_configs), 1)

        candidate = make_builtin_probe_candidates(
            "blocket", history_frames=1, count=1, seed=29
        )[0]
        evaluation_config = candidate.environment.config
        self.assertEqual(evaluation_config, generated_configs[0])
        self.assertEqual(evaluation_config, experiment_f_blocket_world_config())


if __name__ == "__main__":
    unittest.main()
