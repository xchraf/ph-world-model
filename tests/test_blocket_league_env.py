from __future__ import annotations

import unittest

import numpy as np

from blocket_league.data import make_clip
from blocket_league.env import BlocketLeagueEnv


class BlocketLeagueEnvironmentTests(unittest.TestCase):

    def test_continuous_vector_step_matches_cardinal_action(self) -> None:
        first = BlocketLeagueEnv(seed=91)
        second = BlocketLeagueEnv(seed=91)
        first.step(3)
        second.step_vector(np.asarray((1.0, 0.0), dtype=np.float32))
        np.testing.assert_allclose(first.state.vector(), second.state.vector(), atol=1e-7)
    def test_seeded_trajectories_are_exactly_reproducible(self) -> None:
        left = BlocketLeagueEnv(seed=23)
        right = BlocketLeagueEnv(seed=23)
        actions = [0, 3, 3, 2, 1, 8, 7, 6, 5, 4] * 12
        for action in actions:
            left.step(action)
            right.step(action)
        np.testing.assert_array_equal(left.state.vector(), right.state.vector())
        np.testing.assert_array_equal(left.render(), right.render())

    def test_autopilot_distribution_contains_contacts_and_goals(self) -> None:
        env = BlocketLeagueEnv(seed=3)
        events: set[str] = set()
        for _ in range(1_000):
            env.step(env.policy_action())
            events.add(env.state.last_event)
            self.assertTrue(np.isfinite(env.state.vector()).all())
        self.assertGreater(env.state.score, 0)
        self.assertIn("impact", events)
        self.assertIn("wall", events)
        self.assertIn("goal", events)

    def test_clip_contract_aligns_context_future_and_actions(self) -> None:
        clip = make_clip(11, context_frames=4, future_frames=8, image_size=64)
        self.assertEqual(clip["frames"].shape, (12, 64, 64, 3))
        self.assertEqual(clip["context"].shape, (4, 64, 64, 3))
        self.assertEqual(clip["target"].shape, (8, 64, 64, 3))
        self.assertEqual(clip["actions"].shape, (8,))
        self.assertEqual(clip["state"].shape, (8, 10))
        self.assertEqual(clip["events"].shape, (8,))


if __name__ == "__main__":
    unittest.main()
