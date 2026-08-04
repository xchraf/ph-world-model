from __future__ import annotations

import unittest

import numpy as np

from blocket_league.passive_control_systems import (
    PendulumEnv,
    PendulumState,
    make_passive_pendulum_clip,
)


class PassiveControlSystemsTests(unittest.TestCase):
    def test_passive_pendulum_payload_has_no_action_or_state(self) -> None:
        clip = make_passive_pendulum_clip(17, context_frames=3, future_frames=4, image_size=32)
        self.assertEqual(set(clip), {"frames", "context", "target"})
        self.assertEqual(clip["frames"].shape, (7, 32, 32, 3))

    def test_zero_torque_dissipates_energy_over_a_short_step(self) -> None:
        env = PendulumEnv(seed=3)
        env.set_state(PendulumState(angle=0.8, angular_velocity=1.2))
        before = env.energy()
        env.step(0.0)
        self.assertLess(env.energy(), before + 1e-6)

    def test_torque_sign_changes_angular_velocity(self) -> None:
        plus = PendulumEnv(seed=4)
        minus = PendulumEnv(seed=4)
        state = PendulumState(angle=0.0, angular_velocity=0.0)
        plus.set_state(state)
        minus.set_state(state)
        plus.step(1.0)
        minus.step(-1.0)
        self.assertGreater(plus.state.angular_velocity, 0.0)
        self.assertLess(minus.state.angular_velocity, 0.0)
        self.assertGreater(
            abs(plus.state.angular_velocity - minus.state.angular_velocity), 0.1
        )


if __name__ == "__main__":
    unittest.main()

