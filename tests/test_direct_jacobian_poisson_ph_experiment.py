from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from blocket_league.action_free_excitation import (
    action_free_environment_config_sha256,
)
from blocket_league.direct_jacobian_poisson_ph_experiment import (
    ExperimentFConfig,
    generate_sanitized_splits,
    load_sanitized_split,
)


class DirectJacobianPoissonPHExperimentTests(unittest.TestCase):
    def test_producer_archive_contains_no_seed_or_simulator_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ExperimentFConfig(
                fit_trajectories=2,
                validation_trajectories=1,
                test_trajectories=1,
                history_frames=3,
                cache_frames=5,
                image_size=16,
                transitions=2,
                variants=("full",),
            )
            summary = generate_sanitized_splits(
                "pendulum", root, config, producer_seed=998_123
            )
            self.assertFalse(summary["producerSeedSerialized"])
            self.assertEqual(
                summary["generationEnvironmentSha256"],
                action_free_environment_config_sha256(
                    "pendulum", image_size=config.image_size
                ),
            )
            raw = torch.load(
                root / "trainer-mount" / "pendulum" / "fit-pixels.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(set(raw), {"pixels", "manifest"})
            serialized = repr(raw).lower()
            for forbidden in ("action", "control", "state", "torque", "seed"):
                self.assertNotIn(forbidden, serialized)
            pixels, manifest = load_sanitized_split(
                root / "trainer-mount" / "pendulum" / "fit-pixels.pt",
                expected_system="pendulum",
            )
            self.assertEqual(pixels.shape, (2, 5, 16, 16))
            self.assertEqual(manifest.source_schema, ("frames",))
            self.assertFalse(
                (root / "trainer-mount" / "pendulum" / "test-pixels.pt").exists()
            )
            self.assertTrue(
                (root / "heldout" / "pendulum" / "test-pixels.pt").exists()
            )


if __name__ == "__main__":
    unittest.main()
