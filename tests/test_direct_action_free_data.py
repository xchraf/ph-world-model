from __future__ import annotations

import unittest
from dataclasses import asdict
import copy

import torch

from blocket_league.direct_action_free_data import (
    ActionFreeBackboneTrainConfig,
    PixelsOnlyManifest,
    validate_action_free_backbone_checkpoint,
    make_optimization_suite,
)
from blocket_league.direct_experiment_f_producer import (
    collect_action_free_video_cache,
)
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig


class DirectActionFreeDataTests(unittest.TestCase):
    def test_cache_and_suite_have_pixels_only_schemas(self) -> None:
        model_config = PixelDirectConfig(
            image_size=16,
            patch_size=4,
            palette_size=9,
            history_frames=3,
            hidden_size=16,
            depth=2,
            heads=2,
        )
        cache, manifest = collect_action_free_video_cache(
            "pendulum",
            trajectories=2,
            frames_per_trajectory=6,
            image_size=16,
            seed=19,
            log_every=0,
        )
        self.assertEqual(cache.shape, (2, 6, 16, 16))
        self.assertEqual(cache.dtype, torch.uint8)
        self.assertEqual(manifest.source_schema, ("frames",))
        suite = make_optimization_suite(cache, model_config, transitions=3)
        self.assertEqual(set(suite), {"pixelContexts", "frames"})
        self.assertEqual(suite["pixelContexts"].shape, (2, 4, 3, 16, 16))
        self.assertEqual(suite["frames"].shape, (2, 4, 16, 16))

    def test_manifest_hash_is_deterministic(self) -> None:
        arguments = dict(
            system="blocket",
            trajectories=2,
            frames_per_trajectory=5,
            image_size=16,
            seed=23,
            log_every=0,
        )
        first, first_manifest = collect_action_free_video_cache(**arguments)
        second, second_manifest = collect_action_free_video_cache(**arguments)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertEqual(first_manifest.aggregate_sha256, second_manifest.aggregate_sha256)

    def test_backbone_provenance_fails_closed(self) -> None:
        model_config = PixelDirectConfig(
            image_size=8,
            patch_size=4,
            palette_size=9,
            history_frames=2,
            pixel_embedding_size=2,
            hidden_size=8,
            depth=1,
            heads=2,
            mlp_ratio=2.0,
        )
        train_config = ActionFreeBackboneTrainConfig(
            steps=1,
            batch_size=2,
            warmup_steps=0,
            log_every=1,
        )
        manifest = PixelsOnlyManifest(
            system="pendulum",
            trajectories=2,
            frames_per_trajectory=4,
            image_size=8,
            aggregate_sha256="a" * 64,
            sanitized_tensor_sha256="b" * 64,
        )
        valid = {
            "kind": "passive_direct_pixel_world_model",
            "system": "pendulum",
            "actionChannels": 0,
            "optimizationTensorKeys": ["pixels"],
            "pixelsOnlyManifest": asdict(manifest),
            "model": dict(DirectPixelTransformer(model_config).state_dict()),
            "model_config": model_config.to_dict(),
            "train_config": asdict(train_config),
            "step": 1,
        }
        validation_arguments = {
            "expected_manifest_sha256": "a" * 64,
            "expected_sanitized_tensor_sha256": "b" * 64,
            "expected_system": "pendulum",
        }
        validate_action_free_backbone_checkpoint(
            valid, **validation_arguments
        )
        for mutation in (
            {"actionChannels": 1},
            {"optimizationTensorKeys": ["pixels", "actions"]},
            {"pixelsOnlyManifest": None},
            {"actions": torch.zeros(1)},
        ):
            invalid = {**valid, **mutation}
            with self.assertRaises(ValueError):
                validate_action_free_backbone_checkpoint(
                    invalid, **validation_arguments
                )
        for invalid in (
            {**valid, "metadata": {"actions": torch.zeros(1)}},
            {**valid, "torqueSequence": torch.zeros(1)},
            {
                **valid,
                "pixelsOnlyManifest": {
                    **valid["pixelsOnlyManifest"],
                    "aggregate_sha256": "z" * 64,
                },
            },
        ):
            with self.assertRaises(ValueError):
                validate_action_free_backbone_checkpoint(
                    invalid, **validation_arguments
                )

        adversarial = []
        for nested_key, nested_value in (
            ("u", torch.zeros(1)),
            ("notes", "action-conditioned with torques"),
            ("аctions", torch.zeros(1)),  # first letter is Cyrillic
        ):
            candidate = copy.deepcopy(valid)
            candidate["train_config"][nested_key] = nested_value
            adversarial.append(candidate)
        invalid_sanitized = copy.deepcopy(valid)
        invalid_sanitized["pixelsOnlyManifest"]["sanitized_tensor_sha256"] = "invalid"
        adversarial.append(invalid_sanitized)
        hidden_model_tensor = copy.deepcopy(valid)
        hidden_model_tensor["model"]["latent_effort"] = torch.zeros(1)
        adversarial.append(hidden_model_tensor)
        for invalid in adversarial:
            with self.assertRaises(ValueError):
                validate_action_free_backbone_checkpoint(invalid, **validation_arguments)


if __name__ == "__main__":
    unittest.main()
