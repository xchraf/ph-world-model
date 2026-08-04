from __future__ import annotations

import unittest

import torch

from blocket_league.end_to_end_ph_experiment import (
    EndToEndDynamicsBranch,
    EndToEndPHConfig,
    LatentPatchTransformerRenderer,
    TransformerStateEncoder,
    _concatenate_training_suites,
    end_to_end_branch_loss,
)
from blocket_league.neural_port_hamiltonian import NeuralPortHamiltonianConfig
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig


class EndToEndPHExperimentTests(unittest.TestCase):
    def _branch(self) -> EndToEndDynamicsBranch:
        backbone = DirectPixelTransformer(
            PixelDirectConfig(
                image_size=8,
                patch_size=4,
                palette_size=9,
                history_frames=2,
                pixel_embedding_size=4,
                hidden_size=16,
                depth=1,
                heads=4,
                mlp_ratio=2.0,
            )
        )
        encoder = TransformerStateEncoder(
            backbone, state_size=8, hidden_size=16, eval_batch_size=2
        )
        renderer = LatentPatchTransformerRenderer(
            8,
            image_size=8,
            patch_size=4,
            palette_size=9,
            hidden_size=16,
            depth=1,
            heads=4,
        )
        return EndToEndDynamicsBranch(
            encoder,
            renderer,
            core_config=NeuralPortHamiltonianConfig(
                state_size=8,
                input_size=2,
                hidden_size=16,
                hidden_layers=1,
                dt=0.05,
            ),
            structured=True,
        )

    def test_training_suite_contains_pixels_and_actions_only(self) -> None:
        suite = {
            "pixelContexts": torch.zeros(2, 3, 2, 8, 8, dtype=torch.uint8),
            "frames": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
            "actions": torch.zeros(2, 2, dtype=torch.long),
            "actionVectors": torch.zeros(2, 2, 2),
        }
        merged = _concatenate_training_suites(suite, suite)
        self.assertEqual(merged["pixelContexts"].shape[0], 4)
        with self.assertRaises(AssertionError):
            _concatenate_training_suites({**suite, "worldStates": torch.zeros(2, 3, 10)})

    def test_encoder_chunks_arbitrary_leading_dimensions_at_eval(self) -> None:
        branch = self._branch().eval()
        contexts = torch.randint(0, 9, (2, 3, 2, 8, 8))
        with torch.no_grad():
            state = branch.encode(contexts)
        self.assertEqual(state.shape, (2, 3, 8))

    def test_joint_loss_updates_transformer_and_all_ph_functions(self) -> None:
        torch.manual_seed(41_021)
        branch = self._branch()
        contexts = torch.randint(0, 9, (2, 3, 2, 8, 8))
        frames = contexts[:, :, -1]
        actions = torch.randn(2, 2, 2).clamp(-1, 1)
        loss, terms = end_to_end_branch_loss(
            branch,
            contexts,
            frames,
            actions,
            torch.tensor((1, 2)),
            torch.ones(9),
            EndToEndPHConfig(
                fit_policy_trajectories=1,
                fit_cardinal_trajectories=1,
                transitions_per_trajectory=2,
                dynamics_batch_size=2,
                decoder_hidden_size=16,
                decoder_depth=1,
                decoder_heads=4,
            ),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(
            set(terms),
            {
                "reconstruction",
                "teacherLatent",
                "rolloutLatent",
                "rolloutPixel",
                "actionContrast",
                "whitening",
                "energyGauge",
            },
        )
        loss.backward()
        embedding_gradient = branch.encoder.backbone.pixel_embedding.weight.grad
        self.assertIsNotNone(embedding_gradient)
        self.assertGreater(float(embedding_gradient.abs().sum()), 0.0)
        self.assertGreater(
            float(branch.renderer.latent_projection.weight.grad.abs().sum()), 0.0
        )
        for name in (
            "energy_network",
            "interconnection_network",
            "resistance_network",
            "port_network",
        ):
            gradients = [
                parameter.grad for parameter in getattr(branch.core, name).parameters()
            ]
            self.assertTrue(any(gradient is not None for gradient in gradients), name)
            self.assertTrue(
                any(
                    float(gradient.abs().sum()) > 0
                    for gradient in gradients
                    if gradient is not None
                ),
                name,
            )


if __name__ == "__main__":
    unittest.main()
