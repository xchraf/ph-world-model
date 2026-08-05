from __future__ import annotations

import inspect
import unittest

import torch

from blocket_league.direct_action_free_data import class_weights
from blocket_league.direct_poisson_ph import DirectPoissonPHConfig, DirectPoissonPortHamiltonian
from blocket_league.direct_visual_poisson_ph import (
    DirectVideoLossConfig,
    DirectVisualPoissonPH,
    WholeStreamEncoderConfig,
    WholeStreamFrozenEncoder,
    direct_video_objective,
    port_frame_regularizers,
    state_effort_second_moment_independence_loss,
    trainable_parameters_without_backbone,
)
from blocket_league.end_to_end_ph_experiment import LatentPatchTransformerRenderer
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig


class DirectVisualPoissonPHTests(unittest.TestCase):
    def _model(self) -> DirectVisualPoissonPH:
        backbone = DirectPixelTransformer(
            PixelDirectConfig(
                image_size=8,
                patch_size=4,
                palette_size=9,
                history_frames=2,
                pixel_embedding_size=3,
                hidden_size=8,
                depth=2,
                heads=2,
                mlp_ratio=2.0,
            )
        )
        encoder = WholeStreamFrozenEncoder(
            backbone,
            WholeStreamEncoderConfig(
                state_size=2,
                readout_hidden_size=8,
                lens_block=0,
            ),
        )
        renderer = LatentPatchTransformerRenderer(
            2,
            image_size=8,
            patch_size=4,
            palette_size=9,
            hidden_size=8,
            depth=1,
            heads=2,
        )
        core = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(
                state_size=2,
                port_size=1,
                hidden_size=8,
                hidden_layers=1,
                coupling_layers=2,
                dt=0.02,
                implicit_iterations=3,
                implicit_tolerance=0.0,
            )
        )
        return DirectVisualPoissonPH(encoder, renderer, core)

    def test_complete_stream_write_and_no_mask_or_physical_input_api(self) -> None:
        model = self._model()
        context = torch.randint(0, 9, (3, 2, 8, 8))
        activation = model.encoder.prefix_activation(context)
        self.assertEqual(activation.shape, (3, 2, 4, 8))
        write = torch.randn_like(activation, requires_grad=True)
        state = model.encoder.from_activation(activation, intervention=write)
        state.square().sum().backward()
        self.assertIsNotNone(write.grad)
        self.assertGreater(float(write.grad.abs().sum()), 0.0)

        for method in (model.encoder.forward, model.step, model.infer_latent_effort):
            names = tuple(inspect.signature(method).parameters)
            for forbidden in ("action", "control", "mask"):
                self.assertTrue(all(forbidden not in name.lower() for name in names))

    def test_joint_pixels_only_objective_keeps_backbone_exactly_frozen(self) -> None:
        torch.manual_seed(4)
        model = self._model()
        before = model.encoder.sealed_backbone_hash
        contexts = torch.randint(0, 9, (3, 3, 2, 8, 8))
        frames = contexts[:, :, -1]
        weights = class_weights(frames, 9, torch.device("cpu"))
        loss, metrics = direct_video_objective(
            model,
            contexts,
            frames,
            weights,
            DirectVideoLossConfig(rollout_horizons=(1, 2)),
            require_lens_terms=False,
        )
        optimizer = torch.optim.Adam(trainable_parameters_without_backbone(model), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIn("balanceDefectMax", metrics)
        model.encoder.assert_backbone_frozen()
        self.assertEqual(before, model.encoder.sealed_backbone_hash)
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.backbone.parameters()))
        groups = {
            "readout": list(model.encoder.pool_score.parameters())
            + list(model.encoder.readout.parameters()),
            "effort": list(model.effort_inference.parameters()),
            "H": [model.core.energy_curvature, *model.core.energy_network.parameters()],
            "J": list(model.core.coordinate_map.parameters()),
            "R": list(model.core.resistance_network.parameters()),
            "B": list(model.core.port_network.parameters()),
            "renderer": list(model.renderer.parameters()),
        }
        for name, parameters in groups.items():
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients), name)
            self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0, name)

    def test_registered_lens_terms_fail_closed(self) -> None:
        model = self._model()
        contexts = torch.randint(0, 9, (2, 2, 2, 8, 8))
        frames = contexts[:, :, -1]
        weights = class_weights(frames, 9, torch.device("cpu"))
        with self.assertRaises(RuntimeError):
            direct_video_objective(
                model,
                contexts,
                frames,
                weights,
                DirectVideoLossConfig(rollout_horizons=(1,)),
            )

    def test_rank_two_aligned_port_frame_has_finite_backward(self) -> None:
        class ConstantCore(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.port_value = torch.nn.Parameter(torch.eye(2))

            def port(self, state: torch.Tensor) -> torch.Tensor:
                return self.port_value.expand(*state.shape[:-1], 2, 2)

        core = ConstantCore()
        states = torch.randn(4, 3, 2)
        frame, holonomy, rank = port_frame_regularizers(core, states)  # type: ignore[arg-type]
        (frame + holonomy + rank).backward()
        self.assertIsNotNone(core.port_value.grad)
        self.assertTrue(bool(torch.isfinite(core.port_value.grad).all()))

    def test_port_frame_rejects_state_dependent_swap_across_trajectories(self) -> None:
        class SwappedCore(torch.nn.Module):
            def port(self, state: torch.Tensor) -> torch.Tensor:
                base = state.new_zeros(*state.shape[:-1], 4, 2)
                base[..., 0, 0] = 1.0
                base[..., 1, 1] = 1.0
                swap = torch.tensor(
                    [[0.0, 1.0], [1.0, 0.0]],
                    dtype=state.dtype,
                    device=state.device,
                )
                use_swap = (state[..., 0] > 0)[..., None, None]
                return torch.where(use_swap, base @ swap, base)

        states = torch.zeros(4, 3, 4)
        states[:2, :, 0] = -1.0
        states[2:, :, 0] = 1.0
        frame, holonomy, rank = port_frame_regularizers(  # type: ignore[arg-type]
            SwappedCore(), states
        )
        self.assertGreater(float(frame), 0.10)
        self.assertLess(float(holonomy), 1e-6)
        self.assertLess(float(rank), 1e-6)

    def test_rank_barrier_rejects_exact_rank_one_at_two_scales(self) -> None:
        class RankOneCore(torch.nn.Module):
            def __init__(self, scale: float) -> None:
                super().__init__()
                value = torch.zeros(4, 2)
                value[0, 0] = scale
                self.port_value = torch.nn.Parameter(value)

            def port(self, state: torch.Tensor) -> torch.Tensor:
                return self.port_value.expand(*state.shape[:-1], 4, 2)

        for scale in (1.0, 1e-2):
            core = RankOneCore(scale)
            states = torch.randn(2, 3, 4)
            frame, holonomy, rank = port_frame_regularizers(  # type: ignore[arg-type]
                core, states
            )
            self.assertGreater(float(rank), 1.0)
            (frame + holonomy + rank).backward()
            self.assertIsNotNone(core.port_value.grad)
            self.assertTrue(bool(torch.isfinite(core.port_value.grad).all()))

    def test_second_moment_independence_rejects_state_dependent_port_gain(self) -> None:
        state_coordinate = torch.linspace(-2.0, 2.0, 64)
        states = torch.stack((state_coordinate, state_coordinate.square()), dim=-1)
        signs = torch.where(
            torch.arange(64) % 2 == 0,
            torch.tensor(1.0),
            torch.tensor(-1.0),
        )
        constant_effort = signs[:, None]
        locally_rescaled = (signs / (1.0 + state_coordinate.square()))[:, None]
        independent = state_effort_second_moment_independence_loss(
            states, constant_effort
        )
        dependent = state_effort_second_moment_independence_loss(
            states, locally_rescaled
        )
        self.assertLess(float(independent), 1e-8)
        self.assertGreater(float(dependent), 1e-3)


if __name__ == "__main__":
    unittest.main()
