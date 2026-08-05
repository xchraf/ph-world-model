from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from blocket_league.direct_activation_lens import (
    ActivationWriteFieldConfig,
    FrozenSoftPixelActivationLens,
    StateConditionedActivationWriteField,
    differentiable_attention_backend,
    odd_symmetry_loss,
)
from blocket_league.direct_cotangent_bridge import (
    PixelChangeProbeBank,
    activation_observable_covectors,
)
from blocket_league.direct_poisson_ph import (
    DirectPoissonPHConfig,
    DirectPoissonPortHamiltonian,
)
from blocket_league.direct_visual_poisson_ph import (
    WholeStreamEncoderConfig,
    WholeStreamFrozenEncoder,
)
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig


class ExactPerformanceOptimizationTests(unittest.TestCase):
    """Guard optimizations that must not alter values or trainable gradients."""

    def test_exact_energy_gradient_matches_nested_autograd_and_gradients(self) -> None:
        torch.manual_seed(7200)
        core = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(
                state_size=4,
                port_size=2,
                hidden_size=7,
                hidden_layers=2,
                coupling_layers=2,
                implicit_iterations=2,
            )
        ).double()
        state = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        reference_state = state.detach().clone().requires_grad_(True)
        actual_energy, actual_gradient = core._energy_gradient(  # noqa: SLF001
            state, create_graph=True
        )
        reference_energy = core.hamiltonian(reference_state)
        reference_gradient = torch.autograd.grad(
            reference_energy.sum(),
            reference_state,
            create_graph=True,
            retain_graph=True,
        )[0]
        torch.testing.assert_close(
            actual_energy, reference_energy, atol=2e-13, rtol=2e-13
        )
        torch.testing.assert_close(
            actual_gradient, reference_gradient, atol=3e-13, rtol=3e-13
        )

        energy_weight = torch.randn_like(actual_energy)
        gradient_weight = torch.randn_like(actual_gradient)
        energy_parameters = (
            core.energy_curvature,
            *tuple(core.energy_network.parameters()),
        )
        actual_derivatives = torch.autograd.grad(
            (actual_energy * energy_weight).sum()
            + (actual_gradient * gradient_weight).sum(),
            (state, *energy_parameters),
            retain_graph=True,
        )
        reference_derivatives = torch.autograd.grad(
            (reference_energy * energy_weight).sum()
            + (reference_gradient * gradient_weight).sum(),
            (reference_state, *energy_parameters),
        )
        for actual, reference in zip(actual_derivatives, reference_derivatives):
            torch.testing.assert_close(actual, reference, atol=2e-11, rtol=2e-10)

    def test_closed_form_coupling_jacobian_matches_jacrev_and_gradients(self) -> None:
        torch.manual_seed(7201)
        core = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(
                state_size=4,
                port_size=2,
                hidden_size=7,
                hidden_layers=2,
                coupling_layers=4,
                implicit_iterations=2,
            )
        ).double()
        base = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        analytic_value, analytic = core.coordinate_map.value_and_jacobian(base)
        reference_value = core.coordinate_map(base)
        reference = torch.func.vmap(
            torch.func.jacrev(core.coordinate_map.forward)
        )(base)
        torch.testing.assert_close(
            analytic_value, reference_value, atol=2e-13, rtol=2e-13
        )
        torch.testing.assert_close(analytic, reference, atol=4e-13, rtol=4e-13)

        weight = torch.randn_like(analytic)
        parameters = (base, *tuple(core.coordinate_map.parameters()))
        analytic_gradients = torch.autograd.grad(
            (analytic * weight).sum(), parameters, retain_graph=True
        )
        reference_gradients = torch.autograd.grad(
            (reference * weight).sum(), parameters
        )
        for actual, expected in zip(analytic_gradients, reference_gradients):
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)

    def test_inverse_reuses_exact_forward_jacobian_and_gradients(self) -> None:
        torch.manual_seed(7211)
        core = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(
                state_size=4,
                port_size=2,
                hidden_size=7,
                hidden_layers=2,
                coupling_layers=4,
                implicit_iterations=2,
            )
        ).double()
        image = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        reference_image = image.detach().clone().requires_grad_(True)
        actual_base, actual_jacobian = core.coordinate_map.inverse_and_jacobian(image)
        reference_base = core.coordinate_map.inverse(reference_image)
        reference_jacobian = core.coordinate_map.jacobian(reference_base)
        torch.testing.assert_close(
            actual_base, reference_base, atol=4e-13, rtol=4e-13
        )
        torch.testing.assert_close(
            actual_jacobian, reference_jacobian, atol=8e-13, rtol=8e-13
        )

        base_weight = torch.randn_like(actual_base)
        jacobian_weight = torch.randn_like(actual_jacobian)
        parameters = tuple(core.coordinate_map.parameters())
        actual_gradients = torch.autograd.grad(
            (actual_base * base_weight).sum()
            + (actual_jacobian * jacobian_weight).sum(),
            (image, *parameters),
            retain_graph=True,
        )
        reference_gradients = torch.autograd.grad(
            (reference_base * base_weight).sum()
            + (reference_jacobian * jacobian_weight).sum(),
            (reference_image, *parameters),
        )
        for actual, reference in zip(actual_gradients, reference_gradients):
            torch.testing.assert_close(actual, reference, atol=4e-11, rtol=4e-10)

    @staticmethod
    def _backbone() -> DirectPixelTransformer:
        return DirectPixelTransformer(
            PixelDirectConfig(
                image_size=4,
                patch_size=2,
                palette_size=3,
                history_frames=2,
                pixel_embedding_size=2,
                hidden_size=6,
                depth=2,
                heads=2,
                mlp_ratio=2.0,
            )
        )

    def test_batched_state_vjp_matches_coordinate_loop_and_gradients(self) -> None:
        torch.manual_seed(7202)
        encoder = WholeStreamFrozenEncoder(
            self._backbone(),
            WholeStreamEncoderConfig(4, readout_hidden_size=7, lens_block=0),
        )
        pixels = torch.randint(0, 3, (2, 2, 4, 4))
        activation = encoder.prefix_activation(pixels).requires_grad_(True)
        reference_activation = activation.detach().clone().requires_grad_(True)
        with differentiable_attention_backend(activation):
            actual = encoder.state_jacobian_from_activation(
                activation, create_graph=True
            )
            reference_state = encoder.from_activation(reference_activation)
            reference = torch.stack(
                tuple(
                    torch.autograd.grad(
                        reference_state[:, coordinate].sum(),
                        reference_activation,
                        create_graph=True,
                        retain_graph=True,
                    )[0].flatten(1)
                    for coordinate in range(encoder.state_size)
                ),
                dim=1,
            )
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)

        weight = torch.randn_like(actual)
        trainable = tuple(encoder.readout.parameters()) + tuple(
            encoder.pool_score.parameters()
        )
        actual_gradients = torch.autograd.grad(
            (actual * weight).sum(),
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        reference_gradients = torch.autograd.grad(
            (reference * weight).sum(), trainable, allow_unused=True
        )
        for actual_gradient, reference_gradient in zip(
            actual_gradients, reference_gradients
        ):
            self.assertEqual(actual_gradient is None, reference_gradient is None)
            if actual_gradient is None or reference_gradient is None:
                continue
            torch.testing.assert_close(
                actual_gradient,
                reference_gradient,
                atol=3e-5,
                rtol=3e-4,
            )

    def test_shared_state_and_vjp_matches_separate_calls(self) -> None:
        torch.manual_seed(7212)
        encoder = WholeStreamFrozenEncoder(
            self._backbone(),
            WholeStreamEncoderConfig(4, readout_hidden_size=7, lens_block=0),
        )
        pixels = torch.randint(0, 3, (2, 2, 4, 4))
        shared_activation = encoder.prefix_activation(pixels).requires_grad_(True)
        separate_activation = shared_activation.detach().clone().requires_grad_(True)
        with differentiable_attention_backend(shared_activation):
            shared_state, shared_jacobian = (
                encoder.state_and_jacobian_from_activation(
                    shared_activation, create_graph=True
                )
            )
            separate_state = encoder.from_activation(separate_activation)
            separate_jacobian = encoder.state_jacobian_from_activation(
                separate_activation, create_graph=True
            )
        torch.testing.assert_close(shared_state, separate_state, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            shared_jacobian, separate_jacobian, atol=2e-6, rtol=2e-5
        )

        state_weight = torch.randn_like(shared_state)
        jacobian_weight = torch.randn_like(shared_jacobian)
        parameters = tuple(encoder.readout.parameters()) + tuple(
            encoder.pool_score.parameters()
        )
        shared_gradients = torch.autograd.grad(
            (shared_state * state_weight).sum()
            + (shared_jacobian * jacobian_weight).sum(),
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        separate_gradients = torch.autograd.grad(
            (separate_state * state_weight).sum()
            + (separate_jacobian * jacobian_weight).sum(),
            parameters,
            allow_unused=True,
        )
        for shared, separate in zip(shared_gradients, separate_gradients):
            self.assertEqual(shared is None, separate is None)
            if shared is None or separate is None:
                continue
            torch.testing.assert_close(shared, separate, atol=3e-5, rtol=3e-4)

    def test_batched_port_jvp_matches_sequential_jvp_and_gradients(self) -> None:
        torch.manual_seed(7203)
        backbone = self._backbone()
        lens = FrozenSoftPixelActivationLens(
            backbone, intervention_block=0, horizons=(1, 2)
        )
        field = StateConditionedActivationWriteField(
            ActivationWriteFieldConfig(
                latent_size=4,
                port_size=2,
                history_frames=2,
                patch_count=4,
                hidden_size=6,
                network_hidden_size=7,
                network_layers=1,
            )
        )
        pixels = torch.randint(0, 3, (2, 2, 4, 4))
        basis = field(torch.randn(2, 4))
        actual = lens.response_jacobians(
            pixels, basis, horizons=(1, 2), create_graph=True
        )

        zero = basis.new_zeros(2, 2)

        def response_tuple(pulse: torch.Tensor) -> tuple[torch.Tensor, ...]:
            rollout = lens.rollout(pixels, basis, pulse, horizons=(1, 2))
            return tuple(rollout[horizon].reshape(2, -1) for horizon in (1, 2))

        reference_columns: list[list[torch.Tensor]] = [[], []]
        for port_index in range(2):
            tangent = torch.zeros_like(zero)
            tangent[:, port_index] = 1.0
            _, directional = torch.autograd.functional.jvp(
                response_tuple,
                zero,
                tangent,
                create_graph=True,
                strict=False,
            )
            for horizon_index, value in enumerate(directional):
                reference_columns[horizon_index].append(value)
        reference = {
            horizon: torch.stack(reference_columns[index], dim=-1)
            for index, horizon in enumerate((1, 2))
        }
        for horizon in (1, 2):
            torch.testing.assert_close(
                actual.jacobians[horizon],
                reference[horizon],
                atol=3e-6,
                rtol=3e-5,
            )

        weight = {
            horizon: torch.randn_like(actual.jacobians[horizon])
            for horizon in (1, 2)
        }
        field_parameters = tuple(field.parameters())
        actual_loss = sum(
            (actual.jacobians[horizon] * weight[horizon]).sum()
            for horizon in (1, 2)
        )
        reference_loss = sum(
            (reference[horizon] * weight[horizon]).sum()
            for horizon in (1, 2)
        )
        actual_gradients = torch.autograd.grad(
            actual_loss, field_parameters, retain_graph=True
        )
        reference_gradients = torch.autograd.grad(reference_loss, field_parameters)
        for actual_gradient, reference_gradient in zip(
            actual_gradients, reference_gradients
        ):
            torch.testing.assert_close(
                actual_gradient,
                reference_gradient,
                atol=2e-5,
                rtol=2e-4,
            )

    def test_state_response_reuses_exact_future_context_tokens(self) -> None:
        """The fast bridge equals explicit soft rollout then re-encoding."""

        torch.manual_seed(7213)
        backbone = self._backbone()
        encoder = WholeStreamFrozenEncoder(
            backbone,
            WholeStreamEncoderConfig(4, readout_hidden_size=7, lens_block=0),
        )
        lens = FrozenSoftPixelActivationLens(
            backbone, intervention_block=0, horizons=(1, 2)
        )
        field = StateConditionedActivationWriteField(
            ActivationWriteFieldConfig(
                latent_size=4,
                port_size=2,
                history_frames=2,
                patch_count=4,
                hidden_size=6,
                network_hidden_size=7,
                network_layers=1,
            )
        )
        pixels = torch.randint(0, 3, (2, 2, 4, 4))
        basis = field(torch.randn(2, 4))
        actual = lens.state_response_jacobians(
            pixels,
            basis,
            encoder.read_suffix_tokens,
            horizons=(1, 2),
            create_graph=True,
        )

        zero = basis.new_zeros(2, 2)

        def explicit_reencoded_rollout(
            pulse: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            current = lens.pixel_probabilities(pixels)
            write = lens.residual_write(basis, pulse)
            responses: list[torch.Tensor] = []
            for horizon in range(1, 3):
                prediction = lens.soft_forward(
                    current,
                    residual_write=write if horizon == 1 else None,
                )[:, -1]
                current = torch.cat((current[:, 1:], prediction[:, None]), dim=1)
                prefix = lens.soft_prefix_activation(current)
                responses.append(encoder.from_activation(prefix))
            return responses[0], responses[1]

        reference_columns: list[list[torch.Tensor]] = [[], []]
        with differentiable_attention_backend(zero):
            for port_index in range(2):
                tangent = torch.zeros_like(zero)
                tangent[:, port_index] = 1.0
                _, directional = torch.autograd.functional.jvp(
                    explicit_reencoded_rollout,
                    zero,
                    tangent,
                    create_graph=True,
                    strict=False,
                )
                for horizon_index, value in enumerate(directional):
                    reference_columns[horizon_index].append(value)
        reference = {
            horizon: torch.stack(reference_columns[index], dim=-1)
            for index, horizon in enumerate((1, 2))
        }
        for horizon in (1, 2):
            self.assertEqual(actual.jacobians[horizon].shape, (2, 4, 2))
            torch.testing.assert_close(
                actual.jacobians[horizon],
                reference[horizon],
                atol=4e-6,
                rtol=4e-5,
            )

        weights = {
            horizon: torch.randn_like(actual.jacobians[horizon])
            for horizon in (1, 2)
        }
        parameters = tuple(field.parameters()) + tuple(encoder.readout.parameters())
        actual_gradients = torch.autograd.grad(
            sum(
                (actual.jacobians[horizon] * weights[horizon]).sum()
                for horizon in (1, 2)
            ),
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        reference_gradients = torch.autograd.grad(
            sum(
                (reference[horizon] * weights[horizon]).sum()
                for horizon in (1, 2)
            ),
            parameters,
            allow_unused=True,
        )
        for actual_gradient, reference_gradient in zip(
            actual_gradients, reference_gradients
        ):
            self.assertEqual(actual_gradient is None, reference_gradient is None)
            if actual_gradient is None or reference_gradient is None:
                continue
            torch.testing.assert_close(
                actual_gradient,
                reference_gradient,
                atol=8e-5,
                rtol=8e-4,
            )

    def test_reused_proxy_rollouts_match_unoptimized_definition_and_gradients(self) -> None:
        torch.manual_seed(7204)
        lens = FrozenSoftPixelActivationLens(
            self._backbone(), intervention_block=0, horizons=(1, 2)
        )
        field = StateConditionedActivationWriteField(
            ActivationWriteFieldConfig(
                latent_size=4,
                port_size=2,
                history_frames=2,
                patch_count=4,
                hidden_size=6,
                network_hidden_size=7,
                network_layers=1,
            )
        )
        pixels = torch.randint(0, 3, (2, 2, 4, 4))
        basis = field(torch.randn(2, 4))
        amplitude = 0.03
        actual = lens.intervention_proxies(pixels, basis, amplitude=amplitude)

        probabilities = lens.pixel_probabilities(pixels)
        batch, port_size = probabilities.shape[0], basis.shape[-1]
        expanded_context = probabilities[:, None].expand(
            batch, port_size, *probabilities.shape[1:]
        ).reshape(batch * port_size, *probabilities.shape[1:])
        expanded_basis = basis[:, None].expand(
            batch, port_size, *basis.shape[1:]
        ).reshape(batch * port_size, *basis.shape[1:])
        directions = amplitude * torch.eye(
            port_size, dtype=basis.dtype
        )[None].expand(batch, port_size, port_size).reshape(
            batch * port_size, port_size
        )
        zero = torch.zeros_like(directions)
        baseline_all = lens.soft_forward(expanded_context)
        positive_all = lens.soft_forward(
            expanded_context,
            residual_write=lens.residual_write(expanded_basis, directions),
        )
        negative_all = lens.soft_forward(
            expanded_context,
            residual_write=lens.residual_write(expanded_basis, -directions),
        )
        baseline_terminal = baseline_all[:, -1]
        positive_terminal = positive_all[:, -1]
        negative_terminal = negative_all[:, -1]
        signal = (positive_terminal - negative_terminal).square().mean()
        history_change = 0.5 * (
            (positive_all[:, :-1] - baseline_all[:, :-1]).square().mean()
            + (negative_all[:, :-1] - baseline_all[:, :-1]).square().mean()
        )
        terminal_change = 0.5 * (
            (positive_terminal - baseline_terminal).square().mean()
            + (negative_terminal - baseline_terminal).square().mean()
        )
        old_values = (
            odd_symmetry_loss(
                positive_terminal, negative_terminal, baseline_terminal
            ),
            history_change / terminal_change.clamp_min(1e-8),
            (
                lens._scheduled_rollout(  # noqa: SLF001 - equivalence oracle
                    expanded_context,
                    expanded_basis,
                    (directions, -directions),
                )
                - lens._scheduled_rollout(  # noqa: SLF001 - equivalence oracle
                    expanded_context,
                    expanded_basis,
                    (zero, zero),
                )
            ).square().mean()
            / signal.clamp_min(1e-8),
            signal,
        )
        new_values = (
            actual.odd_symmetry,
            actual.current_frame_leakage,
            actual.manifold_cycle,
            actual.first_order_signal,
        )
        for new, old in zip(new_values, old_values):
            torch.testing.assert_close(new, old, atol=3e-6, rtol=3e-5)

        weights = (0.7, -0.2, 0.3, 1.1)
        parameters = tuple(field.parameters())
        new_gradients = torch.autograd.grad(
            sum(weight * value for weight, value in zip(weights, new_values)),
            parameters,
            retain_graph=True,
        )
        old_gradients = torch.autograd.grad(
            sum(weight * value for weight, value in zip(weights, old_values)),
            parameters,
        )
        for new_gradient, old_gradient in zip(new_gradients, old_gradients):
            torch.testing.assert_close(
                new_gradient, old_gradient, atol=3e-4, rtol=2e-3
            )

    def test_batched_visual_covectors_match_sequential_reverse_passes(self) -> None:
        torch.manual_seed(7205)
        lens = FrozenSoftPixelActivationLens(
            self._backbone(), intervention_block=0, horizons=(1, 2)
        )
        probes = PixelChangeProbeBank(torch.randn(2, 3, 4, 4))
        pixels = torch.randint(0, 3, (2, 2, 4, 4))
        actual = activation_observable_covectors(
            lens, pixels, probes, horizons=(1, 2), create_graph=False
        )

        write = torch.zeros(
            2,
            *lens.activation_shape,
            dtype=lens.backbone.pixel_embedding.weight.dtype,
            requires_grad=True,
        )
        current = lens.pixel_probabilities(pixels)
        predictions = {}
        for step in range(1, 3):
            output = lens.soft_forward(
                current, residual_write=write if step == 1 else None
            )
            next_pixels = output[:, -1]
            current = torch.cat((current[:, 1:], next_pixels[:, None]), dim=1)
            predictions[step] = next_pixels
        reference = {}
        for horizon in (1, 2):
            observable = probes(predictions[horizon])
            reference[horizon] = torch.stack(
                tuple(
                    torch.autograd.grad(
                        observable[:, probe].sum(),
                        write,
                        retain_graph=True,
                    )[0].flatten(1)
                    for probe in range(probes.probe_size)
                ),
                dim=-1,
            )
            torch.testing.assert_close(
                actual[horizon], reference[horizon], atol=2e-6, rtol=2e-5
            )

    def test_probe_subsample_before_one_hot_preserves_original_pca_input(self) -> None:
        torch.manual_seed(7206)
        frames = torch.randint(0, 3, (5, 6, 4, 4))
        maximum = 7
        # Original definition: expand every transition, flatten, then take the
        # deterministic linspace subset.
        old_difference = (
            F.one_hot(frames[:, 1:].long(), num_classes=3)
            - F.one_hot(frames[:, :-1].long(), num_classes=3)
        ).permute(0, 1, 4, 2, 3)
        old_flat = old_difference.reshape(-1, 3 * 4 * 4).float()
        indices = torch.linspace(0, old_flat.shape[0] - 1, maximum).long()
        old_flat = old_flat[indices]
        old_flat = old_flat - old_flat.mean(dim=0, keepdim=True)
        torch.manual_seed(91)
        _, _, old_right = torch.pca_lowrank(
            old_flat, q=6, center=False, niter=4
        )
        expected = PixelChangeProbeBank(old_right[:, :2].T.reshape(2, 3, 4, 4))

        torch.manual_seed(91)
        actual = PixelChangeProbeBank.from_pixel_frames(
            frames,
            palette_size=3,
            probe_size=2,
            maximum_differences=maximum,
        )
        torch.testing.assert_close(actual.basis, expected.basis, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
