from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from blocket_league.direct_activation_lens import (
    FrozenSoftPixelActivationLens,
)
from blocket_league.direct_cotangent_bridge import (
    PixelChangeProbeBank,
    activation_observable_covectors,
)
from blocket_league.direct_jacobian_port_extractor import (
    EmpiricalTangentConfig,
    FrozenEmpiricalJacobianActivationPort,
    build_empirical_tangent_artifact,
)
from blocket_league.direct_physical_evaluation import (
    CEMConfig,
    CEMPlan,
    EvaluationSystem,
    FrozenActivationWriteWorldModel,
    FrozenEvaluationSeal,
    FrozenLatentPlannerSpec,
    PhysicalInterface,
    PixelControlEpisode,
    PixelPlant,
    ProbeCandidate,
    activation_calibration_from_response_frame,
    calibrate_activation_interface_after_freeze,
    cem_frozen_world_model_mpc,
    collect_paired_calibration_response_bank,
    collect_paired_heldout_response_bank,
    evaluate_closed_loop_controllers,
    evaluate_heldout_activation_from_response_bank,
    evaluate_heldout_realizability_from_response_bank,
    fit_interface_calibration_from_response_bank,
    select_d_optimal_probe_states,
    select_shared_maximin_probe_states,
)
from blocket_league.direct_visual_poisson_ph import (
    WholeStreamEncoderConfig,
    WholeStreamFrozenEncoder,
)
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig


def _tiny_activation_world_model() -> FrozenActivationWriteWorldModel:
    torch.manual_seed(2027)
    backbone = DirectPixelTransformer(
        PixelDirectConfig(
            image_size=8,
            patch_size=4,
            palette_size=3,
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
        WholeStreamEncoderConfig(state_size=2, readout_hidden_size=6, lens_block=0),
    )
    generator = torch.Generator().manual_seed(2028)
    fit_source = torch.randn(6, 2, 4, 8, generator=generator)
    fit_prediction = fit_source + 0.03 * torch.randn(
        fit_source.shape, generator=generator
    )
    fit_observation = fit_prediction + 0.15 * torch.randn(
        fit_source.shape, generator=generator
    )
    tangent_config = EmpiricalTangentConfig(
        channel_rank=2,
        neighbors=2,
        support_floor_ratio=0.01,
    )
    artifact = build_empirical_tangent_artifact(
        fit_source, fit_observation, fit_prediction, tangent_config
    )
    field = FrozenEmpiricalJacobianActivationPort(
        artifact,
        history_frames=2,
        patch_count=4,
        hidden_size=8,
        port_size=1,
        config=tangent_config,
    )
    lens = FrozenSoftPixelActivationLens(
        backbone, intervention_block=0, horizons=(1, 2)
    )
    probes = PixelChangeProbeBank(torch.randn(1, 3, 8, 8, generator=generator))
    for module in (encoder, field, lens, probes):
        module.eval().requires_grad_(False)
    return FrozenActivationWriteWorldModel(encoder, field, lens, probes)


class FrozenActivationWorldModelTests(unittest.TestCase):
    def test_rollout_is_exact_frozen_backbone_autoregression(self) -> None:
        model = _tiny_activation_world_model()
        contexts = torch.randint(0, 3, (2, 2, 8, 8))
        efforts = torch.zeros(2, 3, 1)
        observed = model(contexts, efforts)

        current = contexts
        expected = []
        for _ in range(3):
            prefix = model.encoder.prefix_activation(current).detach()
            logits = model.lens.soft_logits_from_prefix(prefix)[:, -1]
            expected.append(logits)
            categories = logits.argmax(dim=1).to(current.dtype)
            current = torch.cat((current[:, 1:], categories[:, None]), dim=1)
        torch.testing.assert_close(observed, torch.stack(expected, dim=1))

        intervened = model(contexts, torch.full_like(efforts, 0.7))
        self.assertGreater(float((intervened - observed).abs().max()), 1e-6)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.training)
        self.assertEqual(
            set(model._modules), {"encoder", "write_field", "lens", "probes"}
        )
        model.write_field.assert_frozen_parameter_free()
        self.assertFalse(
            any(
                segment in {"core", "renderer", "effort_inference"}
                for name in model.state_dict()
                for segment in name.split(".")
            )
        )
        seal = FrozenEvaluationSeal.capture({"activation": model})
        seal.assert_unchanged()
        with self.assertRaisesRegex(RuntimeError, "cannot train"):
            model.train()

    def test_activation_port_is_the_exact_encoder_jacobian_write(self) -> None:
        model = _tiny_activation_world_model()
        contexts = torch.randint(0, 3, (1, 2, 8, 8))
        dt = 0.05
        analytic = model.activation_state_rate_port(contexts, dt=dt)[0, :, 0]
        covectors = activation_observable_covectors(
            model.lens,
            contexts,
            model.probes,
            horizons=model.horizons,
            create_graph=False,
        )
        prefix = model.encoder.prefix_activation(contexts).detach()
        basis = model.write_field(
            covectors, prefix
        ).jacobian.write_basis[0, ..., 0]
        epsilon = 1e-3
        plus = model.encoder.from_activation(prefix + epsilon * basis[None])
        minus = model.encoder.from_activation(prefix - epsilon * basis[None])
        finite_difference = ((plus - minus) / (2.0 * epsilon * dt))[0]
        torch.testing.assert_close(analytic, finite_difference, atol=4e-3, rtol=4e-3)

    def test_plain_or_learned_write_field_has_no_evaluation_fallback(self) -> None:
        model = _tiny_activation_world_model()
        with self.assertRaisesRegex(TypeError, "FrozenEmpiricalJacobianActivationPort"):
            FrozenActivationWriteWorldModel(
                model.encoder,
                nn.Identity(),  # type: ignore[arg-type]
                model.lens,
                model.probes,
            )

    def test_exact_extractor_dependencies_are_explicit_and_shared(self) -> None:
        model = _tiny_activation_world_model()
        contexts = torch.randint(0, 3, (1, 2, 8, 8))
        with patch(
            "blocket_league.direct_physical_evaluation.activation_observable_covectors",
            wraps=activation_observable_covectors,
        ) as observed, patch.object(
            model.write_field,
            "forward",
            wraps=model.write_field.forward,
        ) as extracted:
            model.activation_state_rate_port(contexts, dt=0.05)
        observed.assert_called_once_with(
            model.lens,
            contexts,
            model.probes,
            horizons=model.horizons,
            create_graph=False,
        )
        extracted.assert_called_once()
        torch.testing.assert_close(
            extracted.call_args.args[1],
            model.encoder.prefix_activation(contexts).detach(),
        )
        constructor = inspect.signature(FrozenActivationWriteWorldModel)
        self.assertIs(
            constructor.parameters["probes"].default,
            inspect.Parameter.empty,
        )
        source = inspect.getsource(FrozenActivationWriteWorldModel)
        self.assertIn("activation_observable_covectors", source)
        self.assertIn("self.write_field(covectors, source_activation)", source)
        self.assertNotIn("self.write_field(state)", source)

    def test_rollout_reextracts_the_exact_port_at_every_hard_context(self) -> None:
        model = _tiny_activation_world_model()
        contexts = torch.randint(0, 3, (1, 2, 8, 8))
        efforts = torch.zeros(1, 3, 1)
        with patch(
            "blocket_league.direct_physical_evaluation.activation_observable_covectors",
            wraps=activation_observable_covectors,
        ) as observed:
            logits = model(contexts, efforts)
        self.assertEqual(observed.call_count, efforts.shape[1])
        expected_context = contexts
        for step, call in enumerate(observed.call_args_list):
            torch.testing.assert_close(call.args[1], expected_context)
            next_pixels = logits[:, step].argmax(dim=1).to(expected_context.dtype)
            expected_context = torch.cat(
                (expected_context[:, 1:], next_pixels[:, None]), dim=1
            )

    def test_response_frame_change_of_basis_is_exact(self) -> None:
        angle = torch.tensor(0.63)
        frame = torch.stack(
            (
                torch.stack((angle.cos(), -angle.sin())),
                torch.stack((angle.sin(), angle.cos())),
            )
        )
        structured = torch.tensor(((0.8, -0.2), (0.4, 1.1)))
        activation = activation_calibration_from_response_frame(structured, frame)
        ph_response = torch.tensor(((1.2, 0.1), (-0.3, 0.7), (0.4, 1.5)))
        lens_response = ph_response @ frame
        commands = torch.tensor(((0.2, -0.4), (0.9, 0.1)))
        torch.testing.assert_close(
            lens_response @ activation @ commands.T,
            ph_response @ structured @ commands.T,
        )


class _VectorEncoder(nn.Module):
    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        return contexts[:, -1].float()


def _vector_port(state: torch.Tensor) -> torch.Tensor:
    result = state.new_zeros(state.shape[0], 4, 2)
    result[:, 0, 0] = 1.0 + 0.08 * state[:, 0]
    result[:, 1, 1] = 0.9 + 0.06 * state[:, 1]
    result[:, 2, 0] = 0.25 + 0.04 * state[:, 2]
    result[:, 3, 1] = 0.35 + 0.03 * state[:, 3]
    return result


class _ToyActivationCalibrationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _VectorEncoder()

    def assert_frozen_and_unchanged(self) -> None:
        if self.training or any(parameter.requires_grad for parameter in self.parameters()):
            raise AssertionError("toy activation model is not frozen")

    def activation_state_rate_port(
        self, contexts: torch.Tensor, *, dt: float
    ) -> torch.Tensor:
        del dt
        return _vector_port(self.encoder(contexts))


class _ToySharedDynamics(nn.Module):
    def port(self, state: torch.Tensor) -> torch.Tensor:
        return _vector_port(state)

    def step(self, state: torch.Tensor, effort: torch.Tensor) -> torch.Tensor:
        return state + 0.05 * torch.einsum(
            "bnm,bm->bn", self.port(state), effort
        )


@dataclass
class _VectorEnvironment:
    state: np.ndarray


def _vector_candidates(count: int, *, offset: int = 0) -> list[ProbeCandidate]:
    result = []
    for index in range(count):
        value = index + offset
        state = np.asarray(
            (
                -1.4 + 0.31 * value,
                1.2 - 0.17 * value,
                -0.8 + 0.23 * value,
                0.5 + 0.11 * value,
            ),
            dtype=np.float32,
        )
        result.append(
            ProbeCandidate(
                f"candidate-{offset}-{index}",
                torch.from_numpy(state)[None],
                _VectorEnvironment(state.astype(np.float64)),
            )
        )
    return result


class ActivationCalibrationTests(unittest.TestCase):
    def test_separate_four_pair_calibration_is_closed_form_and_heldout(self) -> None:
        model = _ToyActivationCalibrationModel().eval().requires_grad_(False)
        dynamics = _ToySharedDynamics().eval().requires_grad_(False)
        seal = FrozenEvaluationSeal.capture(
            {"activation": model, "sharedDynamics": dynamics}
        )
        system = EvaluationSystem("toy", 2, 0.05, 1, 2, probe_amplitude=0.25)
        interface = PhysicalInterface("unseen", ((0.7, -0.2), (0.3, 1.1)))
        counter = {"steps": 0}

        def clone(environment: _VectorEnvironment) -> _VectorEnvironment:
            return _VectorEnvironment(environment.state.copy())

        def step(
            environment: _VectorEnvironment,
            physical_interface: PhysicalInterface,
            command: np.ndarray,
        ) -> None:
            counter["steps"] += 1
            native = physical_interface.matrix() @ command
            state = torch.from_numpy(environment.state).double()[None]
            port = _vector_port(state)[0].numpy()
            environment.state = environment.state + system.dt * port @ native

        def append(
            context: torch.Tensor, environment: _VectorEnvironment
        ) -> torch.Tensor:
            frame = torch.from_numpy(environment.state.astype(np.float32))
            return torch.cat((context[1:], frame[None]), dim=0)

        plant = PixelPlant(
            clone,
            step,
            append,
            lambda _environment: torch.zeros(1, 1, dtype=torch.uint8),
        )
        candidates = _vector_candidates(12)
        selection = select_shared_maximin_probe_states(
            {
                "full": (model.encoder, dynamics),
                "unstructured": (model.encoder, dynamics),
            },
            model,
            candidates,
            system,
            seal=seal,
        )
        self.assertEqual(counter["steps"], 0)
        self.assertEqual(
            selection.selection_method,
            "shared_maximin_normalized_d_optimal",
        )
        self.assertEqual(
            selection.selection_model_names,
            ("activation", "full", "unstructured"),
        )
        self.assertTrue(
            all(
                indices == selection.indices_by_axis[0]
                for indices in selection.indices_by_axis
            )
        )
        self.assertTrue(all(value > 0.0 for value in selection.normalization_scales))
        forged_identifiers = tuple(
            (("forged",) + identifiers[1:])
            for identifiers in selection.identifiers_by_axis
        )
        with self.assertRaisesRegex(ValueError, "identifiers differ"):
            collect_paired_calibration_response_bank(
                plant,
                candidates,
                replace(selection, identifiers_by_axis=forged_identifiers),
                system,
                interface,
                seal=seal,
            )
        self.assertEqual(counter["steps"], 0)
        response_bank = collect_paired_calibration_response_bank(
            plant,
            candidates,
            selection,
            system,
            interface,
            seal=seal,
        )
        self.assertEqual(counter["steps"], 16)
        original_context = candidates[0].context.clone()
        candidates[0].context.add_(0.125)
        with self.assertRaisesRegex(ValueError, "pixel-pool hash"):
            fit_interface_calibration_from_response_bank(
                model.encoder,
                dynamics,
                candidates,
                response_bank,
                system,
                interface,
                seal=seal,
                model_name="full",
            )
        candidates[0].context.copy_(original_context)
        with self.assertRaisesRegex(ValueError, "detached"):
            replace(
                response_bank,
                plus_contexts=response_bank.plus_contexts.clone().requires_grad_(True),
            )
        nonfinite_contexts = response_bank.plus_contexts.clone()
        nonfinite_contexts.reshape(-1)[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            replace(response_bank, plus_contexts=nonfinite_contexts)
        full_calibration = fit_interface_calibration_from_response_bank(
            model.encoder,
            dynamics,
            candidates,
            response_bank,
            system,
            interface,
            seal=seal,
            model_name="full",
        )
        calibration = calibrate_activation_interface_after_freeze(
            model,
            candidates,
            response_bank,
            system,
            interface,
            seal=seal,
        )
        self.assertEqual(calibration.environment_steps, 16)
        self.assertEqual(calibration.additional_environment_steps, 0)
        self.assertEqual(calibration.gradient_updates, 0)
        self.assertEqual(counter["steps"], 16)
        self.assertEqual(
            calibration.response_evidence_sha256,
            full_calibration.response_evidence_sha256,
        )
        torch.testing.assert_close(
            calibration.latent_from_interface,
            torch.tensor(interface.native_from_interface),
            atol=2e-5,
            rtol=2e-5,
        )
        heldout_candidates = _vector_candidates(10, offset=100)
        heldout_bank = collect_paired_heldout_response_bank(
            plant,
            heldout_candidates,
            system,
            interface,
            seal=seal,
            states_per_axis=10,
        )
        self.assertEqual(counter["steps"], 56)
        full_heldout = evaluate_heldout_realizability_from_response_bank(
            model.encoder,
            dynamics,
            heldout_candidates,
            system,
            interface,
            full_calibration,
            heldout_bank,
            seal=seal,
        )
        heldout = evaluate_heldout_activation_from_response_bank(
            model,
            heldout_candidates,
            system,
            interface,
            calibration,
            heldout_bank,
            seal=seal,
        )
        self.assertEqual(counter["steps"], 56)
        self.assertEqual(
            heldout.response_evidence_sha256,
            full_heldout.response_evidence_sha256,
        )
        self.assertEqual(heldout.additional_environment_steps, 0)
        self.assertGreater(heldout.mean_cosine, 0.99999)
        self.assertGreater(heldout.magnitude_r2, 0.9999)
        self.assertEqual(heldout.environment_steps, 40)
        self.assertEqual(calibration.neural_hashes_before, calibration.neural_hashes_after)
        heldout_bank.plus_contexts[0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "modified"):
            evaluate_heldout_realizability_from_response_bank(
                model.encoder,
                dynamics,
                heldout_candidates,
                system,
                interface,
                full_calibration,
                heldout_bank,
                seal=seal,
            )


@dataclass
class _Environment:
    frame: torch.Tensor


class _OneStateEncoder(nn.Module):
    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        return contexts[:, -1, :1, :1].reshape(contexts.shape[0], 1).float()


class _OneStateDynamics(nn.Module):
    def port(self, state: torch.Tensor) -> torch.Tensor:
        return torch.ones(state.shape[0], 1, 1, device=state.device)

    def step(self, state: torch.Tensor, effort: torch.Tensor) -> torch.Tensor:
        return state + effort


class _OneStateRenderer(nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.stack((-state[:, 0], state[:, 0]), dim=1)[:, :, None, None]


class _ActivationRollout(nn.Module):
    def forward(self, contexts: torch.Tensor, effort: torch.Tensor) -> torch.Tensor:
        score = effort[..., 0].cumsum(dim=1)
        return torch.stack((-score, score), dim=2)[:, :, :, None, None]


class CEMFairnessTests(unittest.TestCase):
    def test_activation_candidates_are_exactly_microbatched_without_budget_loss(self) -> None:
        class RecordingRollout(_ActivationRollout):
            def __init__(self) -> None:
                super().__init__()
                self.batch_sizes: list[int] = []

            def forward(
                self, contexts: torch.Tensor, effort: torch.Tensor
            ) -> torch.Tensor:
                self.batch_sizes.append(contexts.shape[0])
                return super().forward(contexts, effort)

        rollout = RecordingRollout().eval().requires_grad_(False)
        seal = FrozenEvaluationSeal.capture({"activation": rollout})
        config = CEMConfig(
            horizon=2,
            candidates=70,
            iterations=2,
            elites=7,
            activation_rollout_batch_size=32,
        )
        plan = cem_frozen_world_model_mpc(
            rollout,
            torch.zeros(1, 1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.long),
            torch.ones(1, 1),
            config,
            seal=seal,
            seed=73,
        )
        self.assertEqual(rollout.batch_sizes, [32, 32, 6, 32, 32, 6])
        self.assertEqual(plan.candidate_evaluations, 140)
        unchunked = cem_frozen_world_model_mpc(
            rollout,
            torch.zeros(1, 1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.long),
            torch.ones(1, 1),
            CEMConfig(
                horizon=2,
                candidates=70,
                iterations=2,
                elites=7,
                activation_rollout_batch_size=70,
            ),
            seal=seal,
            seed=73,
        )
        torch.testing.assert_close(
            plan.best_interface_sequence, unchunked.best_interface_sequence
        )
        torch.testing.assert_close(
            plan.elite_mean_sequence, unchunked.elite_mean_sequence
        )
        self.assertAlmostEqual(plan.best_cost, unchunked.best_cost, places=7)

    def test_all_learned_planners_receive_identical_noise_and_budget(self) -> None:
        encoder = _OneStateEncoder().eval().requires_grad_(False)
        renderer = _OneStateRenderer().eval().requires_grad_(False)
        unstructured_encoder = _OneStateEncoder().eval().requires_grad_(False)
        unstructured_renderer = _OneStateRenderer().eval().requires_grad_(False)
        structured = _OneStateDynamics().eval().requires_grad_(False)
        unstructured = _OneStateDynamics().eval().requires_grad_(False)
        activation = _ActivationRollout().eval().requires_grad_(False)
        seal = FrozenEvaluationSeal.capture(
            {
                "encoder": encoder,
                "renderer": renderer,
                "unstructured-encoder": unstructured_encoder,
                "unstructured-renderer": unstructured_renderer,
                "structured": structured,
                "unstructured": unstructured,
                "activation": activation,
            }
        )
        system = EvaluationSystem("toy", 1, 0.1, 1, 2, controlled_pixel_values=(1,))

        def clone(environment: _Environment) -> _Environment:
            return _Environment(environment.frame.clone())

        def step(
            environment: _Environment,
            _interface: PhysicalInterface,
            _command: np.ndarray,
        ) -> None:
            environment.frame = environment.frame.clone()

        plant = PixelPlant(
            clone,
            step,
            lambda context, _environment: context.clone(),
            lambda environment: environment.frame,
        )
        episode = PixelControlEpisode(
            "paired",
            _Environment(torch.zeros(1, 1, dtype=torch.uint8)),
            torch.zeros(1, 1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.uint8),
        )
        config = CEMConfig(horizon=2, candidates=8, iterations=2, elites=2)
        observed_seeds: list[int] = []

        def plan(*_args, **kwargs) -> CEMPlan:
            observed_seeds.append(int(kwargs["seed"]))
            return CEMPlan(
                first_interface_command=torch.zeros(1),
                best_interface_sequence=torch.zeros(config.horizon, 1),
                elite_mean_sequence=torch.zeros(config.horizon, 1),
                best_cost=0.0,
                candidate_evaluations=config.candidates * config.iterations,
                candidates_per_iteration=config.candidates,
                iterations=config.iterations,
                elites=config.elites,
            )

        with patch(
            "blocket_league.direct_physical_evaluation.cem_pixel_target_mpc",
            side_effect=plan,
        ), patch(
            "blocket_league.direct_physical_evaluation.cem_frozen_world_model_mpc",
            side_effect=plan,
        ):
            result = evaluate_closed_loop_controllers(
                [episode],
                system,
                plant,
                PhysicalInterface("native", ((1.0,),)),
                encoder,
                renderer,
                structured,
                unstructured,
                torch.ones(1, 1),
                torch.ones(1, 1),
                unstructured_encoder=unstructured_encoder,
                unstructured_renderer=unstructured_renderer,
                seal=seal,
                cem_config=config,
                activation_rollout=activation,
                activation_calibration=torch.ones(1, 1),
                additional_latent_planners={
                    name: FrozenLatentPlannerSpec(
                        encoder=encoder,
                        renderer=renderer,
                        dynamics=structured,
                        calibration=torch.ones(1, 1),
                    )
                    for name in ("no_jacobian", "shuffled_lens")
                },
                seed=991,
            )
        self.assertEqual(observed_seeds, [991, 991, 991, 991, 991])
        self.assertEqual(
            set(result.errors),
            {
                "structured",
                "unstructured",
                "activation",
                "no_jacobian",
                "shuffled_lens",
                "coast",
                "random",
            },
        )
        self.assertEqual(result.planner_budget["candidateEvaluationsPerDecision"], 16)
        self.assertEqual(result.planner_budget["pairedCandidateNoiseAcrossLearnedPlanners"], 1)


if __name__ == "__main__":
    unittest.main()
