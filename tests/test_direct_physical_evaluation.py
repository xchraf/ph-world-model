from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import math
import unittest

import numpy as np
import torch
from torch import nn

from blocket_league.direct_physical_evaluation import (
    CEMConfig,
    ControlResult,
    DOptimalSelection,
    EvaluationSystem,
    FrozenEvaluationSeal,
    InterfaceExecutionEvidence,
    InterfaceTransferEvidence,
    PAIRED_CALIBRATION_STATES_PER_AXIS,
    PhysicalInterface,
    PixelPlant,
    ProbeCandidate,
    RealizabilityMetrics,
    SYSTEMS,
    adapt_dynamics_for_evaluation,
    builtin_pixel_plant,
    calibrate_interface_after_freeze,
    cem_frozen_world_model_mpc,
    cem_pixel_target_mpc,
    control_gate_metrics,
    evaluate_heldout_realizability,
    evaluation_system_from_direct_spec,
    fixed_interfaces,
    interface_transfer_gate_metrics,
    linear_interface_protocol,
    make_puck_only_pixel_objective,
    paired_bootstrap_ci,
    realizability_gate_metrics,
    registered_cem_config,
    registered_linear_interface_command_bound,
    select_d_optimal_probe_states,
)


class ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4))

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        return contexts[:, -1].float() * self.scale


class ToyDynamics(nn.Module):
    def __init__(self, dt: float = 0.05) -> None:
        super().__init__()
        self.dt = dt
        self.gain = nn.Parameter(torch.tensor(1.0))

    def port(self, state: torch.Tensor) -> torch.Tensor:
        result = state.new_zeros(state.shape[0], 4, 2)
        result[:, 0, 0] = self.gain * (1.0 + 0.08 * state[:, 0])
        result[:, 1, 1] = self.gain * (0.9 + 0.06 * state[:, 1])
        result[:, 2, 0] = self.gain * (0.25 + 0.04 * state[:, 2])
        result[:, 3, 1] = self.gain * (0.35 + 0.03 * state[:, 3])
        return result

    def step(self, state: torch.Tensor, latent_effort: torch.Tensor) -> torch.Tensor:
        return state + self.dt * torch.einsum(
            "bnm,bm->bn", self.port(state), latent_effort
        )


@dataclass
class ToyEnvironment:
    state: np.ndarray


def toy_port(state: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 2), dtype=np.float64)
    result[0, 0] = 1.0 + 0.08 * state[0]
    result[1, 1] = 0.9 + 0.06 * state[1]
    result[2, 0] = 0.25 + 0.04 * state[2]
    result[3, 1] = 0.35 + 0.03 * state[3]
    return result


def make_toy_plant(counter: dict[str, int], dt: float = 0.05) -> PixelPlant:
    def clone(environment: ToyEnvironment) -> ToyEnvironment:
        return ToyEnvironment(environment.state.copy())

    def step(
        environment: ToyEnvironment,
        interface: PhysicalInterface,
        command: np.ndarray,
    ) -> None:
        counter["physical_steps"] += 1
        native = interface.matrix() @ np.asarray(command, dtype=np.float64)
        environment.state = environment.state + dt * toy_port(environment.state) @ native

    def append(context: torch.Tensor, environment: ToyEnvironment) -> torch.Tensor:
        frame = torch.from_numpy(environment.state.astype(np.float32))
        return torch.cat((context[1:], frame[None]), dim=0)

    def current(_environment: ToyEnvironment) -> torch.Tensor:
        return torch.zeros(1, 1, dtype=torch.uint8)

    return PixelPlant(clone, step, append, current)


def make_toy_candidates(count: int, offset: int = 0) -> list[ProbeCandidate]:
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
                identifier=f"pixel-{offset}-{index}",
                context=torch.from_numpy(state)[None],
                environment=ToyEnvironment(state.astype(np.float64)),
            )
        )
    return result


def freeze(*modules: nn.Module) -> FrozenEvaluationSeal:
    mapping = {}
    for index, module in enumerate(modules):
        module.eval().requires_grad_(False)
        mapping[f"module{index}"] = module
    return FrozenEvaluationSeal.capture(mapping)


class CalibrationProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.system = EvaluationSystem(
            name="toy",
            physical_action_size=2,
            dt=0.05,
            control_steps=3,
            planning_horizon=2,
            probe_amplitude=0.25,
        )
        self.encoder = ToyEncoder()
        self.dynamics = ToyDynamics(dt=self.system.dt)
        self.seal = freeze(self.encoder, self.dynamics)
        self.counter = {"physical_steps": 0}
        self.plant = make_toy_plant(self.counter, dt=self.system.dt)
        self.candidates = make_toy_candidates(12)
        self.interface = PhysicalInterface(
            "held-out-interface", ((0.70, -0.20), (0.30, 1.10))
        )

    def test_selection_precedes_responses_and_uses_four_states_per_axis(self) -> None:
        selection = select_d_optimal_probe_states(
            self.encoder,
            self.dynamics,
            self.candidates,
            self.system,
            seal=self.seal,
        )
        self.assertEqual(self.counter["physical_steps"], 0)
        self.assertEqual(selection.observed_response_count, 0)
        self.assertEqual(len(selection.indices_by_axis), 2)
        for indices in selection.indices_by_axis:
            self.assertEqual(len(indices), PAIRED_CALIBRATION_STATES_PER_AXIS)
            self.assertEqual(len(set(indices)), PAIRED_CALIBRATION_STATES_PER_AXIS)

    def test_closed_form_calibration_recovers_one_constant_matrix(self) -> None:
        selection = select_d_optimal_probe_states(
            self.encoder,
            self.dynamics,
            self.candidates,
            self.system,
            seal=self.seal,
        )
        result = calibrate_interface_after_freeze(
            self.encoder,
            self.dynamics,
            self.plant,
            self.candidates,
            selection,
            self.system,
            self.interface,
            seal=self.seal,
        )
        self.assertEqual(result.gradient_updates, 0)
        self.assertEqual(result.paired_states_per_axis, 4)
        self.assertEqual(result.environment_steps, 16)
        self.assertEqual(self.counter["physical_steps"], 16)
        torch.testing.assert_close(
            result.latent_from_interface,
            torch.tensor(self.interface.native_from_interface),
            atol=2e-5,
            rtol=2e-5,
        )
        self.assertLess(result.fit_relative_residual, 2e-5)
        self.assertEqual(result.neural_hashes_before, result.neural_hashes_after)
        self.assertIsNone(self.encoder.scale.grad)
        self.assertIsNone(self.dynamics.gain.grad)

    def test_tampered_query_budget_is_rejected(self) -> None:
        bad = DOptimalSelection(
            indices_by_axis=((0, 1, 2), (0, 1, 2)),
            identifiers_by_axis=(("0", "1", "2"), ("0", "1", "2")),
            log_determinants_by_axis=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            paired_states_per_axis=3,
        )
        with self.assertRaises(ValueError):
            calibrate_interface_after_freeze(
                self.encoder,
                self.dynamics,
                self.plant,
                self.candidates,
                bad,
                self.system,
                self.interface,
                seal=self.seal,
            )
        self.assertEqual(self.counter["physical_steps"], 0)

    def test_heldout_realizability_reports_axis_cosine_sign_and_magnitude(self) -> None:
        selection = select_d_optimal_probe_states(
            self.encoder,
            self.dynamics,
            self.candidates,
            self.system,
            seal=self.seal,
        )
        calibration = calibrate_interface_after_freeze(
            self.encoder,
            self.dynamics,
            self.plant,
            self.candidates,
            selection,
            self.system,
            self.interface,
            seal=self.seal,
        )
        heldout = make_toy_candidates(10, offset=100)
        metrics = evaluate_heldout_realizability(
            self.encoder,
            self.dynamics,
            self.plant,
            heldout,
            self.system,
            self.interface,
            calibration,
            seal=self.seal,
            states_per_axis=10,
        )
        self.assertGreater(metrics.mean_cosine, 0.99999)
        self.assertTrue(all(value > 0.99999 for value in metrics.axis_mean_cosines))
        self.assertEqual(metrics.sign_agreement, 1.0)
        self.assertGreater(metrics.magnitude_r2, 0.9999)
        self.assertEqual(metrics.environment_steps, 40)


class FrozenSealTest(unittest.TestCase):
    def test_seal_rejects_trainable_and_changed_modules(self) -> None:
        module = nn.Linear(2, 2)
        module.eval()
        with self.assertRaises(RuntimeError):
            FrozenEvaluationSeal.capture({"module": module})
        module.requires_grad_(False)
        seal = FrozenEvaluationSeal.capture({"module": module})
        with torch.no_grad():
            module.weight.add_(1.0)
        with self.assertRaises(AssertionError):
            seal.assert_unchanged()

    def test_training_spec_and_integrate_named_baseline_adapters(self) -> None:
        from blocket_league.direct_experiment_training import DirectSystemSpec

        definition = evaluation_system_from_direct_spec(
            DirectSystemSpec("pendulum", state_size=2, port_size=1, dt=0.05)
        )
        self.assertEqual(definition.planning_horizon, 24)

        class IntegrateNamed(nn.Module):
            def port(self, state: torch.Tensor) -> torch.Tensor:
                return torch.ones(state.shape[0], 1, 1)

            def integrate(self, state: torch.Tensor, effort: torch.Tensor) -> torch.Tensor:
                return state + effort

        baseline = IntegrateNamed().eval().requires_grad_(False)
        adapted = adapt_dynamics_for_evaluation(baseline)
        torch.testing.assert_close(
            adapted.step(torch.zeros(2, 1), torch.ones(2, 1)), torch.ones(2, 1)
        )


class OneDimensionalDynamics(nn.Module):
    def port(self, state: torch.Tensor) -> torch.Tensor:
        return torch.ones(state.shape[0], 1, 1, device=state.device)

    def step(self, state: torch.Tensor, latent_effort: torch.Tensor) -> torch.Tensor:
        return state + 0.6 * latent_effort


class GoalRenderer(nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        score = 5.0 * state[:, 0]
        return torch.stack((-score, score), dim=1)[:, :, None, None]


class ActivationRollout(nn.Module):
    def forward(self, contexts: torch.Tensor, latent_sequence: torch.Tensor) -> torch.Tensor:
        initial = contexts[:, -1, 0, 0].float()
        trajectory = initial[:, None] + 0.6 * latent_sequence[:, :, 0].cumsum(dim=1)
        score = 5.0 * trajectory
        return torch.stack((-score, score), dim=2)[:, :, :, None, None]


class PlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dynamics = OneDimensionalDynamics()
        self.renderer = GoalRenderer()
        self.activation = ActivationRollout()
        self.seal = freeze(self.dynamics, self.renderer, self.activation)
        self.config = CEMConfig(horizon=3, candidates=64, iterations=3, elites=8)
        self.target = torch.ones(1, 1, dtype=torch.long)

    def test_registered_cem_budget_is_locked(self) -> None:
        expected_bounds = {
            "pendulum": 0.625,
            "blocket": 0.647863548391737,
        }
        for system, expected_bound in expected_bounds.items():
            config = registered_cem_config(system)
            self.assertEqual(config.candidates, 512)
            self.assertEqual(config.iterations, 4)
            self.assertEqual(config.elites, 64)
            self.assertAlmostEqual(config.action_high, expected_bound, places=12)
            self.assertAlmostEqual(config.action_low, -expected_bound, places=12)

    def test_same_cem_controller_works_for_generic_dynamics(self) -> None:
        plan = cem_pixel_target_mpc(
            self.dynamics,
            self.renderer,
            torch.zeros(1),
            self.target,
            torch.ones(1, 1),
            self.config,
            seal=self.seal,
            seed=7,
        )
        self.assertGreater(float(plan.first_interface_command[0]), 0.0)
        self.assertEqual(plan.candidate_evaluations, 64 * 3)
        self.assertEqual(plan.elites, 8)

    def test_generic_frozen_activation_world_model_planner(self) -> None:
        context = torch.zeros(2, 1, 1, dtype=torch.long)
        plan = cem_frozen_world_model_mpc(
            self.activation,
            context,
            self.target,
            torch.ones(1, 1),
            self.config,
            seal=self.seal,
            seed=11,
        )
        self.assertGreater(float(plan.first_interface_command[0]), 0.0)
        self.assertEqual(plan.candidates_per_iteration, 64)

    def test_structured_and_generic_planners_share_exact_puck_cost(self) -> None:
        source = torch.ones(1, 1, dtype=torch.uint8)
        objective = make_puck_only_pixel_objective(
            source, self.target.to(torch.uint8), (1,)
        )
        structured = cem_pixel_target_mpc(
            self.dynamics,
            self.renderer,
            torch.zeros(1),
            self.target,
            torch.ones(1, 1),
            self.config,
            seal=self.seal,
            seed=19,
            pixel_objective=objective,
        )
        generic = cem_frozen_world_model_mpc(
            self.activation,
            torch.zeros(2, 1, 1, dtype=torch.long),
            self.target,
            torch.ones(1, 1),
            self.config,
            seal=self.seal,
            seed=19,
            pixel_objective=objective,
        )
        torch.testing.assert_close(
            structured.best_interface_sequence,
            generic.best_interface_sequence,
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(structured.best_cost, generic.best_cost)


class InterfaceAndGateTest(unittest.TestCase):
    @staticmethod
    def _transfer_evidence() -> InterfaceTransferEvidence:
        system = SYSTEMS["pendulum"]
        interfaces = fixed_interfaces(system)
        identifiers = tuple(f"pendulum-{index}" for index in range(64))
        common = {
            "training_lineage_sha256": "a" * 64,
            "physical_sha256": "b" * 64,
            "module_hashes_before": {"full": "c" * 64},
            "module_hashes_after": {"full": "c" * 64},
            "controller_graph_sha256": "d" * 64,
            "cem_config": {"registered": {"candidates": 512, "iterations": 4}},
            "episode_seed": 17,
            "planner_seed": 23,
            "episodes": 64,
            "control_steps": 80,
            "controller_names": (
                "structured",
                "unstructured",
                "activation",
                "coast",
                "random",
            ),
            "target_source": "categorical_pixels_only",
            "episode_identifiers": identifiers,
            "episode_set_sha256": "e" * 64,
            "target_set_sha256": "f" * 64,
            "planner_seed_schedule_sha256": "1" * 64,
            "calibration_matrix_schema": {
                name: ("torch.float32", (1, 1))
                for name in ("structured", "unstructured", "activation")
            },
        }
        native = InterfaceExecutionEvidence(
            interface_protocol=linear_interface_protocol(system, interfaces["native"]),
            calibration_matrix_sha256={
                name: "2" * 64
                for name in ("structured", "unstructured", "activation")
            },
            **common,
        )
        unseen = InterfaceExecutionEvidence(
            interface_protocol=linear_interface_protocol(system, interfaces["unseen"]),
            calibration_matrix_sha256={
                name: "3" * 64
                for name in ("structured", "unstructured", "activation")
            },
            **common,
        )
        return InterfaceTransferEvidence(native, unseen)

    def test_fixed_unseen_interfaces_match_preregistration(self) -> None:
        pendulum = fixed_interfaces("pendulum")
        np.testing.assert_allclose(pendulum["native"].matrix(), np.eye(1))
        np.testing.assert_allclose(pendulum["unseen"].matrix(), [[-1.6]])
        blocket = fixed_interfaces("blocket")
        angle = math.radians(37.0)
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
        )
        expected = rotation @ np.asarray(((0.0, 0.65), (-1.40, 0.0)))
        np.testing.assert_allclose(blocket["unseen"].matrix(), expected)
        self.assertGreater(abs(np.linalg.det(blocket["unseen"].matrix())), 0.1)

    def test_common_command_box_never_enters_the_saturated_plant_domain(self) -> None:
        expected_bounds = {
            "pendulum": 0.625,
            "blocket": 0.647863548391737,
        }
        for system_name, expected in expected_bounds.items():
            with self.subTest(system=system_name):
                bound = registered_linear_interface_command_bound(system_name)
                self.assertAlmostEqual(bound, expected, places=12)
                interfaces = fixed_interfaces(system_name)
                dimension = interfaces["native"].matrix().shape[0]
                maximum = 0.0
                for interface in interfaces.values():
                    for raw_corner in np.ndindex(*(2,) * dimension):
                        corner = bound * np.asarray(
                            tuple(-1.0 if value == 0 else 1.0 for value in raw_corner)
                        )
                        maximum = max(
                            maximum,
                            float(np.linalg.norm(interface.matrix() @ corner)),
                        )
                self.assertLessEqual(maximum, 1.0 + 1e-12)
                self.assertAlmostEqual(maximum, 1.0, places=12)
                plant = builtin_pixel_plant(system_name)
                beyond = np.full(dimension, bound + 1e-5, dtype=np.float64)
                with self.assertRaisesRegex(ValueError, "linear-domain box"):
                    plant.step_interface(object(), interfaces["unseen"], beyond)

    def test_paired_bootstrap_and_locked_gate_helpers(self) -> None:
        structured = tuple(0.50 + 0.01 * (index % 3) for index in range(64))
        unstructured = tuple(0.82 + 0.01 * (index % 2) for index in range(64))
        activation = tuple(0.72 + 0.01 * (index % 2) for index in range(64))
        coast = tuple(1.00 + 0.01 * (index % 3) for index in range(64))
        random = tuple(1.10 + 0.01 * (index % 4) for index in range(64))
        interval = paired_bootstrap_ci(
            structured, activation, resamples=1_000, seed=3
        )
        self.assertGreater(interval.low, 0.0)
        control = ControlResult(
            errors={
                "structured": structured,
                "unstructured": unstructured,
                "activation": activation,
                "coast": coast,
                "random": random,
            },
            interface_name="native",
            episodes=64,
            control_steps=48,
            planner_budget={
                "candidatesPerDecision": 512,
                "iterationsPerDecision": 4,
                "elitesPerIteration": 64,
                "horizon": 12,
            },
        )
        gate = control_gate_metrics(
            control,
            no_jacobian_errors=tuple(0.65 for _ in range(64)),
            shuffled_lens_errors=tuple(0.68 for _ in range(64)),
            bootstrap_resamples=1_000,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["betterLearnedBaseline"], "activation")

        realizability = RealizabilityMetrics(
            mean_cosine=0.91,
            axis_mean_cosines=(0.90, 0.92),
            sign_agreement=0.90,
            magnitude_r2=0.70,
            axis_magnitude_r2=(0.68, 0.72),
            response_cosines=(0.90, 0.92),
            response_signs=(True, True),
            actual_magnitudes=(1.0, 1.2),
            predicted_magnitudes=(0.98, 1.18),
            samples_per_axis=128,
            environment_steps=512,
        )
        realization_gate = realizability_gate_metrics(realizability)
        self.assertTrue(realization_gate["passed"])
        transfer = interface_transfer_gate_metrics(
            gate,
            {**gate, "improvementVsCoast": 0.9 * gate["improvementVsCoast"]},
            realization_gate,
            evidence=self._transfer_evidence(),
        )
        self.assertTrue(transfer["passed"])

    def test_gate8_cannot_be_satisfied_by_a_caller_boolean(self) -> None:
        self.assertNotIn(
            "only_constant_calibration_changed",
            inspect.signature(interface_transfer_gate_metrics).parameters,
        )
        evidence = self._transfer_evidence()
        tampered_unseen = replace(
            evidence.unseen,
            cem_config={"registered": {"candidates": 511, "iterations": 4}},
            target_set_sha256="0" * 64,
        )
        gate = interface_transfer_gate_metrics(
            {"improvementVsCoast": 0.50},
            {"improvementVsCoast": 0.45},
            {"passed": True},
            evidence=InterfaceTransferEvidence(evidence.native, tampered_unseen),
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["cemConfigExact"])
        self.assertFalse(gate["checks"]["targetSetExact"])
        self.assertFalse(
            gate["checks"]["onlyPhysicalInterfaceAndConstantTMayDiffer"]
        )

    def test_gate8_rejects_each_non_interface_non_T_difference(self) -> None:
        evidence = self._transfer_evidence()
        changed_identifiers = (
            "changed-episode",
            *evidence.unseen.episode_identifiers[1:],
        )
        cases = {
            "controllerGraphExact": {
                "controller_graph_sha256": "4" * 64,
            },
            "moduleHashesAcrossInterfacesExact": {
                "module_hashes_before": {"full": "5" * 64},
                "module_hashes_after": {"full": "5" * 64},
            },
            "episodeSeedExact": {"episode_seed": 18},
            "plannerSeedExact": {"planner_seed": 24},
            "episodeIdentifiersExact": {
                "episode_identifiers": changed_identifiers,
            },
            "episodeSetExact": {"episode_set_sha256": "6" * 64},
            "targetSetExact": {"target_set_sha256": "7" * 64},
            "plannerSeedScheduleExact": {
                "planner_seed_schedule_sha256": "8" * 64,
            },
            "constantCalibrationTSlotsExact": {
                "calibration_matrix_schema": {
                    name: ("torch.float32", (1, 1 if name != "activation" else 2))
                    for name in ("structured", "unstructured", "activation")
                },
            },
        }
        for failed_check, changes in cases.items():
            with self.subTest(check=failed_check):
                unseen = replace(evidence.unseen, **changes)
                gate = interface_transfer_gate_metrics(
                    {"improvementVsCoast": 0.50},
                    {"improvementVsCoast": 0.45},
                    {"passed": True},
                    evidence=InterfaceTransferEvidence(evidence.native, unseen),
                )
                self.assertFalse(gate["passed"])
                self.assertFalse(gate["checks"][failed_check])

    def test_gate7_audits_each_learned_baseline_without_post_selection(self) -> None:
        errors = {
            "structured": tuple(0.50 for _ in range(64)),
            "unstructured": tuple(0.82 for _ in range(64)),
            "activation": tuple(0.75 for _ in range(64)),
            # A future independent WM cannot be hidden by selecting one of the
            # easier baselines after seeing their empirical means.
            "independent_wm": tuple(0.54 for _ in range(64)),
            "coast": tuple(1.00 for _ in range(64)),
            "random": tuple(1.10 for _ in range(64)),
        }
        result = ControlResult(
            errors=errors,
            interface_name="native",
            episodes=64,
            control_steps=48,
            planner_budget={"candidatesPerDecision": 512, "iterationsPerDecision": 4},
        )
        gate = control_gate_metrics(result, bootstrap_resamples=1_000)
        self.assertFalse(gate["passed"])
        self.assertIn("independent_wm", gate["learnedBaselineComparisons"])
        self.assertFalse(
            gate["learnedBaselineComparisons"]["independent_wm"]["checks"][
                "improvementAtLeast0.15"
            ]
        )


if __name__ == "__main__":
    unittest.main()
