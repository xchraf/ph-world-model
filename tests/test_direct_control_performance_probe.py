from __future__ import annotations

import inspect
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import torch

from blocket_league.direct_control_performance_probe import (
    REGISTERED_TIMING_SHARD_EPISODES,
    _assert_exact_control_shard,
    _time_exact_call,
    run_control_performance_probe,
)
from blocket_league.direct_physical_evaluation import (
    SYSTEMS,
    fixed_interfaces,
    linear_interface_protocol,
    registered_cem_config,
)


class ExactCEMPerformanceProbeTests(unittest.TestCase):
    def test_timer_accepts_only_the_registered_512_by_4_budget(self) -> None:
        exact = SimpleNamespace(
            candidate_evaluations=2048,
            candidates_per_iteration=512,
            iterations=4,
            elites=64,
        )
        self.assertGreaterEqual(
            _time_exact_call(torch.device("cpu"), lambda: exact), 0.0
        )
        changed = SimpleNamespace(
            candidate_evaluations=1024,
            candidates_per_iteration=256,
            iterations=4,
            elites=64,
        )
        with self.assertRaisesRegex(AssertionError, "exact CEM budget"):
            _time_exact_call(torch.device("cpu"), lambda: changed)

    def test_nonregistered_shape_cannot_be_reported_as_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "nonregistered timing shapes"):
                run_control_performance_probe(
                    Path(temporary) / "result.json",
                    system_name="pendulum",
                    device=torch.device("cpu"),
                    image_size=8,
                    patch_size=4,
                    backbone_preset="nano",
                    implicit_iterations=4,
                )

    def test_exact_shard_requires_all_seven_registered_controllers(self) -> None:
        system = SYSTEMS["blocket"]
        config = registered_cem_config(system)
        protocol = linear_interface_protocol(system, fixed_interfaces(system)["native"])
        names = (
            "structured",
            "unstructured",
            "activation",
            "no_jacobian",
            "shuffled_lens",
            "coast",
            "random",
        )
        traces = {
            name: tuple(
                tuple((0.0, 0.0) for _ in range(system.control_steps))
                for _ in range(REGISTERED_TIMING_SHARD_EPISODES)
            )
            for name in names
        }
        exact = SimpleNamespace(
            episodes=REGISTERED_TIMING_SHARD_EPISODES,
            control_steps=system.control_steps,
            errors={name: (0.0,) * REGISTERED_TIMING_SHARD_EPISODES for name in names},
            interface_command_traces=traces,
            planner_budget={
                "candidatesPerDecision": config.candidates,
                "iterationsPerDecision": config.iterations,
                "elitesPerIteration": config.elites,
                "horizon": config.horizon,
                "candidateEvaluationsPerDecision": (
                    config.candidates * config.iterations
                ),
                "pairedCandidateNoiseAcrossLearnedPlanners": 1,
                "activationRolloutMicroBatch": config.activation_rollout_batch_size,
                "commonLinearInterfaceCommandBound": (
                    protocol.common_interface_command_bound
                ),
                "linearInterfaceBoundFormula": protocol.bound_formula,
            },
        )
        _assert_exact_control_shard(exact, system_name="blocket")
        missing_independent = SimpleNamespace(
            **{
                **exact.__dict__,
                "errors": {
                    name: values
                    for name, values in exact.errors.items()
                    if name != "unstructured"
                },
            }
        )
        with self.assertRaisesRegex(AssertionError, "exact control shard"):
            _assert_exact_control_shard(
                missing_independent, system_name="blocket"
            )

    def test_probe_routes_the_independent_visual_world_model_and_real_plant(self) -> None:
        source = inspect.getsource(run_control_performance_probe)
        for required in (
            "build_fresh_independent_baseline",
            "unstructured_encoder=independent.model.encoder",
            "unstructured_renderer=independent.model.renderer",
            "adapt_dynamics_for_evaluation(independent.model.dynamics)",
            "full.probes",
            "make_builtin_control_episodes",
            "evaluate_closed_loop_controllers",
            "_replay_control_traces",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
