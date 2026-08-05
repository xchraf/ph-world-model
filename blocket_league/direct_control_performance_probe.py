"""Timing-only execution of one exact four-episode Experiment-F control shard.

The probe builds frozen, randomly initialized modules at every registered
shape and executes the real closed-loop path: CEM replan, plant step, render,
fresh context, and authenticated-trace replay.  It opens no experiment archive
and discards all random-weight outcomes.  Its JSON is never scientific
evidence; it exists only to estimate post-freeze wall time and memory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch

from .direct_cotangent_bridge import PixelChangeProbeBank
from .direct_experiment_training import (
    DIRECT_SYSTEMS,
    DirectTrainingConfig,
    build_direct_bundle,
    seed_everything,
)
from .direct_jacobian_port_extractor import (
    EmpiricalTangentConfig,
    make_synthetic_empirical_tangent_artifact_for_tests,
)
from .direct_physical_evaluation import (
    ControlResult,
    FrozenActivationWriteWorldModel,
    FrozenEvaluationSeal,
    FrozenLatentPlannerSpec,
    PhysicalInterface,
    PixelControlEpisode,
    PixelPlant,
    SYSTEMS,
    adapt_dynamics_for_evaluation,
    builtin_pixel_plant,
    evaluate_closed_loop_controllers,
    fixed_interfaces,
    linear_interface_protocol,
    make_builtin_control_episodes,
    pixel_target_error,
    registered_cem_config,
)
from .direct_unstructured_training import build_fresh_independent_baseline
from .direct_unstructured_world_model import freeze_independent_bundle
from .env import PALETTE
from .pixel_direct_model import DirectPixelTransformer, pixel_direct_config_for_preset


REGISTERED_TIMING_SHARD_EPISODES = 4
_REGISTERED_RESIDENT_VARIANTS = (
    "full",
    "no_jacobian",
    "single_horizon",
    "shuffled_lens",
    "skew_only",
    "constant_port",
)
_REGISTERED_CONTROL_NAMES = (
    "structured",
    "unstructured",
    "activation",
    "no_jacobian",
    "shuffled_lens",
    "coast",
    "random",
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_exact_call(device: torch.device, call: Callable[[], Any]) -> float:
    _synchronize(device)
    started = time.perf_counter()
    value = call()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if (
        value.candidate_evaluations != 512 * 4
        or value.candidates_per_iteration != 512
        or value.iterations != 4
        or value.elites != 64
    ):
        raise AssertionError("performance probe changed the exact CEM budget")
    return elapsed


def _assert_exact_control_shard(
    result: ControlResult,
    *,
    system_name: str,
) -> None:
    system = SYSTEMS[system_name]
    config = registered_cem_config(system)
    protocol = linear_interface_protocol(system, fixed_interfaces(system)["native"])
    expected_budget = {
        "candidatesPerDecision": config.candidates,
        "iterationsPerDecision": config.iterations,
        "elitesPerIteration": config.elites,
        "horizon": config.horizon,
        "candidateEvaluationsPerDecision": config.candidates * config.iterations,
        "pairedCandidateNoiseAcrossLearnedPlanners": 1,
        "activationRolloutMicroBatch": config.activation_rollout_batch_size,
        "commonLinearInterfaceCommandBound": (
            protocol.common_interface_command_bound
        ),
        "linearInterfaceBoundFormula": protocol.bound_formula,
    }
    if (
        result.episodes != REGISTERED_TIMING_SHARD_EPISODES
        or result.control_steps != system.control_steps
        or tuple(result.errors) != _REGISTERED_CONTROL_NAMES
        or tuple(result.interface_command_traces) != _REGISTERED_CONTROL_NAMES
        or dict(result.planner_budget) != expected_budget
        or any(
            len(episode_traces) != REGISTERED_TIMING_SHARD_EPISODES
            or any(len(trace) != system.control_steps for trace in episode_traces)
            for episode_traces in result.interface_command_traces.values()
        )
    ):
        raise AssertionError("performance probe did not execute the exact control shard")


def _replay_control_traces(
    result: ControlResult,
    episodes: list[PixelControlEpisode],
    system_name: str,
    plant: PixelPlant,
    interface: PhysicalInterface,
) -> int:
    """Replay every raw command exactly as the post-freeze finalizer does."""

    system = SYSTEMS[system_name]
    physical_steps = 0
    for controller, traces in result.interface_command_traces.items():
        replayed_errors: list[float] = []
        for episode, commands in zip(episodes, traces, strict=True):
            environment = plant.clone_environment(episode.environment)
            context = episode.context.clone()
            for command in commands:
                value = np.asarray(command, dtype=np.float32)
                plant.step_interface(environment, interface, value)
                context = plant.append_observation(context, environment)
                physical_steps += 1
            replayed_errors.append(
                pixel_target_error(
                    system,
                    plant.current_pixels(environment),
                    episode.target_pixels,
                )
            )
        expected = result.errors[controller]
        if len(expected) != len(replayed_errors) or any(
            not math.isclose(first, second, rel_tol=0.0, abs_tol=1e-8)
            for first, second in zip(replayed_errors, expected, strict=True)
        ):
            raise AssertionError(f"timing-only trace replay changed {controller!r}")
    return physical_steps


def _cuda_memory_snapshot(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "allocatedBytes": int(torch.cuda.memory_allocated(device)),
        "reservedBytes": int(torch.cuda.memory_reserved(device)),
        "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
        "peakReservedBytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run_control_performance_probe(
    output_path: Path,
    *,
    system_name: str,
    device: torch.device,
    image_size: int = 64,
    patch_size: int = 4,
    backbone_preset: str = "tiny",
    implicit_iterations: int = 32,
    float32_matmul_precision: str = "highest",
    episodes_per_shard: int = REGISTERED_TIMING_SHARD_EPISODES,
    allow_nonregistered_smoke: bool = False,
) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError("control performance output must not already exist")
    if system_name not in DIRECT_SYSTEMS:
        raise ValueError("unknown registered control system")
    if (
        type(episodes_per_shard) is not int
        or episodes_per_shard < 1
        or episodes_per_shard > 64
    ):
        raise ValueError("episodes_per_shard must be an integer in [1, 64]")
    if float32_matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError("unknown float32 matmul precision")
    registered_shape_exact = (
        image_size == 64
        and patch_size == 4
        and backbone_preset == "tiny"
        and implicit_iterations == 32
        and float32_matmul_precision == "highest"
        and episodes_per_shard == REGISTERED_TIMING_SHARD_EPISODES
    )
    if not registered_shape_exact and not allow_nonregistered_smoke:
        raise ValueError(
            "nonregistered timing shapes require --allow-nonregistered-smoke"
        )
    torch.set_float32_matmul_precision(float32_matmul_precision)
    seed = 151_910_737 + 992_003
    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    direct_system = DIRECT_SYSTEMS[system_name]
    evaluation_system = SYSTEMS[system_name]
    config = registered_cem_config(system_name)
    if (config.candidates, config.iterations, config.elites) != (512, 4, 64):
        raise AssertionError("registered CEM budget drifted")
    if config.horizon != evaluation_system.planning_horizon:
        raise AssertionError("registered planning horizon drifted")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    preparation_started = time.perf_counter()
    model_config = pixel_direct_config_for_preset(
        backbone_preset,
        image_size=image_size,
        patch_size=patch_size,
        palette_size=len(PALETTE),
        history_frames=8,
    )
    backbone = DirectPixelTransformer(model_config).to(device)
    frames = torch.randint(
        0,
        model_config.palette_size,
        (4, 9, image_size, image_size),
        generator=generator,
        dtype=torch.uint8,
    )
    probes = PixelChangeProbeBank.from_pixel_frames(
        frames,
        palette_size=model_config.palette_size,
        probe_size=direct_system.port_size,
    )
    train_config = DirectTrainingConfig(implicit_iterations=implicit_iterations)
    tangent_config = EmpiricalTangentConfig(
        channel_rank=train_config.port_tangent_channel_rank,
        neighbors=train_config.port_tangent_neighbors,
        support_floor_ratio=train_config.port_support_floor_ratio,
    )
    empirical_tangent = make_synthetic_empirical_tangent_artifact_for_tests(
        history_frames=model_config.history_frames,
        patch_count=model_config.grid_size**2,
        hidden_size=model_config.hidden_size,
        config=tangent_config,
        seed=seed + 41,
    )
    resident_bundles = {}
    for variant in _REGISTERED_RESIDENT_VARIANTS:
        # Experiment F uses matched initial seeds across variants.  Values are
        # random here, but the resident graph and storage footprint are exact.
        seed_everything(train_config.seed)
        resident_bundles[variant] = build_direct_bundle(
            backbone,
            direct_system,
            probes,
            train_config,
            device,
            empirical_tangent=empirical_tangent,
            variant=variant,
        )
    for resident in resident_bundles.values():
        for module in (
            resident.model,
            resident.write_field,
            resident.lens,
            resident.probes,
            resident.response_frame,
            resident.cotangent_frame,
        ):
            module.eval().requires_grad_(False)

    independent = build_fresh_independent_baseline(
        backbone,
        direct_system,
        probes,
        train_config,
        device,
        empirical_tangent=empirical_tangent,
        reference_initialization_seed=train_config.seed,
    )
    freeze_independent_bundle(independent)
    full = resident_bundles["full"]
    activation = FrozenActivationWriteWorldModel(
        full.model.encoder, full.write_field, full.lens, full.probes
    )
    activation.eval().requires_grad_(False)
    seal = FrozenEvaluationSeal.capture(
        {
            **{
                f"direct-{name}": resident.model
                for name, resident in resident_bundles.items()
            },
            "independentUnstructuredWorldModel": independent.model,
            "activationWorldModel": activation,
        }
    )
    all_episodes = make_builtin_control_episodes(
        evaluation_system,
        history_frames=model_config.history_frames,
        count=64,
        seed=151_910_737 + 80_000,
        image_size=image_size,
    )
    episodes = all_episodes[:episodes_per_shard]
    transform = torch.eye(
        direct_system.port_size,
        device=device,
        dtype=torch.float32,
    )
    plant = builtin_pixel_plant(evaluation_system)
    interface = fixed_interfaces(evaluation_system)["native"]
    seal.assert_unchanged()
    _synchronize(device)
    preparation_seconds = time.perf_counter() - preparation_started
    preparation_memory = _cuda_memory_snapshot(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    control_started = time.perf_counter()
    control_result = evaluate_closed_loop_controllers(
        episodes,
        evaluation_system,
        plant,
        interface,
        full.model.encoder,
        full.model.renderer,
        full.model.core,
        adapt_dynamics_for_evaluation(independent.model.dynamics),
        transform,
        transform.clone(),
        unstructured_encoder=independent.model.encoder,
        unstructured_renderer=independent.model.renderer,
        seal=seal,
        activation_rollout=activation,
        activation_calibration=transform.clone(),
        additional_latent_planners={
            name: FrozenLatentPlannerSpec(
                encoder=resident_bundles[name].model.encoder,
                renderer=resident_bundles[name].model.renderer,
                dynamics=resident_bundles[name].model.core,
                calibration=transform.clone(),
            )
            for name in ("no_jacobian", "shuffled_lens")
        },
        seed=151_910_737 + 90_000,
    )
    _synchronize(device)
    control_seconds = time.perf_counter() - control_started
    control_memory = _cuda_memory_snapshot(device)
    _assert_exact_control_shard(control_result, system_name=system_name)
    seal.assert_unchanged()
    activation.assert_frozen_and_unchanged()
    independent.model.encoder.assert_backbone_frozen()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    replay_started = time.perf_counter()
    replayed_physical_steps = _replay_control_traces(
        control_result,
        episodes,
        system_name,
        plant,
        interface,
    )
    _synchronize(device)
    replay_seconds = time.perf_counter() - replay_started
    replay_memory = _cuda_memory_snapshot(device)

    registered_shards_per_interface = 64 // REGISTERED_TIMING_SHARD_EPISODES
    registered_two_interface_shards = 2 * registered_shards_per_interface
    measured_shard_task_seconds = preparation_seconds + control_seconds
    estimated_two_interface_control_seconds = (
        registered_two_interface_shards * measured_shard_task_seconds
    )
    estimated_two_interface_replay_seconds = (
        registered_two_interface_shards * replay_seconds
    )
    estimated_two_interface_seconds_excluding_calibration = (
        estimated_two_interface_control_seconds
        + estimated_two_interface_replay_seconds
    )
    hypothetical_128_shards_seconds = 128.0 * (
        measured_shard_task_seconds + replay_seconds
    )
    learned_controllers = 5
    cem_candidate_sequence_evaluations = (
        episodes_per_shard
        * evaluation_system.control_steps
        * learned_controllers
        * config.candidates
        * config.iterations
    )
    result: dict[str, Any] = {
        "kind": (
            "synthetic_exact_full_control_shard_timing_not_scientific_evidence"
            if registered_shape_exact
            else "synthetic_nonregistered_full_control_smoke_not_scientific_evidence"
        ),
        "system": system_name,
        "registeredShapeExact": registered_shape_exact,
        "registeredFullShardExact": registered_shape_exact,
        "modelConfig": asdict(model_config),
        "shapeCriticalConfig": {
            "implicitIterations": implicit_iterations,
            "planningHorizon": config.horizon,
            "candidatesPerDecision": config.candidates,
            "iterationsPerDecision": config.iterations,
            "elitesPerIteration": config.elites,
            "candidateEvaluationsPerDecision": config.candidates * config.iterations,
            "activationRolloutMicroBatch": config.activation_rollout_batch_size,
            "float32MatmulPrecision": float32_matmul_precision,
            "episodesPerShard": episodes_per_shard,
            "controlSteps": evaluation_system.control_steps,
        },
        "exactControllerOrder": tuple(control_result.errors),
        "exactMeasuredWork": {
            "learnedControllers": learned_controllers,
            "allControllers": len(control_result.errors),
            "recedingHorizonDecisions": (
                episodes_per_shard
                * evaluation_system.control_steps
                * len(control_result.errors)
            ),
            "learnedCEMDecisions": (
                episodes_per_shard
                * evaluation_system.control_steps
                * learned_controllers
            ),
            "candidateSequenceEvaluations": (
                cem_candidate_sequence_evaluations
            ),
            "plannedModelTransitionEvaluations": (
                cem_candidate_sequence_evaluations * config.horizon
            ),
            "plantStepsDuringControl": (
                episodes_per_shard
                * evaluation_system.control_steps
                * len(control_result.errors)
            ),
            "plantStepsDuringReplay": replayed_physical_steps,
        },
        "plannerMultiplicityPerDecision": {
            "structuredPHArchitecture": 3,
            "independentUnstructuredWorldModel": 1,
            "genericFrozenActivationWM": 1,
        },
        "measuredSeconds": {
            "syntheticModuleAndEpisodePreparation": preparation_seconds,
            "exactFourEpisodeClosedLoopShard": control_seconds,
            "perEpisodeWithinMeasuredShard": (
                control_seconds / episodes_per_shard
            ),
            "exactTraceReplayForMeasuredShard": replay_seconds,
            "measuredShardTaskIncludingSyntheticPreparation": (
                measured_shard_task_seconds
            ),
        },
        "memory": {
            "preparation": preparation_memory,
            "control": control_memory,
            "traceReplay": replay_memory,
        },
        "estimatedFourEpisodeShardSeconds": (
            measured_shard_task_seconds if registered_shape_exact else None
        ),
        "estimatedTwoInterfaceControlSeconds": (
            estimated_two_interface_control_seconds
            if registered_shape_exact
            else None
        ),
        "estimatedTwoInterfaceReplaySeconds": (
            estimated_two_interface_replay_seconds
            if registered_shape_exact
            else None
        ),
        "estimatedTwoInterfaceTotalSecondsExcludingCalibration": (
            estimated_two_interface_seconds_excluding_calibration
            if registered_shape_exact
            else None
        ),
        "estimatedTwoInterfaceTotalDaysExcludingCalibration": (
            estimated_two_interface_seconds_excluding_calibration / 86_400.0
            if registered_shape_exact
            else None
        ),
        "registeredShardAccounting": {
            "episodesPerSystem": 64,
            "interfacesPerSystem": 2,
            "fourEpisodeShardsPerInterface": registered_shards_per_interface,
            "fourEpisodeShardsPerSystemAcrossTwoInterfaces": (
                registered_two_interface_shards
            ),
            "episodeInterfaceUnitsPerSystem": 128,
            "fourEpisodeShardsAcrossBothSystems": 64,
        },
        "hypothetical128FourEpisodeShardsSeconds": (
            hypothetical_128_shards_seconds if registered_shape_exact else None
        ),
        "hypothetical128FourEpisodeShardsDays": (
            hypothetical_128_shards_seconds / 86_400.0
            if registered_shape_exact
            else None
        ),
        "estimateScope": (
            "one real native-interface shard plus exact trace replay; unseen uses "
            "the identical graph/work count. Checkpoint I/O, physical calibration, "
            "64-shard merge, bootstrap, and final Gate tables remain excluded"
            if registered_shape_exact
            else "disabled because this is an explicitly nonregistered smoke shape"
        ),
        "targetTask": control_result.target_task,
        "controlledPixelValues": evaluation_system.controlled_pixel_values,
        "calibrationTimingMeasured": False,
        "calibrationEvidenceAdmissible": False,
        "scientificEvidenceAdmissible": False,
        "openedExperimentData": False,
        "openedSimulator": True,
        "randomUntrainedWeights": True,
        "controlOutcomesPersisted": False,
        "gradientUpdates": 0,
    }
    if device.type == "cuda":
        result["deviceName"] = torch.cuda.get_device_name(device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--system", choices=tuple(DIRECT_SYSTEMS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--backbone-preset", default="tiny")
    parser.add_argument("--implicit-iterations", type=int, default=32)
    parser.add_argument(
        "--episodes-per-shard",
        type=int,
        default=REGISTERED_TIMING_SHARD_EPISODES,
    )
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    parser.add_argument("--allow-nonregistered-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_control_performance_probe(
        args.output,
        system_name=args.system,
        device=torch.device(args.device),
        image_size=args.image_size,
        patch_size=args.patch_size,
        backbone_preset=args.backbone_preset,
        implicit_iterations=args.implicit_iterations,
        float32_matmul_precision=args.float32_matmul_precision,
        episodes_per_shard=args.episodes_per_shard,
        allow_nonregistered_smoke=args.allow_nonregistered_smoke,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["REGISTERED_TIMING_SHARD_EPISODES", "run_control_performance_probe"]
