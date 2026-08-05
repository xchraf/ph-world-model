"""Complete typed Gates 1--9 orchestration and fail-closed verification.

This CLI never accepts a caller-supplied gate boolean.  Its distributed stages
collect structural/shared-physical evidence once, run one exact four-episode
control shard per worker, and derive the final result by re-auditing cached raw
evidence and replaying command traces without rerunning CEM.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .direct_physical_evaluation import (
    InterfaceExecutionEvidence,
    InterfaceTransferEvidence,
    SYSTEMS,
    builtin_pixel_plant,
    control_gate_metrics,
    fixed_interfaces,
    interface_transfer_gate_metrics,
    linear_interface_protocol,
    make_builtin_control_episodes,
    pixel_target_error,
    registered_cem_config,
)
from .direct_postfreeze_evidence_io import (
    AuthenticatedControlShard,
    REGISTERED_CONTROL_EPISODE_SEED,
    REGISTERED_CONTROL_PLANNER_SEED,
    build_authenticated_control_shard,
    canonical_evidence_sha256,
    load_control_shard,
    load_physical_result,
    physical_result_sha256,
    save_control_shard,
    save_physical_result,
    training_lineage_sha256,
)
from .direct_postfreeze_force_port import run_gate5_postfreeze
from .direct_postfreeze_energy_semantics import (
    audit_energy_semantics,
    collect_energy_semantic_evidence,
)
from .direct_postfreeze_runner import (
    ControlShard,
    PostFreezePaths,
    assemble_gate1_evidence,
    audit_gate1_postfreeze,
    audit_gate4_postfreeze,
    load_postfreeze_system,
    merge_control_shards,
    physical_gate6_table,
    registered_control_shard_ranges,
    run_control_shard,
    run_physical_realizability,
)
from .direct_postfreeze_structural import (
    collect_gate2_postfreeze,
    collect_gate3_and_rk2_postfreeze,
)
from .direct_postfreeze_staging import (
    audit_prepared_structural_gates,
    build_prepared_system_artifact,
    load_prepared_system_artifact,
    save_prepared_system_artifact,
)


REQUIRED_GATES = tuple(range(1, 10))
REQUIRED_SYSTEMS = ("pendulum", "blocket")
FINAL_KIND = "experiment_f_direct_jacobian_ph_complete_v1"


def _json_native(value: Any) -> Any:
    """Return the exact JSON representation used on disk.

    Canonical evidence distinguishes tuples from lists.  Hashing a Python
    tuple-rich result and only then serializing it to JSON would therefore
    create a file that fails verification after JSON converts tuples to lists.
    """

    return json.loads(json.dumps(value, allow_nan=False))


def _parse_required_gates(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("gate list must contain integers") from error
    if result != REQUIRED_GATES:
        raise argparse.ArgumentTypeError("complete CLI requires exactly Gates 1--9")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _final_artifact(systems: Mapping[str, Any]) -> dict[str, Any]:
    core = _json_native(_final_core(systems))
    return {**core, "artifactSha256": canonical_evidence_sha256(core)}


def _verify_final_artifact_envelope(supplied: Any) -> dict[str, Any]:
    if type(supplied) is not dict or set(supplied) != {
        "kind", "requiredGates", "systems", "gate9", "passed", "outcome", "artifactSha256"
    }:
        raise ValueError("final outcome schema is not exact")
    core = {name: supplied[name] for name in supplied if name != "artifactSha256"}
    if (
        supplied["kind"] != FINAL_KIND
        or supplied["requiredGates"] != list(REQUIRED_GATES)
        or canonical_evidence_sha256(core) != supplied["artifactSha256"]
    ):
        raise ValueError("final outcome digest/provenance is invalid")
    return core


def _schedule_sha256(seed: int, start: int, episodes: int, steps: int) -> str:
    schedule = [
        seed + episode * 100_003 + decision
        for episode in range(start, start + episodes)
        for decision in range(steps)
    ]
    digest = hashlib.sha256()
    digest.update(str((seed, start, episodes, steps)).encode("ascii"))
    digest.update(np.asarray(schedule, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _control_episode_hashes(system: Any, episodes: Sequence[Any]) -> tuple[str, str]:
    """Hash controller initial conditions and targets, including hidden state."""

    episode_payload: list[dict[str, Any]] = []
    target_payload: list[dict[str, Any]] = []
    for episode in episodes:
        environment = episode.environment
        state = environment.state
        if system.name == "pendulum":
            environment_state = {
                "kind": "pendulum",
                "stateVector": torch.as_tensor(state.vector()).detach().cpu(),
                "tick": int(state.tick),
                "config": asdict(environment.config),
            }
        elif system.name == "blocket":
            environment_state = {
                "kind": "blocket",
                "stateVector": torch.as_tensor(state.vector()).detach().cpu(),
                "tick": int(state.tick),
                "resetTimer": int(state.reset_timer),
                "lastEvent": str(state.last_event),
                "config": asdict(environment.config),
            }
        else:  # pragma: no cover - registered systems are closed above
            raise KeyError(f"cannot fingerprint control environment {system.name!r}")
        target = episode.target_pixels.detach().cpu()
        episode_payload.append(
            {
                "identifier": episode.identifier,
                "environment": environment_state,
                "context": episode.context.detach().cpu(),
                "targetPixels": target,
            }
        )
        target_payload.append(
            {"identifier": episode.identifier, "targetPixels": target}
        )
    return (
        canonical_evidence_sha256(
            {"kind": "experiment_f_control_episode_set_v1", "episodes": episode_payload}
        ),
        canonical_evidence_sha256(
            {"kind": "experiment_f_control_target_set_v1", "targets": target_payload}
        ),
    )


def _derive_interface_execution_evidence(
    loaded: Any,
    physical: Any,
    records: Sequence[AuthenticatedControlShard],
    merged: Any,
) -> InterfaceExecutionEvidence:
    """Derive Gate-8 static-graph evidence from authenticated replayed shards."""

    if not records or merged.physical_protocol is None:
        raise ValueError("Gate 8 needs authenticated replayable control records")
    episode_seeds = {record.episode_seed for record in records}
    planner_seeds = {record.planner_seed for record in records}
    if len(episode_seeds) != 1 or len(planner_seeds) != 1:
        raise ValueError("Gate 8 control seeds differ across shards")
    episode_seed = next(iter(episode_seeds))
    planner_seed = next(iter(planner_seeds))
    system = SYSTEMS[loaded.system_name]
    episodes = make_builtin_control_episodes(
        system,
        history_frames=loaded.backbone.config.history_frames,
        count=64,
        seed=episode_seed,
        image_size=loaded.backbone.config.image_size,
    )
    identifiers = tuple(episode.identifier for episode in episodes)
    if identifiers != merged.episode_identifiers:
        raise ValueError("Gate 8 regenerated episode ordering differs from control evidence")
    episode_sha256, target_sha256 = _control_episode_hashes(system, episodes)

    interface_result = physical.interfaces.get(merged.interface_name)
    if interface_result is None:
        raise ValueError("Gate 8 lacks the interface calibration table")
    controller_names = tuple(records[0].result.errors)
    if any(tuple(record.result.errors) != controller_names for record in records):
        raise ValueError("Gate 8 controller graph differs across shards")
    calibration_names = {
        controller: ("full" if controller == "structured" else controller)
        for controller in controller_names
        if controller not in {"coast", "random"}
    }
    calibration_hashes: dict[str, str] = {}
    calibration_schema: dict[str, tuple[str, tuple[int, int]]] = {}
    for controller, calibration_name in calibration_names.items():
        model_result = interface_result.models.get(calibration_name)
        if model_result is None:
            raise ValueError(f"Gate 8 lacks calibration T for {controller!r}")
        calibration = model_result.calibration
        matrix = calibration.latent_from_interface.detach().cpu().contiguous()
        if (
            matrix.requires_grad
            or matrix.grad_fn is not None
            or matrix.ndim != 2
            or calibration.gradient_updates != 0
        ):
            raise ValueError("Gate 8 calibration T is not a frozen constant matrix")
        calibration_hashes[controller] = canonical_evidence_sha256(
            {"kind": "experiment_f_constant_calibration_T_v1", "matrix": matrix}
        )
        calibration_schema[controller] = (str(matrix.dtype), tuple(matrix.shape))

    first = records[0]
    module_before = dict(first.neural_hashes_before)
    module_after = dict(first.neural_hashes_after)
    if any(
        dict(record.neural_hashes_before) != module_before
        or dict(record.neural_hashes_after) != module_after
        for record in records
    ):
        raise ValueError("Gate 8 neural graph hashes differ across shards")
    graph_sha256 = canonical_evidence_sha256(
        {
            "kind": "experiment_f_registered_control_graph_v1",
            "sourceTreeSha256": loaded.source_tree_sha256,
            "moduleRolesAndHashes": module_before,
            "controllerOrder": controller_names,
            "calibrationTSlots": calibration_names,
            "entryPoint": "direct_postfreeze_runner.run_control_shard",
            "freshPixelReencodingEachDecision": True,
        }
    )
    cem = registered_cem_config(system)
    cem_config = {
        "registered": asdict(cem),
        "reportedPlannerBudget": dict(merged.planner_budget),
    }
    if merged.planner_seed_schedule_sha256 is None:
        raise ValueError("Gate 8 control merge lacks its planner-seed schedule seal")
    return InterfaceExecutionEvidence(
        interface_protocol=merged.physical_protocol,
        training_lineage_sha256=first.training_lineage_sha256,
        physical_sha256=first.physical_sha256,
        module_hashes_before=module_before,
        module_hashes_after=module_after,
        controller_graph_sha256=graph_sha256,
        cem_config=cem_config,
        episode_seed=episode_seed,
        planner_seed=planner_seed,
        episodes=merged.episodes,
        control_steps=merged.control_steps,
        controller_names=controller_names,
        target_source=merged.target_source,
        episode_identifiers=identifiers,
        episode_set_sha256=episode_sha256,
        target_set_sha256=target_sha256,
        planner_seed_schedule_sha256=merged.planner_seed_schedule_sha256,
        calibration_matrix_sha256=calibration_hashes,
        calibration_matrix_schema=calibration_schema,
    )


def _verify_and_replay_control_shard(
    record: AuthenticatedControlShard,
    loaded: Any,
    physical: Any,
) -> None:
    if (
        record.training_lineage_sha256 != training_lineage_sha256(loaded)
        or record.physical_sha256 != physical_result_sha256(physical)
        or dict(record.neural_hashes_before) != dict(physical.neural_hashes_before)
        or dict(record.neural_hashes_after) != dict(physical.neural_hashes_after)
    ):
        raise ValueError("control shard differs from frozen neural/physical lineage")
    system = SYSTEMS[record.system_name]
    interface = fixed_interfaces(system)[record.interface_name]
    expected_protocol = linear_interface_protocol(system, interface)
    if record.result.physical_protocol != expected_protocol:
        raise ValueError("control shard physical protocol differs from preregistration")
    cem = registered_cem_config(system)
    budget = record.result.planner_budget
    expected_budget = {
        "candidatesPerDecision": cem.candidates,
        "iterationsPerDecision": cem.iterations,
        "elitesPerIteration": cem.elites,
        "horizon": cem.horizon,
        "candidateEvaluationsPerDecision": cem.candidates * cem.iterations,
        "pairedCandidateNoiseAcrossLearnedPlanners": 1,
        "activationRolloutMicroBatch": cem.activation_rollout_batch_size,
        "commonLinearInterfaceCommandBound": (
            expected_protocol.common_interface_command_bound
        ),
        "linearInterfaceBoundFormula": expected_protocol.bound_formula,
    }
    if dict(budget) != expected_budget or (cem.candidates, cem.iterations) != (512, 4):
        raise ValueError("control shard differs from the exact registered CEM budget")
    expected_schedule = _schedule_sha256(
        record.planner_seed,
        record.start,
        record.stop - record.start,
        system.control_steps,
    )
    if record.result.planner_seed_schedule_sha256 != expected_schedule:
        raise ValueError("control shard planner-seed schedule changed")
    episodes = make_builtin_control_episodes(
        system,
        history_frames=loaded.backbone.config.history_frames,
        count=64,
        seed=record.episode_seed,
        image_size=loaded.backbone.config.image_size,
    )[record.start : record.stop]
    if tuple(episode.identifier for episode in episodes) != record.result.episode_identifiers:
        raise ValueError("control shard episode identifiers changed")
    plant = builtin_pixel_plant(system)
    for controller, traces in record.result.interface_command_traces.items():
        replayed = []
        for episode, commands in zip(episodes, traces, strict=True):
            environment = plant.clone_environment(episode.environment)
            context = episode.context.clone()
            for command in commands:
                value = np.asarray(command, dtype=np.float32)
                plant.step_interface(environment, interface, value)
                context = plant.append_observation(context, environment)
            replayed.append(
                pixel_target_error(
                    system, plant.current_pixels(environment), episode.target_pixels
                )
            )
        stored = record.result.errors[controller]
        if len(replayed) != len(stored) or any(
            not math.isclose(first, second, rel_tol=0.0, abs_tol=1e-8)
            for first, second in zip(replayed, stored, strict=True)
        ):
            raise ValueError(f"control shard replay differs for {controller}")
    loaded.assert_frozen_and_unchanged()


def _control_records(
    loaded: Any,
    physical: Any,
    result_dir: Path,
    interface_name: str,
) -> tuple[list[AuthenticatedControlShard], Any]:
    records = []
    for start, stop in registered_control_shard_ranges():
        path = result_dir / "control" / interface_name / f"{start:02d}-{stop:02d}.pt"
        if path.exists():
            record = load_control_shard(path)
        else:
            raise ValueError(f"missing authenticated control shard {path}")
        if (
            record.system_name != loaded.system_name
            or record.interface_name != interface_name
            or (record.start, record.stop) != (start, stop)
        ):
            raise ValueError("control shard path and typed interval differ")
        _verify_and_replay_control_shard(record, loaded, physical)
        records.append(record)
    merged = merge_control_shards(
        [
            ControlShard(
                record.interface_name,
                record.start,
                record.stop,
                record.total_episodes,
                record.result,
            )
            for record in records
        ]
    )
    return records, merged


def _load_system(
    system_name: str,
    sanitized_root: Path,
    training_root: Path,
    device: torch.device,
) -> Any:
    return load_postfreeze_system(
        system_name,
        PostFreezePaths(sanitized_root, training_root / system_name),
        device,
    )


def prepare_system(args: argparse.Namespace) -> None:
    """Collect exactly one structural and one shared physical artifact."""

    loaded = _load_system(
        args.system,
        args.sanitized_root,
        args.training_root,
        torch.device(args.device),
    )
    system_result = args.result_root / args.system
    physical_path = system_result / "physical-shared.pt"
    prepared_path = system_result / "prepared-system.pt"
    if physical_path.exists():
        physical = load_physical_result(physical_path, loaded)
    else:
        physical = run_physical_realizability(loaded)
        save_physical_result(physical_path, loaded, physical)
        physical = load_physical_result(physical_path, loaded)
    if prepared_path.exists():
        prepared = load_prepared_system_artifact(prepared_path, loaded, physical)
        print(
            json.dumps(
                {
                    "stage": "prepare-system",
                    "system": args.system,
                    "resumed": True,
                    "artifactSha256": prepared.artifact_sha256,
                    "physicalSha256": prepared.physical_sha256,
                },
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )
        return

    gate1_evidence = assemble_gate1_evidence(loaded)
    # Execute every registered audit while its evidence is live.  The finalizer
    # independently recomputes Gates 1--4 from the detached artifact below.
    audit_gate1_postfreeze(loaded)
    gate2 = collect_gate2_postfreeze(loaded)
    gate3 = collect_gate3_and_rk2_postfreeze(loaded)
    _, gate4_evidence = audit_gate4_postfreeze(loaded)
    gate5_artifact, gate5_evidence = run_gate5_postfreeze(loaded)
    energy_evidence = collect_energy_semantic_evidence(loaded, gate5_evidence)
    gate5_energy_artifact = audit_energy_semantics(energy_evidence).to_dict()
    artifact = build_prepared_system_artifact(
        loaded,
        physical,
        gate1_evidence=gate1_evidence,
        gate2_evidence=gate2.evidence,
        gate3_states=gate3.transitions.states,
        gate3_efforts=gate3.transitions.efforts,
        gate3_source_manifest_sha256=gate3.transitions.source_manifest_sha256,
        gate4_evidence=gate4_evidence,
        gate5_artifact=gate5_artifact.to_dict(),
        gate5_energy_artifact=gate5_energy_artifact,
    )
    save_prepared_system_artifact(prepared_path, artifact)
    loaded.assert_frozen_and_unchanged()
    print(
        json.dumps(
            {
                "stage": "prepare-system",
                "system": args.system,
                "resumed": False,
                "artifactSha256": artifact.artifact_sha256,
                "physicalSha256": artifact.physical_sha256,
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


def run_one_control_shard(args: argparse.Namespace) -> None:
    """Run or validate one registered four-episode CEM shard."""

    interval = (args.start, args.stop)
    if interval not in registered_control_shard_ranges():
        raise ValueError("control-shard interval is not a registered four-episode range")
    loaded = _load_system(
        args.system,
        args.sanitized_root,
        args.training_root,
        torch.device(args.device),
    )
    system_result = args.result_root / args.system
    physical = load_physical_result(system_result / "physical-shared.pt", loaded)
    load_prepared_system_artifact(
        system_result / "prepared-system.pt", loaded, physical
    )
    path = (
        system_result
        / "control"
        / args.interface
        / f"{args.start:02d}-{args.stop:02d}.pt"
    )
    resumed = path.exists()
    if resumed:
        record = load_control_shard(path)
    else:
        raw = run_control_shard(
            loaded,
            physical,
            interface_name=args.interface,
            start=args.start,
            stop=args.stop,
            episode_seed=REGISTERED_CONTROL_EPISODE_SEED,
            planner_seed=REGISTERED_CONTROL_PLANNER_SEED,
        )
        record = build_authenticated_control_shard(loaded, physical, raw)
        save_control_shard(path, record)
        record = load_control_shard(path)
    if (
        record.system_name != args.system
        or record.interface_name != args.interface
        or (record.start, record.stop) != interval
    ):
        raise ValueError("control shard file and requested task differ")
    _verify_and_replay_control_shard(record, loaded, physical)
    print(
        json.dumps(
            {
                "stage": "control-shard",
                "system": args.system,
                "interface": args.interface,
                "start": args.start,
                "stop": args.stop,
                "resumed": resumed,
                "artifactSha256": record.artifact_sha256,
                "candidateEvaluationsPerDecision": 512 * 4,
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


def _derive_system(
    system_name: str,
    sanitized_root: Path,
    training_root: Path,
    result_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    system_result = result_root / system_name
    loaded = _load_system(system_name, sanitized_root, training_root, device)
    physical_path = system_result / "physical-shared.pt"
    physical = load_physical_result(physical_path, loaded)
    prepared = load_prepared_system_artifact(
        system_result / "prepared-system.pt", loaded, physical
    )
    structural = audit_prepared_structural_gates(prepared, loaded)
    physical_sha = physical_result_sha256(physical)
    gate6 = physical_gate6_table(physical)

    control_gates: dict[str, Any] = {}
    control_evidence: dict[str, Any] = {}
    execution_evidence: dict[str, InterfaceExecutionEvidence] = {}
    for interface_name in ("native", "unseen"):
        records, merged = _control_records(
            loaded,
            physical,
            system_result,
            interface_name,
        )
        control_gates[interface_name] = control_gate_metrics(
            merged,
            no_jacobian_errors=merged.errors["no_jacobian"],
            shuffled_lens_errors=merged.errors["shuffled_lens"],
            bootstrap_seed=151_910_737 + (0 if interface_name == "native" else 1),
        )
        execution_evidence[interface_name] = _derive_interface_execution_evidence(
            loaded,
            physical,
            records,
            merged,
        )
        control_evidence[interface_name] = {
            "aggregate": merged.as_dict(),
            "errors": {name: list(values) for name, values in merged.errors.items()},
            "shardArtifactSha256": [record.artifact_sha256 for record in records],
        }
    gate7 = {
        "passed": all(value["passed"] for value in control_gates.values()),
        "interfaces": control_gates,
    }
    unseen_realizability = gate6["interfaces"]["unseen"]
    gate8 = interface_transfer_gate_metrics(
        control_gates["native"],
        control_gates["unseen"],
        unseen_realizability,
        evidence=InterfaceTransferEvidence(
            native=execution_evidence["native"],
            unseen=execution_evidence["unseen"],
        ),
    )
    gates = {
        **structural,
        "gate6": gate6,
        "gate7": gate7,
        "gate8": gate8,
    }
    return {
        "system": system_name,
        "trainingLineageSha256": training_lineage_sha256(loaded),
        "preparedArtifactSha256": prepared.artifact_sha256,
        "physicalSha256": physical_sha,
        "gates": gates,
        "controlEvidence": control_evidence,
    }


def _final_core(systems: Mapping[str, Any]) -> dict[str, Any]:
    if set(systems) != set(REQUIRED_SYSTEMS):
        raise ValueError("final outcome requires both registered systems")
    system_passes = {
        system: all(
            systems[system]["gates"][f"gate{gate}"].get("passed") is True
            for gate in range(1, 9)
        )
        for system in REQUIRED_SYSTEMS
    }
    gate9 = {
        "gate": 9,
        "passed": all(system_passes.values()),
        "checks": {
            "pendulumGates1Through8": system_passes["pendulum"],
            "blocketGates1Through8": system_passes["blocket"],
            "singleFrozenSeed": True,
        },
    }
    return {
        "kind": FINAL_KIND,
        "requiredGates": list(REQUIRED_GATES),
        "systems": dict(systems),
        "gate9": gate9,
        "passed": gate9["passed"],
        "outcome": (
            "direct_jacobian_poisson_ph_breakthrough_supported_single_seed_two_systems"
            if gate9["passed"]
            else "direct_jacobian_poisson_ph_breakthrough_not_supported_single_seed"
        ),
    }


def run_complete(args: argparse.Namespace) -> None:
    """Finalize only from complete staged evidence; never run missing work."""

    systems = {
        system: _derive_system(
            system,
            args.sanitized_root,
            args.training_root,
            args.result_root,
            torch.device(args.device),
        )
        for system in REQUIRED_SYSTEMS
    }
    final = _final_artifact(systems)
    path = args.result_root / "final-outcome.json"
    _atomic_json(path, final)
    print(json.dumps(final, indent=2, allow_nan=False), flush=True)


def verify_complete(args: argparse.Namespace) -> None:
    supplied = json.loads(args.final_outcome.read_text(encoding="utf-8"))
    core = _verify_final_artifact_envelope(supplied)
    result_root = args.final_outcome.parent
    recomputed = {
        system: _derive_system(
            system,
            args.sanitized_root,
            args.training_root,
            result_root,
            torch.device(args.device),
        )
        for system in REQUIRED_SYSTEMS
    }
    expected_core = _json_native(_final_core(recomputed))
    if canonical_evidence_sha256(expected_core) != canonical_evidence_sha256(core):
        raise ValueError("final outcome differs from recomputed/replayed evidence")
    print(json.dumps({"verified": True, "artifactSha256": supplied["artifactSha256"]}), flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "finalize"):
        run = subparsers.add_parser(command)
        run.add_argument("sanitized_root", type=Path)
        run.add_argument("training_root", type=Path)
        run.add_argument("result_root", type=Path)
        run.add_argument("--device", default="cuda")
        run.add_argument("--require-gates", type=_parse_required_gates, required=True)
    prepare = subparsers.add_parser("prepare-system")
    prepare.add_argument("sanitized_root", type=Path)
    prepare.add_argument("training_root", type=Path)
    prepare.add_argument("result_root", type=Path)
    prepare.add_argument("--system", choices=REQUIRED_SYSTEMS, required=True)
    prepare.add_argument("--device", default="cuda")
    shard = subparsers.add_parser("control-shard")
    shard.add_argument("sanitized_root", type=Path)
    shard.add_argument("training_root", type=Path)
    shard.add_argument("result_root", type=Path)
    shard.add_argument("--system", choices=REQUIRED_SYSTEMS, required=True)
    shard.add_argument("--interface", choices=("native", "unseen"), required=True)
    shard.add_argument("--start", type=int, required=True)
    shard.add_argument("--stop", type=int, required=True)
    shard.add_argument("--device", default="cuda")
    verify = subparsers.add_parser("verify")
    verify.add_argument("final_outcome", type=Path)
    verify.add_argument("sanitized_root", type=Path)
    verify.add_argument("training_root", type=Path)
    verify.add_argument("--device", default="cpu")
    verify.add_argument("--require-gates", type=_parse_required_gates, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command in {"run", "finalize"}:
        run_complete(args)
    elif args.command == "prepare-system":
        prepare_system(args)
    elif args.command == "control-shard":
        run_one_control_shard(args)
    else:
        verify_complete(args)


if __name__ == "__main__":
    main()


__all__ = [
    "main",
    "parse_args",
    "prepare_system",
    "run_complete",
    "run_one_control_shard",
    "verify_complete",
]
