"""Post-freeze physical realizability audit for one sealed full checkpoint.

This deliberately narrow runner exists for a full-only pilot: it never loads a
baseline or ablation checkpoint, never constructs an optimizer, and never
changes a neural tensor.  It performs the registered four-paired-state analytic
calibration for each physical axis and evaluates the resulting constant port
map on 128 disjoint paired states per axis under both the native and unseen
interfaces.

The result is candidate evidence only.  In particular, it cannot claim the
complete Experiment-F breakthrough because the registered ablation and
closed-loop control comparisons are absent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .direct_distributed_training import (
    _data_seal,
    _load_backbone,
    _load_empirical_tangent,
    _load_registered_probes,
    _variant_loss,
    load_sealed_config,
)
from .direct_experiment_training import (
    DIRECT_SYSTEMS,
    _validate_direct_checkpoint,
    build_direct_bundle,
    seed_everything,
)
from .direct_physical_evaluation import (
    FrozenEvaluationSeal,
    builtin_pixel_plant,
    collect_paired_calibration_response_bank,
    collect_paired_heldout_response_bank,
    evaluate_heldout_realizability_from_response_bank,
    evaluation_system_from_direct_spec,
    fit_interface_calibration_from_response_bank,
    fixed_interfaces,
    make_builtin_probe_candidates,
    realizability_gate_metrics,
    select_d_optimal_probe_states,
)
from .direct_postfreeze_runner import _freeze_bundle
from .source_provenance import load_source_manifest
from .tensor_provenance import module_tensor_hash


CANDIDATE_POOL_SIZE = 64
HELDOUT_STATES_PER_AXIS = 128
CANDIDATE_SEED_OFFSET = 60_000
HELDOUT_SEED_OFFSET = 70_000
_CHECKPOINT_FIELDS = (
    ("model", "model"),
    ("writeField", "write_field"),
    ("responseFrame", "response_frame"),
    ("cotangentFrame", "cotangent_frame"),
    ("probes", "probes"),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def _prepare_result_directory(
    result_dir: Path, evaluator_source_manifest_path: Path
) -> None:
    """Refuse stale evidence while allowing the pre-created source seal."""

    result_dir.mkdir(parents=True, exist_ok=True)
    allowed = set()
    if evaluator_source_manifest_path.parent.resolve() == result_dir.resolve():
        allowed.add(evaluator_source_manifest_path.resolve())
    unexpected = [
        path for path in result_dir.iterdir() if path.resolve() not in allowed
    ]
    if unexpected:
        names = sorted(path.name for path in unexpected)
        raise FileExistsError(
            f"full-only realizability output directory contains stale files: {names}"
        )


def _load_frozen_full_bundle(
    training_system_dir: Path,
    *,
    system_name: str,
    device: torch.device,
):
    config = load_sealed_config(training_system_dir)
    if config.system != system_name:
        raise ValueError("sealed distributed config belongs to another system")
    backbone = _load_backbone(training_system_dir, config, device)
    empirical_tangent = _load_empirical_tangent(
        training_system_dir, config, backbone
    )
    probes = _load_registered_probes(training_system_dir, config)
    seed_everything(config.experiment.seed + 10_003)
    bundle = build_direct_bundle(
        backbone,
        DIRECT_SYSTEMS[system_name],
        probes,
        config.direct,
        device,
        empirical_tangent=empirical_tangent,
        variant="full",
    )
    checkpoint_path = training_system_dir / "direct" / "full" / "best.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if type(payload) is not dict:
        raise ValueError("full checkpoint is not a plain weights-only dictionary")
    _validate_direct_checkpoint(
        payload,
        bundle,
        variant="full",
        system=DIRECT_SYSTEMS[system_name],
        train_config=config.direct,
        loss_config=_variant_loss(config.loss, "full"),
        data_seal=_data_seal(system_name, config.manifests),
        source_tree_sha256=config.source_tree_sha256,
        include_training_state=False,
    )
    for field, attribute in _CHECKPOINT_FIELDS:
        getattr(bundle, attribute).load_state_dict(payload[field], strict=True)
    _freeze_bundle(bundle)
    if module_tensor_hash(bundle.model.encoder.backbone) != payload["backboneHash"]:
        raise ValueError("post-freeze load changed the sealed backbone")
    return config, bundle, payload, checkpoint_path


def _candidate_outcome(interface_gates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    native = bool(interface_gates.get("native", {}).get("passed", False))
    unseen = bool(interface_gates.get("unseen", {}).get("passed", False))
    return {
        "physicalRealizabilityCandidatePass": native and unseen,
        "nativePass": native,
        "unseenPass": unseen,
        "completeBreakthroughClaimAllowed": False,
        "missingRequiredEvidence": [
            "paired_ablation_advantage",
            "closed_loop_control",
            "generic_world_model_planner_comparison",
            "second_system",
            "multi_seed_reproducibility",
        ],
        "interpretation": (
            "postfreeze_full_only_realizability_supported"
            if native and unseen
            else "postfreeze_full_only_realizability_not_supported"
        ),
    }


@torch.no_grad()
def run_full_only_realizability(
    training_system_dir: Path,
    result_dir: Path,
    *,
    system_name: str,
    device: torch.device,
    evaluator_source_manifest_path: Path,
) -> dict[str, Any]:
    """Run the fixed full-only Gate-6 candidate audit without neural updates."""

    if system_name not in DIRECT_SYSTEMS:
        raise KeyError(f"unknown direct system {system_name!r}")
    _prepare_result_directory(result_dir, evaluator_source_manifest_path)
    evaluator_manifest = load_source_manifest(evaluator_source_manifest_path)
    evaluator_source_tree_sha256 = evaluator_manifest["treeSha256"]
    started = time.perf_counter()
    config, bundle, checkpoint, checkpoint_path = _load_frozen_full_bundle(
        training_system_dir, system_name=system_name, device=device
    )
    modules = {
        "fullModel": bundle.model,
        "exactJacobianWriteField": bundle.write_field,
        "responseFrame": bundle.response_frame,
        "cotangentFrame": bundle.cotangent_frame,
        "fixedPixelProbes": bundle.probes,
    }
    seal = FrozenEvaluationSeal.capture(modules)
    system = evaluation_system_from_direct_spec(DIRECT_SYSTEMS[system_name])
    backbone_config = bundle.model.encoder.backbone.config
    candidate_seed = config.experiment.seed + CANDIDATE_SEED_OFFSET
    heldout_seed = config.experiment.seed + HELDOUT_SEED_OFFSET
    candidates = make_builtin_probe_candidates(
        system,
        history_frames=backbone_config.history_frames,
        count=CANDIDATE_POOL_SIZE,
        seed=candidate_seed,
        image_size=backbone_config.image_size,
    )
    heldout = make_builtin_probe_candidates(
        system,
        history_frames=backbone_config.history_frames,
        count=HELDOUT_STATES_PER_AXIS,
        seed=heldout_seed,
        image_size=backbone_config.image_size,
    )
    candidate_ids = {value.identifier for value in candidates}
    heldout_ids = {value.identifier for value in heldout}
    if candidate_ids & heldout_ids:
        raise AssertionError("calibration and held-out physical pools overlap")
    selection = select_d_optimal_probe_states(
        bundle.model.encoder,
        bundle.model.core,
        candidates,
        system,
        seal=seal,
    )
    if (
        selection.selection_method != "single_model_d_optimal"
        or selection.paired_states_per_axis != 4
        or selection.observed_response_count != 0
    ):
        raise AssertionError("full-only response-blind calibration selection changed")

    plant = builtin_pixel_plant(system)
    interfaces: dict[str, Any] = {}
    evidence_interfaces: dict[str, Any] = {}
    interface_gates: dict[str, Mapping[str, Any]] = {}
    total_environment_steps = 0
    for interface_name, interface in fixed_interfaces(system).items():
        response_bank = collect_paired_calibration_response_bank(
            plant,
            candidates,
            selection,
            system,
            interface,
            seal=seal,
        )
        heldout_bank = collect_paired_heldout_response_bank(
            plant,
            heldout,
            system,
            interface,
            seal=seal,
            states_per_axis=HELDOUT_STATES_PER_AXIS,
        )
        calibration = fit_interface_calibration_from_response_bank(
            bundle.model.encoder,
            bundle.model.core,
            candidates,
            response_bank,
            system,
            interface,
            seal=seal,
            model_name="full",
        )
        metrics = evaluate_heldout_realizability_from_response_bank(
            bundle.model.encoder,
            bundle.model.core,
            heldout,
            system,
            interface,
            calibration,
            heldout_bank,
            seal=seal,
        )
        gate = realizability_gate_metrics(metrics)
        interface_gates[interface_name] = gate
        interfaces[interface_name] = {
            "calibration": calibration.as_dict(),
            "heldoutRealizability": metrics.as_dict(),
            "partialGate6WithoutAblations": gate,
        }
        evidence_interfaces[interface_name] = {
            "physicalProtocol": response_bank.protocol.to_dict(),
            "calibrationPlusPixelContexts": response_bank.plus_contexts.cpu(),
            "calibrationMinusPixelContexts": response_bank.minus_contexts.cpu(),
            "heldoutPlusPixelContexts": heldout_bank.plus_contexts.cpu(),
            "heldoutMinusPixelContexts": heldout_bank.minus_contexts.cpu(),
            "calibrationMatrix": calibration.latent_from_interface.cpu(),
            "responseCosines": torch.tensor(metrics.response_cosines),
            "responseSigns": torch.tensor(metrics.response_signs),
            "actualMagnitudes": torch.tensor(metrics.actual_magnitudes),
            "predictedMagnitudes": torch.tensor(metrics.predicted_magnitudes),
            "calibrationResponseEvidenceSha256": response_bank.evidence_sha256,
            "heldoutResponseEvidenceSha256": heldout_bank.evidence_sha256,
        }
        total_environment_steps += (
            response_bank.environment_steps + heldout_bank.environment_steps
        )
    seal.assert_unchanged()
    neural_hashes_after = dict(FrozenEvaluationSeal.capture(modules).hashes)
    outcome = _candidate_outcome(interface_gates)
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    result = {
        "kind": "direct_full_only_postfreeze_realizability_v1",
        "scope": "single_seed_single_system_candidate_evidence",
        "system": system_name,
        "checkpoint": str(checkpoint_path),
        "checkpointSha256": checkpoint_sha256,
        "checkpointStep": int(checkpoint["step"]),
        "trainingSourceTreeSha256": config.source_tree_sha256,
        "evaluatorSourceTreeSha256": evaluator_source_tree_sha256,
        "backboneHash": module_tensor_hash(bundle.model.encoder.backbone),
        "actionGradientUpdates": 0,
        "physicalStateGradientUpdates": 0,
        "postfreezeGradientUpdates": 0,
        "candidatePoolSize": CANDIDATE_POOL_SIZE,
        "candidateSeed": candidate_seed,
        "heldoutStatesPerAxis": HELDOUT_STATES_PER_AXIS,
        "heldoutSeed": heldout_seed,
        "pairedCalibrationStatesPerAxis": 4,
        "selection": asdict(selection),
        "calibrationAndHeldoutPoolsDisjoint": True,
        "totalPhysicalEnvironmentSteps": total_environment_steps,
        "neuralHashesBefore": dict(seal.hashes),
        "neuralHashesAfter": neural_hashes_after,
        "neuralHashesUnchanged": neural_hashes_after == dict(seal.hashes),
        "interfaces": interfaces,
        "outcome": outcome,
        "seconds": float(time.perf_counter() - started),
    }
    evidence = {
        "kind": "direct_full_only_postfreeze_realizability_evidence_v1",
        "system": system_name,
        "checkpointSha256": checkpoint_sha256,
        "trainingSourceTreeSha256": config.source_tree_sha256,
        "evaluatorSourceTreeSha256": evaluator_source_tree_sha256,
        "selection": asdict(selection),
        "interfaces": evidence_interfaces,
    }
    _atomic_torch(result_dir / "evidence.pt", evidence)
    _atomic_json(result_dir / "summary.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_system_dir", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--system", required=True, choices=tuple(DIRECT_SYSTEMS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluator-source-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_full_only_realizability(
        args.training_system_dir,
        args.result_dir,
        system_name=args.system,
        device=torch.device(args.device),
        evaluator_source_manifest_path=args.evaluator_source_manifest,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_POOL_SIZE",
    "HELDOUT_STATES_PER_AXIS",
    "run_full_only_realizability",
]
