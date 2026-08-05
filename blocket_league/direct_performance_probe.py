"""Shape-exact, synthetic performance probe for Experiment F.

This command is deliberately not an experiment or a reduced training run.  It
uses random categorical pixels and a randomly initialized backbone solely to
measure the wall time and peak accelerator memory of the registered tensor
shapes.  It never opens fit, validation, test, calibration, or simulator data,
and its output is therefore inadmissible as scientific evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import types
from typing import Any

import torch

from .direct_cotangent_bridge import PixelChangeProbeBank
from .direct_experiment_training import (
    DIRECT_SYSTEMS,
    DirectTrainingConfig,
    build_direct_bundle,
    seed_everything,
    train_direct_bundle,
)
from .direct_jacobian_port_extractor import (
    EmpiricalTangentConfig,
    make_synthetic_empirical_tangent_artifact_for_tests,
)
from .direct_visual_poisson_ph import DirectVideoLossConfig
from .env import PALETTE
from .pixel_direct_model import DirectPixelTransformer, pixel_direct_config_for_preset


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _synthetic_suite(
    *,
    trajectories: int,
    transitions: int,
    history_frames: int,
    image_size: int,
    palette_size: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    contexts = torch.randint(
        0,
        palette_size,
        (trajectories, transitions + 1, history_frames, image_size, image_size),
        generator=generator,
        dtype=torch.uint8,
    )
    frames = torch.randint(
        0,
        palette_size,
        (trajectories, transitions + 1, image_size, image_size),
        generator=generator,
        dtype=torch.uint8,
    )
    return {"pixelContexts": contexts, "frames": frames}


def _read_training_step_seconds(path: Path) -> tuple[float, ...]:
    elapsed: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value.get("stage") == "joint_direct_jacobian_poisson_ph":
            elapsed.append(float(value["seconds"]))
    if len(elapsed) < 3:
        raise ValueError("performance probe did not record all three optimizer steps")
    return tuple(elapsed[index] - elapsed[index - 1] for index in range(1, len(elapsed)))


def _read_last_validation_audits(path: Path) -> dict[str, float]:
    required = (
        "auditImplicitResidualMax",
        "auditChainRuleDefectMax",
        "auditBalanceDefectMax",
    )
    validation: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value.get("stage") == "pixels_only_validation":
            validation = value
    if validation is None or any(name not in validation for name in required):
        raise ValueError("performance probe is missing float64 validation audits")
    return {name: float(validation[name]) for name in required}


def _nested_autograd_energy_gradient_reference(
    core: Any,
    state: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-optimization energy derivative retained only as a timing oracle."""

    with torch.enable_grad():
        differentiable_state = state
        if not differentiable_state.requires_grad:
            differentiable_state = state.detach().requires_grad_(True)
        energy = core.hamiltonian(differentiable_state)
        gradient = torch.autograd.grad(
            energy.sum(),
            differentiable_state,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
    return energy, gradient


def run_direct_step_probe(
    output_dir: Path,
    *,
    system_name: str,
    variant: str,
    device: torch.device,
    micro_batch_size: int,
    lens_batch_size: int,
    gradient_accumulation: int,
    implicit_iterations: int,
    image_size: int,
    patch_size: int,
    backbone_preset: str,
    float32_matmul_precision: str,
    energy_gradient_implementation: str = "exact",
) -> dict[str, Any]:
    """Run three exact optimizer steps and time the two post-warm-up steps."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("performance-probe output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    if float32_matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError("unknown float32 matmul precision")
    if energy_gradient_implementation not in {"exact", "autograd-reference"}:
        raise ValueError("unknown energy-gradient implementation")
    torch.set_float32_matmul_precision(float32_matmul_precision)
    seed = 151_910_737 + 991_003
    seed_everything(seed)
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    model_config = pixel_direct_config_for_preset(
        backbone_preset,
        image_size=image_size,
        patch_size=patch_size,
        palette_size=len(PALETTE),
        history_frames=8,
    )
    backbone = DirectPixelTransformer(model_config).to(device)
    trajectories = max(micro_batch_size, lens_batch_size)
    fit_suite = _synthetic_suite(
        trajectories=trajectories,
        transitions=8,
        history_frames=model_config.history_frames,
        image_size=image_size,
        palette_size=model_config.palette_size,
        generator=cpu_generator,
    )
    validation_suite = _synthetic_suite(
        trajectories=trajectories,
        transitions=8,
        history_frames=model_config.history_frames,
        image_size=image_size,
        palette_size=model_config.palette_size,
        generator=cpu_generator,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 2)
        probes = PixelChangeProbeBank.from_pixel_frames(
            fit_suite["frames"],
            palette_size=model_config.palette_size,
            probe_size=DIRECT_SYSTEMS[system_name].port_size,
        )
    config = DirectTrainingConfig(
        # Three steps are sufficient to exclude first-call initialization from
        # the two reported deltas.  All tensor-shape and solver settings below
        # are the intended registered settings.
        steps=3,
        micro_batch_size=micro_batch_size,
        gradient_accumulation=gradient_accumulation,
        warmup_steps=0,
        validation_every=3,
        validation_batches=1,
        checkpoint_every=3,
        log_every=1,
        lens_batch_size=lens_batch_size,
        implicit_iterations=implicit_iterations,
    )
    tangent_config = EmpiricalTangentConfig(
        channel_rank=config.port_tangent_channel_rank,
        neighbors=config.port_tangent_neighbors,
        support_floor_ratio=config.port_support_floor_ratio,
    )
    empirical_tangent = make_synthetic_empirical_tangent_artifact_for_tests(
        history_frames=model_config.history_frames,
        patch_count=model_config.grid_size**2,
        hidden_size=model_config.hidden_size,
        config=tangent_config,
        seed=seed + 31,
    )
    bundle = build_direct_bundle(
        backbone,
        DIRECT_SYSTEMS[system_name],
        probes,
        config,
        device,
        empirical_tangent=empirical_tangent,
        variant=variant,  # type: ignore[arg-type]
    )
    if energy_gradient_implementation == "autograd-reference":
        # Performance-probe-only instance override.  It is absent from every
        # scientific training entry point and state dict.
        bundle.model.core._energy_gradient = types.MethodType(  # type: ignore[method-assign]
            _nested_autograd_energy_gradient_reference,
            bundle.model.core,
        )
    weights = torch.ones(model_config.palette_size, device=device)
    data_seal = {
        "system": system_name,
        "fitAggregateSha256": "1" * 64,
        "fitSanitizedTensorSha256": "2" * 64,
        "validationAggregateSha256": "3" * 64,
        "validationSanitizedTensorSha256": "4" * 64,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    summary = train_direct_bundle(
        bundle,
        fit_suite,
        validation_suite,
        weights,
        DIRECT_SYSTEMS[system_name],
        output_dir,
        config,
        DirectVideoLossConfig(),
        variant=variant,  # type: ignore[arg-type]
        data_seal=data_seal,
        resume=False,
    )
    _synchronize(device)
    total_seconds = time.perf_counter() - started
    deltas = _read_training_step_seconds(output_dir / "train.jsonl")
    validation_audits = _read_last_validation_audits(output_dir / "train.jsonl")
    registered_shape = (
        micro_batch_size == 16
        and lens_batch_size == 4
        and gradient_accumulation == 1
        and implicit_iterations == 32
        and image_size == 64
        and patch_size == 4
        and backbone_preset == "tiny"
        and float32_matmul_precision == "highest"
        and energy_gradient_implementation == "exact"
    )
    result: dict[str, Any] = {
        "kind": (
            "synthetic_shape_exact_performance_probe_not_scientific_evidence"
            if registered_shape
            else "synthetic_nonregistered_performance_probe_not_scientific_evidence"
        ),
        "system": system_name,
        "variant": variant,
        "modelConfig": asdict(model_config),
        "shapeCriticalTrainingConfig": {
            "microBatchSize": micro_batch_size,
            "gradientAccumulation": gradient_accumulation,
            "lensBatchSize": lens_batch_size,
            "lensHorizons": list(config.lens_horizons),
            "transitions": 8,
            "implicitIterations": implicit_iterations,
            "energyGradientImplementation": energy_gradient_implementation,
            "float32MatmulPrecision": float32_matmul_precision,
        },
        "postWarmupOptimizerStepSeconds": list(deltas),
        "meanPostWarmupOptimizerStepSeconds": sum(deltas) / len(deltas),
        "totalProbeSecondsIncludingOneValidationAndCheckpoint": total_seconds,
        "trainingSummarySeconds": float(summary["seconds"]),
        "scientificEvidenceAdmissible": False,
        "registeredShapeConfiguration": registered_shape,
        "openedExperimentData": False,
        "openedSimulator": False,
        "float64ValidationAudits": validation_audits,
    }
    if device.type == "cuda":
        result.update(
            {
                "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
                "peakReservedBytes": int(torch.cuda.max_memory_reserved(device)),
                "deviceName": torch.cuda.get_device_name(device),
            }
        )
    (output_dir / "performance-probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--system", choices=tuple(DIRECT_SYSTEMS), required=True)
    parser.add_argument(
        "--variant",
        choices=("full", "no_jacobian", "single_horizon", "shuffled_lens", "skew_only", "constant_port"),
        default="full",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--lens-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--implicit-iterations", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--backbone-preset", default="tiny")
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    parser.add_argument(
        "--energy-gradient-implementation",
        choices=("exact", "autograd-reference"),
        default="exact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_direct_step_probe(
        args.output_dir,
        system_name=args.system,
        variant=args.variant,
        device=torch.device(args.device),
        micro_batch_size=args.micro_batch_size,
        lens_batch_size=args.lens_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        implicit_iterations=args.implicit_iterations,
        image_size=args.image_size,
        patch_size=args.patch_size,
        backbone_preset=args.backbone_preset,
        float32_matmul_precision=args.float32_matmul_precision,
        energy_gradient_implementation=args.energy_gradient_implementation,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["run_direct_step_probe"]
