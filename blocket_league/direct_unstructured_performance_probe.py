"""Shape-exact timing probe for the independent unstructured baseline.

Random categorical tensors are used exclusively.  The command opens no
experiment archive, simulator, action, or physical state, so its JSON is
timing evidence only and is inadmissible for every scientific gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import torch

from .direct_cotangent_bridge import PixelChangeProbeBank
from .direct_experiment_training import DIRECT_SYSTEMS, DirectTrainingConfig, seed_everything
from .direct_jacobian_port_extractor import (
    EmpiricalTangentConfig,
    make_synthetic_empirical_tangent_artifact_for_tests,
)
from .direct_performance_probe import _synthetic_suite, _synchronize
from .direct_unstructured_training import (
    build_fresh_independent_baseline,
    train_independent_unstructured_world_model,
)
from .direct_visual_poisson_ph import DirectVideoLossConfig
from .env import PALETTE
from .pixel_direct_model import DirectPixelTransformer, pixel_direct_config_for_preset


def _optimizer_step_deltas(path: Path) -> tuple[float, ...]:
    elapsed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value.get("stage") == "independent_unstructured_jacobian_lens_world_model":
            elapsed.append(float(value["seconds"]))
    if len(elapsed) != 3:
        raise ValueError("independent performance probe did not log three steps")
    return tuple(elapsed[index] - elapsed[index - 1] for index in (1, 2))


def run_independent_step_probe(
    output_dir: Path,
    *,
    system_name: str,
    device: torch.device,
    micro_batch_size: int = 16,
    lens_batch_size: int = 4,
    gradient_accumulation: int = 1,
    image_size: int = 64,
    patch_size: int = 4,
    backbone_preset: str = "tiny",
    float32_matmul_precision: str = "highest",
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("independent performance output must be new or empty")
    if system_name not in DIRECT_SYSTEMS:
        raise ValueError("unknown independent performance system")
    if float32_matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError("unknown float32 matmul precision")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision(float32_matmul_precision)
    seed = 151_910_737 + 991_004
    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    model_config = pixel_direct_config_for_preset(
        backbone_preset,
        image_size=image_size,
        patch_size=patch_size,
        palette_size=len(PALETTE),
        history_frames=8,
    )
    backbone = DirectPixelTransformer(model_config).to(device).eval().requires_grad_(False)
    trajectories = max(micro_batch_size, lens_batch_size)
    fit = _synthetic_suite(
        trajectories=trajectories,
        transitions=8,
        history_frames=8,
        image_size=image_size,
        palette_size=len(PALETTE),
        generator=generator,
    )
    validation = _synthetic_suite(
        trajectories=trajectories,
        transitions=8,
        history_frames=8,
        image_size=image_size,
        palette_size=len(PALETTE),
        generator=generator,
    )
    system = DIRECT_SYSTEMS[system_name]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 2)
        probes = PixelChangeProbeBank.from_pixel_frames(
            fit["frames"], palette_size=len(PALETTE), probe_size=system.port_size
        )
    config = DirectTrainingConfig(
        steps=3,
        micro_batch_size=micro_batch_size,
        gradient_accumulation=gradient_accumulation,
        warmup_steps=0,
        validation_every=3,
        validation_batches=1,
        checkpoint_every=3,
        log_every=1,
        lens_batch_size=lens_batch_size,
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
        seed=seed + 3,
    )
    bundle = build_fresh_independent_baseline(
        backbone,
        system,
        probes,
        config,
        device,
        empirical_tangent=empirical_tangent,
        reference_initialization_seed=151_910_737 + 10_003,
    )
    archive_root = output_dir / "synthetic-trainer" / system_name
    archive_root.mkdir(parents=True)
    archive_paths = {
        "fit": archive_root / "fit-pixels.pt",
        "validation": archive_root / "validation-pixels.pt",
    }
    for split, path in archive_paths.items():
        torch.save({"manifest": {"synthetic": True}, "frames": fit["frames"] if split == "fit" else validation["frames"]}, path)
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
    summary = train_independent_unstructured_world_model(
        bundle,
        fit,
        validation,
        torch.ones(len(PALETTE), device=device),
        system,
        output_dir / "training",
        config,
        DirectVideoLossConfig(),
        data_seal=data_seal,
        pixel_archive_paths=archive_paths,
        source_tree_sha256="e" * 64,
    )
    _synchronize(device)
    total = time.perf_counter() - started
    deltas = _optimizer_step_deltas(output_dir / "training" / "train.jsonl")
    registered = bool(
        micro_batch_size == 16
        and lens_batch_size == 4
        and gradient_accumulation == 1
        and image_size == 64
        and patch_size == 4
        and backbone_preset == "tiny"
        and float32_matmul_precision == "highest"
    )
    result: dict[str, Any] = {
        "kind": (
            "synthetic_shape_exact_independent_performance_probe_not_scientific_evidence"
            if registered
            else "synthetic_nonregistered_independent_performance_probe_not_scientific_evidence"
        ),
        "system": system_name,
        "modelConfig": asdict(model_config),
        "shapeCriticalTrainingConfig": {
            "microBatchSize": micro_batch_size,
            "gradientAccumulation": gradient_accumulation,
            "lensBatchSize": lens_batch_size,
            "lensHorizons": list(config.lens_horizons),
            "transitions": 8,
            "float32MatmulPrecision": float32_matmul_precision,
        },
        "targetTrainableParameters": bundle.target_trainable_parameters,
        "trainableParameters": bundle.trainable_parameters,
        "relativeParameterGap": bundle.relative_parameter_gap,
        "dynamicsHiddenSize": bundle.dynamics_hidden_size,
        "postWarmupOptimizerStepSeconds": list(deltas),
        "meanPostWarmupOptimizerStepSeconds": sum(deltas) / len(deltas),
        "totalProbeSecondsIncludingOneValidationAndCheckpoint": total,
        "trainingSummarySeconds": float(summary["seconds"]),
        "scientificEvidenceAdmissible": False,
        "registeredShapeConfiguration": registered,
        "openedExperimentData": False,
        "openedSimulator": False,
        "actionChannels": 0,
        "physicalStateChannels": 0,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--system", choices=tuple(DIRECT_SYSTEMS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--lens-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--backbone-preset", default="tiny")
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    args = parser.parse_args()
    result = run_independent_step_probe(
        args.output_dir,
        system_name=args.system,
        device=torch.device(args.device),
        micro_batch_size=args.micro_batch_size,
        lens_batch_size=args.lens_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        image_size=args.image_size,
        patch_size=args.patch_size,
        backbone_preset=args.backbone_preset,
        float32_matmul_precision=args.float32_matmul_precision,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["run_independent_step_probe"]
