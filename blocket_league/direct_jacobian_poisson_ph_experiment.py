"""Experiment F: direct Jacobian–Poisson pH latent from action-erased video.

The command has deliberately separate ``generate`` and ``train`` stages.  The
producer stage sees the private simulator excitation and writes a sanitized
pixel tensor.  The training stage accepts no producer seed and refuses any
cache containing a key beyond pixels and its cryptographic manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import torch

from .action_free_excitation import (
    HiddenExcitationConfig,
    action_free_environment_config_sha256,
    hidden_excitation_config_sha256,
    private_producer_seed_from_file,
)
from .direct_action_free_data import (
    ActionFreeBackboneTrainConfig,
    PixelsOnlyManifest,
    build_validated_action_free_backbone,
    class_weights,
    make_optimization_suite,
    sanitized_pixel_tensor_sha256,
    train_action_free_backbone,
    validate_action_free_backbone_checkpoint,
    _atomic_torch_save,
    _atomic_json_save,
)
from .direct_experiment_f_producer import collect_action_free_video_cache
from .direct_cotangent_bridge import PixelChangeProbeBank
from .direct_jacobian_port_precompute import (
    JacobianPortPrecomputeConfig,
    build_empirical_tangent_from_pixels,
)
from .direct_experiment_training import (
    BaselineTrainingConfig,
    DIRECT_SYSTEMS,
    DirectModelBundle,
    DirectTrainingConfig,
    Variant,
    build_direct_bundle,
    encode_pixel_suite,
    seed_everything,
    train_direct_bundle,
    train_unstructured_action_free_baseline,
)
from .direct_visual_poisson_ph import DirectVideoLossConfig
from .experiment_f_contract import ExperimentFConfig, REGISTERED_VARIANTS
from .runtime_firewall_trace import RuntimeFirewallTrace
from .source_provenance import build_source_manifest, load_source_manifest
from .env import PALETTE
from .passive_jacobian_ph_model import module_tensor_hash
from .pixel_direct_model import PixelDirectConfig, pixel_direct_config_for_preset


def _sanitized_path(root: Path, system: str, split: str) -> Path:
    return root / system / f"{split}-pixels.pt"


def _producer_split_root(root: Path, split: str) -> Path:
    if split in {"fit", "validation"}:
        return root / "trainer-mount"
    if split == "test":
        return root / "heldout"
    raise ValueError(f"unknown sanitized split {split!r}")


def generate_sanitized_splits(
    system: str,
    sanitized_root: Path,
    config: ExperimentFConfig,
    *,
    producer_seed: int,
    runtime_trace: RuntimeFirewallTrace | None = None,
    source_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Producer-only stage; the private seed is never serialized."""

    if system not in DIRECT_SYSTEMS:
        raise ValueError(f"unknown system {system!r}")
    if source_tree_sha256 is None:
        source_tree_sha256 = str(build_source_manifest()["treeSha256"])
    owns_runtime_trace = runtime_trace is None
    if runtime_trace is None:
        runtime_trace = RuntimeFirewallTrace(
            sanitized_root / "seals" / system / "firewall-trace.jsonl",
            stage=f"producer:{system}",
            source_tree_sha256=source_tree_sha256,
        )
    split_counts = {
        "fit": config.fit_trajectories,
        "validation": config.validation_trajectories,
        "test": config.test_trajectories,
    }
    manifests: dict[str, Any] = {}
    for split_index, (split, trajectories) in enumerate(split_counts.items()):
        path = _sanitized_path(_producer_split_root(sanitized_root, split), system, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels, manifest = collect_action_free_video_cache(
            system,
            trajectories=trajectories,
            frames_per_trajectory=config.cache_frames,
            image_size=config.image_size,
            # This value is destroyed at the boundary below and is not part of
            # the cache, manifest, file name, or training command.
            seed=producer_seed + split_index * 1_000_003,
            runtime_trace=runtime_trace,
            trace_phase="producer",
        )
        payload = {"pixels": pixels, "manifest": asdict(manifest)}
        if set(payload) != {"pixels", "manifest"}:
            raise AssertionError("sanitization payload gained an unexpected field")
        _atomic_torch_save(payload, path)
        runtime_trace.record_file_read(
            path,
            role=f"sanitized_archive_published:{split}",
            serialized_keys=tuple(sorted(payload)),
            semantic_sha256=manifest.sanitized_tensor_sha256,
        )
        manifests[split] = asdict(manifest)
    runtime_trace_seal = runtime_trace.snapshot().to_dict()
    if owns_runtime_trace:
        runtime_trace.close()
    excitation_config = HiddenExcitationConfig(
        frames=config.cache_frames, image_size=config.image_size
    )
    summary = {
        "system": system,
        "splits": manifests,
        "generationEnvironmentSha256": action_free_environment_config_sha256(
            system, image_size=config.image_size
        ),
        "producerSeedSerialized": False,
        "physicalCommandsSerialized": False,
        "simulatorStatesSerialized": False,
        "sourceTreeSha256": source_tree_sha256,
        "runtimeTrace": runtime_trace_seal,
        "hiddenExcitationConfig": asdict(excitation_config),
        "hiddenExcitationConfigSha256": hidden_excitation_config_sha256(
            excitation_config
        ),
    }
    seal_directory = sanitized_root / "seals" / system
    seal_directory.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(summary, seal_directory / "manifest.json")
    return summary


def load_sanitized_split(
    path: Path,
    *,
    expected_system: str,
    runtime_trace: RuntimeFirewallTrace | None = None,
    trace_role: str = "sanitized_pixels_archive",
) -> tuple[torch.Tensor, PixelsOnlyManifest]:
    # The archive crosses the producer/trainer boundary.  Safe tensor-only
    # loading occurs before the positive schema and digest checks below.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if runtime_trace is not None:
        runtime_trace.record_file_read(
            path,
            role=trace_role,
            serialized_keys=tuple(sorted(payload)) if type(payload) is dict else (),
        )
    if type(payload) is not dict or set(payload) != {"pixels", "manifest"}:
        raise ValueError("sanitized archive schema is not exactly pixels+manifest")
    pixels = payload["pixels"]
    raw_manifest = payload["manifest"]
    expected_manifest_keys = set(asdict(PixelsOnlyManifest("x", 1, 1, 1, "a" * 64)))
    if type(raw_manifest) is not dict or set(raw_manifest) != expected_manifest_keys:
        raise ValueError("sanitized manifest schema is not exact")
    try:
        manifest = PixelsOnlyManifest(**raw_manifest)
    except (TypeError, ValueError) as error:
        raise ValueError("sanitized manifest is malformed") from error
    if manifest.system != expected_system:
        raise ValueError("sanitized archive belongs to another system")
    if type(manifest.system) is not str:
        raise ValueError("sanitized manifest system must be a string")
    for name in ("trajectories", "frames_per_trajectory", "image_size"):
        value = getattr(manifest, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"sanitized manifest field {name!r} must be positive int")
    if type(manifest.source_schema) is not tuple or manifest.source_schema != ("frames",):
        raise ValueError("sanitized source schema is not exactly frames-only")
    if (
        type(manifest.optimization_schema) is not tuple
        or manifest.optimization_schema != ("pixelContexts", "frames")
    ):
        raise ValueError("sanitized optimization schema is not pixels-only")
    for name in ("aggregate_sha256", "sanitized_tensor_sha256"):
        value = getattr(manifest, name)
        if type(value) is not str or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"sanitized manifest field {name!r} is not SHA-256")
    if type(pixels) is not torch.Tensor:
        raise ValueError("sanitized pixels must be a plain tensor")
    expected_shape = (
        manifest.trajectories,
        manifest.frames_per_trajectory,
        manifest.image_size,
        manifest.image_size,
    )
    if tuple(pixels.shape) != expected_shape:
        raise ValueError("sanitized pixel tensor shape does not match its manifest")
    digest = sanitized_pixel_tensor_sha256(pixels)
    if digest != manifest.sanitized_tensor_sha256:
        raise ValueError("sanitized pixel tensor hash mismatch")
    if int(pixels.max()) >= len(PALETTE):
        raise ValueError("sanitized pixel tensor contains a non-palette class")
    return pixels, manifest


def _model_config(config: ExperimentFConfig) -> PixelDirectConfig:
    return pixel_direct_config_for_preset(
        config.backbone_preset,
        image_size=config.image_size,
        patch_size=config.patch_size,
        palette_size=len(PALETTE),
        history_frames=config.history_frames,
    )


def prepare_action_free_backbone(
    system: str,
    fit_pixels: torch.Tensor,
    fit_manifest: PixelsOnlyManifest,
    output_dir: Path,
    experiment_config: ExperimentFConfig,
    train_config: ActionFreeBackboneTrainConfig,
    device: torch.device,
    *,
    initialization_checkpoint: Path | None = None,
    runtime_trace: RuntimeFirewallTrace | None = None,
    source_tree_sha256: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_action_free_backbone_checkpoint(
            payload,
            expected_manifest_sha256=fit_manifest.aggregate_sha256,
            expected_sanitized_tensor_sha256=fit_manifest.sanitized_tensor_sha256,
            expected_system=system,
        )
        if payload["model_config"] != _model_config(experiment_config).to_dict():
            raise ValueError("existing backbone architecture differs from registered config")
        if payload["train_config"] != asdict(train_config):
            raise ValueError("existing backbone training schedule differs from registered config")
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file():
            raise ValueError(
                "completed backbone checkpoint is missing its atomic summary seal"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if type(summary) is not dict:
            raise ValueError("backbone summary is not a plain dictionary")
        return build_validated_action_free_backbone(
            payload,
            expected_manifest_sha256=fit_manifest.aggregate_sha256,
            expected_sanitized_tensor_sha256=fit_manifest.sanitized_tensor_sha256,
            expected_system=system,
        ), summary
    initial = None
    if initialization_checkpoint is not None and initialization_checkpoint.exists():
        initial = torch.load(initialization_checkpoint, map_location="cpu", weights_only=True)
        validate_action_free_backbone_checkpoint(
            initial,
            expected_manifest_sha256=fit_manifest.aggregate_sha256,
            expected_sanitized_tensor_sha256=fit_manifest.sanitized_tensor_sha256,
            expected_system=system,
        )
    backbone, summary = train_action_free_backbone(
        fit_pixels,
        _model_config(experiment_config),
        train_config,
        system=system,
        manifest=fit_manifest,
        output_dir=output_dir,
        device=device,
        initial_checkpoint=initial,
        runtime_trace=runtime_trace,
        source_tree_sha256=source_tree_sha256,
    )
    return backbone, summary


def _variant_directory(output_dir: Path, variant: Variant) -> Path:
    return output_dir / "direct" / variant


def train_registered_system(
    system_name: str,
    sanitized_root: Path,
    output_dir: Path,
    experiment_config: ExperimentFConfig,
    backbone_config: ActionFreeBackboneTrainConfig,
    direct_config: DirectTrainingConfig,
    baseline_config: BaselineTrainingConfig,
    loss_config: DirectVideoLossConfig,
    device: torch.device,
    *,
    initialization_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run every neural phase without mounting a producer seed or labels."""

    started = time.perf_counter()
    seed_everything(experiment_config.seed)
    system = DIRECT_SYSTEMS[system_name]
    mounted_paths = tuple((sanitized_root / system_name).iterdir())
    mounted_entries = {path.name for path in mounted_paths}
    if (
        mounted_entries != {"fit-pixels.pt", "validation-pixels.pt"}
        or any(not path.is_file() or path.is_symlink() for path in mounted_paths)
    ):
        raise ValueError(
            "training data mount must expose exactly fit-pixels.pt and "
            "validation-pixels.pt; held-out test data must be absent"
        )
    fit_pixels, fit_manifest = load_sanitized_split(
        _sanitized_path(sanitized_root, system_name, "fit"),
        expected_system=system_name,
    )
    validation_pixels, validation_manifest = load_sanitized_split(
        _sanitized_path(sanitized_root, system_name, "validation"),
        expected_system=system_name,
    )
    backbone, backbone_summary = prepare_action_free_backbone(
        system_name,
        fit_pixels,
        fit_manifest,
        output_dir / "backbone",
        experiment_config,
        backbone_config,
        device,
        initialization_checkpoint=initialization_checkpoint,
    )
    backbone = backbone.to(device).eval().requires_grad_(False)
    backbone_hash = module_tensor_hash(backbone)
    model_config = backbone.config
    fit_suite = make_optimization_suite(
        fit_pixels, model_config, transitions=experiment_config.transitions
    )
    validation_suite = make_optimization_suite(
        validation_pixels, model_config, transitions=experiment_config.transitions
    )
    weights = class_weights(
        fit_suite["frames"], model_config.palette_size, device
    )
    seed_everything(experiment_config.seed + 71)
    probes = PixelChangeProbeBank.from_pixel_frames(
        fit_suite["frames"],
        palette_size=model_config.palette_size,
        probe_size=system.port_size,
    )
    data_seal = {
        "system": system_name,
        "fitAggregateSha256": fit_manifest.aggregate_sha256,
        "fitSanitizedTensorSha256": fit_manifest.sanitized_tensor_sha256,
        "validationAggregateSha256": validation_manifest.aggregate_sha256,
        "validationSanitizedTensorSha256": validation_manifest.sanitized_tensor_sha256,
    }
    port_config = JacobianPortPrecomputeConfig(
        lens_block=system.lens_block,
        horizons=direct_config.lens_horizons,
        channel_rank=direct_config.port_tangent_channel_rank,
        neighbors=direct_config.port_tangent_neighbors,
        support_floor_ratio=direct_config.port_support_floor_ratio,
    )
    empirical_tangent, port_summary = build_empirical_tangent_from_pixels(
        backbone,
        fit_suite,
        system=system_name,
        fit_sanitized_tensor_sha256=fit_manifest.sanitized_tensor_sha256,
        output_dir=output_dir / "port-precompute",
        device=device,
        config=port_config,
        source_tree_sha256=str(build_source_manifest()["treeSha256"]),
    )

    variant_summaries: dict[str, Any] = {}
    full_bundle: DirectModelBundle | None = None
    for variant in experiment_config.variants:
        # Resetting the lineage gives matched initialization wherever the
        # architectures share parameters; it is still one master seed.
        seed_everything(experiment_config.seed + 10_003)
        bundle = build_direct_bundle(
            backbone,
            system,
            copy_probe_bank(probes),
            direct_config,
            device,
            empirical_tangent=empirical_tangent,
            variant=variant,
        )
        summary = train_direct_bundle(
            bundle,
            fit_suite,
            validation_suite,
            weights,
            system,
            _variant_directory(output_dir, variant),
            direct_config,
            loss_config,
            variant=variant,
            data_seal=data_seal,
        )
        variant_summaries[variant] = summary
        if variant == "full":
            full_bundle = bundle
        else:
            del bundle
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if full_bundle is None:
        raise RuntimeError("registered run omitted the full direct model")

    encoded_fit = encode_pixel_suite(
        full_bundle.model,
        fit_suite,
        batch_size=max(1, direct_config.micro_batch_size),
        device=device,
    )
    encoded_validation = encode_pixel_suite(
        full_bundle.model,
        validation_suite,
        batch_size=max(1, direct_config.micro_batch_size),
        device=device,
    )
    baseline, baseline_inference, baseline_summary = train_unstructured_action_free_baseline(
        full_bundle.model,
        encoded_fit,
        encoded_validation,
        validation_suite["frames"],
        weights,
        system,
        output_dir / "baseline-unstructured",
        baseline_config,
        device,
        data_seal=data_seal,
    )
    full_bundle.model.encoder.assert_backbone_frozen()
    if module_tensor_hash(full_bundle.model.encoder.backbone) != backbone_hash:
        raise AssertionError("backbone changed across registered variants")
    summary: dict[str, Any] = {
        "kind": "direct_jacobian_poisson_ph_training_complete",
        "system": system_name,
        "experimentConfig": asdict(experiment_config),
        "backboneConfig": asdict(backbone_config),
        "directConfig": asdict(direct_config),
        "baselineConfig": asdict(baseline_config),
        "lossConfig": asdict(loss_config),
        "manifests": {
            "fit": asdict(fit_manifest),
            "validation": asdict(validation_manifest),
        },
        "heldoutTestArchiveOpenedByTraining": False,
        "backbone": backbone_summary,
        "portPrecompute": port_summary,
        "portConfig": asdict(port_config),
        "backboneHash": backbone_hash,
        "variants": variant_summaries,
        "baseline": baseline_summary,
        "seconds": time.perf_counter() - started,
        "neuralParametersFrozenForPhysicalEvaluation": True,
        "actionGradientUpdates": 0,
        "physicalStateGradientUpdates": 0,
    }
    _atomic_json_save(summary, output_dir / "training-complete.json")
    return summary


def copy_probe_bank(probes: PixelChangeProbeBank) -> PixelChangeProbeBank:
    return PixelChangeProbeBank(probes.basis.detach().cpu().clone())


def _parse_variants(value: str) -> tuple[Variant, ...]:
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(requested) - set(REGISTERED_VARIANTS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {sorted(unknown)}")
    return requested  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    producer = subparsers.add_parser("generate")
    producer.add_argument("sanitized_root", type=Path)
    producer.add_argument("--system", choices=tuple(DIRECT_SYSTEMS), required=True)
    producer.add_argument("--producer-seed-file", type=Path, required=True)
    producer.add_argument("--source-manifest", type=Path, required=True)
    trainer = subparsers.add_parser("train")
    trainer.add_argument("sanitized_root", type=Path)
    trainer.add_argument("output_dir", type=Path)
    trainer.add_argument("--system", choices=tuple(DIRECT_SYSTEMS), required=True)
    trainer.add_argument("--initialization-checkpoint", type=Path)
    for subparser in (producer, trainer):
        subparser.add_argument("--fit-trajectories", type=int, default=4_096)
        subparser.add_argument("--validation-trajectories", type=int, default=512)
        subparser.add_argument("--test-trajectories", type=int, default=512)
        subparser.add_argument("--image-size", type=int, default=64)
        subparser.add_argument("--patch-size", type=int, default=4)
        subparser.add_argument("--cache-frames", type=int, default=24)
        subparser.add_argument("--transitions", type=int, default=8)
        subparser.add_argument("--backbone-preset", default="tiny")
    trainer.add_argument("--variants", type=_parse_variants, default=REGISTERED_VARIANTS)
    trainer.add_argument("--backbone-steps", type=int, default=30_000)
    trainer.add_argument("--direct-steps", type=int, default=30_000)
    trainer.add_argument("--baseline-steps", type=int, default=30_000)
    trainer.add_argument("--micro-batch-size", type=int, default=16)
    trainer.add_argument("--lens-batch-size", type=int, default=4)
    trainer.add_argument("--implicit-iterations", type=int, default=32)
    trainer.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = ExperimentFConfig(
        fit_trajectories=args.fit_trajectories,
        validation_trajectories=args.validation_trajectories,
        test_trajectories=args.test_trajectories,
        transitions=args.transitions,
        cache_frames=args.cache_frames,
        image_size=args.image_size,
        patch_size=args.patch_size,
        backbone_preset=args.backbone_preset,
        variants=getattr(args, "variants", REGISTERED_VARIANTS),
    )
    if args.stage == "generate":
        source_manifest = load_source_manifest(args.source_manifest)
        source_tree_sha256 = str(source_manifest["treeSha256"])
        runtime_trace = RuntimeFirewallTrace(
            args.sanitized_root
            / "seals"
            / args.system
            / "firewall-trace.jsonl",
            stage=f"producer:{args.system}",
            source_tree_sha256=source_tree_sha256,
        )
        runtime_trace.record_file_read(
            args.producer_seed_file, role="producer_private_seed_file"
        )
        runtime_trace.record_file_read(
            args.source_manifest,
            role="sealed_source_manifest",
            serialized_keys=tuple(sorted(source_manifest)),
            semantic_sha256=source_tree_sha256,
        )
        summary = generate_sanitized_splits(
            args.system,
            args.sanitized_root,
            experiment,
            producer_seed=private_producer_seed_from_file(
                args.producer_seed_file, system=args.system
            ),
            runtime_trace=runtime_trace,
            source_tree_sha256=source_tree_sha256,
        )
        runtime_trace.close()
    else:
        device = torch.device(args.device)
        summary = train_registered_system(
            args.system,
            args.sanitized_root,
            args.output_dir,
            experiment,
            ActionFreeBackboneTrainConfig(steps=args.backbone_steps),
            DirectTrainingConfig(
                steps=args.direct_steps,
                micro_batch_size=args.micro_batch_size,
                lens_batch_size=args.lens_batch_size,
                implicit_iterations=args.implicit_iterations,
            ),
            BaselineTrainingConfig(steps=args.baseline_steps),
            DirectVideoLossConfig(),
            device,
            initialization_checkpoint=args.initialization_checkpoint,
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
