"""Strict learner-side I/O for sanitized Experiment F pixel archives."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from .direct_action_free_data import (
    ActionFreeBackboneTrainConfig,
    PixelsOnlyManifest,
    build_validated_action_free_backbone,
    sanitized_pixel_tensor_sha256,
    train_action_free_backbone,
    validate_action_free_backbone_checkpoint,
)
from .experiment_f_contract import ExperimentFConfig
from .pixel_direct_model import PixelDirectConfig, pixel_direct_config_for_preset
from .pixel_palette import PALETTE
from .runtime_firewall_trace import RuntimeFirewallTrace


def sanitized_path(root: Path, system: str, split: str) -> Path:
    return root / system / f"{split}-pixels.pt"


def load_sanitized_split(
    path: Path,
    *,
    expected_system: str,
    runtime_trace: RuntimeFirewallTrace | None = None,
    trace_role: str = "sanitized_pixels_archive",
) -> tuple[torch.Tensor, PixelsOnlyManifest]:
    """Load a tensor-only archive through an exact positive schema."""

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
    expected_manifest_keys = set(
        asdict(PixelsOnlyManifest("x", 1, 1, 1, "a" * 64))
    )
    if type(raw_manifest) is not dict or set(raw_manifest) != expected_manifest_keys:
        raise ValueError("sanitized manifest schema is not exact")
    try:
        manifest = PixelsOnlyManifest(**raw_manifest)
    except (TypeError, ValueError) as error:
        raise ValueError("sanitized manifest is malformed") from error
    if type(manifest.system) is not str or manifest.system != expected_system:
        raise ValueError("sanitized archive belongs to another system")
    for name in ("trajectories", "frames_per_trajectory", "image_size"):
        value = getattr(manifest, name)
        if type(value) is not int or value < 1:
            raise ValueError(
                f"sanitized manifest field {name!r} must be positive int"
            )
    if (
        type(manifest.source_schema) is not tuple
        or manifest.source_schema != ("frames",)
    ):
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
    if type(pixels) is not torch.Tensor or pixels.dtype != torch.uint8:
        raise ValueError("sanitized pixels must be a plain uint8 tensor")
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
    if pixels.numel() < 1 or int(pixels.max()) >= len(PALETTE):
        raise ValueError("sanitized pixel tensor contains a non-palette class")
    return pixels, manifest


def experiment_f_model_config(config: ExperimentFConfig) -> PixelDirectConfig:
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
    """Build or resume the frozen backbone from sanitized pixels only."""

    checkpoint_path = output_dir / "checkpoint.pt"
    model_config = experiment_f_model_config(experiment_config)
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_action_free_backbone_checkpoint(
            payload,
            expected_manifest_sha256=fit_manifest.aggregate_sha256,
            expected_sanitized_tensor_sha256=fit_manifest.sanitized_tensor_sha256,
            expected_system=system,
        )
        if payload["model_config"] != model_config.to_dict():
            raise ValueError(
                "existing backbone architecture differs from registered config"
            )
        if payload["train_config"] != asdict(train_config):
            raise ValueError(
                "existing backbone training schedule differs from registered config"
            )
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file():
            raise ValueError(
                "completed backbone checkpoint is missing its atomic summary seal"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if type(summary) is not dict:
            raise ValueError("backbone summary is not a plain dictionary")
        return (
            build_validated_action_free_backbone(
                payload,
                expected_manifest_sha256=fit_manifest.aggregate_sha256,
                expected_sanitized_tensor_sha256=fit_manifest.sanitized_tensor_sha256,
                expected_system=system,
            ),
            summary,
        )
    initial = None
    if initialization_checkpoint is not None and initialization_checkpoint.exists():
        initial = torch.load(
            initialization_checkpoint, map_location="cpu", weights_only=True
        )
        validate_action_free_backbone_checkpoint(
            initial,
            expected_manifest_sha256=fit_manifest.aggregate_sha256,
            expected_sanitized_tensor_sha256=fit_manifest.sanitized_tensor_sha256,
            expected_system=system,
        )
    return train_action_free_backbone(
        fit_pixels,
        model_config,
        train_config,
        system=system,
        manifest=fit_manifest,
        output_dir=output_dir,
        device=device,
        initial_checkpoint=initial,
        runtime_trace=runtime_trace,
        source_tree_sha256=source_tree_sha256,
    )


__all__ = [
    "experiment_f_model_config",
    "load_sanitized_split",
    "prepare_action_free_backbone",
    "sanitized_path",
]
