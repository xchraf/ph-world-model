"""Simulator-free pixel tensors and backbone training for Experiment F.

The producer implementation lives in :mod:`direct_experiment_f_producer` and
is deliberately absent from the learner source bundle.  This module accepts
only already-sanitized palette tensors.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .pixel_palette import PALETTE
from .pixel_direct_model import (
    DirectPixelTransformer,
    PixelDirectConfig,
    build_pixel_direct_from_checkpoint,
)
from .tensor_provenance import module_tensor_hash
from .runtime_firewall_trace import RuntimeFirewallTrace
from .source_provenance import build_source_manifest


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    """Publish a complete checkpoint with one same-filesystem rename."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class PixelsOnlyManifest:
    system: str
    trajectories: int
    frames_per_trajectory: int
    image_size: int
    aggregate_sha256: str
    sanitized_tensor_sha256: str = ""
    source_schema: tuple[str, ...] = ("frames",)
    optimization_schema: tuple[str, ...] = ("pixelContexts", "frames")


@dataclass(frozen=True)
class ActionFreeBackboneTrainConfig:
    steps: int = 30_000
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1_000
    minimum_learning_rate_ratio: float = 0.1
    ema_decay: float = 0.9995
    log_every: int = 100

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "log_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"backbone {name} must be a positive integer")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("backbone warmup_steps must be a non-negative integer")
        finite = (
            self.learning_rate,
            self.weight_decay,
            self.minimum_learning_rate_ratio,
            self.ema_decay,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("backbone scalar configuration must be finite")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("backbone learning_rate/weight_decay are invalid")
        if not 0.0 <= self.minimum_learning_rate_ratio <= 1.0:
            raise ValueError("minimum_learning_rate_ratio must lie in [0,1]")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0,1)")


def classes_from_rgb(frames: np.ndarray) -> torch.Tensor:
    """Map rendered RGB pixels to the fixed game palette exactly."""

    palette = np.stack(tuple(PALETTE.values())).astype(np.int32)
    rgb = frames.astype(np.int32)
    distance = ((rgb[..., None, :] - palette) ** 2).sum(axis=-1)
    return torch.from_numpy(distance.argmin(axis=-1).astype(np.uint8))


def _aggregate_pixel_hash(hashes: list[str], shape: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(str(shape).encode("ascii"))
    for item in hashes:
        digest.update(item.encode("ascii"))
    return digest.hexdigest()


def sanitized_pixel_tensor_sha256(pixels: torch.Tensor) -> str:
    tensor = pixels.detach().cpu().contiguous()
    if tensor.dtype != torch.uint8 or tensor.ndim != 4:
        raise ValueError("sanitized pixels must be uint8 [trajectory,time,height,width]")
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_optimization_suite(
    pixel_videos: torch.Tensor,
    model_config: PixelDirectConfig,
    *,
    transitions: int,
) -> dict[str, torch.Tensor]:
    """Derive sliding histories and targets from palette pixels only."""

    if pixel_videos.ndim != 4:
        raise ValueError("pixel_videos must have shape [trajectory,time,height,width]")
    required = model_config.history_frames + transitions
    if pixel_videos.shape[1] < required:
        raise ValueError(f"pixel videos need at least {required} frames")
    contexts = torch.stack(
        tuple(
            pixel_videos[:, offset : offset + model_config.history_frames]
            for offset in range(transitions + 1)
        ),
        dim=1,
    )
    frames = pixel_videos[
        :, model_config.history_frames - 1 : model_config.history_frames + transitions
    ]
    suite = {"pixelContexts": contexts.contiguous(), "frames": frames.contiguous()}
    if set(suite) != {"pixelContexts", "frames"}:
        raise AssertionError("optimization suite violated the pixels-only firewall")
    return suite


def class_weights(
    frames: torch.Tensor,
    palette_size: int,
    device: torch.device,
) -> torch.Tensor:
    counts = torch.bincount(frames.flatten().long(), minlength=palette_size).float()
    frequencies = counts / counts.sum().clamp_min(1.0)
    weights = (frequencies.max() / frequencies.clamp_min(1e-8)).sqrt().clamp(0.25, 12.0)
    weights /= weights[1].clamp_min(1e-6)
    return weights.to(device)


def weighted_pixel_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-3], logits.shape[-2], logits.shape[-1]).float(),
        targets.reshape(-1, targets.shape[-2], targets.shape[-1]).long(),
        weight=weights,
    )


def _learning_rate(step: int, config: ActionFreeBackboneTrainConfig) -> float:
    if step <= config.warmup_steps:
        multiplier = step / max(config.warmup_steps, 1)
    else:
        progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        multiplier = config.minimum_learning_rate_ratio + (
            1.0 - config.minimum_learning_rate_ratio
        ) * cosine
    return config.learning_rate * multiplier


def train_action_free_backbone(
    pixel_videos: torch.Tensor,
    model_config: PixelDirectConfig,
    train_config: ActionFreeBackboneTrainConfig,
    *,
    system: str,
    manifest: PixelsOnlyManifest,
    output_dir: Path,
    device: torch.device,
    initial_checkpoint: dict[str, Any] | None = None,
    runtime_trace: RuntimeFirewallTrace | None = None,
    source_tree_sha256: str | None = None,
) -> tuple[DirectPixelTransformer, dict[str, Any]]:
    """Pretrain a video transformer using no tensor other than pixels."""

    output_dir.mkdir(parents=True, exist_ok=True)
    owns_runtime_trace = runtime_trace is None
    if runtime_trace is None:
        if source_tree_sha256 is None:
            source_tree_sha256 = str(build_source_manifest()["treeSha256"])
        runtime_trace = RuntimeFirewallTrace(
            output_dir / "firewall-trace.jsonl",
            stage=f"backbone:{system}",
            source_tree_sha256=source_tree_sha256,
        )
    if initial_checkpoint is None:
        model = DirectPixelTransformer(model_config)
    else:
        validate_action_free_backbone_checkpoint(
            initial_checkpoint,
            expected_manifest_sha256=manifest.aggregate_sha256,
            expected_sanitized_tensor_sha256=manifest.sanitized_tensor_sha256,
            expected_system=system,
        )
        if initial_checkpoint["model_config"] != model_config.to_dict():
            raise ValueError("initial checkpoint architecture does not match requested backbone")
        candidate = DirectPixelTransformer(model_config)
        candidate.load_state_dict(initial_checkpoint["model"])
        model = candidate
    model = model.to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    runtime_trace.record_optimizer(
        phase="backbone", named_parameters=dict(model.named_parameters())
    )
    runtime_trace.record_backbone_boundary(
        phase="backbone", boundary="optimization_start", sha256=module_tensor_hash(model)
    )
    pixels = pixel_videos.to(device)
    weights = class_weights(pixel_videos, model_config.palette_size, device)
    available_starts = pixels.shape[1] - model_config.history_frames
    if available_starts < 1:
        raise ValueError("backbone cache needs one target beyond the history")
    offsets = torch.arange(model_config.history_frames, device=device)[None]
    started = time.perf_counter()
    log_path = output_dir / "backbone-train.jsonl"
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, train_config.steps + 1):
            rows = torch.randint(
                0, pixels.shape[0], (train_config.batch_size,), device=device
            )
            starts = torch.randint(
                0, available_starts, (train_config.batch_size,), device=device
            )
            indices = starts[:, None] + offsets
            inputs = pixels[rows[:, None], indices].long()
            targets = pixels[rows[:, None], indices + 1].long()
            # The target is a deterministic view of the same pixels-only
            # tensor.  The trace records the actual externally supplied batch
            # key, not internal loss aliases such as ``targets``.
            runtime_trace.record_gradient_batch(
                phase="backbone", step=step, tensors={"pixels": inputs}
            )
            learning_rate = _learning_rate(step, train_config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = weighted_pixel_cross_entropy(model(inputs), targets, weights)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            decay = min(train_config.ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
                    ema_parameter.lerp_(parameter, 1.0 - decay)
            if step == 1 or step % train_config.log_every == 0 or step == train_config.steps:
                elapsed = time.perf_counter() - started
                record = {
                    "stage": "action_free_backbone",
                    "system": system,
                    "step": step,
                    "steps": train_config.steps,
                    "loss": float(loss.detach()),
                    "gradientNorm": float(gradient_norm),
                    "learningRate": learning_rate,
                    "seconds": elapsed,
                    "estimatedSeconds": elapsed / step * train_config.steps,
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
    ema.eval().requires_grad_(False)
    checkpoint = {
        "kind": "passive_direct_pixel_world_model",
        "system": system,
        "actionChannels": 0,
        "optimizationTensorKeys": ["pixels"],
        "pixelsOnlyManifest": asdict(manifest),
        # Plain dict (not an arbitrary Mapping subclass) is part of the closed
        # serialization contract checked before the frozen model is rebuilt.
        "model": dict(ema.state_dict()),
        "model_config": model_config.to_dict(),
        "train_config": asdict(train_config),
        "step": train_config.steps,
    }
    _atomic_torch_save(checkpoint, output_dir / "checkpoint.pt")
    runtime_trace.record_backbone_boundary(
        phase="backbone", boundary="selected_checkpoint", sha256=module_tensor_hash(ema)
    )
    runtime_trace_seal = runtime_trace.snapshot().to_dict()
    if owns_runtime_trace:
        runtime_trace.close()
    summary = {
        "system": system,
        "seconds": time.perf_counter() - started,
        "finalLoss": float(loss.detach()),
        "checkpoint": str(output_dir / "checkpoint.pt"),
        "pixelsOnlyManifest": asdict(manifest),
        "runtimeTrace": runtime_trace_seal,
    }
    _atomic_json_save(summary, output_dir / "summary.json")
    return ema, summary


def validate_action_free_backbone_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_manifest_sha256: str,
    expected_sanitized_tensor_sha256: str,
    expected_system: str,
) -> None:
    """Fail closed unless a checkpoint has the exact pixels-only schema.

    This is deliberately a positive-schema validator, not a blacklist.  Every
    scalar, string, nested key, and tensor in the archive has one prescribed
    location and type.  Consequently an archive cannot smuggle an effort,
    simulator state, or descriptive side channel under an innocuous spelling.
    The two expected digests and the expected system are supplied by the
    independently loaded sanitized pixel archive.
    """

    if type(checkpoint) is not dict:
        raise ValueError("checkpoint must be a plain dictionary")

    allowed_top_level = {
        "kind",
        "system",
        "actionChannels",
        "optimizationTensorKeys",
        "pixelsOnlyManifest",
        "model",
        "model_config",
        "train_config",
        "step",
    }
    if set(checkpoint) != allowed_top_level:
        raise ValueError(
            f"checkpoint top-level schema mismatch: {sorted(set(checkpoint) ^ allowed_top_level)}"
        )
    if (
        type(checkpoint["kind"]) is not str
        or checkpoint["kind"] != "passive_direct_pixel_world_model"
    ):
        raise ValueError("checkpoint is not a frozen pixel world model")
    if type(expected_system) is not str or re.fullmatch(r"[a-z][a-z0-9_-]*", expected_system) is None:
        raise ValueError("expected system must be a canonical ASCII identifier")
    if type(checkpoint["system"]) is not str or checkpoint["system"] != expected_system:
        raise ValueError("checkpoint system does not match the sanitized archive")
    if type(checkpoint["actionChannels"]) is not int or checkpoint["actionChannels"] != 0:
        raise ValueError("checkpoint does not seal zero action channels")
    if (
        type(checkpoint["optimizationTensorKeys"]) is not list
        or checkpoint["optimizationTensorKeys"] != ["pixels"]
    ):
        raise ValueError("backbone optimization schema was not exactly pixels-only")
    manifest = checkpoint["pixelsOnlyManifest"]
    if type(manifest) is not dict:
        raise ValueError("checkpoint is missing its sealed pixels-only manifest")
    manifest_keys = {field.name for field in fields(PixelsOnlyManifest)}
    if set(manifest) != manifest_keys:
        raise ValueError("pixels-only manifest schema is not exact")
    if type(manifest["system"]) is not str or manifest["system"] != expected_system:
        raise ValueError("pixels-only manifest system does not match")
    for name in ("trajectories", "frames_per_trajectory", "image_size"):
        if type(manifest[name]) is not int or manifest[name] < 1:
            raise ValueError(f"pixels-only manifest field {name!r} must be positive int")
    if (
        type(manifest["source_schema"]) is not tuple
        or manifest["source_schema"] != ("frames",)
    ):
        raise ValueError("backbone source schema was not the one-key frame payload")
    if (
        type(manifest["optimization_schema"]) is not tuple
        or manifest["optimization_schema"] != ("pixelContexts", "frames")
    ):
        raise ValueError("backbone optimization manifest schema is not pixels-only")
    digest = manifest["aggregate_sha256"]
    sanitized_digest = manifest["sanitized_tensor_sha256"]
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("backbone manifest is missing a SHA-256 digest")
    if (
        type(sanitized_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", sanitized_digest) is None
    ):
        raise ValueError("backbone manifest is missing its sanitized tensor SHA-256")
    for name, expected in (
        ("manifest", expected_manifest_sha256),
        ("sanitized tensor", expected_sanitized_tensor_sha256),
    ):
        if type(expected) is not str or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"expected {name} SHA-256 must be 64 lowercase hex characters")
    if digest != expected_manifest_sha256:
        raise ValueError("backbone pixels-only manifest hash does not match")
    if sanitized_digest != expected_sanitized_tensor_sha256:
        raise ValueError("backbone sanitized pixel tensor hash does not match")

    model_config = checkpoint["model_config"]
    model_config_keys = {field.name for field in fields(PixelDirectConfig)}
    if type(model_config) is not dict or set(model_config) != model_config_keys:
        raise ValueError("pixel transformer configuration schema is not exact")
    integer_config_fields = {
        "image_size",
        "patch_size",
        "palette_size",
        "history_frames",
        "pixel_embedding_size",
        "hidden_size",
        "depth",
        "heads",
    }
    for name in integer_config_fields:
        if type(model_config[name]) is not int or model_config[name] < 1:
            raise ValueError(f"model configuration field {name!r} must be positive int")
    if type(model_config["mlp_ratio"]) not in (int, float):
        raise ValueError("model configuration mlp_ratio must be numeric")
    if not math.isfinite(float(model_config["mlp_ratio"])) or model_config["mlp_ratio"] <= 0:
        raise ValueError("model configuration mlp_ratio must be finite and positive")
    if model_config["image_size"] != manifest["image_size"]:
        raise ValueError("model image size does not match the pixels-only manifest")
    try:
        parsed_model_config = PixelDirectConfig(**model_config)
        reference_state = DirectPixelTransformer(parsed_model_config).state_dict()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("invalid pixel transformer configuration") from error

    train_config = checkpoint["train_config"]
    train_config_keys = {field.name for field in fields(ActionFreeBackboneTrainConfig)}
    if type(train_config) is not dict or set(train_config) != train_config_keys:
        raise ValueError("backbone training configuration schema is not exact")
    integer_train_fields = {"steps", "batch_size", "warmup_steps", "log_every"}
    for name in integer_train_fields:
        minimum = 0 if name in {"warmup_steps", "log_every"} else 1
        if type(train_config[name]) is not int or train_config[name] < minimum:
            raise ValueError(f"training configuration field {name!r} has invalid type/value")
    for name in train_config_keys - integer_train_fields:
        value = train_config[name]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError(f"training configuration field {name!r} must be finite numeric")
    if train_config["learning_rate"] <= 0 or train_config["weight_decay"] < 0:
        raise ValueError("training learning rate/weight decay are invalid")
    if not 0 <= train_config["minimum_learning_rate_ratio"] <= 1:
        raise ValueError("minimum learning-rate ratio must lie in [0,1]")
    if not 0 <= train_config["ema_decay"] < 1:
        raise ValueError("EMA decay must lie in [0,1)")
    if type(checkpoint["step"]) is not int or checkpoint["step"] != train_config["steps"]:
        raise ValueError("checkpoint step does not match its sealed training configuration")

    state = checkpoint["model"]
    if type(state) is not dict or not state:
        raise ValueError("checkpoint model must be a non-empty tensor mapping")
    if set(state) != set(reference_state):
        raise ValueError("checkpoint model state schema does not match the declared transformer")
    ascii_parameter_name = re.compile(r"[A-Za-z0-9_.]+")
    for name, expected_tensor in reference_state.items():
        tensor = state[name]
        if type(name) is not str or ascii_parameter_name.fullmatch(name) is None:
            raise ValueError("checkpoint contains a non-canonical parameter name")
        if type(tensor) is not torch.Tensor:
            raise ValueError(f"checkpoint parameter {name!r} is not a plain tensor")
        if tensor.shape != expected_tensor.shape or tensor.dtype != expected_tensor.dtype:
            raise ValueError(f"checkpoint parameter {name!r} shape/dtype mismatch")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"checkpoint parameter {name!r} is not finite")


def build_validated_action_free_backbone(
    checkpoint: dict[str, Any],
    *,
    expected_manifest_sha256: str,
    expected_sanitized_tensor_sha256: str,
    expected_system: str,
) -> DirectPixelTransformer:
    validate_action_free_backbone_checkpoint(
        checkpoint,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_sanitized_tensor_sha256=expected_sanitized_tensor_sha256,
        expected_system=expected_system,
    )
    return build_pixel_direct_from_checkpoint(checkpoint).eval().requires_grad_(False)


__all__ = [
    "ActionFreeBackboneTrainConfig",
    "PixelsOnlyManifest",
    "class_weights",
    "build_validated_action_free_backbone",
    "classes_from_rgb",
    "make_optimization_suite",
    "sanitized_pixel_tensor_sha256",
    "train_action_free_backbone",
    "validate_action_free_backbone_checkpoint",
    "weighted_pixel_cross_entropy",
]
