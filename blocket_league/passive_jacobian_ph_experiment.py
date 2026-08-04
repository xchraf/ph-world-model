from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import make_passive_clip
from .end_to_end_ph_experiment import LatentPatchTransformerRenderer
from .env import BlocketLeagueEnv, PALETTE, WorldConfig, WorldState
from .neural_port_hamiltonian import NeuralPortHamiltonian, NeuralPortHamiltonianConfig
from .passive_control_systems import (
    PendulumConfig,
    PendulumEnv,
    PendulumState,
    make_passive_pendulum_clip,
    pendulum_target_frames,
    wrap_angle,
)
from .passive_jacobian_ph_model import (
    FrozenTransformerStateAdapter,
    PassiveVisualPHModel,
    UnstructuredPortConfig,
    UnstructuredPortDynamics,
    matched_unstructured_hidden_size,
    module_tensor_hash,
    parameter_count,
)
from .pixel_direct_model import (
    DirectPixelTransformer,
    PixelDirectConfig,
    build_pixel_direct_from_checkpoint,
    pixel_direct_config_for_preset,
)


@dataclass(frozen=True)
class PassiveJacobianPHConfig:
    seed: int = 141_810_733
    fit_trajectories: int = 2_048
    test_trajectories: int = 256
    transitions: int = 8
    visual_steps: int = 1_200
    passive_dynamics_steps: int = 4_000
    baseline_steps: int = 3_000
    port_fit_samples: int = 512
    port_test_samples: int = 256
    port_steps: int = 1_500
    batch_size: int = 16
    state_hidden_size: int = 192
    decoder_hidden_size: int = 192
    decoder_depth: int = 3
    decoder_heads: int = 6
    ph_hidden_size: int = 64
    ph_hidden_layers: int = 2
    learning_rate: float = 3e-4
    port_learning_rate: float = 2e-4
    lens_strength: float = 4.0
    calibration_states_per_axis: int = 4
    calibration_amplitude: float = 0.35
    realizability_states: int = 64
    control_episodes: int = 32
    control_steps: int = 12
    planner_horizon: int = 8
    planner_iterations: int = 30
    planner_learning_rate: float = 0.25
    pendulum_backbone_steps: int = 12_000
    pendulum_backbone_cache: int = 4_096
    pendulum_backbone_batch_size: int = 16
    log_every: int = 100


@dataclass(frozen=True)
class SystemDefinition:
    name: str
    state_size: int
    input_size: int
    dt: float
    entity_values: tuple[int, ...]
    observable: str
    lens_block: int


SYSTEMS = {
    "blocket": SystemDefinition(
        name="blocket",
        state_size=8,
        input_size=2,
        dt=WorldConfig().dt,
        entity_values=(5, 6),
        observable="centroid",
        lens_block=4,
    ),
    "pendulum": SystemDefinition(
        name="pendulum",
        state_size=2,
        input_size=1,
        dt=PendulumConfig().dt,
        entity_values=(7, 8),
        observable="pendulum_angle",
        lens_block=4,
    ),
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _classes_from_rgb(frames: np.ndarray) -> torch.Tensor:
    # Squared 8-bit colour differences reach 65,025 per channel and therefore
    # overflow int16.  int32 keeps the exact palette matching used by every
    # visual audit and prevents absent-entity centroids from silently becoming 0.
    palette = np.stack(tuple(PALETTE.values())).astype(np.int32)
    rgb = frames.astype(np.int32)
    distance = ((rgb[..., None, :] - palette) ** 2).sum(axis=-1)
    return torch.from_numpy(distance.argmin(axis=-1).astype(np.uint8))


def _soft_centroid(logits: torch.Tensor, values: tuple[int, ...]) -> torch.Tensor:
    probability = logits.float().softmax(dim=1)[:, list(values)].sum(dim=1)
    height, width = probability.shape[-2:]
    x = torch.arange(width, device=logits.device, dtype=torch.float32) + 0.5
    y = torch.arange(height, device=logits.device, dtype=torch.float32) + 0.5
    mass = probability.sum(dim=(-2, -1)).clamp_min(1e-7)
    return torch.stack(
        (
            (probability * x).sum(dim=(-2, -1)) / mass,
            (probability * y[:, None]).sum(dim=(-2, -1)) / mass,
        ),
        dim=-1,
    )


def _hard_centroid(classes: torch.Tensor, values: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(classes, dtype=torch.float32)
    for value in values:
        mask.add_(classes.eq(value))
    height, width = classes.shape[-2:]
    x = torch.arange(width, device=classes.device, dtype=torch.float32) + 0.5
    y = torch.arange(height, device=classes.device, dtype=torch.float32) + 0.5
    mass = mask.sum(dim=(-2, -1)).clamp_min(1e-7)
    return torch.stack(
        ((mask * x).sum(dim=(-2, -1)) / mass, (mask * y[:, None]).sum(dim=(-2, -1)) / mass),
        dim=-1,
    )


def _observable_from_logits(logits: torch.Tensor, system: SystemDefinition) -> torch.Tensor:
    centroid = _soft_centroid(logits, system.entity_values)
    if system.observable == "centroid":
        return centroid
    pivot = logits.shape[-1] * torch.tensor((0.5, 0.43), device=logits.device)
    delta = centroid - pivot
    return torch.atan2(delta[:, 0], delta[:, 1])[:, None]


def _observable_from_classes(classes: torch.Tensor, system: SystemDefinition) -> torch.Tensor:
    centroid = _hard_centroid(classes, system.entity_values)
    if system.observable == "centroid":
        return centroid
    pivot = classes.shape[-1] * torch.tensor((0.5, 0.43), device=classes.device)
    delta = centroid - pivot
    return torch.atan2(delta[:, 0], delta[:, 1])[:, None]


def _wrapped_difference(first: torch.Tensor, second: torch.Tensor, system: SystemDefinition) -> torch.Tensor:
    difference = first - second
    if system.observable == "pendulum_angle":
        difference = torch.atan2(torch.sin(difference), torch.cos(difference))
    return difference


def _entity_token_mask(
    model: DirectPixelTransformer,
    contexts: torch.Tensor,
    values: tuple[int, ...],
) -> torch.Tensor:
    centroid = _hard_centroid(contexts[:, -1], values)
    patch_x = (centroid[:, 0] / model.config.patch_size).long().clamp(0, model.config.grid_size - 1)
    patch_y = (centroid[:, 1] / model.config.patch_size).long().clamp(0, model.config.grid_size - 1)
    token = patch_y * model.config.grid_size + patch_x
    mask = torch.zeros(
        contexts.shape[0], contexts.shape[1], model.config.grid_size**2, device=contexts.device
    )
    mask[torch.arange(contexts.shape[0], device=contexts.device), -1, token] = 1.0
    return mask


def _passive_clip_frames(system: SystemDefinition, seed: int, frames: int, image_size: int) -> np.ndarray:
    if system.name == "blocket":
        payload = make_passive_clip(
            seed, context_frames=1, future_frames=frames - 1, image_size=image_size
        )
    else:
        payload = make_passive_pendulum_clip(
            seed, context_frames=1, future_frames=frames - 1, image_size=image_size
        )
    return payload["frames"]


@torch.no_grad()
def collect_passive_pixels(
    system: SystemDefinition,
    model_config: PixelDirectConfig,
    *,
    trajectories: int,
    transitions: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    contexts = []
    frames = []
    count = model_config.history_frames + transitions
    started = time.perf_counter()
    for index in range(trajectories):
        classes = _classes_from_rgb(
            _passive_clip_frames(
                system,
                seed + index * 9_973,
                count,
                model_config.image_size,
            )
        )
        contexts.append(
            torch.stack(
                [classes[offset : offset + model_config.history_frames] for offset in range(transitions + 1)]
            )
        )
        frames.append(classes[model_config.history_frames - 1 :])
        if (index + 1) % 256 == 0 or index + 1 == trajectories:
            print(
                json.dumps(
                    {
                        "stage": "collect_passive_pixels",
                        "system": system.name,
                        "trajectories": index + 1,
                        "total": trajectories,
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    result = {"pixelContexts": torch.stack(contexts), "frames": torch.stack(frames)}
    if set(result) != {"pixelContexts", "frames"}:
        raise AssertionError("passive optimization suite violated the information firewall")
    return result


def _class_weights(frames: torch.Tensor, palette_size: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(frames.flatten().long(), minlength=palette_size).float()
    frequency = counts / counts.sum()
    weights = (frequency.max() / frequency.clamp_min(1e-8)).sqrt().clamp(0.25, 12.0)
    weights /= weights[1].clamp_min(1e-6)
    return weights.to(device)


def _pixel_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-3], logits.shape[-2], logits.shape[-1]),
        target.reshape(-1, target.shape[-2], target.shape[-1]).long(),
        weight=weights,
    )


def _whitening_loss(state: torch.Tensor) -> torch.Tensor:
    flat = state.reshape(-1, state.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False).clamp_min(1e-4)
    standardized = centered / scale
    covariance = standardized.T @ standardized / max(flat.shape[0], 1)
    identity = torch.eye(flat.shape[-1], device=flat.device)
    return (covariance - identity).square().mean() + (scale.log().square()).mean() * 0.02


def train_pendulum_backbone(
    output_dir: Path,
    config: PassiveJacobianPHConfig,
    device: torch.device,
) -> Path:
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        return checkpoint_path
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = pixel_direct_config_for_preset(
        "tiny", image_size=64, patch_size=4, palette_size=len(PALETTE), history_frames=8
    )
    cache = []
    for index in range(config.pendulum_backbone_cache):
        cache.append(
            _classes_from_rgb(
                make_passive_pendulum_clip(
                    config.seed + 10_000_000 + index * 9_973,
                    context_frames=1,
                    future_frames=23,
                    image_size=64,
                )["frames"]
            )
        )
    cached = torch.stack(cache).to(device)
    weights = _class_weights(cached.cpu(), len(PALETTE), device)
    model = DirectPixelTransformer(model_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95), fused=device.type == "cuda"
    )
    offsets = torch.arange(model_config.history_frames, device=device)[None]
    started = time.perf_counter()
    log_path = output_dir / "train.jsonl"
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.pendulum_backbone_steps + 1):
            rows = torch.randint(0, cached.shape[0], (config.pendulum_backbone_batch_size,), device=device)
            starts = torch.randint(0, cached.shape[1] - model_config.history_frames, (rows.shape[0],), device=device)
            indices = starts[:, None] + offsets
            inputs = cached[rows[:, None], indices].long()
            targets = cached[rows[:, None], indices + 1].long()
            progress = step / config.pendulum_backbone_steps
            learning_rate = 3e-4 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = _pixel_loss(logits, targets, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            decay = min(0.9995, (1 + step) / (10 + step))
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
                    ema_parameter.lerp_(parameter, 1.0 - decay)
            if step == 1 or step % config.log_every == 0 or step == config.pendulum_backbone_steps:
                record = {
                    "stage": "pretrain_passive_pendulum_transformer",
                    "step": step,
                    "steps": config.pendulum_backbone_steps,
                    "loss": float(loss.detach()),
                    "seconds": time.perf_counter() - started,
                    "estimatedSeconds": (time.perf_counter() - started) / step * config.pendulum_backbone_steps,
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
    payload = {
        "kind": "passive_direct_pixel_world_model",
        "system": "pendulum",
        "actionChannels": 0,
        "model": ema.state_dict(),
        "model_config": model_config.to_dict(),
        "step": config.pendulum_backbone_steps,
        "seed": config.seed,
    }
    torch.save(payload, checkpoint_path)
    del cached, model, ema, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_path


def _make_visual_model(
    backbone: DirectPixelTransformer,
    system: SystemDefinition,
    config: PassiveJacobianPHConfig,
    device: torch.device,
) -> PassiveVisualPHModel:
    adapter = FrozenTransformerStateAdapter(
        backbone,
        system.state_size,
        config.state_hidden_size,
        lens_block=min(system.lens_block, len(backbone.blocks) - 1),
    )
    renderer = LatentPatchTransformerRenderer(
        system.state_size,
        image_size=backbone.config.image_size,
        patch_size=backbone.config.patch_size,
        palette_size=backbone.config.palette_size,
        hidden_size=config.decoder_hidden_size,
        depth=config.decoder_depth,
        heads=config.decoder_heads,
    )
    core = NeuralPortHamiltonian(
        NeuralPortHamiltonianConfig(
            state_size=system.state_size,
            input_size=system.input_size,
            hidden_size=config.ph_hidden_size,
            hidden_layers=config.ph_hidden_layers,
            dt=system.dt,
            integration_method="passivity",
            integration_substeps=1,
            resistance_floor=1e-5,
        )
    )
    return PassiveVisualPHModel(adapter, renderer, core).to(device)


def _move_suite(suite: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in suite.items()}


def train_passive_visual_ph(
    model: PassiveVisualPHModel,
    suite: dict[str, torch.Tensor],
    weights: torch.Tensor,
    config: PassiveJacobianPHConfig,
    output_dir: Path,
) -> dict[str, Any]:
    device = suite["frames"].device
    for parameter in model.core.port_network.parameters():
        parameter.requires_grad_(False)
    port_hash_before = module_tensor_hash(model.core.port_network)
    visual_parameters = [
        parameter
        for module in (model.adapter.pool_score, model.adapter.readout, model.renderer)
        for parameter in module.parameters()
    ]
    visual_optimizer = torch.optim.AdamW(visual_parameters, lr=config.learning_rate, weight_decay=1e-5)
    started = time.perf_counter()
    for step in range(1, config.visual_steps + 1):
        rows = torch.randint(0, suite["frames"].shape[0], (config.batch_size,), device=device)
        endpoints = torch.randint(0, config.transitions + 1, (config.batch_size,), device=device)
        visual_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            state = model.encode(suite["pixelContexts"][rows, endpoints].long())
            reconstruction = _pixel_loss(model.renderer(state), suite["frames"][rows, endpoints], weights)
            whitening = _whitening_loss(state)
            loss = reconstruction + 0.05 * whitening
        loss.backward()
        torch.nn.utils.clip_grad_norm_(visual_parameters, 5.0)
        visual_optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.visual_steps:
            print(json.dumps({"stage": "passive_visual_adapter", "step": step, "steps": config.visual_steps,
                              "loss": float(loss.detach()), "reconstruction": float(reconstruction.detach()),
                              "seconds": time.perf_counter() - started}), flush=True)

    passive_parameters = visual_parameters + [
        parameter for name, parameter in model.core.named_parameters() if not name.startswith("port_network")
    ]
    optimizer = torch.optim.AdamW(passive_parameters, lr=config.learning_rate, weight_decay=1e-5)
    log_path = output_dir / "passive-training.jsonl"
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.passive_dynamics_steps + 1):
            rows = torch.randint(0, suite["frames"].shape[0], (config.batch_size,), device=device)
            anchors = torch.randint(1, config.transitions + 1, (config.batch_size,), device=device)
            batch_rows = torch.arange(config.batch_size, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                initial = model.encode(suite["pixelContexts"][rows, 0].long())
                teacher = model.encode(suite["pixelContexts"][rows, anchors].long())
            current = initial.float()
            zeros = torch.zeros(config.batch_size, model.core.config.input_size, device=device)
            states = []
            for _ in range(config.transitions):
                current = model.step(current, zeros)
                states.append(current)
            rollout = torch.stack(states, dim=1)
            predicted = rollout[batch_rows, anchors - 1]
            latent = (predicted - teacher.float().detach()).square().mean()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pixel = _pixel_loss(
                    model.renderer(predicted), suite["frames"][rows, anchors], weights
                )
                reconstruction = _pixel_loss(
                    model.renderer(torch.stack((initial, teacher), dim=1)),
                    torch.stack((suite["frames"][rows, 0], suite["frames"][rows, anchors]), dim=1),
                    weights,
                )
                whitening = _whitening_loss(torch.cat((initial, teacher), dim=0))
            energy = model.core.hamiltonian(torch.cat((initial.float(), teacher.float()), dim=0))
            energy_gauge = energy.mean().square() + (energy.std(unbiased=False) - 1.0).square()
            loss = latent + pixel + 0.4 * reconstruction + 0.04 * whitening + 0.02 * energy_gauge
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(passive_parameters, 5.0)
            optimizer.step()
            if step == 1 or step % config.log_every == 0 or step == config.passive_dynamics_steps:
                record = {
                    "stage": "fit_action_free_ph_latent",
                    "step": step,
                    "steps": config.passive_dynamics_steps,
                    "loss": float(loss.detach()),
                    "latent": float(latent.detach()),
                    "pixel": float(pixel.detach()),
                    "reconstruction": float(reconstruction.detach()),
                    "gradientNorm": float(gradient_norm),
                    "seconds": time.perf_counter() - started,
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
    if module_tensor_hash(model.core.port_network) != port_hash_before:
        raise AssertionError("B changed during passive trajectory fitting")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {"seconds": time.perf_counter() - started, "portUnchangedDuringPassiveFit": True}


@torch.no_grad()
def encode_suite(
    adapter: FrozenTransformerStateAdapter,
    suite: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> torch.Tensor:
    contexts = suite["pixelContexts"]
    flat = contexts.reshape(-1, *contexts.shape[-3:])
    states = []
    for start in range(0, flat.shape[0], batch_size):
        states.append(adapter(flat[start : start + batch_size].long()).float())
    return torch.cat(states).reshape(*contexts.shape[:2], -1)


def train_unstructured_baseline(
    ph_core: NeuralPortHamiltonian,
    states: torch.Tensor,
    system: SystemDefinition,
    config: PassiveJacobianPHConfig,
) -> tuple[UnstructuredPortDynamics, dict[str, Any]]:
    target = parameter_count(ph_core)
    hidden = matched_unstructured_hidden_size(
        target,
        state_size=system.state_size,
        input_size=system.input_size,
        hidden_layers=config.ph_hidden_layers,
        dt=system.dt,
    )
    mean = states[:, :-1].reshape(-1, system.state_size).mean(dim=0)
    scale = states[:, :-1].reshape(-1, system.state_size).std(dim=0).clamp_min(1e-4)
    baseline = UnstructuredPortDynamics(
        UnstructuredPortConfig(
            state_size=system.state_size,
            input_size=system.input_size,
            hidden_size=hidden,
            hidden_layers=config.ph_hidden_layers,
            dt=system.dt,
        ),
        state_mean=mean,
        state_scale=scale,
    ).to(states.device)
    for parameter in baseline.port_network.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(baseline.drift_network.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    started = time.perf_counter()
    for step in range(1, config.baseline_steps + 1):
        rows = torch.randint(0, states.shape[0], (config.batch_size,), device=states.device)
        anchors = torch.randint(1, config.transitions + 1, (config.batch_size,), device=states.device)
        batch_rows = torch.arange(config.batch_size, device=states.device)
        current = states[rows, 0]
        zero = torch.zeros(config.batch_size, system.input_size, device=states.device)
        rollout = []
        for _ in range(config.transitions):
            current = baseline(current, zero)
            rollout.append(current)
        predicted = torch.stack(rollout, dim=1)[batch_rows, anchors - 1]
        target_state = states[rows, anchors]
        loss = ((predicted - target_state) / scale).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline.drift_network.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.baseline_steps:
            print(json.dumps({"stage": "fit_action_free_unstructured_baseline", "step": step,
                              "steps": config.baseline_steps, "loss": float(loss.detach()),
                              "seconds": time.perf_counter() - started}), flush=True)
    baseline.eval().requires_grad_(False)
    gap = abs(parameter_count(ph_core) - parameter_count(baseline)) / max(parameter_count(ph_core), 1)
    return baseline, {
        "hiddenSize": hidden,
        "phParameters": parameter_count(ph_core),
        "baselineParameters": parameter_count(baseline),
        "relativeParameterGap": gap,
        "seconds": time.perf_counter() - started,
    }


def extract_jacobian_ports(
    model: PassiveVisualPHModel,
    contexts: torch.Tensor,
    system: SystemDefinition,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states, ports, directions = [], [], []
    backbone = model.adapter.backbone
    for start in range(0, contexts.shape[0], batch_size):
        batch = contexts[start : start + batch_size].long()
        mask = _entity_token_mask(backbone, batch, system.entity_values)
        write = torch.zeros(
            batch.shape[0], backbone.config.hidden_size, device=batch.device, requires_grad=True
        )
        logits = backbone(
            batch,
            intervention_block=model.adapter.lens_block,
            intervention=write,
            intervention_mask=mask,
        )[:, -1]
        observable = _observable_from_logits(logits, system)
        local_directions = []
        for axis in range(system.input_size):
            gradient = torch.autograd.grad(
                observable[:, axis].sum(), write, retain_graph=True
            )[0]
            local_directions.append(gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        direction = torch.stack(local_directions, dim=1).detach()

        adapter_write = torch.zeros_like(write, requires_grad=True)
        latent = model.adapter(batch, intervention=adapter_write, intervention_mask=mask)
        columns = []
        for axis in range(system.input_size):
            coordinate_effects = []
            for coordinate in range(system.state_size):
                derivative = torch.autograd.grad(
                    latent[:, coordinate].sum(),
                    adapter_write,
                    retain_graph=True,
                )[0]
                coordinate_effects.append((derivative * direction[:, axis]).sum(dim=-1))
            column = torch.stack(coordinate_effects, dim=1)
            column = column / column.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            columns.append(column / system.dt)
        states.append(model.adapter(batch).detach().float())
        ports.append(torch.stack(columns, dim=-1).detach().float())
        directions.append(direction)
    return torch.cat(states), torch.cat(ports), torch.cat(directions)


def fit_port_networks(
    ph: NeuralPortHamiltonian,
    baseline: UnstructuredPortDynamics,
    states: torch.Tensor,
    targets: torch.Tensor,
    config: PassiveJacobianPHConfig,
) -> dict[str, Any]:
    for module in (ph.port_network, baseline.port_network):
        module.requires_grad_(True)
    optimizers = {
        "portHamiltonian": torch.optim.AdamW(ph.port_network.parameters(), lr=config.port_learning_rate),
        "unstructured": torch.optim.AdamW(baseline.port_network.parameters(), lr=config.port_learning_rate),
    }
    started = time.perf_counter()
    final_losses = {}
    for step in range(1, config.port_steps + 1):
        rows = torch.randint(0, states.shape[0], (config.batch_size,), device=states.device)
        for name, module in (("portHamiltonian", ph), ("unstructured", baseline)):
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            prediction = module.port(states[rows])
            loss = (prediction - targets[rows]).square().mean()
            loss.backward()
            optimizer.step()
            final_losses[name] = float(loss.detach())
        if step == 1 or step % config.log_every == 0 or step == config.port_steps:
            print(json.dumps({"stage": "fit_jacobian_ports_without_actions", "step": step,
                              "steps": config.port_steps, **final_losses,
                              "seconds": time.perf_counter() - started}), flush=True)
    for module in (ph, baseline):
        module.eval().requires_grad_(False)
    return {"finalLoss": final_losses, "seconds": time.perf_counter() - started}


@torch.no_grad()
def evaluate_passive_prediction(
    ph: NeuralPortHamiltonian,
    baseline: UnstructuredPortDynamics,
    states: torch.Tensor,
    system: SystemDefinition,
    horizon: int,
) -> dict[str, Any]:
    scale = states[:, :-1].reshape(-1, system.state_size).std(dim=0).clamp_min(1e-4)
    zero = torch.zeros(states.shape[0], system.input_size, device=states.device)
    result = {}
    for name, dynamics in (("portHamiltonian", ph), ("unstructured", baseline)):
        current = states[:, 0]
        for _ in range(horizon):
            current = dynamics(current, zero)
        error = ((current - states[:, horizon]) / scale).square().mean().sqrt()
        result[name] = {"normalizedRmse": float(error), "horizon": horizon}
    result["phToUnstructuredRatio"] = (
        result["portHamiltonian"]["normalizedRmse"]
        / max(result["unstructured"]["normalizedRmse"], 1e-8)
    )
    return result


def evaluate_causal_lens(
    model: PassiveVisualPHModel,
    contexts: torch.Tensor,
    system: SystemDefinition,
    directions: torch.Tensor,
    *,
    strength: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    backbone = model.adapter.backbone
    generator = torch.Generator(device=contexts.device).manual_seed(seed)
    expected_effects = []
    random_effects = []
    signs = []
    for start in range(0, contexts.shape[0], batch_size):
        batch = contexts[start : start + batch_size].long()
        local = directions[start : start + batch.shape[0]]
        mask = _entity_token_mask(backbone, batch, system.entity_values)
        for axis in range(system.input_size):
            direction = local[:, axis]
            plus = backbone(
                batch,
                intervention_block=model.adapter.lens_block,
                intervention=direction * strength,
                intervention_mask=mask,
            )[:, -1]
            minus = backbone(
                batch,
                intervention_block=model.adapter.lens_block,
                intervention=-direction * strength,
                intervention_mask=mask,
            )[:, -1]
            effect = 0.5 * _wrapped_difference(
                _observable_from_logits(plus, system), _observable_from_logits(minus, system), system
            )[:, axis]
            random_direction = torch.randn(
                direction.shape, generator=generator, device=direction.device
            )
            projection = (random_direction * direction).sum(dim=-1, keepdim=True)
            random_direction = random_direction - projection * direction
            random_direction /= random_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            random_plus = backbone(
                batch,
                intervention_block=model.adapter.lens_block,
                intervention=random_direction * strength,
                intervention_mask=mask,
            )[:, -1]
            random_minus = backbone(
                batch,
                intervention_block=model.adapter.lens_block,
                intervention=-random_direction * strength,
                intervention_mask=mask,
            )[:, -1]
            random_effect = 0.5 * _wrapped_difference(
                _observable_from_logits(random_plus, system),
                _observable_from_logits(random_minus, system),
                system,
            )[:, axis]
            expected_effects.append(effect.detach())
            random_effects.append(random_effect.detach())
            signs.append(effect.gt(0).detach())
    expected = torch.cat(expected_effects).float()
    random_values = torch.cat(random_effects).float()
    return {
        "expectedSignFraction": float(torch.cat(signs).float().mean()),
        "meanAbsoluteCausalEffect": float(expected.abs().mean()),
        "meanAbsoluteRandomEffect": float(random_values.abs().mean()),
        "causalToRandomEffectRatio": float(
            expected.abs().mean() / random_values.abs().mean().clamp_min(1e-8)
        ),
        "samples": int(expected.numel()),
        "strength": strength,
    }


def _copy_world_state(state: WorldState) -> WorldState:
    return WorldState(
        player_position=state.player_position.copy(),
        player_velocity=state.player_velocity.copy(),
        puck_position=state.puck_position.copy(),
        puck_velocity=state.puck_velocity.copy(),
        score=state.score,
        tick=state.tick,
        reset_timer=state.reset_timer,
        last_event=state.last_event,
    )


def _clone_environment(system: SystemDefinition, environment: Any) -> Any:
    if system.name == "blocket":
        clone = BlocketLeagueEnv(seed=0, config=environment.config)
        clone.state = _copy_world_state(environment.state)
        return clone
    clone = PendulumEnv(seed=0, config=environment.config)
    clone.set_state(environment.state)
    return clone


def _deployment_context(
    system: SystemDefinition,
    seed: int,
    history_frames: int,
) -> tuple[Any, torch.Tensor]:
    rng = np.random.default_rng(seed)
    if system.name == "blocket":
        environment = BlocketLeagueEnv(
            seed=seed,
            config=WorldConfig(player_drag=0.12, puck_drag=0.12),
        )
        state = environment.state
        for _ in range(100):
            player = rng.uniform((0.16, 0.16), (0.58, 0.84)).astype(np.float32)
            puck = rng.uniform((0.42, 0.16), (0.84, 0.84)).astype(np.float32)
            if np.linalg.norm(player - puck) > 0.19:
                break
        state.player_position = player
        state.puck_position = puck
        state.player_velocity = rng.uniform(-0.45, 0.45, size=2).astype(np.float32)
        state.puck_velocity = rng.uniform(-0.38, 0.38, size=2).astype(np.float32)
        state.reset_timer = 0
        frames = [environment.render()]
        for _ in range(history_frames - 1):
            environment.step_vector(np.zeros(2, dtype=np.float32))
            frames.append(environment.render())
    else:
        environment = PendulumEnv(seed=seed)
        frames = [environment.render()]
        for _ in range(history_frames - 1):
            environment.step(0.0)
            frames.append(environment.render())
    return environment, _classes_from_rgb(np.stack(frames))


def _apply_interface_command(
    system: SystemDefinition,
    environment: Any,
    interface: np.ndarray,
    command: np.ndarray,
) -> None:
    physical = interface @ command.astype(np.float64)
    if system.name == "blocket":
        environment.step_vector(np.asarray(physical, dtype=np.float32))
    else:
        normalized = float(np.clip(physical[0], -1.0, 1.0))
        environment.step(normalized * environment.config.max_torque)


def _append_rendered(history: torch.Tensor, environment: Any) -> torch.Tensor:
    frame = _classes_from_rgb(environment.render()[None])[0]
    return torch.cat((history[1:], frame[None]), dim=0)


def _interface_matrices(system: SystemDefinition) -> dict[str, np.ndarray]:
    if system.name == "blocket":
        return {
            "native": np.eye(2, dtype=np.float32),
            "unseen": np.asarray(((0.0, -1.25), (0.70, 0.0)), dtype=np.float32),
        }
    return {
        "native": np.ones((1, 1), dtype=np.float32),
        "unseen": np.asarray(((-1.60,),), dtype=np.float32),
    }


@torch.no_grad()
def calibrate_interface(
    adapter: FrozenTransformerStateAdapter,
    dynamics: NeuralPortHamiltonian | UnstructuredPortDynamics,
    system: SystemDefinition,
    interface: np.ndarray,
    config: PassiveJacobianPHConfig,
    *,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = next(adapter.readout.parameters()).device
    design_blocks: list[torch.Tensor] = []
    effects_by_axis: list[list[torch.Tensor]] = [[] for _ in range(system.input_size)]
    amplitude = config.calibration_amplitude
    steps = 0
    for sample in range(config.calibration_states_per_axis):
        environment, history = _deployment_context(
            system, seed + sample * 100_003, adapter.backbone.config.history_frames
        )
        context = history[None].to(device)
        state = adapter(context).float()
        design_blocks.append(system.dt * dynamics.port(state)[0])
        for axis in range(system.input_size):
            command = np.zeros(system.input_size, dtype=np.float32)
            command[axis] = amplitude
            plus = _clone_environment(system, environment)
            minus = _clone_environment(system, environment)
            _apply_interface_command(system, plus, interface, command)
            _apply_interface_command(system, minus, interface, -command)
            plus_state = adapter(_append_rendered(history, plus)[None].to(device)).float()[0]
            minus_state = adapter(_append_rendered(history, minus)[None].to(device)).float()[0]
            effects_by_axis[axis].append((plus_state - minus_state) / (2.0 * amplitude))
            steps += 2
    design = torch.cat(design_blocks, dim=0)
    ridge = 1e-4 * torch.eye(system.input_size, device=device)
    columns = []
    for axis in range(system.input_size):
        target = torch.stack(effects_by_axis[axis]).reshape(-1)
        columns.append(torch.linalg.solve(design.T @ design + ridge, design.T @ target))
    calibration = torch.stack(columns, dim=1)
    return calibration, {
        "pairedStatesPerAxis": config.calibration_states_per_axis,
        "environmentSteps": steps,
        "gradientUpdates": 0,
        "method": "analytic_ridge_on_paired_one_step_pixel_reencodings",
        "matrixVirtualFromInterface": calibration.cpu().tolist(),
    }


@torch.no_grad()
def evaluate_realizability(
    adapter: FrozenTransformerStateAdapter,
    dynamics: NeuralPortHamiltonian | UnstructuredPortDynamics,
    system: SystemDefinition,
    interface: np.ndarray,
    calibration: torch.Tensor,
    config: PassiveJacobianPHConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    device = next(adapter.readout.parameters()).device
    cosines = []
    projections = []
    amplitude = config.calibration_amplitude
    for sample in range(config.realizability_states):
        environment, history = _deployment_context(
            system, seed + sample * 100_003, adapter.backbone.config.history_frames
        )
        state = adapter(history[None].to(device)).float()
        axis = sample % system.input_size
        command = np.zeros(system.input_size, dtype=np.float32)
        command[axis] = amplitude
        plus = _clone_environment(system, environment)
        minus = _clone_environment(system, environment)
        _apply_interface_command(system, plus, interface, command)
        _apply_interface_command(system, minus, interface, -command)
        actual = (
            adapter(_append_rendered(history, plus)[None].to(device)).float()[0]
            - adapter(_append_rendered(history, minus)[None].to(device)).float()[0]
        ) / (2.0 * amplitude)
        predicted = system.dt * dynamics.port(state)[0] @ calibration[:, axis]
        cosine = F.cosine_similarity(actual[None], predicted[None], dim=-1)[0]
        cosines.append(cosine)
        projections.append((actual * predicted).sum().gt(0))
    values = torch.stack(cosines)
    return {
        "meanCosine": float(values.mean()),
        "positiveProjectionFraction": float(torch.stack(projections).float().mean()),
        "samples": len(cosines),
        "evaluationEnvironmentSteps": len(cosines) * 2,
    }


def _plan_virtual_control(
    dynamics: NeuralPortHamiltonian | UnstructuredPortDynamics,
    initial: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    config: PassiveJacobianPHConfig,
) -> torch.Tensor:
    raw = torch.zeros(
        initial.shape[0], config.planner_horizon, dynamics.config.input_size,
        device=initial.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam((raw,), lr=config.planner_learning_rate)
    for _ in range(config.planner_iterations):
        control = torch.tanh(raw)
        current = initial
        path_loss = initial.new_zeros(())
        for step in range(config.planner_horizon):
            current = dynamics(current, control[:, step])
            if step >= config.planner_horizon // 2:
                path_loss = path_loss + ((current - target) / scale).square().mean()
        loss = path_loss / max(config.planner_horizon - config.planner_horizon // 2, 1)
        loss = loss + 0.01 * control.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return torch.tanh(raw[:, 0]).detach()


def _control_episode(
    system: SystemDefinition,
    seed: int,
    history_frames: int,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    environment, history = _deployment_context(system, seed, history_frames)
    rng = np.random.default_rng(seed + 41)
    target_environment = _clone_environment(system, environment)
    if system.name == "blocket":
        offset = rng.uniform(-0.30, 0.30, size=2).astype(np.float32)
        if np.linalg.norm(offset) < 0.18:
            offset *= 0.18 / max(float(np.linalg.norm(offset)), 1e-6)
        target_environment.state.player_position = np.clip(
            target_environment.state.player_position + offset,
            0.13,
            0.87,
        ).astype(np.float32)
        target_environment.state.player_velocity[:] = 0.0
        target_frame = _classes_from_rgb(target_environment.render()[None])[0]
        target_history = target_frame[None].repeat(history_frames, 1, 1)
    else:
        offset = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.75, 1.35))
        target_angle = float(wrap_angle(environment.state.angle + offset))
        target_rgb = pendulum_target_frames(
            target_angle, frames=history_frames, image_size=environment.config.image_size
        )
        target_history = _classes_from_rgb(target_rgb)
        target_frame = target_history[-1]
    target_observable = _observable_from_classes(target_frame[None], system)[0]
    return environment, history, target_history, target_observable


def _final_visual_error(
    system: SystemDefinition,
    environment: Any,
    target_observable: torch.Tensor,
) -> float:
    classes = _classes_from_rgb(environment.render()[None])
    observable = _observable_from_classes(classes, system)[0]
    error = _wrapped_difference(observable[None], target_observable[None], system)[0]
    if system.name == "blocket":
        error = error / classes.shape[-1]
    return float(torch.linalg.vector_norm(error))


def evaluate_closed_loop_control(
    adapter: FrozenTransformerStateAdapter,
    ph: NeuralPortHamiltonian,
    baseline: UnstructuredPortDynamics,
    states: torch.Tensor,
    system: SystemDefinition,
    calibrations: dict[str, dict[str, torch.Tensor]],
    interfaces: dict[str, np.ndarray],
    config: PassiveJacobianPHConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    device = states.device
    scale = states[:, :-1].reshape(-1, system.state_size).std(dim=0).clamp_min(0.05)
    episode_specs = [
        _control_episode(
            system,
            seed + episode * 100_003,
            adapter.backbone.config.history_frames,
        )
        for episode in range(config.control_episodes)
    ]

    coast_errors = []
    for environment, _, _, target_observable in episode_specs:
        coast = _clone_environment(system, environment)
        for _ in range(config.control_steps):
            if system.name == "blocket":
                coast.step_vector(np.zeros(2, dtype=np.float32))
            else:
                coast.step(0.0)
        coast_errors.append(_final_visual_error(system, coast, target_observable))

    result: dict[str, Any] = {"coast": {"meanFinalError": float(np.mean(coast_errors))}}
    raw_errors: dict[str, dict[str, list[float]]] = {}
    for interface_name, interface in interfaces.items():
        raw_errors[interface_name] = {}
        for branch_name, dynamics in (("portHamiltonian", ph), ("unstructured", baseline)):
            environments = [_clone_environment(system, item[0]) for item in episode_specs]
            histories = [item[1].clone() for item in episode_specs]
            target_contexts = torch.stack([item[2] for item in episode_specs]).to(device)
            with torch.no_grad():
                targets = adapter(target_contexts).float()
            calibration = calibrations[interface_name][branch_name]
            inverse = torch.linalg.pinv(calibration)
            for _ in range(config.control_steps):
                context = torch.stack(histories).to(device)
                with torch.no_grad():
                    initial = adapter(context).float()
                virtual = _plan_virtual_control(dynamics, initial, targets, scale, config)
                command = virtual @ inverse.T
                command = command.clamp(-1.0, 1.0).cpu().numpy()
                for index, environment in enumerate(environments):
                    _apply_interface_command(system, environment, interface, command[index])
                    histories[index] = _append_rendered(histories[index], environment)
            errors = [
                _final_visual_error(system, environment, episode_specs[index][3])
                for index, environment in enumerate(environments)
            ]
            raw_errors[interface_name][branch_name] = errors

        ph_errors = np.asarray(raw_errors[interface_name]["portHamiltonian"])
        baseline_errors = np.asarray(raw_errors[interface_name]["unstructured"])
        coast = np.asarray(coast_errors)
        result[interface_name] = {
            "portHamiltonian": {
                "meanFinalError": float(ph_errors.mean()),
                "improvementVsCoast": float((coast.mean() - ph_errors.mean()) / max(coast.mean(), 1e-8)),
            },
            "unstructured": {
                "meanFinalError": float(baseline_errors.mean()),
                "improvementVsCoast": float((coast.mean() - baseline_errors.mean()) / max(coast.mean(), 1e-8)),
            },
            "phImprovementVsUnstructured": float(
                (baseline_errors.mean() - ph_errors.mean()) / max(baseline_errors.mean(), 1e-8)
            ),
            "phWinsFraction": float((ph_errors < baseline_errors).mean()),
            "episodes": config.control_episodes,
        }
    native_improvement = result["native"]["portHamiltonian"]["improvementVsCoast"]
    unseen_improvement = result["unseen"]["portHamiltonian"]["improvementVsCoast"]
    result["transferRetention"] = float(
        unseen_improvement / max(native_improvement, 1e-8)
    )
    return result


def evaluate_energy_structure(
    core: NeuralPortHamiltonian,
    states: torch.Tensor,
    system: SystemDefinition,
) -> dict[str, Any]:
    sample = states[:, :-1].reshape(-1, system.state_size)[:512]
    zero = torch.zeros(sample.shape[0], system.input_size, device=sample.device)
    terms = core.power_terms(sample, zero, create_graph=False)
    with torch.no_grad():
        next_state = core(sample, zero)
        current_energy = core.hamiltonian(sample)
        next_energy = core.hamiltonian(next_state)
    return {
        "powerBalanceMaxAbsDefect": float(terms["balanceDefect"].abs().max()),
        "zeroInputEnergyIncreaseFraction": float((next_energy > current_energy + 1e-6).float().mean()),
        "jSkewMaxAbsDefect": float(
            (core.interconnection(sample) + core.interconnection(sample).transpose(-1, -2)).abs().max()
        ),
        "rMinimumEigenvalue": float(torch.linalg.eigvalsh(core.resistance(sample)).min()),
        "samples": int(sample.shape[0]),
    }


def _system_decision(summary: dict[str, Any]) -> dict[str, Any]:
    causal = summary["causalLens"]
    passive = summary["passivePrediction"]
    structure = summary["structure"]
    control = summary["control"]
    realizability = summary["realizability"]
    gates = {
        "actionFirewall": bool(
            summary["trainingAudit"]["actionGradientUpdates"] == 0
            and summary["trainingAudit"]["physicalStateGradientUpdates"] == 0
            and summary["trainingAudit"]["backboneHashBefore"]
            == summary["trainingAudit"]["backboneHashAfter"]
            and summary["trainingAudit"]["portUnchangedDuringPassiveFit"]
        ),
        "passivePredictionParity": passive["phToUnstructuredRatio"] <= 1.15,
        "causalLens": causal["expectedSignFraction"] >= 0.75
        and causal["causalToRandomEffectRatio"] >= 2.0,
        "nativeRealizability": realizability["native"]["portHamiltonian"]["meanCosine"] >= 0.70
        and realizability["native"]["portHamiltonian"]["positiveProjectionFraction"] >= 0.75,
        "unseenRealizability": realizability["unseen"]["portHamiltonian"]["meanCosine"] >= 0.70
        and realizability["unseen"]["portHamiltonian"]["positiveProjectionFraction"] >= 0.75,
        "nativeRealControl": control["native"]["phImprovementVsUnstructured"] >= 0.10
        and control["native"]["phWinsFraction"] >= 0.60,
        "unseenRealControl": control["unseen"]["phImprovementVsUnstructured"] >= 0.10
        and control["unseen"]["phWinsFraction"] >= 0.60,
        "interfaceTransfer": control["transferRetention"] >= 0.80,
        "powerAccounting": structure["powerBalanceMaxAbsDefect"] <= 1e-5
        and structure["zeroInputEnergyIncreaseFraction"] <= 0.01,
    }
    return {"allGatesPass": all(gates.values()), "gates": gates}


def run_system_experiment(
    system: SystemDefinition,
    checkpoint_path: Path,
    output_dir: Path,
    config: PassiveJacobianPHConfig,
    device: torch.device,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = build_pixel_direct_from_checkpoint(payload)
    backbone_hash_before = module_tensor_hash(backbone)
    fit_cpu = collect_passive_pixels(
        system,
        backbone.config,
        trajectories=config.fit_trajectories,
        transitions=config.transitions,
        seed=config.seed + (20_000_000 if system.name == "blocket" else 30_000_000),
    )
    test_cpu = collect_passive_pixels(
        system,
        backbone.config,
        trajectories=config.test_trajectories,
        transitions=config.transitions,
        seed=config.seed + (40_000_000 if system.name == "blocket" else 50_000_000),
    )
    weights = _class_weights(fit_cpu["frames"], backbone.config.palette_size, device)
    fit = _move_suite(fit_cpu, device)
    test = _move_suite(test_cpu, device)
    model = _make_visual_model(backbone, system, config, device)
    passive_training = train_passive_visual_ph(model, fit, weights, config, output_dir)
    backbone_hash_after = module_tensor_hash(model.adapter.backbone)
    fit_states = encode_suite(model.adapter, fit, batch_size=config.batch_size)
    test_states = encode_suite(model.adapter, test, batch_size=config.batch_size)
    baseline, capacity = train_unstructured_baseline(
        model.core, fit_states, system, config
    )

    fit_contexts = fit["pixelContexts"][: config.port_fit_samples, 0]
    test_contexts = test["pixelContexts"][: config.port_test_samples, 0]
    lens_fit_states, lens_fit_ports, _ = extract_jacobian_ports(
        model, fit_contexts, system, batch_size=config.batch_size
    )
    lens_test_states, lens_test_ports, lens_test_directions = extract_jacobian_ports(
        model, test_contexts, system, batch_size=config.batch_size
    )
    port_fitting = fit_port_networks(
        model.core, baseline, lens_fit_states, lens_fit_ports, config
    )
    with torch.no_grad():
        port_test_error = {
            "portHamiltonian": float(
                (model.core.port(lens_test_states) - lens_test_ports).square().mean().sqrt()
            ),
            "unstructured": float(
                (baseline.port(lens_test_states) - lens_test_ports).square().mean().sqrt()
            ),
        }
    causal_lens = evaluate_causal_lens(
        model,
        test_contexts,
        system,
        lens_test_directions,
        strength=config.lens_strength,
        batch_size=config.batch_size,
        seed=config.seed + 61,
    )
    passive_prediction = evaluate_passive_prediction(
        model.core,
        baseline,
        test_states,
        system,
        min(config.planner_horizon, config.transitions),
    )

    interfaces = _interface_matrices(system)
    calibrations: dict[str, dict[str, torch.Tensor]] = {}
    calibration_audit: dict[str, Any] = {}
    realizability: dict[str, Any] = {}
    for interface_index, (interface_name, interface) in enumerate(interfaces.items()):
        calibrations[interface_name] = {}
        calibration_audit[interface_name] = {}
        realizability[interface_name] = {}
        for branch_index, (branch_name, dynamics) in enumerate(
            (("portHamiltonian", model.core), ("unstructured", baseline))
        ):
            calibration, audit = calibrate_interface(
                model.adapter,
                dynamics,
                system,
                interface,
                config,
                seed=config.seed + 60_000_000 + interface_index * 1_000_000,
            )
            calibrations[interface_name][branch_name] = calibration
            calibration_audit[interface_name][branch_name] = audit
            realizability[interface_name][branch_name] = evaluate_realizability(
                model.adapter,
                dynamics,
                system,
                interface,
                calibration,
                config,
                seed=config.seed + 70_000_000 + interface_index * 1_000_000 + branch_index * 100_000,
            )
    control = evaluate_closed_loop_control(
        model.adapter,
        model.core,
        baseline,
        fit_states,
        system,
        calibrations,
        interfaces,
        config,
        seed=config.seed + (80_000_000 if system.name == "blocket" else 90_000_000),
    )
    structure = evaluate_energy_structure(model.core, test_states, system)
    checkpoint = {
        "kind": "passive_jacobian_port_hamiltonian_control",
        "system": system.name,
        "seed": config.seed,
        "backboneCheckpoint": str(checkpoint_path),
        "adapter": model.adapter.state_dict(),
        "renderer": model.renderer.state_dict(),
        "portHamiltonian": model.core.state_dict(),
        "unstructured": baseline.state_dict(),
        "calibrations": {
            interface_name: {name: value.cpu() for name, value in branches.items()}
            for interface_name, branches in calibrations.items()
        },
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    summary: dict[str, Any] = {
        "kind": checkpoint["kind"],
        "system": system.name,
        "seed": config.seed,
        "trainingAudit": {
            "optimizationTensorKeys": ["pixelContexts", "frames"],
            "actionGradientUpdates": 0,
            "physicalStateGradientUpdates": 0,
            "backboneFrozen": True,
            "backboneHashBefore": backbone_hash_before,
            "backboneHashAfter": backbone_hash_after,
            "portUnchangedDuringPassiveFit": passive_training["portUnchangedDuringPassiveFit"],
            "portTargetSource": "activation_jacobian_only",
        },
        "capacity": capacity,
        "passiveTraining": passive_training,
        "portFitting": {**port_fitting, "heldOutRmse": port_test_error},
        "causalLens": causal_lens,
        "passivePrediction": passive_prediction,
        "calibration": calibration_audit,
        "realizability": realizability,
        "control": control,
        "structure": structure,
    }
    summary["decision"] = _system_decision(summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "system_complete", "system": system.name,
                      "decision": summary["decision"]}, indent=2), flush=True)
    return summary


def run_passive_jacobian_ph_experiment(
    blocket_checkpoint: Path,
    output_dir: Path,
    *,
    config: PassiveJacobianPHConfig = PassiveJacobianPHConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    _seed_everything(config.seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pendulum_checkpoint = train_pendulum_backbone(
        output_dir / "pendulum-passive-backbone", config, device
    )
    summaries = {}
    for system_name, checkpoint in (
        ("blocket", blocket_checkpoint),
        ("pendulum", pendulum_checkpoint),
    ):
        summaries[system_name] = run_system_experiment(
            SYSTEMS[system_name],
            checkpoint,
            output_dir / system_name,
            config,
            device,
        )
    all_pass = all(summary["decision"]["allGatesPass"] for summary in summaries.values())
    result = {
        "kind": "passive_jacobian_port_hamiltonian_two_system_breakthrough_test",
        "seed": config.seed,
        "config": asdict(config),
        "systems": summaries,
        "decision": {
            "outcome": (
                "breakthrough_supported_single_seed_two_systems"
                if all_pass
                else "breakthrough_not_supported_single_seed"
            ),
            "allGatesPass": all_pass,
            "systemConjunction": {
                name: summary["decision"]["allGatesPass"] for name, summary in summaries.items()
            },
            "scope": "single_seed_falsification_not_reproducibility",
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "totalSeconds": time.perf_counter() - started,
        "preregistration": "docs/passive-jacobian-ph-control-experiment.md",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("blocket_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=PassiveJacobianPHConfig.seed)
    parser.add_argument("--fit-trajectories", type=int, default=PassiveJacobianPHConfig.fit_trajectories)
    parser.add_argument("--test-trajectories", type=int, default=PassiveJacobianPHConfig.test_trajectories)
    parser.add_argument("--visual-steps", type=int, default=PassiveJacobianPHConfig.visual_steps)
    parser.add_argument("--passive-dynamics-steps", type=int, default=PassiveJacobianPHConfig.passive_dynamics_steps)
    parser.add_argument("--baseline-steps", type=int, default=PassiveJacobianPHConfig.baseline_steps)
    parser.add_argument("--port-steps", type=int, default=PassiveJacobianPHConfig.port_steps)
    parser.add_argument("--pendulum-backbone-steps", type=int, default=PassiveJacobianPHConfig.pendulum_backbone_steps)
    parser.add_argument("--control-episodes", type=int, default=PassiveJacobianPHConfig.control_episodes)
    parser.add_argument("--planner-iterations", type=int, default=PassiveJacobianPHConfig.planner_iterations)
    parser.add_argument("--log-every", type=int, default=PassiveJacobianPHConfig.log_every)
    args = parser.parse_args()
    config = PassiveJacobianPHConfig(
        seed=args.seed,
        fit_trajectories=args.fit_trajectories,
        test_trajectories=args.test_trajectories,
        visual_steps=args.visual_steps,
        passive_dynamics_steps=args.passive_dynamics_steps,
        baseline_steps=args.baseline_steps,
        port_steps=args.port_steps,
        pendulum_backbone_steps=args.pendulum_backbone_steps,
        control_episodes=args.control_episodes,
        planner_iterations=args.planner_iterations,
        log_every=args.log_every,
    )
    run_passive_jacobian_ph_experiment(
        args.blocket_checkpoint,
        args.output,
        config=config,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
