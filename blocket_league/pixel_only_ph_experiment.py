from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .action_port_pixel_experiment import action_vectors
from .data import make_clip, make_excitation_clip
from .env import ACTION_VECTORS, PALETTE, BlocketLeagueEnv, WorldConfig, WorldState
from .neural_ph_experiment import (
    _fit_affine_r2,
    _matched_control_hidden_size,
    _parameter_count,
)
from .neural_port_hamiltonian import (
    NeuralODE,
    NeuralPortHamiltonian,
    NeuralPortHamiltonianConfig,
)
from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .pixel_probe import PLAYER_CLASSES, _soft_centroid, _visual_centroid
from .port_hamiltonian_audit import canonical_state
from .position_write_probe import PUCK_CLASSES
from .train_pixel_direct import frames_to_classes, palette_tensor


@dataclass(frozen=True)
class PixelOnlyPHConfig:
    fit_policy_trajectories: int = 3_072
    fit_cardinal_trajectories: int = 3_072
    test_policy_trajectories: int = 512
    test_diagonal_trajectories: int = 256
    test_reversal_trajectories: int = 256
    audit_trajectories: int = 1_024
    transitions_per_trajectory: int = 8
    feature_batch_size: int = 64
    block_count: int = 5
    state_size: int = 8
    encoder_hidden_size: int = 128
    renderer_hidden_size: int = 96
    core_hidden_size: int = 64
    core_hidden_layers: int = 2
    autoencoder_steps: int = 3_000
    autoencoder_batch_size: int = 256
    dynamics_steps: int = 5_000
    dynamics_batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 300
    min_learning_rate_ratio: float = 0.1
    reconstruction_weight: float = 0.50
    teacher_latent_weight: float = 1.00
    rollout_latent_weight: float = 1.00
    rollout_pixel_weight: float = 1.00
    action_contrast_weight: float = 0.20
    whitening_weight: float = 0.05
    energy_gauge_weight: float = 0.02
    action_contrast_margin: float = 0.02
    planner_samples: int = 64
    planner_horizon: int = 8
    planner_steps: int = 40
    planner_learning_rate: float = 0.35
    log_every: int = 50
    seed: int = 131_610_731

    @property
    def fit_trajectories(self) -> int:
        return self.fit_policy_trajectories + self.fit_cardinal_trajectories


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _learning_rate_multiplier(step: int, steps: int, config: PixelOnlyPHConfig) -> float:
    if step <= config.warmup_steps:
        return step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate_ratio + (1.0 - config.min_learning_rate_ratio) * cosine


@torch.no_grad()
def _generic_block_features(
    model: nn.Module,
    classes: torch.Tensor,
    block_count: int,
) -> torch.Tensor:
    """Object-agnostic visual readout: no entity mask, coordinate, or state label."""

    tokens = (
        model.patch_projection(model.patch_tokens(classes))
        + model.spatial_position
        + model.temporal_position[:, : classes.shape[1]]
    )
    for block in model.blocks[:block_count]:
        tokens = block(tokens)
    latest = tokens[:, -1].float()
    return torch.cat(
        (
            latest.mean(dim=1),
            latest.std(dim=1, unbiased=False),
            latest[:, -1],
        ),
        dim=-1,
    )


@torch.no_grad()
def collect_visual_action_suite(
    model: nn.Module,
    *,
    trajectories: int,
    transitions: int,
    seed: int,
    family: str,
    feature_batch_size: int,
    block_count: int,
    device: torch.device,
    include_world_states: bool,
) -> dict[str, torch.Tensor]:
    """Collect pixel-derived tensors; physical states are optional audit-only data."""

    if family not in {"policy", "cardinal", "diagonal", "reversal"}:
        raise ValueError(f"unknown family {family!r}")
    history = model.config.history_frames
    endpoints = transitions + 1
    pending_contexts: list[np.ndarray] = []
    feature_chunks: list[torch.Tensor] = []
    frame_sequences: list[torch.Tensor] = []
    action_sequences: list[torch.Tensor] = []
    state_sequences: list[torch.Tensor] = []
    palette = palette_tensor(device)

    def flush() -> None:
        if not pending_contexts:
            return
        videos = torch.from_numpy(np.stack(pending_contexts)).permute(0, 1, 4, 2, 3)
        videos = videos.to(device, non_blocking=True).float().div(127.5).sub(1.0)
        classes = frames_to_classes(videos, palette)
        feature_chunks.append(
            _generic_block_features(model, classes, block_count).half().cpu()
        )
        pending_contexts.clear()

    started = time.perf_counter()
    for trajectory in range(trajectories):
        clip_seed = seed + trajectory * 9_973
        arguments = {
            "context_frames": 1,
            "future_frames": history + transitions - 1,
            "image_size": model.config.image_size,
        }
        clip = (
            make_clip(clip_seed, **arguments)
            if family == "policy"
            else make_excitation_clip(clip_seed, action_family=family, **arguments)
        )
        frames = clip["frames"]
        frame_sequences.append(_rgb_frames_to_classes_cpu(frames))
        all_actions = torch.from_numpy(clip["all_actions"]).long()
        action_sequences.append(
            torch.stack([all_actions[offset + history] for offset in range(transitions)])
        )
        if include_world_states:
            all_states = torch.from_numpy(clip["all_state"]).float()
            state_sequences.append(
                torch.stack([all_states[offset + history - 1] for offset in range(endpoints)])
            )
        for offset in range(endpoints):
            pending_contexts.append(frames[offset : offset + history])
            if len(pending_contexts) >= feature_batch_size:
                flush()
        if (trajectory + 1) % 256 == 0 or trajectory + 1 == trajectories:
            print(
                json.dumps(
                    {
                        "stage": "collect_pixel_only_ph",
                        "family": family,
                        "auditLabels": include_world_states,
                        "trajectories": trajectory + 1,
                        "total": trajectories,
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    flush()
    result = {
        "features": torch.cat(feature_chunks).reshape(trajectories, endpoints, -1),
        "frames": torch.stack(frame_sequences)[:, history - 1 : history - 1 + endpoints],
        "actions": torch.stack(action_sequences),
    }
    result["actionVectors"] = action_vectors(result["actions"])
    if include_world_states:
        result["worldStates"] = torch.stack(state_sequences)
    return result


def _rgb_frames_to_classes_cpu(frames: np.ndarray) -> torch.Tensor:
    rgb = frames.astype(np.int32)
    keys = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    classes = np.zeros(keys.shape, dtype=np.uint8)
    for index, color in enumerate(PALETTE.values()):
        key = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
        classes[keys == key] = index
    return torch.from_numpy(classes)


def _concatenate_training_suites(*suites: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    required = {"features", "frames", "actions", "actionVectors"}
    if any(set(suite) != required for suite in suites):
        raise AssertionError("training suites must contain pixels, features, and actions only")
    return {name: torch.cat([suite[name] for suite in suites]) for name in required}


class GenericStateEncoder(nn.Module):
    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        state_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_scale", feature_scale.float())
        self.network = nn.Sequential(
            nn.Linear(feature_mean.numel(), hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, state_size),
        )
        nn.init.normal_(self.network[-1].weight, std=0.03)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network((features.float() - self.feature_mean) / self.feature_scale)


class GenericSpatialRenderer(nn.Module):
    """Spatial-broadcast neural renderer with no semantic coordinate assignment."""

    def __init__(
        self,
        state_size: int,
        image_size: int,
        palette_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        axis = (torch.arange(image_size, dtype=torch.float32) + 0.5) / image_size
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        harmonics = [coordinates]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            phase = 2.0 * math.pi * frequency * coordinates
            harmonics.extend((torch.sin(phase), torch.cos(phase)))
        coordinate_features = torch.cat(harmonics, dim=-1)
        self.register_buffer("coordinate_features", coordinate_features)
        self.state_size = state_size
        self.image_size = image_size
        self.palette_size = palette_size
        input_size = state_size + coordinate_features.shape[-1]
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, palette_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        shape = state.shape[:-1]
        flat = state.reshape(-1, self.state_size)
        pixels = self.coordinate_features.shape[0]
        coordinates = self.coordinate_features[None].expand(flat.shape[0], -1, -1)
        broadcast_state = flat[:, None].expand(-1, pixels, -1)
        logits = self.network(torch.cat((broadcast_state, coordinates), dim=-1))
        return logits.transpose(1, 2).reshape(
            *shape, self.palette_size, self.image_size, self.image_size
        )


class VisualAutoencoder(nn.Module):
    def __init__(self, encoder: GenericStateEncoder, renderer: GenericSpatialRenderer) -> None:
        super().__init__()
        self.encoder = encoder
        self.renderer = renderer

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encoder(features)
        return state, self.renderer(state)


class PixelOnlyDynamicsBranch(nn.Module):
    def __init__(
        self,
        encoder: GenericStateEncoder,
        renderer: GenericSpatialRenderer,
        *,
        core_config: NeuralPortHamiltonianConfig,
        structured: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.renderer = renderer
        self.structured = structured
        self.core: NeuralPortHamiltonian | NeuralODE
        self.core = (
            NeuralPortHamiltonian(core_config)
            if structured
            else NeuralODE(core_config)
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.core(state, action)


def _class_weights(frames: torch.Tensor, palette_size: int) -> torch.Tensor:
    counts = torch.bincount(frames.reshape(-1).long(), minlength=palette_size).float()
    weights = (counts.max() / counts.clamp_min(1.0)).sqrt().clamp(0.25, 15.0)
    return weights / weights[1].clamp_min(1e-6)


def _pixel_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-3], logits.shape[-2], logits.shape[-1]).float(),
        targets.reshape(-1, targets.shape[-2], targets.shape[-1]).long(),
        weight=class_weights,
    )


def _whitening_loss(state: torch.Tensor) -> torch.Tensor:
    flat = state.reshape(-1, state.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    std = centered.square().mean(dim=0).add(1e-6).sqrt()
    normalized = centered / std
    covariance = normalized.T @ normalized / max(normalized.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    return (
        flat.mean(dim=0).square().mean()
        + (std - 1.0).square().mean()
        + off_diagonal.square().mean()
    )


def _energy_gauge(core: NeuralPortHamiltonian, state: torch.Tensor) -> torch.Tensor:
    _, gradient, _, _, _ = core.components(state, create_graph=True)
    return (gradient.square().mean().add(1e-12).sqrt() - 1.0).square()


def pixel_only_branch_loss(
    branch: PixelOnlyDynamicsBranch,
    features: torch.Tensor,
    frames: torch.Tensor,
    actions: torch.Tensor,
    class_weights: torch.Tensor,
    config: PixelOnlyPHConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Training loss whose arguments deliberately exclude all physical labels."""

    encoded = branch.encode(features)
    reconstruction_indices = torch.tensor((0, 4, 8), device=features.device)
    reconstructed_logits = branch.renderer(encoded[:, reconstruction_indices])
    reconstruction = _pixel_cross_entropy(
        reconstructed_logits, frames[:, reconstruction_indices], class_weights
    )

    teacher_state = encoded[:, :-1].reshape(-1, encoded.shape[-1])
    teacher_actions = actions.reshape(-1, actions.shape[-1])
    teacher_prediction = branch.step(teacher_state, teacher_actions)
    teacher_target = encoded[:, 1:].detach().reshape_as(teacher_prediction)
    teacher_per_sample = (teacher_prediction - teacher_target).square().mean(dim=-1)
    teacher = teacher_per_sample.mean()

    shuffled_actions = torch.roll(teacher_actions, shifts=1, dims=0)
    wrong_prediction = branch.step(teacher_state, shuffled_actions)
    wrong_error = (wrong_prediction - teacher_target).square().mean(dim=-1)
    action_contrast = F.relu(
        config.action_contrast_margin + teacher_per_sample - wrong_error
    ).mean()

    current = encoded[:, 0]
    rollout_states = []
    for step in range(actions.shape[1]):
        current = branch.step(current, actions[:, step])
        rollout_states.append(current)
    rollout = torch.stack(rollout_states, dim=1)
    rollout_target = encoded[:, 1:].detach()
    horizon_weights = torch.linspace(1.0, 2.0, actions.shape[1], device=features.device)
    rollout_error = (rollout - rollout_target).square().mean(dim=(0, 2))
    rollout_latent = (rollout_error * horizon_weights).sum() / horizon_weights.sum()

    pixel_horizons = torch.tensor((0, 1, 3, 7), device=features.device)
    rollout_logits = branch.renderer(rollout[:, pixel_horizons])
    rollout_pixel = _pixel_cross_entropy(
        rollout_logits, frames[:, pixel_horizons + 1], class_weights
    )
    whitening = _whitening_loss(encoded)
    energy_gauge = (
        _energy_gauge(branch.core, encoded[:, 0])
        if isinstance(branch.core, NeuralPortHamiltonian)
        else encoded.square().mean() * 0.0
    )
    total = (
        config.reconstruction_weight * reconstruction
        + config.teacher_latent_weight * teacher
        + config.rollout_latent_weight * rollout_latent
        + config.rollout_pixel_weight * rollout_pixel
        + config.action_contrast_weight * action_contrast
        + config.whitening_weight * whitening
        + config.energy_gauge_weight * energy_gauge
    )
    return total, {
        "reconstruction": reconstruction,
        "teacherLatent": teacher,
        "rolloutLatent": rollout_latent,
        "rolloutPixel": rollout_pixel,
        "actionContrast": action_contrast,
        "whitening": whitening,
        "energyGauge": energy_gauge,
    }


def _fit_ridge(inputs: torch.Tensor, targets: torch.Tensor, ridge: float = 1e-2):
    mean = inputs.mean(dim=0, keepdim=True)
    scale = inputs.std(dim=0, keepdim=True).clamp_min(1e-5)
    normalized = (inputs - mean) / scale
    design = torch.cat((normalized, torch.ones_like(normalized[:, :1])), dim=-1)
    identity = torch.eye(design.shape[1], device=inputs.device)
    identity[-1, -1] = 0.0
    weight = torch.linalg.solve(design.T @ design + ridge * identity, design.T @ targets)
    return mean, scale, weight


def _ridge_predict(fit, inputs: torch.Tensor) -> torch.Tensor:
    mean, scale, weight = fit
    normalized = (inputs - mean) / scale
    return torch.cat((normalized, torch.ones_like(normalized[:, :1])), dim=-1) @ weight


def _ridge_jacobian(fit) -> torch.Tensor:
    _, scale, weight = fit
    return (weight[:-1] / scale.T).T


def _r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    residual = (prediction - target).square().sum()
    total = (target - target.mean(dim=0, keepdim=True)).square().sum().clamp_min(1e-12)
    return float(1.0 - residual / total)


def _pixel_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    result = {"accuracy": float((prediction == target).float().mean())}
    for name, values in (("player", PLAYER_CLASSES), ("puck", PUCK_CLASSES)):
        predicted_mask = sum(prediction.eq(value) for value in values).bool()
        target_mask = sum(target.eq(value) for value in values).bool()
        intersection = (predicted_mask & target_mask).sum(dim=(-2, -1)).float()
        union = (predicted_mask | target_mask).sum(dim=(-2, -1)).float().clamp_min(1.0)
        result[f"{name}Iou"] = float((intersection / union).mean())
        predicted_position = _visual_centroid(prediction, values)
        target_position = _visual_centroid(target, values)
        result[f"{name}CentroidErrorPx"] = float(
            torch.linalg.vector_norm(predicted_position - target_position, dim=-1).mean()
        )
    return result


def _cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (first * second).sum(dim=-1) / (
        torch.linalg.vector_norm(first, dim=-1)
        * torch.linalg.vector_norm(second, dim=-1)
    ).clamp_min(1e-12)


@torch.no_grad()
def _decode_classes(
    branch: PixelOnlyDynamicsBranch,
    states: torch.Tensor,
    *,
    batch_size: int = 64,
) -> torch.Tensor:
    chunks = []
    flat = states.reshape(-1, states.shape[-1])
    for start in range(0, flat.shape[0], batch_size):
        chunks.append(branch.renderer(flat[start : start + batch_size]).argmax(dim=1).cpu())
    return torch.cat(chunks).reshape(*states.shape[:-1], branch.renderer.image_size, branch.renderer.image_size)


def _physical_probe(
    branch: PixelOnlyDynamicsBranch,
    audit: dict[str, torch.Tensor],
) -> tuple[Any, dict[str, Any]]:
    with torch.no_grad():
        encoded = branch.encode(_suite_model_inputs(audit))
    physical = canonical_state(audit["worldStates"])
    split = encoded.shape[0] // 2
    fit = _fit_ridge(
        encoded[:split].reshape(-1, encoded.shape[-1]),
        physical[:split].reshape(-1, physical.shape[-1]),
    )
    test_latent = encoded[split:].reshape(-1, encoded.shape[-1])
    test_physical = physical[split:].reshape(-1, physical.shape[-1])
    prediction = _ridge_predict(fit, test_latent)
    return fit, {
        "fitTrajectories": split,
        "testTrajectories": encoded.shape[0] - split,
        "fullR2": _r2(prediction, test_physical),
        "qR2": _r2(prediction[:, :4], test_physical[:, :4]),
        "pR2": _r2(prediction[:, 4:8], test_physical[:, 4:8]),
        "qRmse": float((prediction[:, :4] - test_physical[:, :4]).square().mean().sqrt()),
        "pRmse": float((prediction[:, 4:8] - test_physical[:, 4:8]).square().mean().sqrt()),
    }


@torch.no_grad()
def _structured_audit(
    branch: PixelOnlyDynamicsBranch,
    latent: torch.Tensor,
    actions: torch.Tensor,
    physical: torch.Tensor,
    physical_fit: Any,
) -> dict[str, Any]:
    if not isinstance(branch.core, NeuralPortHamiltonian):
        raise TypeError("structured audit requires a neural pH core")
    count = min(256, latent.shape[0])
    sample = latent[:count]
    controls = actions[:count]
    true_state = physical[:count]
    energy, gradient, interconnection, resistance, port = branch.core.components(
        sample, create_graph=False
    )
    power = branch.core.power_terms(sample, controls, create_graph=False)
    world = WorldConfig()
    kinetic = (
        true_state[:, 4:6].square().sum(dim=-1) / (2.0 * world.player_mass)
        + true_state[:, 6:8].square().sum(dim=-1) / (2.0 * world.puck_mass)
    )
    physical_jacobian = _ridge_jacobian(physical_fit)
    physical_port = torch.einsum("ij,bjm->bim", physical_jacobian, port)
    expected_port = torch.zeros(8, 2, device=latent.device)
    expected_port[4, 0] = world.player_mass * world.player_acceleration
    expected_port[5, 1] = world.player_mass * world.player_acceleration
    expected = expected_port[None].expand_as(physical_port)
    port_cosine = _cosine(physical_port.flatten(1), expected.flatten(1))

    jacobi = branch.core.jacobi_tensor(sample[: min(16, count)], create_graph=False)
    current = sample[: min(128, count)]
    zero = torch.zeros(current.shape[0], 2, device=current.device)
    energy_path = [branch.core.hamiltonian(current)]
    finite = True
    for _ in range(64):
        current = branch.core(current, zero)
        finite = finite and bool(torch.isfinite(current).all())
        energy_path.append(branch.core.hamiltonian(current))
    energy_path = torch.stack(energy_path, dim=1)
    changes = energy_path[:, 1:] - energy_path[:, :-1]
    return {
        "powerBalance": {
            "maxAbsDefect": float(power["balanceDefect"].abs().max()),
            "rmsDefect": float(power["balanceDefect"].square().mean().sqrt()),
            "minimumDissipation": float(power["dissipation"].min()),
        },
        "hamiltonian": {
            "physicalKineticEnergyAffineR2": _fit_affine_r2(kinetic, energy),
            "gradientRms": float(gradient.square().mean().sqrt()),
        },
        "portAfterPostHocPhysicalAlignment": {
            "physicalIncidenceCosine": float(port_cosine.mean()),
            "positiveFraction": float((port_cosine > 0).float().mean()),
            "positionRowFraction": float(
                physical_port[:, :4].square().sum().sqrt()
                / physical_port.square().sum().sqrt().clamp_min(1e-12)
            ),
            "puckMomentumFraction": float(
                physical_port[:, 6:8].square().sum().sqrt()
                / physical_port.square().sum().sqrt().clamp_min(1e-12)
            ),
        },
        "interconnection": {
            "skewDefect": float(
                (interconnection + interconnection.transpose(-1, -2)).abs().max()
            ),
            "jacobiRms": float(jacobi.square().mean().sqrt()),
        },
        "resistance": {
            "minimumEigenvalue": float(torch.linalg.eigvalsh(resistance).min()),
        },
        "zeroInputDiscreteEnergy": {
            "finite": finite,
            "increaseFraction": float((changes > 1e-7).float().mean()),
            "meanNetChange": float((energy_path[:, -1] - energy_path[:, 0]).mean()),
            "maximumStepIncrease": float(changes.max()),
        },
    }


def _world_from_vector(vector: np.ndarray, *, seed: int = 0) -> BlocketLeagueEnv:
    env = BlocketLeagueEnv(seed=seed)
    env.state = WorldState(
        player_position=vector[0:2].astype(np.float32).copy(),
        player_velocity=vector[2:4].astype(np.float32).copy(),
        puck_position=vector[4:6].astype(np.float32).copy(),
        puck_velocity=vector[6:8].astype(np.float32).copy(),
        score=int(round(float(vector[8]))),
        reset_timer=int(round(float(vector[9]))),
        tick=0,
        last_event="coast",
    )
    return env


def _counterfactual_truth(
    world_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis_actions = ((3, 7), (5, 1))
    effects = []
    plus_frames = []
    minus_frames = []
    for vector_tensor in world_states.cpu():
        vector = vector_tensor.numpy()
        sample_effects = []
        sample_plus = []
        sample_minus = []
        for plus_action, minus_action in axis_actions:
            plus_env = _world_from_vector(vector)
            minus_env = _world_from_vector(vector)
            plus_state = torch.from_numpy(plus_env.step(plus_action).vector()).float()[None]
            minus_state = torch.from_numpy(minus_env.step(minus_action).vector()).float()[None]
            sample_effects.append(
                0.5 * (canonical_state(plus_state)[0] - canonical_state(minus_state)[0])
            )
            sample_plus.append(_rgb_frames_to_classes_cpu(plus_env.render()[None])[0])
            sample_minus.append(_rgb_frames_to_classes_cpu(minus_env.render()[None])[0])
        effects.append(torch.stack(sample_effects))
        plus_frames.append(torch.stack(sample_plus))
        minus_frames.append(torch.stack(sample_minus))
    return torch.stack(effects), torch.stack(plus_frames), torch.stack(minus_frames)


@torch.no_grad()
def _counterfactual_audit(
    branch: PixelOnlyDynamicsBranch,
    latent: torch.Tensor,
    world_states: torch.Tensor,
    physical_fit: Any,
    *,
    samples: int = 128,
) -> dict[str, Any]:
    valid = (
        world_states[:, 9].eq(0)
        & world_states[:, 0].gt(0.15)
        & world_states[:, 0].lt(0.85)
        & world_states[:, 1].gt(0.15)
        & world_states[:, 1].lt(0.85)
    )
    indices = valid.nonzero(as_tuple=False).flatten()[:samples]
    selected_latent = latent[indices]
    selected_world = world_states[indices]
    truth, truth_plus, truth_minus = _counterfactual_truth(selected_world)
    truth = truth.to(latent.device)
    axes = torch.eye(2, device=latent.device)
    predicted_effects = []
    predicted_pixel_effects = []
    for axis in range(2):
        action = axes[axis][None].expand(selected_latent.shape[0], -1)
        plus = branch.step(selected_latent, action)
        minus = branch.step(selected_latent, -action)
        effect = 0.5 * (plus - minus)
        predicted_effects.append(effect)
        plus_logits = branch.renderer(plus)
        minus_logits = branch.renderer(minus)
        predicted_pixel_effects.append(
            0.5
            * (
                _soft_centroid(plus_logits, PLAYER_CLASSES)
                - _soft_centroid(minus_logits, PLAYER_CLASSES)
            )
        )
    predicted_latent = torch.stack(predicted_effects, dim=1)
    physical_jacobian = _ridge_jacobian(physical_fit)
    predicted = torch.einsum("ij,baj->bai", physical_jacobian, predicted_latent)
    full_cosine = _cosine(predicted, truth)
    momentum_cosine = _cosine(predicted[..., 4:6], truth[..., 4:6])
    predicted_pixels = torch.stack(predicted_pixel_effects, dim=1)
    truth_pixels = 0.5 * (
        _visual_centroid(truth_plus, PLAYER_CLASSES)
        - _visual_centroid(truth_minus, PLAYER_CLASSES)
    ).to(latent.device)
    pixel_cosine = _cosine(predicted_pixels, truth_pixels)
    target_predicted = torch.stack(
        (predicted_pixels[:, 0, 0], predicted_pixels[:, 1, 1]), dim=1
    )
    target_truth = torch.stack((truth_pixels[:, 0, 0], truth_pixels[:, 1, 1]), dim=1)
    return {
        "samples": int(indices.numel()),
        "alignedStateCosine": float(full_cosine.mean()),
        "alignedPlayerMomentumCosine": float(momentum_cosine.mean()),
        "renderedPlayerEffectCosine": float(pixel_cosine.mean()),
        "renderedTargetSignAgreement": float(
            ((target_predicted * target_truth) > 0).float().mean()
        ),
        "predictedTargetEffectPx": float(target_predicted.mean()),
        "trueTargetEffectPx": float(target_truth.mean()),
    }


def _nearest_actions(continuous: torch.Tensor) -> torch.Tensor:
    table = torch.as_tensor(ACTION_VECTORS, device=continuous.device, dtype=continuous.dtype)
    table_norm = table / torch.linalg.vector_norm(table, dim=-1, keepdim=True).clamp_min(1.0)
    distances = (continuous[..., None, :] - table_norm).square().sum(dim=-1)
    actions = distances.argmin(dim=-1)
    return torch.where(
        torch.linalg.vector_norm(continuous, dim=-1) < 0.15,
        torch.zeros_like(actions),
        actions,
    )


def _simulate_plan(
    world_states: torch.Tensor,
    action_indices: torch.Tensor,
) -> torch.Tensor:
    final = []
    for sample, actions in zip(world_states.cpu(), action_indices.cpu()):
        env = _world_from_vector(sample.numpy())
        for action in actions:
            env.step(int(action))
        final.append(torch.from_numpy(env.state.player_position.copy()))
    return torch.stack(final).float()


def _closed_loop_control(
    branch: PixelOnlyDynamicsBranch,
    latent: torch.Tensor,
    world_states: torch.Tensor,
    config: PixelOnlyPHConfig,
) -> dict[str, Any]:
    valid = world_states[:, 9].eq(0)
    indices = valid.nonzero(as_tuple=False).flatten()[: config.planner_samples]
    initial = latent[indices].detach()
    selected_world = world_states[indices]
    initial_position = selected_world[:, :2].to(latent.device)
    directions = torch.tensor(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        device=latent.device,
    )
    target = (initial_position + 0.12 * directions[torch.arange(indices.numel(), device=latent.device) % 4]).clamp(0.14, 0.86)
    raw_actions = torch.zeros(
        indices.numel(), config.planner_horizon, 2,
        device=latent.device, requires_grad=True,
    )
    optimizer = torch.optim.Adam((raw_actions,), lr=config.planner_learning_rate)
    for _ in range(config.planner_steps):
        optimizer.zero_grad(set_to_none=True)
        actions = torch.tanh(raw_actions)
        current = initial
        for step in range(config.planner_horizon):
            current = branch.step(current, actions[:, step])
        logits = branch.renderer(current)
        predicted_position = _soft_centroid(logits, PLAYER_CLASSES) / branch.renderer.image_size
        loss = (predicted_position - target).square().sum(dim=-1).mean()
        loss = loss + 1e-3 * actions.square().mean()
        loss.backward()
        optimizer.step()
    continuous = torch.tanh(raw_actions).detach()
    action_indices = _nearest_actions(continuous)
    planned_final = _simulate_plan(selected_world, action_indices).to(latent.device)
    coast_final = _simulate_plan(
        selected_world,
        torch.zeros(indices.numel(), config.planner_horizon, dtype=torch.long),
    ).to(latent.device)
    generator = torch.Generator().manual_seed(config.seed + 91_003)
    random_actions = torch.randint(
        0, len(ACTION_VECTORS),
        (indices.numel(), config.planner_horizon),
        generator=generator,
    )
    random_final = _simulate_plan(selected_world, random_actions).to(latent.device)
    planned_error = torch.linalg.vector_norm(planned_final - target, dim=-1)
    coast_error = torch.linalg.vector_norm(coast_final - target, dim=-1)
    random_error = torch.linalg.vector_norm(random_final - target, dim=-1)
    with torch.no_grad():
        predicted = initial
        for step in range(config.planner_horizon):
            predicted = branch.step(predicted, continuous[:, step])
        predicted_position = _soft_centroid(
            branch.renderer(predicted), PLAYER_CLASSES
        ) / branch.renderer.image_size
    return {
        "samples": int(indices.numel()),
        "horizon": config.planner_horizon,
        "optimizationSteps": config.planner_steps,
        "modelPredictedTargetError": float(
            torch.linalg.vector_norm(predicted_position - target, dim=-1).mean()
        ),
        "realSimulatorTargetError": float(planned_error.mean()),
        "coastTargetError": float(coast_error.mean()),
        "randomTargetError": float(random_error.mean()),
        "realImprovementVsCoast": float(
            (coast_error.mean() - planned_error.mean()) / coast_error.mean().clamp_min(1e-12)
        ),
        "realImprovementVsRandom": float(
            (random_error.mean() - planned_error.mean()) / random_error.mean().clamp_min(1e-12)
        ),
        "beatsCoastFraction": float((planned_error < coast_error).float().mean()),
        "meanContinuousActionNorm": float(
            torch.linalg.vector_norm(continuous, dim=-1).mean()
        ),
    }


def _evaluate_branch(
    branch: PixelOnlyDynamicsBranch,
    suite: dict[str, torch.Tensor],
    audit: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    config: PixelOnlyPHConfig,
    *,
    full_audit: bool = True,
) -> dict[str, Any]:
    branch.eval().requires_grad_(False)
    physical_fit, probe_metrics = _physical_probe(branch, audit)
    with torch.no_grad():
        encoded = branch.encode(_suite_model_inputs(suite))
        physical = canonical_state(suite["worldStates"])
        current_physical = _ridge_predict(
            physical_fit, encoded.reshape(-1, encoded.shape[-1])
        ).reshape_as(physical)
        reconstruction = _decode_classes(branch, encoded[:, 0]).to(encoded.device)
        reconstruction_metrics = _pixel_metrics(reconstruction, suite["frames"][:, 0])

        current = encoded[:, 0]
        rollout_states = []
        for step in range(suite["actionVectors"].shape[1]):
            current = branch.step(current, suite["actionVectors"][:, step])
            rollout_states.append(current)
        rollout = torch.stack(rollout_states, dim=1)
        horizons = {}
        for horizon in (1, 2, 4, 8):
            state = rollout[:, horizon - 1]
            prediction = _decode_classes(branch, state).to(encoded.device)
            target_frame = suite["frames"][:, horizon]
            logits = branch.renderer(state)
            aligned = _ridge_predict(physical_fit, state)
            horizons[str(horizon)] = {
                "pixels": _pixel_metrics(prediction, target_frame),
                "weightedCrossEntropy": float(
                    _pixel_cross_entropy(logits, target_frame, class_weights)
                ),
                "alignedPhysical": {
                    "qR2": _r2(aligned[:, :4], physical[:, horizon, :4]),
                    "pR2": _r2(aligned[:, 4:8], physical[:, horizon, 4:8]),
                    "qRmse": float(
                        (aligned[:, :4] - physical[:, horizon, :4]).square().mean().sqrt()
                    ),
                    "pRmse": float(
                        (aligned[:, 4:8] - physical[:, horizon, 4:8]).square().mean().sqrt()
                    ),
                },
            }

        shuffled_actions = torch.roll(suite["actionVectors"], shifts=1, dims=0)
        shuffled = encoded[:, 0]
        zero = encoded[:, 0]
        for step in range(suite["actionVectors"].shape[1]):
            shuffled = branch.step(shuffled, shuffled_actions[:, step])
            zero = branch.step(zero, torch.zeros_like(suite["actionVectors"][:, step]))
        target_h8 = suite["frames"][:, -1]
        correct_logits = branch.renderer(rollout[:, -1])
        shuffled_logits = branch.renderer(shuffled)
        zero_logits = branch.renderer(zero)
        correct_ce = _pixel_cross_entropy(correct_logits, target_h8, class_weights)
        shuffled_ce = _pixel_cross_entropy(shuffled_logits, target_h8, class_weights)
        zero_ce = _pixel_cross_entropy(zero_logits, target_h8, class_weights)
        action_controls = {
            "correctH8CrossEntropy": float(correct_ce),
            "shuffledH8CrossEntropy": float(shuffled_ce),
            "zeroH8CrossEntropy": float(zero_ce),
            "shuffledRelativeDegradation": float(
                (shuffled_ce - correct_ce) / correct_ce.clamp_min(1e-12)
            ),
            "zeroRelativeDegradation": float(
                (zero_ce - correct_ce) / correct_ce.clamp_min(1e-12)
            ),
        }
        counterfactual = (
            _counterfactual_audit(
                branch,
                encoded[:, 0],
                suite["worldStates"][:, 0],
                physical_fit,
            )
            if full_audit
            else None
        )
        structure = (
            _structured_audit(
                branch,
                encoded[:, 0],
                suite["actionVectors"][:, 0],
                physical[:, 0],
                physical_fit,
            )
            if full_audit and isinstance(branch.core, NeuralPortHamiltonian)
            else None
        )

    closed_loop = (
        _closed_loop_control(
            branch,
            encoded[:, 0],
            suite["worldStates"][:, 0],
            config,
        )
        if full_audit
        else None
    )
    return {
        "postHocPhysicalProbe": probe_metrics,
        "currentStatePhysicalReadout": {
            "qR2": _r2(current_physical[..., :4], physical[..., :4]),
            "pR2": _r2(current_physical[..., 4:8], physical[..., 4:8]),
        },
        "currentFrameReconstruction": reconstruction_metrics,
        "rolloutByHorizon": horizons,
        "actionControls": action_controls,
        **(
            {
                "oneStepCounterfactualAction": counterfactual,
                "closedLoopPixelTargetControl": closed_loop,
            }
            if full_audit
            else {}
        ),
        **({"structure": structure} if structure is not None else {}),
    }


def _parameter_change_payload(
    branch: PixelOnlyDynamicsBranch,
    initial: dict[str, torch.Tensor],
) -> dict[str, float]:
    if not isinstance(branch.core, NeuralPortHamiltonian):
        return {}
    current = branch.core.state_dict()
    groups = {
        "H": "energy_network",
        "J": "interconnection_network",
        "R": "resistance_network",
        "B": "port_network",
    }
    return {
        label: float(
            torch.stack(
                [
                    (value.detach().cpu() - initial[name]).square().sum()
                    for name, value in current.items()
                    if name.startswith(prefix) and name in initial
                ]
            ).sum().sqrt()
        )
        for label, prefix in groups.items()
    }


def _decision(evaluation: dict[str, Any]) -> dict[str, Any]:
    ph = evaluation["policy"]["pixelOnlyPortHamiltonian"]
    control = evaluation["policy"]["pixelOnlyNeuralOde"]
    structure = ph["structure"]
    ph_h4 = ph["rolloutByHorizon"]["4"]["pixels"]["playerCentroidErrorPx"]
    control_h4 = control["rolloutByHorizon"]["4"]["pixels"]["playerCentroidErrorPx"]
    gates = {
        "pixelReconstructionPlayerIouAtLeast0.70": ph[
            "currentFrameReconstruction"
        ]["playerIou"] >= 0.70,
        "pixelReconstructionPuckIouAtLeast0.50": ph[
            "currentFrameReconstruction"
        ]["puckIou"] >= 0.50,
        "unsupervisedPositionDiscoveryQ_R2AtLeast0.80": ph[
            "postHocPhysicalProbe"
        ]["qR2"] >= 0.80,
        "unsupervisedMomentumDiscoveryP_R2AtLeast0.50": ph[
            "postHocPhysicalProbe"
        ]["pR2"] >= 0.50,
        "alignedPortPhysicalCosineAtLeast0.80": structure[
            "portAfterPostHocPhysicalAlignment"
        ]["physicalIncidenceCosine"] >= 0.80,
        "counterfactualMomentumCosineAtLeast0.80": ph[
            "oneStepCounterfactualAction"
        ]["alignedPlayerMomentumCosine"] >= 0.80,
        "renderedCounterfactualSignAtLeast0.80": ph[
            "oneStepCounterfactualAction"
        ]["renderedTargetSignAgreement"] >= 0.80,
        "usesActionsShuffledDegradationAtLeast0.05": ph["actionControls"][
            "shuffledRelativeDegradation"
        ] >= 0.05,
        "closedLoopRealImprovementVsCoastAtLeast0.20": ph[
            "closedLoopPixelTargetControl"
        ]["realImprovementVsCoast"] >= 0.20,
        "closedLoopBeatsCoastFractionAtLeast0.65": ph[
            "closedLoopPixelTargetControl"
        ]["beatsCoastFraction"] >= 0.65,
        "pHPredictiveParityAtH4": ph_h4 <= 1.15 * control_h4,
        "exactContinuousPowerBalance": structure["powerBalance"][
            "maxAbsDefect"
        ] <= 1e-5,
        "zeroInputEnergyMonotone": structure["zeroInputDiscreteEnergy"][
            "increaseFraction"
        ] <= 1e-3,
    }
    supported = all(gates.values())
    return {
        "outcome": (
            "provisional_breakthrough_supported_single_seed"
            if supported
            else "provisional_breakthrough_not_supported_single_seed"
        ),
        "allGatesPass": supported,
        "gates": gates,
        "scope": (
            "One pH training seed on one visual system. Physical states are used "
            "only for post-training audits and never enter optimization losses."
        ),
    }


def _move_suite(suite: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in suite.items()}


def _suite_model_inputs(suite: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return either cached readouts or raw pixel contexts for end-to-end models."""

    if "pixelContexts" in suite:
        return suite["pixelContexts"]
    return suite["features"]


def run_pixel_only_ph_experiment(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config: PixelOnlyPHConfig = PixelOnlyPHConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    if config.block_count != 5:
        raise ValueError("the pixel-only experiment is registered at block 5")
    if config.state_size != 8:
        raise ValueError("the first registered run uses an eight-dimensional latent")
    if config.transitions_per_trajectory != 8:
        raise ValueError("the first registered run uses eight transitions")
    if min(config.fit_policy_trajectories, config.fit_cardinal_trajectories) < 1:
        raise ValueError("both fit families must be present")
    _seed_everything(config.seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = (
        build_pixel_direct_from_checkpoint(payload).to(device).eval().requires_grad_(False)
    )
    model_config = backbone.config
    collection_started = time.perf_counter()
    fit_policy = collect_visual_action_suite(
        backbone,
        trajectories=config.fit_policy_trajectories,
        transitions=config.transitions_per_trajectory,
        seed=config.seed + 1_000_000,
        family="policy",
        feature_batch_size=config.feature_batch_size,
        block_count=config.block_count,
        device=device,
        include_world_states=False,
    )
    fit_cardinal = collect_visual_action_suite(
        backbone,
        trajectories=config.fit_cardinal_trajectories,
        transitions=config.transitions_per_trajectory,
        seed=config.seed + 2_000_000,
        family="cardinal",
        feature_batch_size=config.feature_batch_size,
        block_count=config.block_count,
        device=device,
        include_world_states=False,
    )
    fit_cpu = _concatenate_training_suites(fit_policy, fit_cardinal)
    collection_seconds = time.perf_counter() - collection_started
    fit = _move_suite(fit_cpu, device)
    fit["features"] = fit["features"].float()
    del fit_policy, fit_cardinal, fit_cpu

    flat_features = fit["features"].reshape(-1, fit["features"].shape[-1])
    feature_mean = flat_features.mean(dim=0)
    feature_scale = flat_features.std(dim=0).clamp_min(1e-5)
    class_weights = _class_weights(fit["frames"], model_config.palette_size).to(device)
    autoencoder = VisualAutoencoder(
        GenericStateEncoder(
            feature_mean,
            feature_scale,
            config.state_size,
            config.encoder_hidden_size,
        ),
        GenericSpatialRenderer(
            config.state_size,
            model_config.image_size,
            model_config.palette_size,
            config.renderer_hidden_size,
        ),
    ).to(device)
    autoencoder_optimizer = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    flat_frames = fit["frames"].reshape(-1, model_config.image_size, model_config.image_size)
    autoencoder_started = time.perf_counter()
    autoencoder_log = output_dir / "autoencoder.jsonl"
    with autoencoder_log.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.autoencoder_steps + 1):
            indices = torch.randint(
                0, flat_features.shape[0],
                (config.autoencoder_batch_size,),
                device=device,
            )
            learning_rate = config.learning_rate * _learning_rate_multiplier(
                step, config.autoencoder_steps, config
            )
            for group in autoencoder_optimizer.param_groups:
                group["lr"] = learning_rate
            autoencoder_optimizer.zero_grad(set_to_none=True)
            state, logits = autoencoder(flat_features[indices])
            reconstruction = _pixel_cross_entropy(logits, flat_frames[indices], class_weights)
            whitening = _whitening_loss(state)
            loss = reconstruction + config.whitening_weight * whitening
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 5.0)
            autoencoder_optimizer.step()
            if step == 1 or step % config.log_every == 0 or step == config.autoencoder_steps:
                elapsed = time.perf_counter() - autoencoder_started
                record = {
                    "stage": "pixel_only_autoencoder",
                    "step": step,
                    "steps": config.autoencoder_steps,
                    "loss": float(loss.detach()),
                    "reconstruction": float(reconstruction.detach()),
                    "whitening": float(whitening.detach()),
                    "gradientNorm": float(gradient_norm),
                    "stepsPerSecond": step / max(elapsed, 1e-8),
                    "estimatedSeconds": elapsed / step * config.autoencoder_steps,
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
    autoencoder_seconds = time.perf_counter() - autoencoder_started

    core_config = NeuralPortHamiltonianConfig(
        state_size=config.state_size,
        input_size=2,
        hidden_size=config.core_hidden_size,
        hidden_layers=config.core_hidden_layers,
        dt=WorldConfig().dt,
        integration_method="midpoint",
        integration_substeps=1,
        resistance_floor=1e-5,
    )
    ph = PixelOnlyDynamicsBranch(
        copy.deepcopy(autoencoder.encoder),
        copy.deepcopy(autoencoder.renderer),
        core_config=core_config,
        structured=True,
    ).to(device)
    control_hidden = _matched_control_hidden_size(
        _parameter_count(ph.core), core_config
    )
    control = PixelOnlyDynamicsBranch(
        copy.deepcopy(autoencoder.encoder),
        copy.deepcopy(autoencoder.renderer),
        core_config=replace(core_config, hidden_size=control_hidden),
        structured=False,
    ).to(device)
    branches = {
        "pixelOnlyPortHamiltonian": ph,
        "pixelOnlyNeuralOde": control,
    }
    ph_initial_core = {
        name: value.detach().cpu().clone() for name, value in ph.core.state_dict().items()
    }
    capacity_gap = abs(_parameter_count(ph.core) - _parameter_count(control.core))
    capacity_gap /= max(_parameter_count(ph.core), 1)
    if capacity_gap > 0.01:
        raise AssertionError("pH and Neural ODE core capacity must match within one percent")
    del autoencoder, autoencoder_optimizer
    optimizer = torch.optim.AdamW(
        [parameter for branch in branches.values() for parameter in branch.parameters()],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    dynamics_started = time.perf_counter()
    dynamics_log = output_dir / "dynamics.jsonl"
    with dynamics_log.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.dynamics_steps + 1):
            indices = torch.randint(
                0, config.fit_trajectories,
                (config.dynamics_batch_size,),
                device=device,
            )
            learning_rate = config.learning_rate * _learning_rate_multiplier(
                step, config.dynamics_steps, config
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            terms = {}
            for name, branch in branches.items():
                branch.train()
                loss, branch_terms = pixel_only_branch_loss(
                    branch,
                    fit["features"][indices],
                    fit["frames"][indices],
                    fit["actionVectors"][indices],
                    class_weights,
                    config,
                )
                losses[name] = loss
                terms[name] = branch_terms
            total = sum(losses.values()) / len(losses)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for branch in branches.values() for parameter in branch.parameters()],
                5.0,
            )
            optimizer.step()
            if step == 1 or step % config.log_every == 0 or step == config.dynamics_steps:
                elapsed = time.perf_counter() - dynamics_started
                record = {
                    "stage": "train_pixel_only_dynamics",
                    "step": step,
                    "steps": config.dynamics_steps,
                    "loss": float(total.detach()),
                    "gradientNorm": float(gradient_norm),
                    "stepsPerSecond": step / max(elapsed, 1e-8),
                    "estimatedSeconds": elapsed / step * config.dynamics_steps,
                    **{
                        name: {
                            "loss": float(losses[name].detach()),
                            **{
                                key: float(value.detach())
                                for key, value in terms[name].items()
                            },
                        }
                        for name in branches
                    },
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
    dynamics_seconds = time.perf_counter() - dynamics_started
    del fit
    if device.type == "cuda":
        torch.cuda.empty_cache()

    evaluation_collection_started = time.perf_counter()
    suite_specs = {
        "policy": (config.test_policy_trajectories, "policy", config.seed + 3_000_000),
        "diagonalOod": (
            config.test_diagonal_trajectories, "diagonal", config.seed + 4_000_000
        ),
        "reversalOod": (
            config.test_reversal_trajectories, "reversal", config.seed + 5_000_000
        ),
    }
    evaluation_suites = {
        name: _move_suite(
            collect_visual_action_suite(
                backbone,
                trajectories=count,
                transitions=config.transitions_per_trajectory,
                seed=seed,
                family=family,
                feature_batch_size=config.feature_batch_size,
                block_count=config.block_count,
                device=device,
                include_world_states=True,
            ),
            device,
        )
        for name, (count, family, seed) in suite_specs.items()
    }
    audit = _move_suite(
        collect_visual_action_suite(
            backbone,
            trajectories=config.audit_trajectories,
            transitions=config.transitions_per_trajectory,
            seed=config.seed + 6_000_000,
            family="policy",
            feature_batch_size=config.feature_batch_size,
            block_count=config.block_count,
            device=device,
            include_world_states=True,
        ),
        device,
    )
    for suite in (*evaluation_suites.values(), audit):
        suite["features"] = suite["features"].float()
    evaluation_collection_seconds = time.perf_counter() - evaluation_collection_started
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    evaluation_started = time.perf_counter()
    evaluation = {
        suite_name: {
            branch_name: _evaluate_branch(
                branch, suite, audit, class_weights, config
            )
            for branch_name, branch in branches.items()
        }
        for suite_name, suite in evaluation_suites.items()
    }
    evaluation_seconds = time.perf_counter() - evaluation_started
    conclusion = _decision(evaluation)
    checkpoint = {
        "kind": "pixel_action_only_port_hamiltonian_world_model",
        "version": 1,
        "baseCheckpoint": str(checkpoint_path),
        "baseCheckpointStep": int(payload["step"]),
        "config": asdict(config),
        "controlHiddenSize": control_hidden,
        "branches": {name: branch.state_dict() for name, branch in branches.items()},
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    summary = {
        "kind": checkpoint["kind"],
        "version": 1,
        "baseCheckpointStep": int(payload["step"]),
        "singleTrainingSeed": True,
        "trainingSupervision": {
            "inputs": ["rendered categorical pixels", "two-dimensional action vectors"],
            "excludedFromEveryOptimizationLoss": [
                "simulator state",
                "positions",
                "velocities",
                "momenta",
                "energy",
                "events and collision labels",
                "object masks and entity tokens",
            ],
            "physicalStatesUsedOnlyAfterTraining": True,
            "frozenBackboneWasPretrainedFromPixelsOnly": True,
        },
        "architecture": {
            "frozenPixelBackbone": True,
            "genericBlock5Readout": ["spatial mean", "spatial standard deviation", "fixed global token"],
            "latentStateSize": config.state_size,
            "genericSpatialBroadcastRenderer": True,
            "learnedFunctions": ["H(x)", "J(x)", "R(x)", "B(x)"],
            "exactConstraints": ["J(x)=-J(x)^T", "R(x)=L(x)L(x)^T"],
            "matchedNeuralOdeControl": True,
            "noStateOrPixelSkipConnection": True,
        },
        "config": asdict(config),
        "capacity": {
            "controlHiddenSize": control_hidden,
            "core": {name: _parameter_count(branch.core) for name, branch in branches.items()},
            "completeBranch": {name: _parameter_count(branch) for name, branch in branches.items()},
            "relativeCoreGap": capacity_gap,
        },
        "parameterChangeNorm": _parameter_change_payload(ph, ph_initial_core),
        "timing": {
            "fitCollectionSeconds": collection_seconds,
            "autoencoderSeconds": autoencoder_seconds,
            "dynamicsSeconds": dynamics_seconds,
            "evaluationCollectionSeconds": evaluation_collection_seconds,
            "evaluationSeconds": evaluation_seconds,
            "totalSeconds": (
                collection_seconds
                + autoencoder_seconds
                + dynamics_seconds
                + evaluation_collection_seconds
                + evaluation_seconds
            ),
        },
        "evaluation": evaluation,
        "conclusion": conclusion,
        "artifacts": ["checkpoint.pt", "autoencoder.jsonl", "dynamics.jsonl", "summary.json"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fit-policy-trajectories", type=int, default=3_072)
    parser.add_argument("--fit-cardinal-trajectories", type=int, default=3_072)
    parser.add_argument("--test-policy-trajectories", type=int, default=512)
    parser.add_argument("--test-diagonal-trajectories", type=int, default=256)
    parser.add_argument("--test-reversal-trajectories", type=int, default=256)
    parser.add_argument("--audit-trajectories", type=int, default=1_024)
    parser.add_argument("--autoencoder-steps", type=int, default=3_000)
    parser.add_argument("--autoencoder-batch-size", type=int, default=256)
    parser.add_argument("--dynamics-steps", type=int, default=5_000)
    parser.add_argument("--dynamics-batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--planner-samples", type=int, default=64)
    parser.add_argument("--planner-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=131_610_731)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_pixel_only_ph_experiment(
        args.checkpoint,
        args.output,
        config=PixelOnlyPHConfig(
            fit_policy_trajectories=args.fit_policy_trajectories,
            fit_cardinal_trajectories=args.fit_cardinal_trajectories,
            test_policy_trajectories=args.test_policy_trajectories,
            test_diagonal_trajectories=args.test_diagonal_trajectories,
            test_reversal_trajectories=args.test_reversal_trajectories,
            audit_trajectories=args.audit_trajectories,
            autoencoder_steps=args.autoencoder_steps,
            autoencoder_batch_size=args.autoencoder_batch_size,
            dynamics_steps=args.dynamics_steps,
            dynamics_batch_size=args.dynamics_batch_size,
            feature_batch_size=args.feature_batch_size,
            planner_samples=args.planner_samples,
            planner_steps=args.planner_steps,
            seed=args.seed,
            log_every=args.log_every,
        ),
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
