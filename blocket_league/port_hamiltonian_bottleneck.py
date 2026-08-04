from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import make_passive_clip
from .env import WorldConfig
from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .pixel_probe import PLAYER_CLASSES, _visual_centroid
from .port_hamiltonian_audit import (
    REGIMES,
    _effective_physics,
    _entity_token,
    _state_metrics,
    _transition_metrics,
    canonical_state,
    transition_regimes,
)
from .position_write_probe import PUCK_CLASSES
from .train_pixel_direct import frames_to_classes, palette_tensor


@dataclass(frozen=True)
class BottleneckExperimentConfig:
    trajectories: int = 8_192
    transitions_per_trajectory: int = 8
    fit_fraction: float = 0.75
    goal_centered_fraction: float = 0.20
    feature_batch_size: int = 64
    block_count: int = 5
    steps: int = 6_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    min_learning_rate_ratio: float = 0.1
    hidden_size: int = 64
    state_loss_weight: float = 1.0
    teacher_dynamics_weight: float = 1.0
    rollout_loss_weight: float = 2.0
    event_loss_weight: float = 0.20
    free_port_weight: float = 0.20
    ridge: float = 1e-2
    log_every: int = 100
    seed: int = 91_410_731


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bottleneck_state(states: torch.Tensor) -> torch.Tensor:
    """Canonical mechanical state plus two observable hybrid-mode coordinates."""

    canonical = canonical_state(states)
    score_phase = torch.remainder(states[..., 8:9], 5.0) / 4.0
    reset_phase = states[..., 9:10] / float(WorldConfig().goal_pause_steps)
    return torch.cat((canonical, score_phase, reset_phase), dim=-1)


def regime_labels(
    states_t: torch.Tensor,
    states_tp1: torch.Tensor,
    events_tp1: torch.Tensor,
) -> torch.Tensor:
    masks = transition_regimes(states_t, states_tp1, events_tp1)
    labels = torch.zeros_like(events_tp1, dtype=torch.long)
    for index, name in enumerate(REGIMES):
        labels[masks[name]] = index
    return labels


@torch.no_grad()
def _block5_entity_features(
    model: nn.Module,
    classes: torch.Tensor,
    block_count: int,
) -> torch.Tensor:
    if not 1 <= block_count <= len(model.blocks):
        raise ValueError("block_count must select an existing transformer block")
    tokens = (
        model.patch_projection(model.patch_tokens(classes))
        + model.spatial_position
        + model.temporal_position[:, : classes.shape[1]]
    )
    for block in model.blocks[:block_count]:
        tokens = block(tokens)
    latest = tokens[:, -1]
    player_position = _visual_centroid(classes[:, -1], PLAYER_CLASSES)
    puck_position = _visual_centroid(classes[:, -1], PUCK_CLASSES)
    player_token = _entity_token(model, player_position)
    puck_token = _entity_token(model, puck_position)
    batch = torch.arange(classes.shape[0], device=classes.device)
    return torch.cat(
        (latest[batch, player_token], latest[batch, puck_token]),
        dim=-1,
    ).float()


@torch.no_grad()
def collect_bottleneck_sequences(
    model: nn.Module,
    config: BottleneckExperimentConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Collect complete trajectories so no rollout crosses a split boundary."""

    feature_chunks: list[torch.Tensor] = []
    state_sequences: list[torch.Tensor] = []
    event_sequences: list[torch.Tensor] = []
    pending_contexts: list[np.ndarray] = []
    palette = palette_tensor(device)
    goal_period = None
    if config.goal_centered_fraction > 0:
        goal_period = max(1, round(1.0 / config.goal_centered_fraction))

    def flush() -> None:
        if not pending_contexts:
            return
        videos = torch.from_numpy(np.stack(pending_contexts)).permute(0, 1, 4, 2, 3)
        videos = videos.to(device, non_blocking=True).float().div(127.5).sub(1.0)
        classes = frames_to_classes(videos, palette)
        feature_chunks.append(
            _block5_entity_features(model, classes, config.block_count).cpu()
        )
        pending_contexts.clear()

    started = time.perf_counter()
    history = model.config.history_frames
    endpoints = config.transitions_per_trajectory + 1
    for trajectory in range(config.trajectories):
        clip = make_passive_clip(
            config.seed + trajectory * 9_973,
            context_frames=1,
            future_frames=history + config.transitions_per_trajectory - 1,
            image_size=model.config.image_size,
            goal_centered=goal_period is not None and trajectory % goal_period == 0,
        )
        frames = clip["frames"]
        all_states = torch.from_numpy(clip["all_state"]).float()
        all_events = torch.from_numpy(clip["all_events"]).long()
        endpoint_states = torch.stack(
            [all_states[offset + history - 1] for offset in range(endpoints)]
        )
        transition_events = torch.stack(
            [
                all_events[offset + history]
                for offset in range(config.transitions_per_trajectory)
            ]
        )
        state_sequences.append(endpoint_states)
        event_sequences.append(transition_events)
        for offset in range(endpoints):
            pending_contexts.append(frames[offset : offset + history])
            if len(pending_contexts) >= config.feature_batch_size:
                flush()
        if (trajectory + 1) % 256 == 0 or trajectory + 1 == config.trajectories:
            print(
                json.dumps(
                    {
                        "stage": "collect_bottleneck",
                        "trajectories": trajectory + 1,
                        "total": config.trajectories,
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    flush()
    features = torch.cat(feature_chunks).reshape(config.trajectories, endpoints, -1)
    states = torch.stack(state_sequences)
    events = torch.stack(event_sequences)
    labels = regime_labels(states[:, :-1], states[:, 1:], events)
    return {
        "features": features,
        "worldStates": states,
        "states": bottleneck_state(states),
        "events": events,
        "regimes": labels,
    }


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _physical_free_parameters() -> tuple[torch.Tensor, torch.Tensor]:
    world = WorldConfig()
    decay = math.exp(-0.12 * world.dt)
    gains = torch.tensor(
        [
            (1.0 - decay) / (0.12 * world.player_mass),
            (1.0 - decay) / (0.12 * world.puck_mass),
        ],
        dtype=torch.float32,
    )
    return gains, torch.full((2,), decay, dtype=torch.float32)


class PortHamiltonianFreeCore(nn.Module):
    """Four-parameter passive free flow with tied x/y particle coefficients."""

    def __init__(self) -> None:
        super().__init__()
        gains, decays = _physical_free_parameters()
        self.raw_gain = nn.Parameter(
            torch.tensor([_inverse_softplus(float(value)) for value in gains])
        )
        self.raw_decay = nn.Parameter(
            torch.tensor([_logit(float(value)) for value in decays])
        )

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.softplus(self.raw_gain) + 1e-6, torch.sigmoid(self.raw_decay)

    def forward(
        self,
        state: torch.Tensor,
        external_port: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gains, decays = self.coefficients()
        output = state.clone()
        for entity in range(2):
            q_slice = slice(entity * 2, entity * 2 + 2)
            p_slice = slice(4 + entity * 2, 4 + entity * 2 + 2)
            output[..., q_slice] = state[..., q_slice] + gains[entity] * state[..., p_slice]
            output[..., p_slice] = decays[entity] * state[..., p_slice]
        if external_port is not None:
            output[..., 4:8] = output[..., 4:8] + external_port
        return output


class SignFreeMatchedCore(nn.Module):
    """The same four coefficients and map, without pH sign/passivity constraints."""

    def __init__(self) -> None:
        super().__init__()
        gains, decays = _physical_free_parameters()
        self.gain = nn.Parameter(gains)
        self.decay = nn.Parameter(decays)

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gain, self.decay

    def forward(
        self,
        state: torch.Tensor,
        external_port: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gains, decays = self.coefficients()
        output = state.clone()
        for entity in range(2):
            q_slice = slice(entity * 2, entity * 2 + 2)
            p_slice = slice(4 + entity * 2, 4 + entity * 2 + 2)
            output[..., q_slice] = state[..., q_slice] + gains[entity] * state[..., p_slice]
            output[..., p_slice] = decays[entity] * state[..., p_slice]
        if external_port is not None:
            output[..., 4:8] = output[..., 4:8] + external_port
        return output


class HybridJumpPort(nn.Module):
    """Predict event mode and the instantaneous state jump outside smooth flow."""

    def __init__(self, state_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(REGIMES) + state_size),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        normalized_state: torch.Tensor,
        state_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.network(normalized_state)
        event_logits = output[..., : len(REGIMES)]
        event_probability = event_logits.softmax(dim=-1)
        nonfree_gate = 1.0 - event_probability[..., :1]
        raw_jump = torch.tanh(output[..., len(REGIMES) :]) * (2.0 * state_scale)
        return event_logits, nonfree_gate * raw_jump, nonfree_gate


class CausalBottleneckBranch(nn.Module):
    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        state_mean: torch.Tensor,
        state_scale: torch.Tensor,
        *,
        hidden_size: int,
        structured: bool,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_scale", feature_scale)
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_scale", state_scale)
        self.encoder = nn.Linear(feature_mean.numel(), state_mean.numel())
        self.core: PortHamiltonianFreeCore | SignFreeMatchedCore
        self.core = PortHamiltonianFreeCore() if structured else SignFreeMatchedCore()
        self.hybrid_port = HybridJumpPort(state_mean.numel(), hidden_size)
        self.structured = structured

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_scale

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.state_mean + self.state_scale * self.encoder(normalized)

    def step(
        self,
        state: torch.Tensor,
        external_port: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        smooth = state.clone()
        smooth[..., :8] = self.core(state[..., :8], external_port)
        event_logits, jump, gate = self.hybrid_port(
            self.normalize_state(state), self.state_scale
        )
        return smooth + jump, event_logits, jump, gate


def _fit_linear_encoder(
    branch: CausalBottleneckBranch,
    features: torch.Tensor,
    states: torch.Tensor,
    ridge: float,
) -> None:
    normalized_features = (features - branch.feature_mean) / branch.feature_scale
    normalized_states = (states - branch.state_mean) / branch.state_scale
    design = torch.cat(
        (normalized_features, torch.ones(normalized_features.shape[0], 1, device=features.device)),
        dim=1,
    )
    penalty = torch.eye(design.shape[1], device=features.device)
    penalty[-1, -1] = 0
    weight = torch.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ normalized_states,
    )
    branch.encoder.weight.data.copy_(weight[:-1].T)
    branch.encoder.bias.data.copy_(weight[-1])


def _learning_rate_multiplier(
    step: int,
    config: BottleneckExperimentConfig,
) -> float:
    if step <= config.warmup_steps:
        return step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate_ratio + (1.0 - config.min_learning_rate_ratio) * cosine


def _event_class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels.flatten(), minlength=len(REGIMES)).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights.sqrt().clamp(0.25, 8.0)
    return weights / weights.mean()


def _branch_loss(
    branch: CausalBottleneckBranch,
    features: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    event_weights: torch.Tensor,
    config: BottleneckExperimentConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    encoded = branch.encode(features)
    state_loss = ((encoded - targets) / branch.state_scale).square().mean()

    flat_encoded = encoded[:, :-1].reshape(-1, encoded.shape[-1])
    teacher_next, teacher_logits, teacher_jump, _ = branch.step(flat_encoded)
    flat_targets = targets[:, 1:].reshape_as(teacher_next)
    teacher_loss = ((teacher_next - flat_targets) / branch.state_scale).square().mean()
    event_loss = F.cross_entropy(
        teacher_logits,
        labels.flatten(),
        weight=event_weights,
    )
    free = labels.flatten() == 0
    if bool(free.any()):
        free_port_loss = (
            teacher_jump[free] / branch.state_scale
        ).square().mean()
    else:
        free_port_loss = teacher_jump.square().mean() * 0.0

    current = encoded[:, 0]
    rollout_predictions = []
    for _ in range(labels.shape[1]):
        current, _, _, _ = branch.step(current)
        rollout_predictions.append(current)
    rollout = torch.stack(rollout_predictions, dim=1)
    normalized_rollout_error = (rollout - targets[:, 1:]) / branch.state_scale
    horizon_weights = torch.linspace(
        1.0,
        2.0,
        labels.shape[1],
        device=features.device,
    )
    rollout_loss = (
        normalized_rollout_error.square().mean(dim=(0, 2)) * horizon_weights
    ).sum() / horizon_weights.sum()
    total = (
        config.state_loss_weight * state_loss
        + config.teacher_dynamics_weight * teacher_loss
        + config.rollout_loss_weight * rollout_loss
        + config.event_loss_weight * event_loss
        + config.free_port_weight * free_port_loss
    )
    return total, {
        "state": state_loss,
        "teacher": teacher_loss,
        "rollout": rollout_loss,
        "event": event_loss,
        "freePort": free_port_loss,
    }


def _coefficient_payload(branch: CausalBottleneckBranch) -> dict[str, Any]:
    gain, decay = branch.core.coefficients()
    parameters = {
        "positionGain": gain.detach().cpu().tolist(),
        "momentumDecay": decay.detach().cpu().tolist(),
    }
    return {
        **parameters,
        **_effective_physics(parameters, WorldConfig().dt),
        "withinPortHamiltonianDomain": bool(
            (gain > 0).all() and (decay >= 0).all() and (decay <= 1).all()
        ),
    }


def _r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    residual = (prediction - target).square().sum()
    centered = (target - target.mean(dim=0, keepdim=True)).square().sum().clamp_min(1e-12)
    return float(1.0 - residual / centered)


@torch.no_grad()
def evaluate_branch(
    branch: CausalBottleneckBranch,
    features: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    branch.eval()
    encoded = branch.encode(features)
    encoded_metrics = {
        **_state_metrics(encoded[..., :8].reshape(-1, 8), targets[..., :8].reshape(-1, 8)),
        "modeR2": _r2(encoded[..., 8:].reshape(-1, 2), targets[..., 8:].reshape(-1, 2)),
    }

    current = encoded[:, 0]
    rollout = []
    rollout_logits = []
    rollout_jumps = []
    for _ in range(labels.shape[1]):
        current, logits, jump, _ = branch.step(current)
        rollout.append(current)
        rollout_logits.append(logits)
        rollout_jumps.append(jump)
    rollout_tensor = torch.stack(rollout, dim=1)
    horizons: dict[str, Any] = {}
    for horizon in (1, 2, 4, labels.shape[1]):
        if horizon > labels.shape[1]:
            continue
        prediction = rollout_tensor[:, horizon - 1, :8]
        target = targets[:, horizon, :8]
        persistence = targets[:, 0, :8]
        scale = (target - persistence).square().mean().sqrt().clamp_min(1e-12)
        horizons[str(horizon)] = {
            "deltaNrmse": float((prediction - target).square().mean().sqrt() / scale),
            "qR2": _r2(prediction[:, :4], target[:, :4]),
            "pR2": _r2(prediction[:, 4:], target[:, 4:]),
        }

    shuffled_current = torch.roll(encoded[:, 0], shifts=1, dims=0)
    for _ in range(labels.shape[1]):
        shuffled_current, _, _, _ = branch.step(shuffled_current)
    shuffled_target = targets[:, -1, :8]
    shuffled_scale = (
        shuffled_target - targets[:, 0, :8]
    ).square().mean().sqrt().clamp_min(1e-12)
    shuffled_control = {
        "horizon": labels.shape[1],
        "deltaNrmse": float(
            (shuffled_current[:, :8] - shuffled_target).square().mean().sqrt()
            / shuffled_scale
        ),
    }

    teacher_current = encoded[:, :-1].reshape(-1, encoded.shape[-1])
    teacher_next, teacher_logits, teacher_jump, _ = branch.step(teacher_current)
    true_next = targets[:, 1:].reshape(-1, targets.shape[-1])
    flat_labels = labels.flatten()
    regimes: dict[str, Any] = {}
    for index, name in enumerate(REGIMES):
        mask = flat_labels == index
        regimes[name] = {
            "samples": int(mask.sum()),
            "transition": (
                _transition_metrics(
                    teacher_next[mask, :8],
                    teacher_current[mask, :8],
                    true_next[mask, :8],
                )
                if bool(mask.any())
                else None
            ),
            "meanPortNorm": (
                float(teacher_jump[mask, :8].norm(dim=-1).mean())
                if bool(mask.any())
                else None
            ),
        }

    predicted_events = teacher_logits.argmax(dim=-1)
    recalls = []
    for index in range(len(REGIMES)):
        mask = flat_labels == index
        if bool(mask.any()):
            recalls.append(float((predicted_events[mask] == index).float().mean()))
    event_metrics = {
        "accuracy": float((predicted_events == flat_labels).float().mean()),
        "balancedAccuracy": float(sum(recalls) / max(len(recalls), 1)),
        "perRegimeRecall": {
            name: (
                float((predicted_events[flat_labels == index] == index).float().mean())
                if bool((flat_labels == index).any())
                else None
            )
            for index, name in enumerate(REGIMES)
        },
    }

    gains, decays = branch.core.coefficients()
    core_current = encoded[:, :-1, :8].reshape(-1, 8)
    core_next = branch.core(core_current)
    free = flat_labels == 0
    masses = _coefficient_payload(branch)["effectiveMass"]

    def kinetic(value: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(value.shape[0], device=value.device)
        for entity, mass in enumerate(masses):
            momentum = value[:, 4 + entity * 2 : 6 + entity * 2]
            result += momentum.square().sum(dim=-1) / (2.0 * max(float(mass), 1e-8))
        return result

    energy_t = kinetic(core_current[free])
    energy_tp1 = kinetic(core_next[free])
    tolerance = 1e-6 * energy_t.mean().clamp_min(1e-8)
    passivity = {
        "coreViolationRate": float((energy_tp1 > energy_t + tolerance).float().mean()),
        "meanEnergyRatio": float(energy_tp1.mean() / energy_t.mean().clamp_min(1e-8)),
    }

    stability_state = encoded[: min(256, encoded.shape[0]), 0, :8]
    stability_energy = [kinetic(stability_state)]
    finite = True
    for _ in range(64):
        stability_state = branch.core(stability_state)
        finite = finite and bool(torch.isfinite(stability_state).all())
        stability_energy.append(kinetic(stability_state))
    energy_stack = torch.stack(stability_energy, dim=1)
    stability = {
        "steps": 64,
        "finite": finite,
        "energyGrowthFraction": float(
            (energy_stack[:, 1:] > energy_stack[:, :-1] + tolerance).float().mean()
        ),
        "finalToInitialEnergy": float(
            energy_stack[:, -1].mean() / energy_stack[:, 0].mean().clamp_min(1e-8)
        ),
        "maxAbsoluteState": float(stability_state.abs().max()),
    }

    intervention_state = encoded[: min(256, encoded.shape[0]), 0]
    zero_port = torch.zeros(intervention_state.shape[0], 4, device=features.device)
    zero_prediction = branch.step(intervention_state, zero_port)[0]
    control_results = []
    for coordinate in range(4):
        port = torch.zeros_like(zero_port)
        port[:, coordinate] = 0.05
        controlled = branch.step(intervention_state, port)[0]
        response = controlled[:, 4:8] - zero_prediction[:, 4:8]
        expected = port
        control_results.append(
            {
                "coordinate": coordinate,
                "cosine": float(
                    F.cosine_similarity(response, expected, dim=-1).mean()
                ),
                "gain": float(
                    response[:, coordinate].mean() / expected[:, coordinate].mean()
                ),
                "crossTalkRmse": float(
                    (response - expected).square().mean().sqrt()
                ),
            }
        )

    return {
        "stateReadout": encoded_metrics,
        "rolloutByHorizon": horizons,
        "shuffledInitialStateControl": shuffled_control,
        "regimes": regimes,
        "eventPrediction": event_metrics,
        "freeCore": {
            "parameters": _coefficient_payload(branch),
            "parameterCount": sum(parameter.numel() for parameter in branch.core.parameters()),
            "passivity": passivity,
        },
        "stability": stability,
        "externalPortInterventions": control_results,
    }


def run_bottleneck_experiment(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config: BottleneckExperimentConfig = BottleneckExperimentConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    if not 0 < config.fit_fraction < 1:
        raise ValueError("fit_fraction must be in (0, 1)")
    if config.transitions_per_trajectory < 2:
        raise ValueError("at least two transitions are required")
    if config.block_count != 5:
        raise ValueError("Experiment B is preregistered at block 5")
    _seed_everything(config.seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = build_pixel_direct_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    collection_started = time.perf_counter()
    collected = collect_bottleneck_sequences(backbone, config, device)
    collection_seconds = time.perf_counter() - collection_started
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fit_trajectories = int(config.trajectories * config.fit_fraction)
    features = collected["features"].to(device)
    states = collected["states"].to(device)
    labels = collected["regimes"].to(device)
    fit_features = features[:fit_trajectories]
    fit_states = states[:fit_trajectories]
    fit_labels = labels[:fit_trajectories]
    test_features = features[fit_trajectories:]
    test_states = states[fit_trajectories:]
    test_labels = labels[fit_trajectories:]

    flattened_features = fit_features.reshape(-1, fit_features.shape[-1])
    flattened_states = fit_states.reshape(-1, fit_states.shape[-1])
    feature_mean = flattened_features.mean(dim=0)
    feature_scale = flattened_features.std(dim=0).clamp_min(1e-5)
    state_mean = flattened_states.mean(dim=0)
    state_scale = flattened_states.std(dim=0).clamp_min(1e-5)
    structured = CausalBottleneckBranch(
        feature_mean,
        feature_scale,
        state_mean,
        state_scale,
        hidden_size=config.hidden_size,
        structured=True,
    ).to(device)
    _fit_linear_encoder(
        structured,
        flattened_features,
        flattened_states,
        config.ridge,
    )
    control = CausalBottleneckBranch(
        feature_mean,
        feature_scale,
        state_mean,
        state_scale,
        hidden_size=config.hidden_size,
        structured=False,
    ).to(device)
    control.encoder.load_state_dict(copy.deepcopy(structured.encoder.state_dict()))
    control.hybrid_port.load_state_dict(copy.deepcopy(structured.hybrid_port.state_dict()))
    if sum(p.numel() for p in structured.parameters()) != sum(
        p.numel() for p in control.parameters()
    ):
        raise AssertionError("structured and control branches must have equal capacity")

    branches = {"portHamiltonian": structured, "signFreeControl": control}
    optimizer = torch.optim.AdamW(
        [parameter for branch in branches.values() for parameter in branch.parameters()],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    event_weights = _event_class_weights(fit_labels).to(device)
    train_started = time.perf_counter()
    log_path = output_dir / "train.jsonl"
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.steps + 1):
            indices = torch.randint(
                0,
                fit_trajectories,
                (config.batch_size,),
                device=device,
            )
            batch_features = fit_features[indices]
            batch_states = fit_states[indices]
            batch_labels = fit_labels[indices]
            learning_rate = config.learning_rate * _learning_rate_multiplier(step, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            terms = {}
            for name, branch in branches.items():
                loss, branch_terms = _branch_loss(
                    branch,
                    batch_features,
                    batch_states,
                    batch_labels,
                    event_weights,
                    config,
                )
                losses[name] = loss
                terms[name] = branch_terms
            loss = sum(losses.values()) / len(losses)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for branch in branches.values() for parameter in branch.parameters()],
                5.0,
            )
            optimizer.step()
            if step == 1 or step % config.log_every == 0 or step == config.steps:
                elapsed = time.perf_counter() - train_started
                steps_per_second = step / max(elapsed, 1e-8)
                payload: dict[str, Any] = {
                    "stage": "train_bottleneck",
                    "step": step,
                    "steps": config.steps,
                    "loss": float(loss.detach()),
                    "gradientNorm": float(gradient_norm),
                    "learningRate": learning_rate,
                    "stepsPerSecond": steps_per_second,
                    "estimatedTrainingSeconds": config.steps / max(steps_per_second, 1e-8),
                }
                for name in branches:
                    payload[name] = {
                        "loss": float(losses[name].detach()),
                        **{
                            term_name: float(value.detach())
                            for term_name, value in terms[name].items()
                        },
                        "core": _coefficient_payload(branches[name]),
                    }
                print(json.dumps(payload), flush=True)
                log_file.write(json.dumps(payload) + "\n")
                log_file.flush()
    training_seconds = time.perf_counter() - train_started

    evaluation_started = time.perf_counter()
    evaluation = {
        name: evaluate_branch(branch, test_features, test_states, test_labels)
        for name, branch in branches.items()
    }
    evaluation_seconds = time.perf_counter() - evaluation_started
    checkpoint_payload = {
        "kind": "causal_port_hamiltonian_bottleneck",
        "version": 1,
        "baseCheckpoint": str(checkpoint_path),
        "baseCheckpointStep": int(checkpoint["step"]),
        "config": asdict(config),
        "featureMean": feature_mean.cpu(),
        "featureScale": feature_scale.cpu(),
        "stateMean": state_mean.cpu(),
        "stateScale": state_scale.cpu(),
        "branches": {name: branch.state_dict() for name, branch in branches.items()},
    }
    torch.save(checkpoint_payload, output_dir / "checkpoint.pt")
    summary = {
        "kind": checkpoint_payload["kind"],
        "version": checkpoint_payload["version"],
        "question": (
            "Does forcing block-5 entity features through a canonical hybrid pH state "
            "improve causal rollout, passivity, and control relative to an equal-capacity "
            "sign-free core?"
        ),
        "baseCheckpoint": str(checkpoint_path),
        "baseCheckpointStep": int(checkpoint["step"]),
        "frozenBackbone": True,
        "hardBottleneck": (
            "Every evaluated future state is rolled from the first block-5 feature through "
            "the 10D canonical-plus-mode state; future backbone features are not consumed."
        ),
        "config": asdict(config),
        "data": {
            "fitTrajectories": fit_trajectories,
            "testTrajectories": config.trajectories - fit_trajectories,
            "transitionsPerTrajectory": config.transitions_per_trajectory,
            "regimeCounts": {
                name: int((labels == index).sum())
                for index, name in enumerate(REGIMES)
            },
        },
        "capacity": {
            name: sum(parameter.numel() for parameter in branch.parameters())
            for name, branch in branches.items()
        },
        "timing": {
            "collectionSeconds": collection_seconds,
            "trainingSeconds": training_seconds,
            "evaluationSeconds": evaluation_seconds,
            "totalSeconds": collection_seconds + training_seconds + evaluation_seconds,
        },
        "evaluation": evaluation,
        "artifacts": ["checkpoint.pt", "train.jsonl", "summary.json"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trajectories", type=int, default=8_192)
    parser.add_argument("--transitions", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=91_410_731)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_bottleneck_experiment(
        args.checkpoint,
        args.output,
        config=BottleneckExperimentConfig(
            trajectories=args.trajectories,
            transitions_per_trajectory=args.transitions,
            steps=args.steps,
            batch_size=args.batch_size,
            feature_batch_size=args.feature_batch_size,
            seed=args.seed,
            log_every=args.log_every,
        ),
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
