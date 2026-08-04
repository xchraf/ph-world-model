from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import make_passive_clip
from .env import WorldConfig
from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .pixel_probe import PLAYER_CLASSES, _visual_centroid
from .position_write_probe import PUCK_CLASSES
from .train_pixel_direct import frames_to_classes, palette_tensor


READOUTS = ("entity_pair", "spatial_mean", "fixed_bottom_right")
REGIMES = ("free", "disc_impact", "wall", "goal_entry", "goal_pause", "kickoff")


@dataclass(frozen=True)
class AuditConfig:
    trajectories: int = 1024
    transitions_per_trajectory: int = 8
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    fit_fraction: float = 0.75
    goal_centered_fraction: float = 0.20
    batch_size: int = 32
    seed: int = 81_240_731
    ridge: float = 1e-2
    temporal_delta_weight: float = 1.0
    mlp_steps: int = 300
    mlp_hidden: int = 16
    mlp_learning_rate: float = 3e-3


def canonical_state(
    states: torch.Tensor,
    *,
    player_mass: float = 1.8,
    puck_mass: float = 1.0,
) -> torch.Tensor:
    """Return canonical [q_player, q_puck, p_player, p_puck] coordinates."""

    q = torch.cat((states[..., 0:2], states[..., 4:6]), dim=-1)
    p = torch.cat(
        (states[..., 2:4] * player_mass, states[..., 6:8] * puck_mass),
        dim=-1,
    )
    return torch.cat((q, p), dim=-1)


def transition_regimes(
    states_t: torch.Tensor,
    states_tp1: torch.Tensor,
    events_tp1: torch.Tensor,
) -> dict[str, torch.Tensor]:
    score_changed = states_tp1[:, 8] != states_t[:, 8]
    reset_active = (states_t[:, 9] > 0) | (states_tp1[:, 9] > 0)
    return {
        "free": (events_tp1 == 0) & ~reset_active & ~score_changed,
        "disc_impact": events_tp1 == 2,
        "wall": events_tp1 == 3,
        "goal_entry": (events_tp1 == 4) & score_changed,
        "goal_pause": (events_tp1 == 4) & ~score_changed,
        "kickoff": events_tp1 == 5,
    }


def _entity_token(model, position_px: torch.Tensor) -> torch.Tensor:
    patch_x = (position_px[:, 0] / model.config.patch_size).long().clamp(
        0, model.config.grid_size - 1
    )
    patch_y = (position_px[:, 1] / model.config.patch_size).long().clamp(
        0, model.config.grid_size - 1
    )
    return patch_y * model.config.grid_size + patch_x


@torch.no_grad()
def _activation_readouts(model, classes: torch.Tensor) -> dict[str, list[torch.Tensor]]:
    tokens = (
        model.patch_projection(model.patch_tokens(classes))
        + model.spatial_position
        + model.temporal_position[:, : classes.shape[1]]
    )
    states = [tokens]
    for block in model.blocks:
        tokens = block(tokens)
        states.append(tokens)

    player_position = _visual_centroid(classes[:, -1], PLAYER_CLASSES)
    puck_position = _visual_centroid(classes[:, -1], PUCK_CLASSES)
    player_token = _entity_token(model, player_position)
    puck_token = _entity_token(model, puck_position)
    batch = torch.arange(classes.shape[0], device=classes.device)

    result = {name: [] for name in READOUTS}
    for state in states:
        latest = state[:, -1]
        result["entity_pair"].append(
            torch.cat(
                (latest[batch, player_token], latest[batch, puck_token]),
                dim=-1,
            ).float().cpu()
        )
        result["spatial_mean"].append(latest.mean(dim=1).float().cpu())
        result["fixed_bottom_right"].append(latest[:, -1].float().cpu())
    return result


def _rgb_contexts_to_classes(
    contexts: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    videos = torch.from_numpy(np.stack(contexts)).permute(0, 1, 4, 2, 3).float()
    videos = videos.to(device).div(127.5).sub(1.0)
    return frames_to_classes(videos, palette_tensor(device))


@torch.no_grad()
def collect_frozen_transitions(
    model,
    config: AuditConfig,
    device: torch.device,
) -> dict[str, Any]:
    feature_chunks = {
        endpoint: {
            readout: [[] for _ in range(model.config.depth + 1)]
            for readout in READOUTS
        }
        for endpoint in ("t", "tp1")
    }
    state_t_chunks: list[torch.Tensor] = []
    state_tp1_chunks: list[torch.Tensor] = []
    event_chunks: list[torch.Tensor] = []
    trajectory_chunks: list[torch.Tensor] = []
    pending_t: list[np.ndarray] = []
    pending_tp1: list[np.ndarray] = []
    pending_state_t: list[np.ndarray] = []
    pending_state_tp1: list[np.ndarray] = []
    pending_events: list[int] = []
    pending_trajectories: list[int] = []

    goal_period = None
    if config.goal_centered_fraction > 0:
        goal_period = max(1, round(1.0 / config.goal_centered_fraction))

    def flush() -> None:
        if not pending_t:
            return
        classes_t = _rgb_contexts_to_classes(pending_t, device)
        classes_tp1 = _rgb_contexts_to_classes(pending_tp1, device)
        readouts_t = _activation_readouts(model, classes_t)
        readouts_tp1 = _activation_readouts(model, classes_tp1)
        for readout in READOUTS:
            for stage in range(model.config.depth + 1):
                feature_chunks["t"][readout][stage].append(readouts_t[readout][stage])
                feature_chunks["tp1"][readout][stage].append(readouts_tp1[readout][stage])
        state_t_chunks.append(torch.from_numpy(np.stack(pending_state_t)).float())
        state_tp1_chunks.append(torch.from_numpy(np.stack(pending_state_tp1)).float())
        event_chunks.append(torch.tensor(pending_events, dtype=torch.long))
        trajectory_chunks.append(torch.tensor(pending_trajectories, dtype=torch.long))
        pending_t.clear()
        pending_tp1.clear()
        pending_state_t.clear()
        pending_state_tp1.clear()
        pending_events.clear()
        pending_trajectories.clear()

    for trajectory in range(config.trajectories):
        clip_seed = config.seed + trajectory * 9_973
        clip = make_passive_clip(
            clip_seed,
            context_frames=1,
            future_frames=model.config.history_frames + config.transitions_per_trajectory - 1,
            image_size=model.config.image_size,
            goal_centered=goal_period is not None and trajectory % goal_period == 0,
        )
        frames = clip["frames"]
        states = clip["all_state"]
        events = clip["all_events"]
        history = model.config.history_frames
        for offset in range(config.transitions_per_trajectory):
            pending_t.append(frames[offset : offset + history])
            pending_tp1.append(frames[offset + 1 : offset + history + 1])
            pending_state_t.append(states[offset + history - 1])
            pending_state_tp1.append(states[offset + history])
            pending_events.append(int(events[offset + history]))
            pending_trajectories.append(trajectory)
            if len(pending_t) >= config.batch_size:
                flush()
        if (trajectory + 1) % 128 == 0 or trajectory + 1 == config.trajectories:
            print(
                json.dumps(
                    {
                        "stage": "collect",
                        "trajectories": trajectory + 1,
                        "total": config.trajectories,
                    }
                ),
                flush=True,
            )
    flush()

    return {
        "features": {
            endpoint: {
                readout: [torch.cat(chunks) for chunks in stages]
                for readout, stages in readouts.items()
            }
            for endpoint, readouts in feature_chunks.items()
        },
        "state_t": torch.cat(state_t_chunks),
        "state_tp1": torch.cat(state_tp1_chunks),
        "events_tp1": torch.cat(event_chunks),
        "trajectory": torch.cat(trajectory_chunks),
    }


def _fit_ridge(
    features: torch.Tensor,
    targets: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0, keepdim=True)
    scale = features.std(dim=0, keepdim=True).clamp_min(1e-5)
    normalized = (features - mean) / scale
    augmented = torch.cat(
        (normalized, torch.ones(normalized.shape[0], 1, dtype=normalized.dtype)),
        dim=1,
    )
    penalty = torch.eye(augmented.shape[1], dtype=augmented.dtype)
    penalty[-1, -1] = 0
    weight = torch.linalg.solve(
        augmented.T @ augmented + ridge * penalty,
        augmented.T @ targets,
    )
    return mean, scale, weight


def _ridge_predict(
    fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    features: torch.Tensor,
) -> torch.Tensor:
    mean, scale, weight = fit
    normalized = (features - mean) / scale
    augmented = torch.cat(
        (normalized, torch.ones(normalized.shape[0], 1, dtype=normalized.dtype)),
        dim=1,
    )
    return augmented @ weight


def _fit_temporally_aligned_ridge(
    features_t: torch.Tensor,
    features_tp1: torch.Tensor,
    targets_t: torch.Tensor,
    targets_tp1: torch.Tensor,
    ridge: float,
    delta_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a linear state readout while also resolving one-step changes.

    A state-only probe can have high R² while its independent endpoint errors
    are much larger than the physical one-frame displacement. This decoder is
    still linear, but it adds normalized feature-difference/target-difference
    rows. The bias column is zero on those rows because a bias cancels between
    consecutive times.
    """

    endpoint_features = torch.cat((features_t, features_tp1))
    endpoint_targets = torch.cat((targets_t, targets_tp1))
    mean = endpoint_features.mean(dim=0, keepdim=True)
    scale = endpoint_features.std(dim=0, keepdim=True).clamp_min(1e-5)
    normalized_endpoints = (endpoint_features - mean) / scale
    normalized_differences = (features_tp1 - features_t) / scale
    endpoint_design = torch.cat(
        (
            normalized_endpoints,
            torch.ones(normalized_endpoints.shape[0], 1),
        ),
        dim=1,
    )
    difference_design = torch.cat(
        (
            normalized_differences,
            torch.zeros(normalized_differences.shape[0], 1),
        ),
        dim=1,
    )
    target_differences = targets_tp1 - targets_t
    penalty = torch.eye(endpoint_design.shape[1])
    penalty[-1, -1] = 0
    weights = []
    for coordinate in range(endpoint_targets.shape[1]):
        state_scale = endpoint_targets[:, coordinate].std().clamp_min(1e-8)
        change_scale = target_differences[:, coordinate].std().clamp_min(1e-8)
        relative_weight = delta_weight * state_scale / change_scale
        design = torch.cat((endpoint_design, difference_design * relative_weight))
        target = torch.cat(
            (
                endpoint_targets[:, coordinate],
                target_differences[:, coordinate] * relative_weight,
            )
        )
        weights.append(
            torch.linalg.solve(
                design.T @ design + ridge * penalty,
                design.T @ target,
            )
        )
    return mean, scale, torch.stack(weights, dim=1)


def _state_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    residual = target - prediction
    total = (target - target.mean(dim=0, keepdim=True)).square().sum().clamp_min(1e-12)
    q_total = (target[:, :4] - target[:, :4].mean(dim=0, keepdim=True)).square().sum().clamp_min(1e-12)
    p_total = (target[:, 4:] - target[:, 4:].mean(dim=0, keepdim=True)).square().sum().clamp_min(1e-12)
    return {
        "r2": float(1.0 - residual.square().sum() / total),
        "qR2": float(1.0 - residual[:, :4].square().sum() / q_total),
        "pR2": float(1.0 - residual[:, 4:].square().sum() / p_total),
        "qRmse": float(residual[:, :4].square().mean().sqrt()),
        "pRmse": float(residual[:, 4:].square().mean().sqrt()),
    }


def _fit_ratio(x: torch.Tensor, y: torch.Tensor, *, lower: float, upper: float) -> float:
    denominator = x.square().sum().clamp_min(1e-12)
    return float((x.mul(y).sum() / denominator).clamp(lower, upper))


def fit_structured_free_map(
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
    *,
    dissipative: bool,
) -> dict[str, list[float]]:
    gains: list[float] = []
    decays: list[float] = []
    for entity in range(2):
        q_slice = slice(entity * 2, entity * 2 + 2)
        p_slice = slice(4 + entity * 2, 4 + entity * 2 + 2)
        momentum = z_t[:, p_slice]
        gains.append(
            _fit_ratio(
                momentum,
                z_tp1[:, q_slice] - z_t[:, q_slice],
                lower=1e-6,
                upper=2.0,
            )
        )
        decays.append(
            _fit_ratio(
                momentum,
                z_tp1[:, p_slice],
                lower=0.0 if dissipative else 1.0,
                upper=1.0,
            )
            if dissipative
            else 1.0
        )
    return {"positionGain": gains, "momentumDecay": decays}


def predict_structured_free_map(
    z_t: torch.Tensor,
    parameters: dict[str, list[float]],
) -> torch.Tensor:
    result = z_t.clone()
    for entity in range(2):
        q_slice = slice(entity * 2, entity * 2 + 2)
        p_slice = slice(4 + entity * 2, 4 + entity * 2 + 2)
        result[:, q_slice] = (
            z_t[:, q_slice] + parameters["positionGain"][entity] * z_t[:, p_slice]
        )
        result[:, p_slice] = parameters["momentumDecay"][entity] * z_t[:, p_slice]
    return result


def _fit_affine_map(
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    augmented = torch.cat((z_t, torch.ones(z_t.shape[0], 1)), dim=1)
    penalty = torch.eye(augmented.shape[1])
    penalty[-1, -1] = 0
    return torch.linalg.solve(
        augmented.T @ augmented + ridge * penalty,
        augmented.T @ z_tp1,
    )


def _predict_affine(z_t: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    augmented = torch.cat((z_t, torch.ones(z_t.shape[0], 1)), dim=1)
    return augmented @ weight


def _transition_metrics(
    prediction: torch.Tensor,
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
) -> dict[str, float]:
    error = prediction - z_tp1
    true_delta = z_tp1 - z_t
    predicted_delta = prediction - z_t

    def group(start: int, end: int) -> dict[str, float]:
        group_error = error[:, start:end]
        group_delta = true_delta[:, start:end]
        scale = group_delta.square().mean().sqrt().clamp_min(1e-12)
        return {
            "nextRmse": float(group_error.square().mean().sqrt()),
            "deltaNrmse": float(group_error.square().mean().sqrt() / scale),
            "deltaCosine": float(
                torch.nn.functional.cosine_similarity(
                    predicted_delta[:, start:end],
                    group_delta,
                    dim=1,
                    eps=1e-8,
                ).mean()
            ),
        }

    return {
        "nextRmse": float(error.square().mean().sqrt()),
        "deltaNrmse": float(
            error.square().mean().sqrt()
            / true_delta.square().mean().sqrt().clamp_min(1e-12)
        ),
        "q": group(0, 4),
        "p": group(4, 8),
    }


def _effective_physics(
    parameters: dict[str, list[float]],
    dt: float,
) -> dict[str, list[float]]:
    masses: list[float] = []
    drags: list[float] = []
    for gain, decay in zip(parameters["positionGain"], parameters["momentumDecay"]):
        if decay > 0 and abs(decay - 1.0) > 1e-7:
            drag = -math.log(decay) / dt
            mass = (1.0 - decay) / max(drag * gain, 1e-12)
        else:
            drag = 0.0
            mass = dt / max(gain, 1e-12)
        masses.append(mass)
        drags.append(drag)
    return {"effectiveMass": masses, "effectiveDrag": drags}


def _energy_metrics(
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
    parameters: dict[str, list[float]],
    dt: float,
) -> dict[str, float]:
    physics = _effective_physics(parameters, dt)
    energy_t = torch.zeros(z_t.shape[0])
    energy_tp1 = torch.zeros(z_t.shape[0])
    expected_dissipation = torch.zeros(z_t.shape[0])
    for entity, mass in enumerate(physics["effectiveMass"]):
        p_slice = slice(4 + entity * 2, 4 + entity * 2 + 2)
        current = z_t[:, p_slice].square().sum(dim=1) / (2.0 * max(mass, 1e-8))
        following = z_tp1[:, p_slice].square().sum(dim=1) / (2.0 * max(mass, 1e-8))
        energy_t += current
        energy_tp1 += following
        decay = parameters["momentumDecay"][entity]
        expected_dissipation += current * (1.0 - decay**2)
    residual = energy_tp1 - energy_t + expected_dissipation
    normalizer = energy_t.mean().clamp_min(1e-8)
    tolerance = 1e-5 * normalizer
    return {
        "meanEnergy": float(energy_t.mean()),
        "normalizedBalanceRmse": float(residual.square().mean().sqrt() / normalizer),
        "passivityViolationRate": float((energy_tp1 > energy_t + tolerance).float().mean()),
    }


def _conformal_symplectic_defect(
    affine_weight: torch.Tensor,
    parameters: dict[str, list[float]],
) -> float:
    linear = affine_weight[:8].T
    identity = torch.eye(4)
    zero = torch.zeros(4, 4)
    omega = torch.cat((torch.cat((zero, identity), dim=1), torch.cat((-identity, zero), dim=1)), dim=0)
    decay = torch.diag(
        torch.tensor(
            [
                parameters["momentumDecay"][0],
                parameters["momentumDecay"][0],
                parameters["momentumDecay"][1],
                parameters["momentumDecay"][1],
            ]
        )
    )
    target = torch.cat((torch.cat((zero, decay), dim=1), torch.cat((-decay, zero), dim=1)), dim=0)
    defect = linear.T @ omega @ linear - target
    return float(torch.linalg.matrix_norm(defect) / torch.linalg.matrix_norm(target).clamp_min(1e-12))


class _DeltaMLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(8, hidden), nn.Tanh(), nn.Linear(hidden, 8))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _fit_mlp_predictions(
    train_t: torch.Tensor,
    train_tp1: torch.Tensor,
    test_t: torch.Tensor,
    *,
    hidden: int,
    steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    torch.manual_seed(seed)
    model = _DeltaMLP(hidden).to(device)
    input_mean = train_t.mean(dim=0, keepdim=True).to(device)
    input_scale = train_t.std(dim=0, keepdim=True).clamp_min(1e-5).to(device)
    train_delta = train_tp1 - train_t
    delta_mean = train_delta.mean(dim=0, keepdim=True).to(device)
    delta_scale = train_delta.std(dim=0, keepdim=True).clamp_min(1e-5).to(device)
    normalized_input = (train_t.to(device) - input_mean) / input_scale
    normalized_target = (train_delta.to(device) - delta_mean) / delta_scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(steps):
        prediction = model(normalized_input)
        loss = torch.nn.functional.mse_loss(prediction, normalized_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        normalized_test = (test_t.to(device) - input_mean) / input_scale
        delta = model(normalized_test) * delta_scale + delta_mean
        prediction = test_t.to(device) + delta
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return prediction.cpu(), parameters


def _regime_counts(regimes: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, int]:
    return {name: int((values & mask).sum()) for name, values in regimes.items()}


def _evaluate_models_by_regime(
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
    regimes: dict[str, torch.Tensor],
    test_mask: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for regime, regime_mask in regimes.items():
        mask = test_mask & regime_mask
        if not mask.any():
            result[regime] = {"samples": 0, "models": {}}
            continue
        result[regime] = {
            "samples": int(mask.sum()),
            "models": {
                name: _transition_metrics(prediction[mask], z_t[mask], z_tp1[mask])
                for name, prediction in predictions.items()
            },
        }
    return result


def _horizon_endpoints(
    values_t: torch.Tensor,
    values_tp1: torch.Tensor,
    *,
    trajectories: int,
    transitions_per_trajectory: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 1 <= horizon <= transitions_per_trajectory:
        raise ValueError("horizon must fit inside transitions_per_trajectory")
    tail = values_t.shape[1:]
    shaped_t = values_t.reshape(trajectories, transitions_per_trajectory, *tail)
    shaped_tp1 = values_tp1.reshape(trajectories, transitions_per_trajectory, *tail)
    count = transitions_per_trajectory - horizon + 1
    return (
        shaped_t[:, :count].reshape(-1, *tail),
        shaped_tp1[:, horizon - 1 :].reshape(-1, *tail),
    )


def _horizon_masks(
    one_step_free: torch.Tensor,
    *,
    trajectories: int,
    transitions_per_trajectory: int,
    fit_trajectories: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    free = one_step_free.reshape(trajectories, transitions_per_trajectory)
    count = transitions_per_trajectory - horizon + 1
    interval_free = torch.stack(
        [free[:, offset : offset + horizon].all(dim=1) for offset in range(count)],
        dim=1,
    )
    fit_by_trajectory = torch.arange(trajectories) < fit_trajectories
    fit = fit_by_trajectory[:, None].expand(trajectories, count)
    return (interval_free & fit).reshape(-1), (interval_free & ~fit).reshape(-1)


def _free_horizon_analysis(
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
    fit_free: torch.Tensor,
    test_free: torch.Tensor,
    *,
    dt: float,
    ridge: float,
    device: torch.device,
    control_seed: int,
    mlp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if int(fit_free.sum()) < 32 or int(test_free.sum()) < 16:
        return {"fitSamples": int(fit_free.sum()), "testSamples": int(test_free.sum())}
    ph = fit_structured_free_map(z_t[fit_free], z_tp1[fit_free], dissipative=True)
    hamiltonian = fit_structured_free_map(
        z_t[fit_free], z_tp1[fit_free], dissipative=False
    )
    affine = _fit_affine_map(z_t[fit_free], z_tp1[fit_free], ridge)
    predictions = {
        "persistence": z_t,
        "hamiltonian": predict_structured_free_map(z_t, hamiltonian),
        "portHamiltonian": predict_structured_free_map(z_t, ph),
        "affine": _predict_affine(z_t, affine),
    }
    if mlp is not None:
        mlp_prediction, mlp_parameters = _fit_mlp_predictions(
            z_t[fit_free],
            z_tp1[fit_free],
            z_t,
            hidden=int(mlp["hidden"]),
            steps=int(mlp["steps"]),
            learning_rate=float(mlp["learningRate"]),
            seed=int(mlp["seed"]),
            device=device,
        )
        predictions["mlp"] = mlp_prediction
    else:
        mlp_parameters = None
    test_indices = torch.nonzero(test_free, as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(control_seed)
    shuffled_indices = test_indices[
        torch.randperm(len(test_indices), generator=generator)
    ]
    pairing_control = {
        name: _transition_metrics(
            prediction[test_indices],
            z_t[test_indices],
            z_tp1[shuffled_indices],
        )
        for name, prediction in predictions.items()
    }
    reverse_ph = fit_structured_free_map(
        z_tp1[fit_free], z_t[fit_free], dissipative=True
    )
    reverse_affine = _fit_affine_map(z_tp1[fit_free], z_t[fit_free], ridge)
    reverse_predictions = {
        "persistence": z_tp1,
        "portHamiltonian": predict_structured_free_map(z_tp1, reverse_ph),
        "affine": _predict_affine(z_tp1, reverse_affine),
    }
    return {
        "fitSamples": int(fit_free.sum()),
        "testSamples": int(test_free.sum()),
        "portHamiltonianParameters": {**ph, **_effective_physics(ph, dt)},
        "models": {
            name: _transition_metrics(
                prediction[test_free], z_t[test_free], z_tp1[test_free]
            )
            for name, prediction in predictions.items()
        },
        "energyBalance": _energy_metrics(
            z_t[test_free], z_tp1[test_free], ph, dt
        ),
        "affineConformalSymplecticDefect": _conformal_symplectic_defect(affine, ph),
        "parameterCounts": {
            "persistence": 0,
            "hamiltonian": 2,
            "portHamiltonian": 4,
            "affine": int(affine.numel()),
            **({"mlp": mlp_parameters} if mlp_parameters is not None else {}),
        },
        "pairingControl": pairing_control,
        "reverseTimeControl": {
            "portHamiltonianParameters": {
                **reverse_ph,
                **_effective_physics(reverse_ph, dt),
            },
            "models": {
                name: _transition_metrics(
                    prediction[test_free],
                    z_tp1[test_free],
                    z_t[test_free],
                )
                for name, prediction in reverse_predictions.items()
            },
        },
    }


def run_port_hamiltonian_audit(
    checkpoint_path: Path,
    output_path: Path,
    *,
    config: AuditConfig = AuditConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    if not 0 < config.fit_fraction < 1:
        raise ValueError("fit_fraction must be in (0, 1)")
    if not config.horizons or any(
        horizon < 1 or horizon > config.transitions_per_trajectory
        for horizon in config.horizons
    ):
        raise ValueError("every audit horizon must fit inside each trajectory segment")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_pixel_direct_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    collected = collect_frozen_transitions(model, config, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    states_t = collected["state_t"]
    states_tp1 = collected["state_tp1"]
    z_t = canonical_state(states_t)
    z_tp1 = canonical_state(states_tp1)
    regimes = transition_regimes(states_t, states_tp1, collected["events_tp1"])
    fit_trajectories = int(config.trajectories * config.fit_fraction)
    fit_mask = collected["trajectory"] < fit_trajectories
    test_mask = ~fit_mask
    fit_free = fit_mask & regimes["free"]
    test_free = test_mask & regimes["free"]
    if int(fit_free.sum()) < 32 or int(test_free.sum()) < 16:
        raise RuntimeError("not enough collision-free transitions for the audit")

    world_ph = fit_structured_free_map(z_t[fit_free], z_tp1[fit_free], dissipative=True)
    world_hamiltonian = fit_structured_free_map(
        z_t[fit_free], z_tp1[fit_free], dissipative=False
    )
    world_affine = _fit_affine_map(z_t[fit_free], z_tp1[fit_free], config.ridge)
    world_mlp_prediction, world_mlp_parameters = _fit_mlp_predictions(
        z_t[fit_free],
        z_tp1[fit_free],
        z_t,
        hidden=config.mlp_hidden,
        steps=config.mlp_steps,
        learning_rate=config.mlp_learning_rate,
        seed=config.seed + 17,
        device=device,
    )
    world_predictions = {
        "persistence": z_t,
        "hamiltonian": predict_structured_free_map(z_t, world_hamiltonian),
        "portHamiltonian": predict_structured_free_map(z_t, world_ph),
        "affine": _predict_affine(z_t, world_affine),
        "mlp": world_mlp_prediction,
    }

    world_reference = {
        "portHamiltonianParameters": {
            **world_ph,
            **_effective_physics(world_ph, WorldConfig().dt),
        },
        "parameterCounts": {
            "persistence": 0,
            "hamiltonian": 2,
            "portHamiltonian": 4,
            "affine": int(world_affine.numel()),
            "mlp": world_mlp_parameters,
        },
        "regimes": _evaluate_models_by_regime(
            z_t, z_tp1, regimes, test_mask, world_predictions
        ),
        "freeEnergyBalance": _energy_metrics(
            z_t[test_free], z_tp1[test_free], world_ph, WorldConfig().dt
        ),
        "freeAffineConformalSymplecticDefect": _conformal_symplectic_defect(
            world_affine, world_ph
        ),
    }
    world_reference["freeDynamicsByHorizon"] = {}
    for horizon in config.horizons:
        horizon_t, horizon_tp1 = _horizon_endpoints(
            z_t,
            z_tp1,
            trajectories=config.trajectories,
            transitions_per_trajectory=config.transitions_per_trajectory,
            horizon=horizon,
        )
        horizon_fit, horizon_test = _horizon_masks(
            regimes["free"],
            trajectories=config.trajectories,
            transitions_per_trajectory=config.transitions_per_trajectory,
            fit_trajectories=fit_trajectories,
            horizon=horizon,
        )
        world_reference["freeDynamicsByHorizon"][str(horizon)] = (
            _free_horizon_analysis(
                horizon_t,
                horizon_tp1,
                horizon_fit,
                horizon_test,
                dt=WorldConfig().dt * horizon,
                ridge=config.ridge,
                device=device,
                control_seed=config.seed + 40_000 + horizon,
                mlp=(
                    {
                        "hidden": config.mlp_hidden,
                        "steps": config.mlp_steps,
                        "learningRate": config.mlp_learning_rate,
                        "seed": config.seed + 17 + horizon,
                    }
                    if horizon > 1
                    else None
                ),
            )
        )

    features = collected["features"]
    readout_results: dict[str, Any] = {}
    for readout in READOUTS:
        stages: list[dict[str, Any]] = []
        for stage_index in range(len(features["t"][readout])):
            features_t = features["t"][readout][stage_index].float()
            features_tp1 = features["tp1"][readout][stage_index].float()
            decoder_fits = {
                "stateOnly": _fit_ridge(
                    torch.cat((features_t[fit_mask], features_tp1[fit_mask])),
                    torch.cat((z_t[fit_mask], z_tp1[fit_mask])),
                    config.ridge,
                ),
                "statePlusDelta": _fit_temporally_aligned_ridge(
                    features_t[fit_mask],
                    features_tp1[fit_mask],
                    z_t[fit_mask],
                    z_tp1[fit_mask],
                    config.ridge,
                    config.temporal_delta_weight,
                ),
            }
            decoder_results: dict[str, Any] = {}
            for decoder_index, (decoder_name, decoder_fit) in enumerate(
                decoder_fits.items()
            ):
                decoded_t = _ridge_predict(decoder_fit, features_t)
                decoded_tp1 = _ridge_predict(decoder_fit, features_tp1)
                decoded_ph = fit_structured_free_map(
                    decoded_t[fit_free], decoded_tp1[fit_free], dissipative=True
                )
                decoded_hamiltonian = fit_structured_free_map(
                    decoded_t[fit_free], decoded_tp1[fit_free], dissipative=False
                )
                decoded_affine = _fit_affine_map(
                    decoded_t[fit_free], decoded_tp1[fit_free], config.ridge
                )
                decoded_predictions = {
                    "persistence": decoded_t,
                    "hamiltonian": predict_structured_free_map(
                        decoded_t, decoded_hamiltonian
                    ),
                    "portHamiltonian": predict_structured_free_map(
                        decoded_t, decoded_ph
                    ),
                    "affine": _predict_affine(decoded_t, decoded_affine),
                }
                if readout == "entity_pair":
                    mlp_prediction, _ = _fit_mlp_predictions(
                        decoded_t[fit_free],
                        decoded_tp1[fit_free],
                        decoded_t,
                        hidden=config.mlp_hidden,
                        steps=config.mlp_steps,
                        learning_rate=config.mlp_learning_rate,
                        seed=config.seed + 100 + stage_index * 2 + decoder_index,
                        device=device,
                    )
                    decoded_predictions["mlp"] = mlp_prediction
                free_by_horizon: dict[str, Any] = {}
                for horizon in config.horizons:
                    horizon_t, horizon_tp1 = _horizon_endpoints(
                        decoded_t,
                        decoded_tp1,
                        trajectories=config.trajectories,
                        transitions_per_trajectory=config.transitions_per_trajectory,
                        horizon=horizon,
                    )
                    horizon_fit, horizon_test = _horizon_masks(
                        regimes["free"],
                        trajectories=config.trajectories,
                        transitions_per_trajectory=config.transitions_per_trajectory,
                        fit_trajectories=fit_trajectories,
                        horizon=horizon,
                    )
                    free_by_horizon[str(horizon)] = _free_horizon_analysis(
                        horizon_t,
                        horizon_tp1,
                        horizon_fit,
                        horizon_test,
                        dt=WorldConfig().dt * horizon,
                        ridge=config.ridge,
                        device=device,
                        control_seed=(
                            config.seed
                            + 50_000
                            + stage_index * 100
                            + decoder_index * 10
                            + horizon
                        ),
                    )
                decoder_results[decoder_name] = {
                    "stateReadout": _state_metrics(
                        torch.cat((decoded_t[test_mask], decoded_tp1[test_mask])),
                        torch.cat((z_t[test_mask], z_tp1[test_mask])),
                    ),
                    "decodedPortHamiltonianParameters": {
                        **decoded_ph,
                        **_effective_physics(decoded_ph, WorldConfig().dt),
                    },
                    "decodedEndpointDynamics": _evaluate_models_by_regime(
                        decoded_t,
                        decoded_tp1,
                        regimes,
                        test_mask,
                        decoded_predictions,
                    ),
                    "worldStateDynamics": _evaluate_models_by_regime(
                        decoded_t,
                        z_tp1,
                        regimes,
                        test_mask,
                        decoded_predictions,
                    ),
                    "freeEnergyBalance": _energy_metrics(
                        decoded_t[test_free],
                        decoded_tp1[test_free],
                        decoded_ph,
                        WorldConfig().dt,
                    ),
                    "freeAffineConformalSymplecticDefect": (
                        _conformal_symplectic_defect(decoded_affine, decoded_ph)
                    ),
                    "freeDynamicsByHorizon": free_by_horizon,
                }
            stages.append(
                {
                    "stage": (
                        "embedding" if stage_index == 0 else f"block_{stage_index}"
                    ),
                    "decoders": decoder_results,
                }
            )
            state_only = decoder_results["stateOnly"]["stateReadout"]
            aligned = decoder_results["statePlusDelta"]["stateReadout"]
            print(
                json.dumps(
                    {
                        "stage": "fit",
                        "readout": readout,
                        "layer": stages[-1]["stage"],
                        "stateOnlyQR2": state_only["qR2"],
                        "stateOnlyPR2": state_only["pR2"],
                        "alignedQR2": aligned["qR2"],
                        "alignedPR2": aligned["pR2"],
                    }
                ),
                flush=True,
            )
        readout_results[readout] = stages

    result = {
        "version": 1,
        "question": (
            "At which transformer stage does a canonical two-disc state become linearly "
            "readable and evolve like a low-capacity port-Hamiltonian free-flow map?"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpointStep": int(checkpoint["step"]),
        "frozenCheckpoint": True,
        "physicalTimeStep": WorldConfig().dt,
        "canonicalStateOrder": [
            "player_qx",
            "player_qy",
            "puck_qx",
            "puck_qy",
            "player_px",
            "player_py",
            "puck_px",
            "puck_py",
        ],
        "massesUsedForCanonicalMomenta": {
            "player": WorldConfig().player_mass,
            "puck": WorldConfig().puck_mass,
        },
        "config": config.__dict__,
        "split": {
            "unit": "complete simulator trajectory",
            "fitTrajectories": fit_trajectories,
            "testTrajectories": config.trajectories - fit_trajectories,
            "fitTransitions": int(fit_mask.sum()),
            "testTransitions": int(test_mask.sum()),
            "fitRegimes": _regime_counts(regimes, fit_mask),
            "testRegimes": _regime_counts(regimes, test_mask),
        },
        "method": {
            "stateDecoders": {
                "stateOnly": "ridge-linear map trained jointly on t and t+1 endpoints",
                "statePlusDelta": (
                    "same linear map with additional normalized one-step feature and state "
                    "difference constraints"
                ),
            },
            "tokenSelection": "entity patches selected only from rendered categorical pixels",
            "freeFlowFit": "event-free transitions with no active reset timer",
            "hamiltonianMap": "two tied position gains, unit momentum decay",
            "portHamiltonianMap": "two tied position gains and two non-negative momentum decays",
            "affineControl": "unconstrained 8D affine next-state map",
            "mlpControl": "two-layer tanh delta predictor; entity-pair readout only",
            "importantDistinction": (
                "transformer depth is computation depth; every dynamics fit compares the same "
                "stage across consecutive physical times"
            ),
        },
        "worldStateReference": world_reference,
        "readouts": readout_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen port-Hamiltonian layer audit")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trajectories", type=int, default=1024)
    parser.add_argument("--transitions-per-trajectory", type=int, default=8)
    parser.add_argument("--horizons", default="1,2,4,8")
    parser.add_argument("--fit-fraction", type=float, default=0.75)
    parser.add_argument("--goal-centered-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=81_240_731)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--temporal-delta-weight", type=float, default=1.0)
    parser.add_argument("--mlp-steps", type=int, default=300)
    parser.add_argument("--mlp-hidden", type=int, default=16)
    parser.add_argument("--mlp-learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    horizons = tuple(int(value) for value in args.horizons.split(",") if value)
    config = AuditConfig(
        trajectories=args.trajectories,
        transitions_per_trajectory=args.transitions_per_trajectory,
        horizons=horizons,
        fit_fraction=args.fit_fraction,
        goal_centered_fraction=args.goal_centered_fraction,
        batch_size=args.batch_size,
        seed=args.seed,
        ridge=args.ridge,
        temporal_delta_weight=args.temporal_delta_weight,
        mlp_steps=args.mlp_steps,
        mlp_hidden=args.mlp_hidden,
        mlp_learning_rate=args.mlp_learning_rate,
    )
    run_port_hamiltonian_audit(
        args.checkpoint,
        args.output,
        config=config,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
