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

from .data import make_clip
from .env import ACTION_VECTORS, WorldConfig
from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .port_hamiltonian_audit import (
    REGIMES,
    _effective_physics,
    _state_metrics,
    _transition_metrics,
)
from .port_hamiltonian_bottleneck import (
    HybridJumpPort,
    _block5_entity_features,
    _event_class_weights,
    _fit_linear_encoder,
    _inverse_softplus,
    _logit,
    _r2,
    bottleneck_state,
    regime_labels,
)
from .train_pixel_direct import frames_to_classes, palette_tensor


@dataclass(frozen=True)
class ActionPortPixelConfig:
    trajectories: int = 8_192
    transitions_per_trajectory: int = 8
    fit_fraction: float = 0.75
    feature_batch_size: int = 64
    block_count: int = 5
    dynamics_steps: int = 6_000
    dynamics_batch_size: int = 256
    renderer_steps: int = 5_000
    renderer_batch_size: int = 64
    learning_rate: float = 3e-4
    renderer_learning_rate: float = 5e-4
    renderer_entity_dice_weight: float = 1.0
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    min_learning_rate_ratio: float = 0.1
    hidden_size: int = 64
    renderer_hidden_size: int = 64
    state_loss_weight: float = 1.0
    teacher_dynamics_weight: float = 1.0
    rollout_loss_weight: float = 2.0
    event_loss_weight: float = 0.20
    free_port_weight: float = 0.20
    ridge: float = 1e-2
    log_every: int = 100
    seed: int = 101_510_731


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def action_vectors(actions: torch.Tensor) -> torch.Tensor:
    table = torch.as_tensor(ACTION_VECTORS, device=actions.device, dtype=torch.float32)
    vectors = table[actions.long()]
    norm = vectors.norm(dim=-1, keepdim=True)
    return vectors / norm.clamp_min(1.0)


def _dynamics_learning_rate_multiplier(
    step: int,
    config: ActionPortPixelConfig,
) -> float:
    if step <= config.warmup_steps:
        return step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(
        config.dynamics_steps - config.warmup_steps,
        1,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate_ratio + (
        1.0 - config.min_learning_rate_ratio
    ) * cosine


@torch.no_grad()
def collect_action_sequences(
    model: nn.Module,
    config: ActionPortPixelConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    feature_chunks: list[torch.Tensor] = []
    frame_chunks: list[torch.Tensor] = []
    state_sequences: list[torch.Tensor] = []
    event_sequences: list[torch.Tensor] = []
    action_sequences: list[torch.Tensor] = []
    pending_contexts: list[np.ndarray] = []
    palette = palette_tensor(device)

    def flush() -> None:
        if not pending_contexts:
            return
        videos = torch.from_numpy(np.stack(pending_contexts)).permute(0, 1, 4, 2, 3)
        videos = videos.to(device, non_blocking=True).float().div(127.5).sub(1.0)
        classes = frames_to_classes(videos, palette)
        feature_chunks.append(
            _block5_entity_features(model, classes, config.block_count).cpu()
        )
        frame_chunks.append(classes[:, -1].byte().cpu())
        pending_contexts.clear()

    started = time.perf_counter()
    history = model.config.history_frames
    endpoints = config.transitions_per_trajectory + 1
    for trajectory in range(config.trajectories):
        clip = make_clip(
            config.seed + trajectory * 9_973,
            context_frames=1,
            future_frames=history + config.transitions_per_trajectory - 1,
            image_size=model.config.image_size,
        )
        frames = clip["frames"]
        all_states = torch.from_numpy(clip["all_state"]).float()
        all_events = torch.from_numpy(clip["all_events"]).long()
        all_actions = torch.from_numpy(clip["all_actions"]).long()
        state_sequences.append(
            torch.stack(
                [all_states[offset + history - 1] for offset in range(endpoints)]
            )
        )
        event_sequences.append(
            torch.stack(
                [
                    all_events[offset + history]
                    for offset in range(config.transitions_per_trajectory)
                ]
            )
        )
        action_sequences.append(
            torch.stack(
                [
                    all_actions[offset + history]
                    for offset in range(config.transitions_per_trajectory)
                ]
            )
        )
        for offset in range(endpoints):
            pending_contexts.append(frames[offset : offset + history])
            if len(pending_contexts) >= config.feature_batch_size:
                flush()
        if (trajectory + 1) % 256 == 0 or trajectory + 1 == config.trajectories:
            print(
                json.dumps(
                    {
                        "stage": "collect_action_port",
                        "trajectories": trajectory + 1,
                        "total": config.trajectories,
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    flush()
    states = torch.stack(state_sequences)
    events = torch.stack(event_sequences)
    actions = torch.stack(action_sequences)
    return {
        "features": torch.cat(feature_chunks).reshape(
            config.trajectories, endpoints, -1
        ),
        "frames": torch.cat(frame_chunks).reshape(
            config.trajectories,
            endpoints,
            model.config.image_size,
            model.config.image_size,
        ),
        "worldStates": states,
        "states": bottleneck_state(states),
        "events": events,
        "actions": actions,
        "actionVectors": action_vectors(actions),
        "regimes": regime_labels(states[:, :-1], states[:, 1:], events),
    }


def _active_initial_coefficients() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world = WorldConfig()
    drags = (world.player_drag, world.puck_drag)
    masses = (world.player_mass, world.puck_mass)
    substep = world.dt / world.substeps
    substep_decays = [math.exp(-drag * substep) for drag in drags]
    decays = torch.tensor(
        [value**world.substeps for value in substep_decays],
        dtype=torch.float32,
    )
    position_gains = []
    for mass, substep_decay in zip(masses, substep_decays):
        decay_sum = sum(
            substep_decay**index for index in range(1, world.substeps + 1)
        )
        position_gains.append(substep * decay_sum / mass)
    gains = torch.tensor(position_gains, dtype=torch.float32)

    player_substep_decay = substep_decays[0]
    momentum_action_gain = (
        world.player_mass
        * world.player_acceleration
        * substep
        * sum(
            player_substep_decay**index
            for index in range(1, world.substeps + 1)
        )
    )
    position_action_gain = (
        world.player_acceleration
        * substep**2
        * sum(
            (world.substeps - index + 1) * player_substep_decay**index
            for index in range(1, world.substeps + 1)
        )
    )
    action_gains = torch.tensor(
        [position_action_gain, momentum_action_gain],
        dtype=torch.float32,
    )
    return gains, decays, action_gains


class ActionPortHamiltonianCore(nn.Module):
    """Sampled passive free flow with an explicit player-force port."""

    def __init__(self) -> None:
        super().__init__()
        gains, decays, action_gains = _active_initial_coefficients()
        self.raw_gain = nn.Parameter(
            torch.tensor([_inverse_softplus(float(value)) for value in gains])
        )
        self.raw_decay = nn.Parameter(
            torch.tensor([_logit(float(value)) for value in decays])
        )
        self.raw_action_gain = nn.Parameter(
            torch.tensor(
                [_inverse_softplus(float(value)) for value in action_gains]
            )
        )

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            F.softplus(self.raw_gain),
            torch.sigmoid(self.raw_decay),
            F.softplus(self.raw_action_gain),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        gains, decays, action_gains = self.coefficients()
        output = state.clone()
        for entity in range(2):
            q_slice = slice(entity * 2, entity * 2 + 2)
            p_slice = slice(4 + entity * 2, 4 + entity * 2 + 2)
            output[..., q_slice] = (
                state[..., q_slice] + gains[entity] * state[..., p_slice]
            )
            output[..., p_slice] = decays[entity] * state[..., p_slice]
        output[..., 0:2] = output[..., 0:2] + action_gains[0] * action
        output[..., 4:6] = output[..., 4:6] + action_gains[1] * action
        return output


class TangentMatchedActionCore(nn.Module):
    """Unbounded control with the pH map's exact initial value and Jacobian."""

    def __init__(self) -> None:
        super().__init__()
        gains, decays, action_gains = _active_initial_coefficients()
        gain_raw = torch.tensor([_inverse_softplus(float(value)) for value in gains])
        decay_raw = torch.tensor([_logit(float(value)) for value in decays])
        action_raw = torch.tensor(
            [_inverse_softplus(float(value)) for value in action_gains]
        )
        self.raw_gain = nn.Parameter(gain_raw.clone())
        self.raw_decay = nn.Parameter(decay_raw.clone())
        self.raw_action_gain = nn.Parameter(action_raw.clone())
        self.register_buffer("initial_gain_raw", gain_raw)
        self.register_buffer("initial_decay_raw", decay_raw)
        self.register_buffer("initial_action_raw", action_raw)
        self.register_buffer("initial_gain", gains)
        self.register_buffer("initial_decay", decays)
        self.register_buffer("initial_action_gain", action_gains)
        self.register_buffer("gain_slope", torch.sigmoid(gain_raw))
        self.register_buffer("decay_slope", decays * (1.0 - decays))
        self.register_buffer("action_slope", torch.sigmoid(action_raw))

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.initial_gain + self.gain_slope * (self.raw_gain - self.initial_gain_raw),
            self.initial_decay
            + self.decay_slope * (self.raw_decay - self.initial_decay_raw),
            self.initial_action_gain
            + self.action_slope * (self.raw_action_gain - self.initial_action_raw),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        gains, decays, action_gains = self.coefficients()
        output = state.clone()
        for entity in range(2):
            q_slice = slice(entity * 2, entity * 2 + 2)
            p_slice = slice(4 + entity * 2, 4 + entity * 2 + 2)
            output[..., q_slice] = (
                state[..., q_slice] + gains[entity] * state[..., p_slice]
            )
            output[..., p_slice] = decays[entity] * state[..., p_slice]
        output[..., 0:2] = output[..., 0:2] + action_gains[0] * action
        output[..., 4:6] = output[..., 4:6] + action_gains[1] * action
        return output


class ActionCausalBranch(nn.Module):
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
        self.core: ActionPortHamiltonianCore | TangentMatchedActionCore
        self.core = (
            ActionPortHamiltonianCore()
            if structured
            else TangentMatchedActionCore()
        )
        self.hybrid_port = HybridJumpPort(state_mean.numel(), hidden_size)
        self.structured = structured

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.state_mean + self.state_scale * self.encoder(normalized)

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        smooth = state.clone()
        smooth[..., :8] = self.core(state[..., :8], action)
        logits, soft_jump, soft_gate = self.hybrid_port(
            (state - self.state_mean) / self.state_scale,
            self.state_scale,
        )
        if self.training:
            jump = soft_jump
            gate = soft_gate
        else:
            hard_gate = (logits.argmax(dim=-1, keepdim=True) != 0).to(state.dtype)
            raw_jump = soft_jump / soft_gate.clamp_min(1e-6)
            jump = hard_gate * raw_jump
            gate = hard_gate
        return smooth + jump, logits, jump, gate


class ImplicitStateRenderer(nn.Module):
    """Render pixels only from state and pixel coordinates; no visual bypass."""

    def __init__(
        self,
        image_size: int,
        palette_size: int,
        state_mean: torch.Tensor,
        state_scale: torch.Tensor,
        hidden_size: int,
    ) -> None:
        super().__init__()
        axis = (torch.arange(image_size, dtype=torch.float32) + 0.5) / image_size
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack((xx, yy), dim=-1).reshape(-1, 2))
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_scale", state_scale)
        self.image_size = image_size
        self.palette_size = palette_size
        self.network = nn.Sequential(
            nn.Linear(18, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, palette_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        coordinates = self.coordinates[None].expand(batch, -1, -1)
        player_relative = coordinates - state[:, None, 0:2]
        puck_relative = coordinates - state[:, None, 2:4]
        distances = torch.stack(
            (
                player_relative.square().sum(dim=-1),
                puck_relative.square().sum(dim=-1),
            ),
            dim=-1,
        )
        normalized_state = ((state - self.state_mean) / self.state_scale)[:, None]
        normalized_state = normalized_state.expand(-1, coordinates.shape[1], -1)
        features = torch.cat(
            (
                coordinates,
                player_relative,
                puck_relative,
                distances,
                normalized_state,
            ),
            dim=-1,
        )
        logits = self.network(features)
        return logits.transpose(1, 2).reshape(
            batch,
            self.palette_size,
            self.image_size,
            self.image_size,
        )


def _entity_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    losses = []
    for classes in ((5, 6), (7, 8)):
        predicted_mask = probabilities[:, classes].sum(dim=1)
        target_mask = torch.zeros_like(predicted_mask)
        for class_index in classes:
            target_mask = target_mask + (targets == class_index).to(logits.dtype)
        intersection = (predicted_mask * target_mask).sum(dim=(-2, -1))
        denominator = (
            predicted_mask.sum(dim=(-2, -1))
            + target_mask.sum(dim=(-2, -1))
        )
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        losses.append(1.0 - dice.mean())
    return torch.stack(losses).mean()


def _branch_loss(
    branch: ActionCausalBranch,
    features: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    labels: torch.Tensor,
    event_weights: torch.Tensor,
    config: ActionPortPixelConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    encoded = branch.encode(features)
    state_loss = ((encoded - targets) / branch.state_scale).square().mean()
    flat_encoded = encoded[:, :-1].reshape(-1, encoded.shape[-1])
    flat_actions = actions.reshape(-1, 2)
    teacher_next, teacher_logits, teacher_jump, _ = branch.step(
        flat_encoded,
        flat_actions,
    )
    flat_targets = targets[:, 1:].reshape_as(teacher_next)
    teacher_loss = ((teacher_next - flat_targets) / branch.state_scale).square().mean()
    event_loss = F.cross_entropy(
        teacher_logits,
        labels.flatten(),
        weight=event_weights,
    )
    free = labels.flatten() == 0
    free_port_loss = (
        (teacher_jump[free] / branch.state_scale).square().mean()
        if bool(free.any())
        else teacher_jump.square().mean() * 0.0
    )
    current = encoded[:, 0]
    predictions = []
    for step in range(actions.shape[1]):
        current, _, _, _ = branch.step(current, actions[:, step])
        predictions.append(current)
    rollout = torch.stack(predictions, dim=1)
    normalized_error = (rollout - targets[:, 1:]) / branch.state_scale
    weights = torch.linspace(1.0, 2.0, actions.shape[1], device=features.device)
    rollout_loss = (
        normalized_error.square().mean(dim=(0, 2)) * weights
    ).sum() / weights.sum()
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


def _coefficient_payload(branch: ActionCausalBranch) -> dict[str, Any]:
    gains, decays, action_gains = branch.core.coefficients()
    parameters = {
        "positionGain": gains.detach().cpu().tolist(),
        "momentumDecay": decays.detach().cpu().tolist(),
    }
    return {
        **parameters,
        "actionPositionGain": float(action_gains[0].detach()),
        "actionMomentumGain": float(action_gains[1].detach()),
        **_effective_physics(parameters, WorldConfig().dt),
        "withinPortHamiltonianDomain": bool(
            (gains > 0).all()
            and (decays >= 0).all()
            and (decays <= 1).all()
            and (action_gains >= 0).all()
        ),
    }


def _pixel_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    result = {"accuracy": float((prediction == target).float().mean())}
    for name, values in (("player", (5, 6)), ("puck", (7, 8))):
        predicted_mask = torch.zeros_like(prediction, dtype=torch.bool)
        target_mask = torch.zeros_like(target, dtype=torch.bool)
        for value in values:
            predicted_mask |= prediction == value
            target_mask |= target == value
        intersection = (predicted_mask & target_mask).sum(dim=(-2, -1)).float()
        union = (predicted_mask | target_mask).sum(dim=(-2, -1)).float().clamp_min(1.0)
        result[f"{name}Iou"] = float((intersection / union).mean())
    return result


@torch.no_grad()
def _render_evaluation(
    renderer: ImplicitStateRenderer,
    states: torch.Tensor,
    frames: torch.Tensor,
    batch_size: int = 64,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    for start in range(0, states.shape[0], batch_size):
        stop = min(start + batch_size, states.shape[0])
        prediction = renderer(states[start:stop]).argmax(dim=1)
        metrics = _pixel_metrics(prediction, frames[start:stop])
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * (stop - start)
        count += stop - start
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def evaluate_branch(
    branch: ActionCausalBranch,
    renderer: ImplicitStateRenderer,
    features: torch.Tensor,
    targets: torch.Tensor,
    frames: torch.Tensor,
    actions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    branch.eval()
    encoded = branch.encode(features)
    current = encoded[:, 0]
    rollout = []
    for step in range(actions.shape[1]):
        current, _, _, _ = branch.step(current, actions[:, step])
        rollout.append(current)
    rollout_tensor = torch.stack(rollout, dim=1)
    horizons: dict[str, Any] = {}
    for horizon in (1, 2, 4, 8):
        prediction = rollout_tensor[:, horizon - 1]
        target = targets[:, horizon]
        persistence = targets[:, 0, :8]
        scale = (target[:, :8] - persistence).square().mean().sqrt().clamp_min(1e-12)
        horizons[str(horizon)] = {
            "deltaNrmse": float(
                (prediction[:, :8] - target[:, :8]).square().mean().sqrt() / scale
            ),
            "qR2": _r2(prediction[:, :4], target[:, :4]),
            "pR2": _r2(prediction[:, 4:8], target[:, 4:8]),
            "pixels": _render_evaluation(
                renderer,
                prediction,
                frames[:, horizon],
            ),
        }

    shuffled_initial = torch.roll(encoded[:, 0], shifts=1, dims=0)
    shuffled_actions = torch.roll(actions, shifts=1, dims=0)
    initial_control = shuffled_initial
    action_control = encoded[:, 0]
    zero_action = encoded[:, 0]
    for step in range(actions.shape[1]):
        initial_control = branch.step(initial_control, actions[:, step])[0]
        action_control = branch.step(action_control, shuffled_actions[:, step])[0]
        zero_action = branch.step(zero_action, torch.zeros_like(actions[:, step]))[0]
    target_h8 = targets[:, -1, :8]
    persistence_h8 = targets[:, 0, :8]
    scale_h8 = (target_h8 - persistence_h8).square().mean().sqrt().clamp_min(1e-12)

    teacher = encoded[:, :-1].reshape(-1, encoded.shape[-1])
    teacher_actions = actions.reshape(-1, 2)
    teacher_next, teacher_logits, teacher_jump, _ = branch.step(teacher, teacher_actions)
    true_next = targets[:, 1:].reshape(-1, targets.shape[-1])
    flat_labels = labels.flatten()
    regimes = {}
    for index, name in enumerate(REGIMES):
        mask = flat_labels == index
        regimes[name] = {
            "samples": int(mask.sum()),
            "deltaNrmse": (
                _transition_metrics(
                    teacher_next[mask, :8],
                    teacher[mask, :8],
                    true_next[mask, :8],
                )["deltaNrmse"]
                if bool(mask.any())
                else None
            ),
            "meanJumpNorm": (
                float(teacher_jump[mask, :8].norm(dim=-1).mean())
                if bool(mask.any())
                else None
            ),
        }
    predicted_events = teacher_logits.argmax(dim=-1)
    recalls = [
        float((predicted_events[flat_labels == index] == index).float().mean())
        for index in range(len(REGIMES))
        if bool((flat_labels == index).any())
    ]

    stability = encoded[: min(256, encoded.shape[0]), 0, :8]
    zero = torch.zeros(stability.shape[0], 2, device=stability.device)
    energy = []
    masses = _coefficient_payload(branch)["effectiveMass"]

    def kinetic(value: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(value.shape[0], device=value.device)
        for entity, mass in enumerate(masses):
            momentum = value[:, 4 + entity * 2 : 6 + entity * 2]
            total += momentum.square().sum(dim=-1) / (2.0 * max(float(mass), 1e-8))
        return total

    energy.append(kinetic(stability))
    finite = True
    for _ in range(64):
        stability = branch.core(stability, zero)
        finite = finite and bool(torch.isfinite(stability).all())
        energy.append(kinetic(stability))
    energy_tensor = torch.stack(energy, dim=1)

    direction_results = []
    intervention_state = encoded[: min(256, encoded.shape[0]), 0, :8]
    for index, direction in enumerate(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    ):
        action = torch.tensor(direction, device=features.device).expand(
            intervention_state.shape[0], -1
        )
        response = branch.core(intervention_state, action) - branch.core(
            intervention_state,
            torch.zeros_like(action),
        )
        expected = action
        direction_results.append(
            {
                "direction": index,
                "momentumCosine": float(
                    F.cosine_similarity(response[:, 4:6], expected, dim=-1).mean()
                ),
                "puckCrossTalkRmse": float(response[:, 6:8].square().mean().sqrt()),
            }
        )

    return {
        "stateReadout": {
            **_state_metrics(
                encoded[..., :8].reshape(-1, 8),
                targets[..., :8].reshape(-1, 8),
            ),
            "modeR2": _r2(
                encoded[..., 8:].reshape(-1, 2),
                targets[..., 8:].reshape(-1, 2),
            ),
        },
        "rolloutByHorizon": horizons,
        "negativeControls": {
            "shuffledInitialStateH8": float(
                (initial_control[:, :8] - target_h8).square().mean().sqrt() / scale_h8
            ),
            "shuffledActionsH8": float(
                (action_control[:, :8] - target_h8).square().mean().sqrt() / scale_h8
            ),
            "zeroActionsH8": float(
                (zero_action[:, :8] - target_h8).square().mean().sqrt() / scale_h8
            ),
        },
        "regimes": regimes,
        "eventBalancedAccuracy": float(sum(recalls) / max(len(recalls), 1)),
        "core": {
            "parameterCount": sum(parameter.numel() for parameter in branch.core.parameters()),
            "parameters": _coefficient_payload(branch),
            "stability": {
                "finite": finite,
                "energyGrowthFraction": float(
                    (energy_tensor[:, 1:] > energy_tensor[:, :-1] + 1e-8)
                    .float()
                    .mean()
                ),
                "finalToInitialEnergy": float(
                    energy_tensor[:, -1].mean()
                    / energy_tensor[:, 0].mean().clamp_min(1e-8)
                ),
            },
        },
        "actionInterventions": direction_results,
    }


def run_action_port_pixel_experiment(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config: ActionPortPixelConfig = ActionPortPixelConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    if config.block_count != 5:
        raise ValueError("the pixel-reinjection experiment is preregistered at block 5")
    if config.transitions_per_trajectory < 8:
        raise ValueError("at least eight transitions are required for horizon-8 evaluation")
    fit_count = int(config.trajectories * config.fit_fraction)
    if not 0 < fit_count < config.trajectories:
        raise ValueError("fit_fraction must leave at least one fit and one test trajectory")
    _seed_everything(config.seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu"
    ) if device_name == "auto" else torch.device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = (
        build_pixel_direct_from_checkpoint(checkpoint)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    model_config = backbone.config
    collection_started = time.perf_counter()
    collected = collect_action_sequences(backbone, config, device)
    collection_seconds = time.perf_counter() - collection_started
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    features = collected["features"].to(device)
    states = collected["states"].to(device)
    frames = collected["frames"].to(device)
    actions = collected["actionVectors"].to(device)
    labels = collected["regimes"].to(device)
    fit_features, test_features = features[:fit_count], features[fit_count:]
    fit_states, test_states = states[:fit_count], states[fit_count:]
    fit_frames, test_frames = frames[:fit_count], frames[fit_count:]
    fit_actions, test_actions = actions[:fit_count], actions[fit_count:]
    fit_labels, test_labels = labels[:fit_count], labels[fit_count:]

    flat_features = fit_features.reshape(-1, fit_features.shape[-1])
    flat_states = fit_states.reshape(-1, fit_states.shape[-1])
    feature_mean = flat_features.mean(dim=0)
    feature_scale = flat_features.std(dim=0).clamp_min(1e-5)
    state_mean = flat_states.mean(dim=0)
    state_scale = flat_states.std(dim=0).clamp_min(1e-5)
    ph = ActionCausalBranch(
        feature_mean,
        feature_scale,
        state_mean,
        state_scale,
        hidden_size=config.hidden_size,
        structured=True,
    ).to(device)
    _fit_linear_encoder(ph, flat_features, flat_states, config.ridge)
    tangent = ActionCausalBranch(
        feature_mean,
        feature_scale,
        state_mean,
        state_scale,
        hidden_size=config.hidden_size,
        structured=False,
    ).to(device)
    tangent.encoder.load_state_dict(copy.deepcopy(ph.encoder.state_dict()))
    tangent.hybrid_port.load_state_dict(copy.deepcopy(ph.hybrid_port.state_dict()))
    if sum(p.numel() for p in ph.parameters()) != sum(
        p.numel() for p in tangent.parameters()
    ):
        raise AssertionError("pH and tangent controls must have equal capacity")
    branches = {"portHamiltonian": ph, "tangentMatchedControl": tangent}
    optimizer = torch.optim.AdamW(
        [parameter for branch in branches.values() for parameter in branch.parameters()],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    event_weights = _event_class_weights(fit_labels).to(device)
    dynamics_started = time.perf_counter()
    dynamics_log = output_dir / "dynamics.jsonl"
    with dynamics_log.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.dynamics_steps + 1):
            indices = torch.randint(
                0,
                fit_count,
                (config.dynamics_batch_size,),
                device=device,
            )
            lr = config.learning_rate * _dynamics_learning_rate_multiplier(
                step,
                config,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            terms = {}
            for name, branch in branches.items():
                branch.train()
                loss, branch_terms = _branch_loss(
                    branch,
                    fit_features[indices],
                    fit_states[indices],
                    fit_actions[indices],
                    fit_labels[indices],
                    event_weights,
                    config,
                )
                losses[name] = loss
                terms[name] = branch_terms
            total = sum(losses.values()) / len(losses)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for branch in branches.values()
                    for parameter in branch.parameters()
                ],
                5.0,
            )
            optimizer.step()
            if step == 1 or step % config.log_every == 0 or step == config.dynamics_steps:
                elapsed = time.perf_counter() - dynamics_started
                payload = {
                    "stage": "train_action_dynamics",
                    "step": step,
                    "steps": config.dynamics_steps,
                    "loss": float(total.detach()),
                    "gradientNorm": float(gradient_norm),
                    "stepsPerSecond": step / max(elapsed, 1e-8),
                    **{
                        name: {
                            "loss": float(losses[name].detach()),
                            **{key: float(value.detach()) for key, value in terms[name].items()},
                            "core": _coefficient_payload(branches[name]),
                        }
                        for name in branches
                    },
                }
                print(json.dumps(payload), flush=True)
                log_file.write(json.dumps(payload) + "\n")
                log_file.flush()
    dynamics_seconds = time.perf_counter() - dynamics_started

    renderer = ImplicitStateRenderer(
        model_config.image_size,
        model_config.palette_size,
        state_mean,
        state_scale,
        config.renderer_hidden_size,
    ).to(device)
    counts = torch.bincount(
        fit_frames.flatten().long(),
        minlength=model_config.palette_size,
    ).float()
    renderer_weights = (counts.max() / counts.clamp_min(1.0)).sqrt().clamp(0.25, 12.0)
    renderer_weights /= renderer_weights[1].clamp_min(1e-6)
    renderer_optimizer = torch.optim.AdamW(
        renderer.parameters(),
        lr=config.renderer_learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    flat_fit_states = fit_states.reshape(-1, fit_states.shape[-1])
    flat_fit_frames = fit_frames.reshape(-1, fit_frames.shape[-2], fit_frames.shape[-1])
    renderer_started = time.perf_counter()
    renderer_log = output_dir / "renderer.jsonl"
    with renderer_log.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.renderer_steps + 1):
            indices = torch.randint(
                0,
                flat_fit_states.shape[0],
                (config.renderer_batch_size,),
                device=device,
            )
            renderer_optimizer.zero_grad(set_to_none=True)
            logits = renderer(flat_fit_states[indices])
            cross_entropy = F.cross_entropy(
                logits,
                flat_fit_frames[indices].long(),
                weight=renderer_weights,
            )
            entity_dice = _entity_dice_loss(logits, flat_fit_frames[indices])
            loss = (
                cross_entropy
                + config.renderer_entity_dice_weight * entity_dice
            )
            loss.backward()
            renderer_optimizer.step()
            if step == 1 or step % config.log_every == 0 or step == config.renderer_steps:
                elapsed = time.perf_counter() - renderer_started
                payload = {
                    "stage": "train_state_renderer",
                    "step": step,
                    "steps": config.renderer_steps,
                    "loss": float(loss.detach()),
                    "crossEntropy": float(cross_entropy.detach()),
                    "entityDice": float(entity_dice.detach()),
                    "stepsPerSecond": step / max(elapsed, 1e-8),
                }
                print(json.dumps(payload), flush=True)
                log_file.write(json.dumps(payload) + "\n")
                log_file.flush()
    renderer_seconds = time.perf_counter() - renderer_started

    evaluation_started = time.perf_counter()
    renderer.eval()
    evaluation = {
        "rendererOracle": _render_evaluation(
            renderer,
            test_states[:, -1],
            test_frames[:, -1],
        ),
        **{
            name: evaluate_branch(
                branch,
                renderer,
                test_features,
                test_states,
                test_frames,
                test_actions,
                test_labels,
            )
            for name, branch in branches.items()
        },
    }
    evaluation_seconds = time.perf_counter() - evaluation_started
    payload = {
        "kind": "action_port_pixel_reinjection",
        "version": 1,
        "baseCheckpoint": str(checkpoint_path),
        "baseCheckpointStep": int(checkpoint["step"]),
        "config": asdict(config),
        "branches": {name: branch.state_dict() for name, branch in branches.items()},
        "renderer": renderer.state_dict(),
    }
    torch.save(payload, output_dir / "checkpoint.pt")
    summary = {
        "kind": payload["kind"],
        "version": 1,
        "baseCheckpointStep": int(checkpoint["step"]),
        "frozenBackbone": True,
        "noPixelBypass": True,
        "config": asdict(config),
        "capacity": {
            name: sum(parameter.numel() for parameter in branch.parameters())
            for name, branch in branches.items()
        }
        | {"renderer": sum(parameter.numel() for parameter in renderer.parameters())},
        "timing": {
            "collectionSeconds": collection_seconds,
            "dynamicsSeconds": dynamics_seconds,
            "rendererSeconds": renderer_seconds,
            "evaluationSeconds": evaluation_seconds,
            "totalSeconds": (
                collection_seconds + dynamics_seconds + renderer_seconds + evaluation_seconds
            ),
        },
        "data": {
            "fitTrajectories": fit_count,
            "testTrajectories": config.trajectories - fit_count,
            "regimeCounts": {
                name: int((labels == index).sum())
                for index, name in enumerate(REGIMES)
            },
        },
        "evaluation": evaluation,
        "artifacts": ["checkpoint.pt", "dynamics.jsonl", "renderer.jsonl", "summary.json"],
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
    parser.add_argument("--dynamics-steps", type=int, default=6_000)
    parser.add_argument("--renderer-steps", type=int, default=5_000)
    parser.add_argument("--dynamics-batch-size", type=int, default=256)
    parser.add_argument("--renderer-batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=101_510_731)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_action_port_pixel_experiment(
        args.checkpoint,
        args.output,
        config=ActionPortPixelConfig(
            trajectories=args.trajectories,
            dynamics_steps=args.dynamics_steps,
            renderer_steps=args.renderer_steps,
            dynamics_batch_size=args.dynamics_batch_size,
            renderer_batch_size=args.renderer_batch_size,
            feature_batch_size=args.feature_batch_size,
            seed=args.seed,
            log_every=args.log_every,
        ),
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
