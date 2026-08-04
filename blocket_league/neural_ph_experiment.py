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

from .action_port_pixel_experiment import (
    ImplicitStateRenderer,
    _branch_loss,
    _dynamics_learning_rate_multiplier,
    _render_evaluation,
    action_vectors,
)
from .data import make_clip, make_excitation_clip
from .env import WorldConfig
from .neural_port_hamiltonian import (
    NeuralODE,
    NeuralPortHamiltonian,
    NeuralPortHamiltonianConfig,
)
from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .port_hamiltonian_audit import REGIMES, _state_metrics, _transition_metrics
from .port_hamiltonian_bottleneck import (
    HybridJumpPort,
    _block5_entity_features,
    _event_class_weights,
    _fit_linear_encoder,
    _r2,
    bottleneck_state,
    regime_labels,
)
from .train_pixel_direct import frames_to_classes, palette_tensor


@dataclass(frozen=True)
class NeuralPHExperimentConfig:
    fit_policy_trajectories: int = 3_072
    fit_cardinal_trajectories: int = 3_072
    test_policy_trajectories: int = 1_024
    test_diagonal_trajectories: int = 512
    test_reversal_trajectories: int = 512
    transitions_per_trajectory: int = 8
    feature_batch_size: int = 64
    block_count: int = 5
    dynamics_steps: int = 4_000
    dynamics_batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 300
    min_learning_rate_ratio: float = 0.1
    hidden_size: int = 64
    hidden_layers: int = 2
    integration_method: str = "midpoint"
    integration_substeps: int = 1
    resistance_floor: float = 1e-5
    state_loss_weight: float = 1.0
    teacher_dynamics_weight: float = 1.0
    rollout_loss_weight: float = 2.0
    event_loss_weight: float = 0.20
    free_port_weight: float = 0.20
    energy_gradient_weight: float = 0.02
    energy_probe_size: int = 64
    ridge: float = 1e-2
    log_every: int = 50
    seed: int = 111_610_731

    @property
    def fit_trajectories(self) -> int:
        return self.fit_policy_trajectories + self.fit_cardinal_trajectories


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_sequence_suite(
    model: nn.Module,
    *,
    trajectories: int,
    transitions: int,
    seed: int,
    family: str,
    feature_batch_size: int,
    block_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Collect one policy or intervention suite using the same visual probe."""

    if family not in {"policy", "cardinal", "diagonal", "reversal"}:
        raise ValueError(f"unknown suite family {family!r}")
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
            _block5_entity_features(model, classes, block_count).cpu()
        )
        frame_chunks.append(classes[:, -1].byte().cpu())
        pending_contexts.clear()

    history = model.config.history_frames
    endpoints = transitions + 1
    started = time.perf_counter()
    for trajectory in range(trajectories):
        clip_seed = seed + trajectory * 9_973
        clip_arguments = {
            "context_frames": 1,
            "future_frames": history + transitions - 1,
            "image_size": model.config.image_size,
        }
        if family == "policy":
            clip = make_clip(clip_seed, **clip_arguments)
        else:
            clip = make_excitation_clip(
                clip_seed,
                action_family=family,
                **clip_arguments,
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
                [all_events[offset + history] for offset in range(transitions)]
            )
        )
        action_sequences.append(
            torch.stack(
                [all_actions[offset + history] for offset in range(transitions)]
            )
        )
        for offset in range(endpoints):
            pending_contexts.append(frames[offset : offset + history])
            if len(pending_contexts) >= feature_batch_size:
                flush()
        if (trajectory + 1) % 256 == 0 or trajectory + 1 == trajectories:
            print(
                json.dumps(
                    {
                        "stage": "collect_neural_ph",
                        "family": family,
                        "trajectories": trajectory + 1,
                        "total": trajectories,
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
        "features": torch.cat(feature_chunks).reshape(trajectories, endpoints, -1),
        "frames": torch.cat(frame_chunks).reshape(
            trajectories,
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


def _concatenate_suites(*suites: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not suites:
        raise ValueError("at least one suite is required")
    return {
        name: torch.cat([suite[name] for suite in suites], dim=0)
        for name in suites[0]
    }


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _matched_control_hidden_size(
    target_parameters: int,
    base_config: NeuralPortHamiltonianConfig,
) -> int:
    candidates = range(4, max(512, base_config.hidden_size * 4) + 1)
    return min(
        candidates,
        key=lambda hidden: abs(
            _parameter_count(NeuralODE(replace(base_config, hidden_size=hidden)))
            - target_parameters
        ),
    )


class NeuralPHBranch(nn.Module):
    """Shared visual/hybrid shell around either a learned pH or Neural ODE core."""

    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        state_mean: torch.Tensor,
        state_scale: torch.Tensor,
        *,
        hidden_size: int,
        hidden_layers: int,
        integration_method: str,
        integration_substeps: int,
        resistance_floor: float,
        structured: bool,
        control_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_scale", feature_scale)
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_scale", state_scale)
        self.encoder = nn.Linear(feature_mean.numel(), state_mean.numel())
        core_config = NeuralPortHamiltonianConfig(
            state_size=8,
            input_size=2,
            hidden_size=(
                hidden_size
                if structured or control_hidden_size is None
                else control_hidden_size
            ),
            hidden_layers=hidden_layers,
            dt=WorldConfig().dt,
            integration_method=integration_method,  # type: ignore[arg-type]
            integration_substeps=integration_substeps,
            resistance_floor=resistance_floor,
        )
        self.core: NeuralPortHamiltonian | NeuralODE
        if structured:
            self.core = NeuralPortHamiltonian(
                core_config,
                state_mean=state_mean[:8],
                state_scale=state_scale[:8],
            )
        else:
            self.core = NeuralODE(
                core_config,
                state_mean=state_mean[:8],
                state_scale=state_scale[:8],
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


def _load_frozen_renderer(
    checkpoint_path: Path,
    *,
    image_size: int,
    palette_size: int,
    device: torch.device,
) -> ImplicitStateRenderer:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["renderer"]
    hidden_size = int(state_dict["network.0.weight"].shape[0])
    renderer = ImplicitStateRenderer(
        image_size,
        palette_size,
        state_dict["state_mean"],
        state_dict["state_scale"],
        hidden_size,
    )
    renderer.load_state_dict(state_dict)
    return renderer.to(device).eval().requires_grad_(False)


def _energy_gauge_loss(branch: NeuralPHBranch, states: torch.Tensor) -> torch.Tensor:
    if not isinstance(branch.core, NeuralPortHamiltonian):
        return states.square().mean() * 0.0
    _, gradient, _, _, _ = branch.core.components(states, create_graph=True)
    normalized_gradient = gradient * branch.core.state_scale
    rms = normalized_gradient.square().mean().add(1e-12).sqrt()
    return (rms - 1.0).square()


def _fit_affine_r2(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    design = torch.stack((inputs, torch.ones_like(inputs)), dim=-1)
    coefficients = torch.linalg.lstsq(design, targets[:, None]).solution[:, 0]
    prediction = design @ coefficients
    residual = (prediction - targets).square().sum()
    total = (targets - targets.mean()).square().sum().clamp_min(1e-12)
    return float(1.0 - residual / total)


def _matrix_cosine(matrix: torch.Tensor, reference: torch.Tensor) -> float:
    expanded = reference.expand_as(matrix)
    return float(
        F.cosine_similarity(matrix.flatten(1), expanded.flatten(1), dim=-1).mean()
    )


def _relative_state_variation(values: torch.Tensor) -> float:
    centered = values - values.mean(dim=0, keepdim=True)
    return float(
        centered.square().mean().sqrt()
        / values.square().mean().sqrt().clamp_min(1e-12)
    )


@torch.no_grad()
def _structured_core_audit(
    core: NeuralPortHamiltonian,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, Any]:
    sample_count = min(256, states.shape[0])
    sample = states[:sample_count, :8]
    control = actions[:sample_count]
    energy, gradient, interconnection, resistance, port = core.components(
        sample, create_graph=False
    )
    powers = core.power_terms(sample, control, create_graph=False)

    world = WorldConfig()
    kinetic = (
        sample[:, 4:6].square().sum(dim=-1) / (2.0 * world.player_mass)
        + sample[:, 6:8].square().sum(dim=-1) / (2.0 * world.puck_mass)
    )
    true_gradient = torch.zeros_like(sample)
    true_gradient[:, 4:6] = sample[:, 4:6] / world.player_mass
    true_gradient[:, 6:8] = sample[:, 6:8] / world.puck_mass
    gradient_scale = (
        (gradient * true_gradient).sum()
        / gradient.square().sum().clamp_min(1e-12)
    )
    aligned_gradient = gradient_scale * gradient

    canonical_j = torch.zeros(8, 8, device=sample.device)
    canonical_j[torch.arange(4), torch.arange(4) + 4] = 1.0
    canonical_j[torch.arange(4) + 4, torch.arange(4)] = -1.0
    expected_r = torch.zeros(8, 8, device=sample.device)
    expected_r[4, 4] = expected_r[5, 5] = world.player_mass * world.player_drag
    expected_r[6, 6] = expected_r[7, 7] = world.puck_mass * world.puck_drag
    expected_b = torch.zeros(8, 2, device=sample.device)
    expected_b[4, 0] = world.player_mass * world.player_acceleration
    expected_b[5, 1] = world.player_mass * world.player_acceleration

    jacobi_sample = sample[: min(16, sample.shape[0])]
    jacobi = core.jacobi_tensor(jacobi_sample, create_graph=False)
    passive = sample[: min(128, sample.shape[0])]
    zero = torch.zeros(passive.shape[0], 2, device=sample.device)
    energy_path = [core.hamiltonian(passive)]
    finite = True
    for _ in range(64):
        passive = core(passive, zero)
        finite = finite and bool(torch.isfinite(passive).all())
        energy_path.append(core.hamiltonian(passive))
    energy_path_tensor = torch.stack(energy_path, dim=1)
    energy_changes = energy_path_tensor[:, 1:] - energy_path_tensor[:, :-1]

    return {
        "powerBalance": {
            "maxAbsDefect": float(powers["balanceDefect"].abs().max()),
            "rmsDefect": float(powers["balanceDefect"].square().mean().sqrt()),
            "minimumDissipation": float(powers["dissipation"].min()),
        },
        "hamiltonian": {
            "kineticEnergyAffineR2": _fit_affine_r2(kinetic, energy),
            "normalizedGradientRms": float(
                (gradient * core.state_scale).square().mean().sqrt()
            ),
            "gradientScaleToKinetic": float(gradient_scale),
            "kineticGradientCosine": float(
                F.cosine_similarity(aligned_gradient, true_gradient, dim=-1).mean()
            ),
            "stateVariation": _relative_state_variation(energy[:, None]),
        },
        "interconnection": {
            "canonicalCosine": _matrix_cosine(interconnection, canonical_j),
            "stateVariation": _relative_state_variation(interconnection),
            "skewDefect": float(
                (interconnection + interconnection.transpose(-1, -2)).abs().max()
            ),
            "jacobiRms": float(jacobi.square().mean().sqrt()),
            "jacobiMaxAbs": float(jacobi.abs().max()),
        },
        "resistance": {
            "physicalDragCosine": _matrix_cosine(resistance, expected_r),
            "stateVariation": _relative_state_variation(resistance),
            "minimumEigenvalue": float(torch.linalg.eigvalsh(resistance).min()),
        },
        "port": {
            "physicalIncidenceCosine": _matrix_cosine(port, expected_b),
            "stateVariation": _relative_state_variation(port),
            "positionRowFraction": float(
                port[:, :4].square().sum().sqrt()
                / port.square().sum().sqrt().clamp_min(1e-12)
            ),
            "puckMomentumFraction": float(
                port[:, 6:8].square().sum().sqrt()
                / port.square().sum().sqrt().clamp_min(1e-12)
            ),
        },
        "zeroInputDiscreteEnergy": {
            "finite": finite,
            "increaseFraction": float((energy_changes > 1e-7).float().mean()),
            "meanNetChange": float(
                (energy_path_tensor[:, -1] - energy_path_tensor[:, 0]).mean()
            ),
            "maximumStepIncrease": float(energy_changes.max()),
        },
    }


@torch.no_grad()
def evaluate_branch(
    branch: NeuralPHBranch,
    renderer: ImplicitStateRenderer,
    suite: dict[str, torch.Tensor],
) -> dict[str, Any]:
    branch.eval()
    features = suite["features"]
    targets = suite["states"]
    frames = suite["frames"]
    actions = suite["actionVectors"]
    labels = suite["regimes"]
    encoded = branch.encode(features)
    current = encoded[:, 0]
    rollout = []
    for step in range(actions.shape[1]):
        current = branch.step(current, actions[:, step])[0]
        rollout.append(current)
    predictions = torch.stack(rollout, dim=1)
    horizons: dict[str, Any] = {}
    for horizon in (1, 2, 4, 8):
        prediction = predictions[:, horizon - 1]
        target = targets[:, horizon]
        persistence = targets[:, 0, :8]
        scale = (target[:, :8] - persistence).square().mean().sqrt().clamp_min(1e-12)
        horizons[str(horizon)] = {
            "deltaNrmse": float(
                (prediction[:, :8] - target[:, :8]).square().mean().sqrt() / scale
            ),
            "qR2": _r2(prediction[:, :4], target[:, :4]),
            "pR2": _r2(prediction[:, 4:8], target[:, 4:8]),
            "pixels": _render_evaluation(renderer, prediction, frames[:, horizon]),
        }

    shuffled_actions = torch.roll(actions, shifts=1, dims=0)
    shuffled_initial = torch.roll(encoded[:, 0], shifts=1, dims=0)
    initial_control = shuffled_initial
    action_control = encoded[:, 0]
    zero_control = encoded[:, 0]
    for step in range(actions.shape[1]):
        initial_control = branch.step(initial_control, actions[:, step])[0]
        action_control = branch.step(action_control, shuffled_actions[:, step])[0]
        zero_control = branch.step(
            zero_control, torch.zeros_like(actions[:, step])
        )[0]
    target_h8 = targets[:, -1, :8]
    scale_h8 = (
        target_h8 - targets[:, 0, :8]
    ).square().mean().sqrt().clamp_min(1e-12)

    teacher = encoded[:, :-1].reshape(-1, encoded.shape[-1])
    teacher_actions = actions.reshape(-1, 2)
    teacher_next, teacher_logits, teacher_jump, _ = branch.step(
        teacher, teacher_actions
    )
    true_next = targets[:, 1:].reshape(-1, targets.shape[-1])
    flat_labels = labels.flatten()
    regimes = {}
    for index, name in enumerate(REGIMES):
        mask = flat_labels == index
        regimes[name] = {
            "samples": int(mask.sum()),
            "deltaNrmse": (
                _transition_metrics(
                    teacher_next[mask, :8], teacher[mask, :8], true_next[mask, :8]
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
    result: dict[str, Any] = {
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
                (zero_control[:, :8] - target_h8).square().mean().sqrt() / scale_h8
            ),
        },
        "regimes": regimes,
        "eventBalancedAccuracy": float(sum(recalls) / max(len(recalls), 1)),
        "coreParameterCount": _parameter_count(branch.core),
    }
    if isinstance(branch.core, NeuralPortHamiltonian):
        true_transition_states = targets[:, :-1, :8].reshape(-1, 8)
        true_transition_actions = actions.reshape(-1, 2)
        result["structure"] = _structured_core_audit(
            branch.core,
            true_transition_states,
            true_transition_actions,
        )
    return result


def _parameter_change_payload(
    branch: NeuralPHBranch,
    initial_state: dict[str, torch.Tensor],
) -> dict[str, float]:
    current = branch.core.state_dict()
    prefixes = (
        ("H", "energy_network"),
        ("J", "interconnection_network"),
        ("R", "resistance_network"),
        ("B", "port_network"),
    )
    result = {}
    for label, prefix in prefixes:
        changes = [
            (value.detach().cpu() - initial_state[name]).square().sum()
            for name, value in current.items()
            if name.startswith(prefix) and name in initial_state
        ]
        result[label] = float(torch.stack(changes).sum().sqrt())
    return result


def run_neural_ph_experiment(
    checkpoint_path: Path,
    renderer_checkpoint_path: Path,
    output_dir: Path,
    *,
    config: NeuralPHExperimentConfig = NeuralPHExperimentConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    if config.block_count != 5:
        raise ValueError("the experiment is preregistered at block 5")
    if config.transitions_per_trajectory < 8:
        raise ValueError("at least eight transitions are required")
    if config.fit_trajectories < 2:
        raise ValueError("at least two fit trajectories are required")
    if min(
        config.test_policy_trajectories,
        config.test_diagonal_trajectories,
        config.test_reversal_trajectories,
    ) < 1:
        raise ValueError("every evaluation suite must be non-empty")
    _seed_everything(config.seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
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
    fit_policy = collect_sequence_suite(
        backbone,
        trajectories=config.fit_policy_trajectories,
        transitions=config.transitions_per_trajectory,
        seed=config.seed + 1_000_000,
        family="policy",
        feature_batch_size=config.feature_batch_size,
        block_count=config.block_count,
        device=device,
    )
    fit_cardinal = collect_sequence_suite(
        backbone,
        trajectories=config.fit_cardinal_trajectories,
        transitions=config.transitions_per_trajectory,
        seed=config.seed + 2_000_000,
        family="cardinal",
        feature_batch_size=config.feature_batch_size,
        block_count=config.block_count,
        device=device,
    )
    evaluation_suites = {
        "policy": collect_sequence_suite(
            backbone,
            trajectories=config.test_policy_trajectories,
            transitions=config.transitions_per_trajectory,
            seed=config.seed + 3_000_000,
            family="policy",
            feature_batch_size=config.feature_batch_size,
            block_count=config.block_count,
            device=device,
        ),
        "diagonalOod": collect_sequence_suite(
            backbone,
            trajectories=config.test_diagonal_trajectories,
            transitions=config.transitions_per_trajectory,
            seed=config.seed + 4_000_000,
            family="diagonal",
            feature_batch_size=config.feature_batch_size,
            block_count=config.block_count,
            device=device,
        ),
        "reversalOod": collect_sequence_suite(
            backbone,
            trajectories=config.test_reversal_trajectories,
            transitions=config.transitions_per_trajectory,
            seed=config.seed + 5_000_000,
            family="reversal",
            feature_batch_size=config.feature_batch_size,
            block_count=config.block_count,
            device=device,
        ),
    }
    collection_seconds = time.perf_counter() - collection_started
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fit = {
        name: value.to(device)
        for name, value in _concatenate_suites(fit_policy, fit_cardinal).items()
    }
    evaluation_suites = {
        suite_name: {name: value.to(device) for name, value in suite.items()}
        for suite_name, suite in evaluation_suites.items()
    }
    flat_features = fit["features"].reshape(-1, fit["features"].shape[-1])
    flat_states = fit["states"].reshape(-1, fit["states"].shape[-1])
    feature_mean = flat_features.mean(dim=0)
    feature_scale = flat_features.std(dim=0).clamp_min(1e-5)
    state_mean = flat_states.mean(dim=0)
    state_scale = flat_states.std(dim=0).clamp_min(1e-5)

    ph = NeuralPHBranch(
        feature_mean,
        feature_scale,
        state_mean,
        state_scale,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        integration_method=config.integration_method,
        integration_substeps=config.integration_substeps,
        resistance_floor=config.resistance_floor,
        structured=True,
    ).to(device)
    _fit_linear_encoder(ph, flat_features, flat_states, config.ridge)
    ph_initial_core = {
        name: value.detach().cpu().clone() for name, value in ph.core.state_dict().items()
    }
    base_core_config = ph.core.config
    control_hidden_size = _matched_control_hidden_size(
        _parameter_count(ph.core), base_core_config
    )
    control = NeuralPHBranch(
        feature_mean,
        feature_scale,
        state_mean,
        state_scale,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        integration_method=config.integration_method,
        integration_substeps=config.integration_substeps,
        resistance_floor=config.resistance_floor,
        structured=False,
        control_hidden_size=control_hidden_size,
    ).to(device)
    control.encoder.load_state_dict(copy.deepcopy(ph.encoder.state_dict()))
    control.hybrid_port.load_state_dict(copy.deepcopy(ph.hybrid_port.state_dict()))
    branches = {"neuralPortHamiltonian": ph, "neuralOdeControl": control}
    capacity_gap = abs(_parameter_count(ph.core) - _parameter_count(control.core))
    capacity_gap /= max(_parameter_count(ph.core), 1)
    if capacity_gap > 0.01:
        raise AssertionError("the control core must match pH capacity within one percent")

    optimizer = torch.optim.AdamW(
        [parameter for branch in branches.values() for parameter in branch.parameters()],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    event_weights = _event_class_weights(fit["regimes"]).to(device)
    dynamics_started = time.perf_counter()
    dynamics_log = output_dir / "dynamics.jsonl"
    with dynamics_log.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.dynamics_steps + 1):
            indices = torch.randint(
                0,
                config.fit_trajectories,
                (config.dynamics_batch_size,),
                device=device,
            )
            learning_rate = config.learning_rate * _dynamics_learning_rate_multiplier(
                step, config  # type: ignore[arg-type]
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            terms = {}
            for name, branch in branches.items():
                branch.train()
                loss, branch_terms = _branch_loss(
                    branch,  # type: ignore[arg-type]
                    fit["features"][indices],
                    fit["states"][indices],
                    fit["actionVectors"][indices],
                    fit["regimes"][indices],
                    event_weights,
                    config,  # type: ignore[arg-type]
                )
                if branch.structured:
                    encoded_probe = branch.encode(fit["features"][indices, 0])[
                        : config.energy_probe_size, :8
                    ]
                    gauge = _energy_gauge_loss(branch, encoded_probe)
                    loss = loss + config.energy_gradient_weight * gauge
                    branch_terms["energyGauge"] = gauge
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
                    "stage": "train_neural_ph",
                    "step": step,
                    "steps": config.dynamics_steps,
                    "loss": float(total.detach()),
                    "gradientNorm": float(gradient_norm),
                    "stepsPerSecond": step / max(elapsed, 1e-8),
                    "estimatedTrainingSeconds": (
                        elapsed / step * config.dynamics_steps
                    ),
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
                print(json.dumps(payload), flush=True)
                log_file.write(json.dumps(payload) + "\n")
                log_file.flush()
    dynamics_seconds = time.perf_counter() - dynamics_started

    renderer = _load_frozen_renderer(
        renderer_checkpoint_path,
        image_size=model_config.image_size,
        palette_size=model_config.palette_size,
        device=device,
    )
    evaluation_started = time.perf_counter()
    evaluation = {}
    for suite_name, suite in evaluation_suites.items():
        evaluation[suite_name] = {
            "rendererOracle": _render_evaluation(
                renderer, suite["states"][:, -1], suite["frames"][:, -1]
            ),
            **{
                branch_name: evaluate_branch(branch, renderer, suite)
                for branch_name, branch in branches.items()
            },
        }
    evaluation_seconds = time.perf_counter() - evaluation_started

    checkpoint_payload = {
        "kind": "state_dependent_neural_port_hamiltonian",
        "version": 1,
        "baseCheckpoint": str(checkpoint_path),
        "rendererCheckpoint": str(renderer_checkpoint_path),
        "baseCheckpointStep": int(checkpoint["step"]),
        "config": asdict(config),
        "controlHiddenSize": control_hidden_size,
        "branches": {name: branch.state_dict() for name, branch in branches.items()},
    }
    torch.save(checkpoint_payload, output_dir / "checkpoint.pt")
    summary = {
        "kind": checkpoint_payload["kind"],
        "version": 1,
        "baseCheckpointStep": int(checkpoint["step"]),
        "frozenBackbone": True,
        "frozenSharedRenderer": True,
        "noPixelBypass": True,
        "learnedFunctions": ["H(x)", "J(x)", "R(x)", "B(x)"],
        "exactConstraints": ["J(x)=-J(x)^T", "R(x)=L(x)L(x)^T"],
        "config": asdict(config),
        "capacity": {
            "controlHiddenSize": control_hidden_size,
            "core": {
                name: _parameter_count(branch.core) for name, branch in branches.items()
            },
            "completeBranch": {
                name: _parameter_count(branch) for name, branch in branches.items()
            },
            "relativeCoreGap": capacity_gap,
        },
        "parameterChangeNorm": _parameter_change_payload(ph, ph_initial_core),
        "timing": {
            "collectionSeconds": collection_seconds,
            "dynamicsSeconds": dynamics_seconds,
            "evaluationSeconds": evaluation_seconds,
            "totalSeconds": collection_seconds + dynamics_seconds + evaluation_seconds,
        },
        "data": {
            "fitPolicyTrajectories": config.fit_policy_trajectories,
            "fitCardinalTrajectories": config.fit_cardinal_trajectories,
            "testSuites": {
                name: int(suite["states"].shape[0])
                for name, suite in evaluation_suites.items()
            },
            "fitRegimeCounts": {
                name: int((fit["regimes"] == index).sum())
                for index, name in enumerate(REGIMES)
            },
        },
        "evaluation": evaluation,
        "artifacts": ["checkpoint.pt", "dynamics.jsonl", "summary.json"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("renderer_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fit-policy-trajectories", type=int, default=3_072)
    parser.add_argument("--fit-cardinal-trajectories", type=int, default=3_072)
    parser.add_argument("--test-policy-trajectories", type=int, default=1_024)
    parser.add_argument("--test-diagonal-trajectories", type=int, default=512)
    parser.add_argument("--test-reversal-trajectories", type=int, default=512)
    parser.add_argument("--dynamics-steps", type=int, default=4_000)
    parser.add_argument("--dynamics-batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--integration-method", choices=("euler", "midpoint", "rk4"), default="midpoint")
    parser.add_argument("--integration-substeps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111_610_731)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_neural_ph_experiment(
        args.checkpoint,
        args.renderer_checkpoint,
        args.output,
        config=NeuralPHExperimentConfig(
            fit_policy_trajectories=args.fit_policy_trajectories,
            fit_cardinal_trajectories=args.fit_cardinal_trajectories,
            test_policy_trajectories=args.test_policy_trajectories,
            test_diagonal_trajectories=args.test_diagonal_trajectories,
            test_reversal_trajectories=args.test_reversal_trajectories,
            dynamics_steps=args.dynamics_steps,
            dynamics_batch_size=args.dynamics_batch_size,
            feature_batch_size=args.feature_batch_size,
            hidden_size=args.hidden_size,
            integration_method=args.integration_method,
            integration_substeps=args.integration_substeps,
            seed=args.seed,
            log_every=args.log_every,
        ),
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
