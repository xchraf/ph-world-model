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
from .direct_model import DirectFactorizedBlock
from .env import PALETTE, WorldConfig
from .neural_ph_experiment import _matched_control_hidden_size, _parameter_count
from .neural_port_hamiltonian import (
    NeuralODE,
    NeuralPortHamiltonian,
    NeuralPortHamiltonianConfig,
)
from .pixel_direct_model import DirectPixelTransformer, build_pixel_direct_from_checkpoint
from .pixel_only_ph_experiment import (
    _class_weights,
    _energy_gauge,
    _evaluate_branch,
    _learning_rate_multiplier,
    _move_suite,
    _parameter_change_payload,
    _pixel_cross_entropy,
    _rgb_frames_to_classes_cpu,
    _whitening_loss,
)


@dataclass(frozen=True)
class EndToEndPHConfig:
    fit_policy_trajectories: int = 3_072
    fit_cardinal_trajectories: int = 3_072
    test_policy_trajectories: int = 512
    test_diagonal_trajectories: int = 256
    test_reversal_trajectories: int = 256
    audit_trajectories: int = 1_024
    transitions_per_trajectory: int = 8
    state_size: int = 8
    encoder_hidden_size: int = 192
    decoder_hidden_size: int = 192
    decoder_depth: int = 3
    decoder_heads: int = 6
    core_hidden_size: int = 64
    core_hidden_layers: int = 2
    visual_warmup_steps: int = 800
    visual_warmup_batch_size: int = 32
    dynamics_steps: int = 5_000
    dynamics_batch_size: int = 8
    learning_rate: float = 3e-4
    backbone_learning_rate: float = 3e-5
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
    eval_encoder_batch_size: int = 16
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


@torch.no_grad()
def collect_end_to_end_suite(
    model_config: Any,
    *,
    trajectories: int,
    transitions: int,
    seed: int,
    family: str,
    include_world_states: bool,
) -> dict[str, torch.Tensor]:
    """Collect raw pixel histories and actions; labels are audit-only."""

    if family not in {"policy", "cardinal", "diagonal", "reversal"}:
        raise ValueError(f"unknown family {family!r}")
    history = model_config.history_frames
    endpoints = transitions + 1
    context_sequences: list[torch.Tensor] = []
    frame_sequences: list[torch.Tensor] = []
    action_sequences: list[torch.Tensor] = []
    state_sequences: list[torch.Tensor] = []
    started = time.perf_counter()
    for trajectory in range(trajectories):
        clip_seed = seed + trajectory * 9_973
        arguments = {
            "context_frames": 1,
            "future_frames": history + transitions - 1,
            "image_size": model_config.image_size,
        }
        clip = (
            make_clip(clip_seed, **arguments)
            if family == "policy"
            else make_excitation_clip(clip_seed, action_family=family, **arguments)
        )
        classes = _rgb_frames_to_classes_cpu(clip["frames"])
        context_sequences.append(
            torch.stack([classes[offset : offset + history] for offset in range(endpoints)])
        )
        frame_sequences.append(classes[history - 1 : history - 1 + endpoints])
        all_actions = torch.from_numpy(clip["all_actions"]).long()
        action_sequences.append(
            torch.stack([all_actions[offset + history] for offset in range(transitions)])
        )
        if include_world_states:
            all_states = torch.from_numpy(clip["all_state"]).float()
            state_sequences.append(
                torch.stack([all_states[offset + history - 1] for offset in range(endpoints)])
            )
        if (trajectory + 1) % 256 == 0 or trajectory + 1 == trajectories:
            print(
                json.dumps(
                    {
                        "stage": "collect_end_to_end_ph",
                        "family": family,
                        "auditLabels": include_world_states,
                        "trajectories": trajectory + 1,
                        "total": trajectories,
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    result = {
        "pixelContexts": torch.stack(context_sequences),
        "frames": torch.stack(frame_sequences),
        "actions": torch.stack(action_sequences),
    }
    result["actionVectors"] = action_vectors(result["actions"])
    if include_world_states:
        result["worldStates"] = torch.stack(state_sequences)
    return result


def _concatenate_training_suites(
    *suites: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    required = {"pixelContexts", "frames", "actions", "actionVectors"}
    if any(set(suite) != required for suite in suites):
        raise AssertionError(
            "end-to-end training suites must contain raw pixels and actions only"
        )
    return {name: torch.cat([suite[name] for suite in suites]) for name in required}


class TransformerStateEncoder(nn.Module):
    """Trainable pixel transformer followed by an object-agnostic 8-D readout."""

    def __init__(
        self,
        backbone: DirectPixelTransformer,
        state_size: int,
        hidden_size: int,
        *,
        eval_batch_size: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.state_size = state_size
        self.eval_batch_size = eval_batch_size
        token_size = backbone.config.hidden_size
        self.pool_score = nn.Linear(token_size, 1)
        self.readout = nn.Sequential(
            nn.LayerNorm(3 * token_size),
            nn.Linear(3 * token_size, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, state_size),
        )
        nn.init.normal_(self.readout[-1].weight, std=0.03)
        nn.init.zeros_(self.readout[-1].bias)

    def _forward_flat(self, contexts: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        tokens = (
            model.patch_projection(model.patch_tokens(contexts))
            + model.spatial_position
            + model.temporal_position[:, : contexts.shape[1]]
        )
        for block in model.blocks:
            tokens = block(tokens)
        latest = tokens[:, -1]
        attention = self.pool_score(latest).squeeze(-1).softmax(dim=-1)
        attended = torch.einsum("bp,bph->bh", attention, latest)
        features = torch.cat(
            (
                latest.mean(dim=1),
                latest.std(dim=1, unbiased=False),
                attended,
            ),
            dim=-1,
        )
        return self.readout(features.float())

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        if contexts.shape[-3:] != (
            self.backbone.config.history_frames,
            self.backbone.config.image_size,
            self.backbone.config.image_size,
        ):
            raise ValueError("pixel context shape does not match transformer configuration")
        leading = contexts.shape[:-3]
        flat = contexts.reshape(-1, *contexts.shape[-3:])
        if not self.training and flat.shape[0] > self.eval_batch_size:
            states = [
                self._forward_flat(flat[start : start + self.eval_batch_size])
                for start in range(0, flat.shape[0], self.eval_batch_size)
            ]
            output = torch.cat(states)
        else:
            output = self._forward_flat(flat)
        return output.reshape(*leading, self.state_size)


class LatentPatchTransformerRenderer(nn.Module):
    """Decode the pH state through learned spatial tokens and transformer blocks."""

    def __init__(
        self,
        state_size: int,
        *,
        image_size: int,
        patch_size: int,
        palette_size: int,
        hidden_size: int,
        depth: int,
        heads: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.state_size = state_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.palette_size = palette_size
        self.grid_size = image_size // patch_size
        self.latent_projection = nn.Linear(state_size, hidden_size)
        self.spatial_position = nn.Parameter(
            torch.randn(1, 1, self.grid_size**2, hidden_size) * 0.02
        )
        self.blocks = nn.ModuleList(
            DirectFactorizedBlock(hidden_size, heads, 4.0) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(
            hidden_size, patch_size**2 * palette_size
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        leading = state.shape[:-1]
        flat = state.reshape(-1, self.state_size)
        tokens = self.latent_projection(flat)[:, None, None] + self.spatial_position
        for block in self.blocks:
            tokens = block(tokens)
        logits = self.output_projection(self.output_norm(tokens)).reshape(
            flat.shape[0],
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            self.palette_size,
        )
        logits = logits.permute(0, 5, 1, 3, 2, 4).reshape(
            flat.shape[0], self.palette_size, self.image_size, self.image_size
        )
        return logits.reshape(
            *leading, self.palette_size, self.image_size, self.image_size
        )


class EndToEndDynamicsBranch(nn.Module):
    def __init__(
        self,
        encoder: TransformerStateEncoder,
        renderer: LatentPatchTransformerRenderer,
        *,
        core_config: NeuralPortHamiltonianConfig,
        structured: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.renderer = renderer
        self.structured = structured
        self.core: NeuralPortHamiltonian | NeuralODE = (
            NeuralPortHamiltonian(core_config)
            if structured
            else NeuralODE(core_config)
        )

    def encode(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.encoder(contexts)

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.core(state, action)


class _VisualBottleneck(nn.Module):
    def __init__(
        self,
        encoder: TransformerStateEncoder,
        renderer: LatentPatchTransformerRenderer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.renderer = renderer


def end_to_end_branch_loss(
    branch: EndToEndDynamicsBranch,
    contexts: torch.Tensor,
    frames: torch.Tensor,
    actions: torch.Tensor,
    anchor_horizons: torch.Tensor,
    class_weights: torch.Tensor,
    config: EndToEndPHConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pixel/action-only loss backpropagated through transformer, pH core and decoder."""

    batch = contexts.shape[0]
    rows = torch.arange(batch, device=contexts.device)
    initial = branch.encode(contexts[:, 0])
    anchored = branch.encode(contexts[rows, anchor_horizons])
    reconstruction_logits = branch.renderer(torch.stack((initial, anchored), dim=1))
    reconstruction_targets = torch.stack(
        (frames[:, 0], frames[rows, anchor_horizons]), dim=1
    )
    reconstruction = _pixel_cross_entropy(
        reconstruction_logits, reconstruction_targets, class_weights
    )

    current = initial
    rollout_states = []
    for step in range(actions.shape[1]):
        current = branch.step(current, actions[:, step])
        rollout_states.append(current)
    rollout = torch.stack(rollout_states, dim=1)
    anchored_prediction = rollout[rows, anchor_horizons - 1]
    teacher_latent = (anchored_prediction - anchored.detach()).square().mean()
    horizon_weights = anchor_horizons.float() / actions.shape[1] + 1.0
    rollout_latent = (
        (anchored_prediction - anchored.detach()).square().mean(dim=-1) * horizon_weights
    ).mean()

    horizon_indices = torch.tensor(
        [index for index in (0, 1, 3, 7) if index < actions.shape[1]],
        device=contexts.device,
    )
    rollout_logits = branch.renderer(rollout[:, horizon_indices])
    rollout_pixel = _pixel_cross_entropy(
        rollout_logits, frames[:, horizon_indices + 1], class_weights
    )

    shuffled_actions = torch.roll(actions, shifts=1, dims=0)
    wrong = initial
    for step in range(actions.shape[1]):
        wrong = branch.step(wrong, shuffled_actions[:, step])
    correct_final_ce = _pixel_cross_entropy(
        branch.renderer(rollout[:, -1]), frames[:, -1], class_weights
    )
    wrong_final_ce = _pixel_cross_entropy(
        branch.renderer(wrong), frames[:, -1], class_weights
    )
    action_contrast = F.relu(
        config.action_contrast_margin + correct_final_ce - wrong_final_ce
    )
    whitening = _whitening_loss(torch.cat((initial, anchored), dim=0))
    energy_gauge = (
        _energy_gauge(branch.core, initial)
        if isinstance(branch.core, NeuralPortHamiltonian)
        else initial.square().mean() * 0.0
    )
    total = (
        config.reconstruction_weight * reconstruction
        + config.teacher_latent_weight * teacher_latent
        + config.rollout_latent_weight * rollout_latent
        + config.rollout_pixel_weight * rollout_pixel
        + config.action_contrast_weight * action_contrast
        + config.whitening_weight * whitening
        + config.energy_gauge_weight * energy_gauge
    )
    return total, {
        "reconstruction": reconstruction,
        "teacherLatent": teacher_latent,
        "rolloutLatent": rollout_latent,
        "rolloutPixel": rollout_pixel,
        "actionContrast": action_contrast,
        "whitening": whitening,
        "energyGauge": energy_gauge,
    }


def _optimizer_for(
    module: nn.Module,
    config: EndToEndPHConfig,
) -> torch.optim.Optimizer:
    backbone = module.encoder.backbone  # type: ignore[attr-defined]
    backbone_ids = {id(parameter) for parameter in backbone.parameters()}
    backbone_parameters = [
        parameter for parameter in module.parameters() if id(parameter) in backbone_ids
    ]
    new_parameters = [
        parameter for parameter in module.parameters() if id(parameter) not in backbone_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": config.backbone_learning_rate,
                "lrScale": config.backbone_learning_rate / config.learning_rate,
            },
            {"params": new_parameters, "lr": config.learning_rate, "lrScale": 1.0},
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: int,
    steps: int,
    config: EndToEndPHConfig,
) -> float:
    multiplier = _learning_rate_multiplier(step, steps, config)  # type: ignore[arg-type]
    learning_rate = config.learning_rate * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * float(group["lrScale"])
    return learning_rate


def _decision(evaluation: dict[str, Any]) -> dict[str, Any]:
    ph = evaluation["policy"]["endToEndPortHamiltonian"]
    control = evaluation["policy"]["endToEndNeuralOde"]
    structure = ph["structure"]
    ph_h4 = ph["rolloutByHorizon"]["4"]["pixels"]["playerCentroidErrorPx"]
    control_h4 = control["rolloutByHorizon"]["4"]["pixels"]["playerCentroidErrorPx"]
    gates = {
        "pixelReconstructionPlayerIouAtLeast0.70": ph["currentFrameReconstruction"]["playerIou"] >= 0.70,
        "pixelReconstructionPuckIouAtLeast0.50": ph["currentFrameReconstruction"]["puckIou"] >= 0.50,
        "unsupervisedPositionDiscoveryQ_R2AtLeast0.80": ph["postHocPhysicalProbe"]["qR2"] >= 0.80,
        "unsupervisedMomentumDiscoveryP_R2AtLeast0.50": ph["postHocPhysicalProbe"]["pR2"] >= 0.50,
        "alignedPortPhysicalCosineAtLeast0.80": structure["portAfterPostHocPhysicalAlignment"]["physicalIncidenceCosine"] >= 0.80,
        "counterfactualMomentumCosineAtLeast0.80": ph["oneStepCounterfactualAction"]["alignedPlayerMomentumCosine"] >= 0.80,
        "renderedCounterfactualSignAtLeast0.80": ph["oneStepCounterfactualAction"]["renderedTargetSignAgreement"] >= 0.80,
        "usesActionsShuffledDegradationAtLeast0.05": ph["actionControls"]["shuffledRelativeDegradation"] >= 0.05,
        "closedLoopRealImprovementVsCoastAtLeast0.20": ph["closedLoopPixelTargetControl"]["realImprovementVsCoast"] >= 0.20,
        "closedLoopBeatsCoastFractionAtLeast0.65": ph["closedLoopPixelTargetControl"]["beatsCoastFraction"] >= 0.65,
        "pHPredictiveParityAtH4": ph_h4 <= 1.15 * control_h4,
        "exactContinuousPowerBalance": structure["powerBalance"]["maxAbsDefect"] <= 1e-5,
        "zeroInputEnergyMonotone": structure["zeroInputDiscreteEnergy"]["increaseFraction"] <= 1e-3,
    }
    supported = all(gates.values())
    return {
        "outcome": (
            "provisional_end_to_end_breakthrough_supported_single_seed"
            if supported
            else "provisional_end_to_end_breakthrough_not_supported_single_seed"
        ),
        "allGatesPass": supported,
        "gates": gates,
        "scope": (
            "One training seed. The visual transformer, latent readout, H/J/R/B and "
            "transformer decoder are optimized jointly from pixels and actions. Physical "
            "states enter post-training audits only."
        ),
    }


def run_end_to_end_ph_experiment(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config: EndToEndPHConfig = EndToEndPHConfig(),
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
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = build_pixel_direct_from_checkpoint(payload)
    model_config = backbone.config

    collection_started = time.perf_counter()
    fit_policy = collect_end_to_end_suite(
        model_config,
        trajectories=config.fit_policy_trajectories,
        transitions=config.transitions_per_trajectory,
        seed=config.seed + 1_000_000,
        family="policy",
        include_world_states=False,
    )
    fit_cardinal = collect_end_to_end_suite(
        model_config,
        trajectories=config.fit_cardinal_trajectories,
        transitions=config.transitions_per_trajectory,
        seed=config.seed + 2_000_000,
        family="cardinal",
        include_world_states=False,
    )
    fit_cpu = _concatenate_training_suites(fit_policy, fit_cardinal)
    del fit_policy, fit_cardinal
    fit = _move_suite(fit_cpu, device)
    del fit_cpu
    collection_seconds = time.perf_counter() - collection_started
    class_weights = _class_weights(fit["frames"], model_config.palette_size).to(device)

    visual = _VisualBottleneck(
        TransformerStateEncoder(
            backbone.to(device),
            config.state_size,
            config.encoder_hidden_size,
            eval_batch_size=config.eval_encoder_batch_size,
        ),
        LatentPatchTransformerRenderer(
            config.state_size,
            image_size=model_config.image_size,
            patch_size=model_config.patch_size,
            palette_size=model_config.palette_size,
            hidden_size=config.decoder_hidden_size,
            depth=config.decoder_depth,
            heads=config.decoder_heads,
        ).to(device),
    ).to(device)
    visual_optimizer = _optimizer_for(visual, config)
    visual_log = output_dir / "visual-warmup.jsonl"
    visual_started = time.perf_counter()
    visual.train()
    with visual_log.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.visual_warmup_steps + 1):
            rows = torch.randint(
                0, config.fit_trajectories,
                (config.visual_warmup_batch_size,),
                device=device,
            )
            endpoints = torch.randint(
                0, config.transitions_per_trajectory + 1,
                (config.visual_warmup_batch_size,),
                device=device,
            )
            learning_rate = _set_learning_rate(
                visual_optimizer, step, config.visual_warmup_steps, config
            )
            visual_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                state = visual.encoder(fit["pixelContexts"][rows, endpoints])
                logits = visual.renderer(state)
                reconstruction = _pixel_cross_entropy(
                    logits, fit["frames"][rows, endpoints], class_weights
                )
                whitening = _whitening_loss(state)
                loss = reconstruction + config.whitening_weight * whitening
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(visual.parameters(), 5.0)
            visual_optimizer.step()
            if step == 1 or step % config.log_every == 0 or step == config.visual_warmup_steps:
                elapsed = time.perf_counter() - visual_started
                record = {
                    "stage": "end_to_end_visual_warmup",
                    "step": step,
                    "steps": config.visual_warmup_steps,
                    "loss": float(loss.detach()),
                    "reconstruction": float(reconstruction.detach()),
                    "whitening": float(whitening.detach()),
                    "gradientNorm": float(gradient_norm),
                    "learningRate": learning_rate,
                    "stepsPerSecond": step / max(elapsed, 1e-8),
                    "estimatedSeconds": elapsed / step * config.visual_warmup_steps,
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
    visual_seconds = time.perf_counter() - visual_started

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
    ph = EndToEndDynamicsBranch(
        copy.deepcopy(visual.encoder),
        copy.deepcopy(visual.renderer),
        core_config=core_config,
        structured=True,
    ).to(device)
    control_hidden = _matched_control_hidden_size(_parameter_count(ph.core), core_config)
    control = EndToEndDynamicsBranch(
        copy.deepcopy(visual.encoder),
        copy.deepcopy(visual.renderer),
        core_config=replace(core_config, hidden_size=control_hidden),
        structured=False,
    ).to(device)
    branches = {
        "endToEndPortHamiltonian": ph,
        "endToEndNeuralOde": control,
    }
    ph_initial_core = {
        name: value.detach().cpu().clone() for name, value in ph.core.state_dict().items()
    }
    capacity_gap = abs(_parameter_count(ph.core) - _parameter_count(control.core))
    capacity_gap /= max(_parameter_count(ph.core), 1)
    if capacity_gap > 0.01:
        raise AssertionError("pH and Neural ODE core capacity must match within one percent")
    optimizers = {name: _optimizer_for(branch, config) for name, branch in branches.items()}
    del visual, visual_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dynamics_started = time.perf_counter()
    dynamics_log = output_dir / "joint-training.jsonl"
    with dynamics_log.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.dynamics_steps + 1):
            rows = torch.randint(
                0, config.fit_trajectories,
                (config.dynamics_batch_size,),
                device=device,
            )
            anchor_horizons = torch.randint(
                1,
                config.transitions_per_trajectory + 1,
                (config.dynamics_batch_size,),
                device=device,
            )
            losses: dict[str, torch.Tensor] = {}
            terms: dict[str, dict[str, torch.Tensor]] = {}
            gradient_norms: dict[str, float] = {}
            learning_rate = 0.0
            for name, branch in branches.items():
                branch.train()
                optimizer = optimizers[name]
                learning_rate = _set_learning_rate(
                    optimizer, step, config.dynamics_steps, config
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    loss, branch_terms = end_to_end_branch_loss(
                        branch,
                        fit["pixelContexts"][rows],
                        fit["frames"][rows],
                        fit["actionVectors"][rows],
                        anchor_horizons,
                        class_weights,
                        config,
                    )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(branch.parameters(), 5.0)
                optimizer.step()
                losses[name] = loss.detach()
                terms[name] = {key: value.detach() for key, value in branch_terms.items()}
                gradient_norms[name] = float(gradient_norm)
            if step == 1 or step % config.log_every == 0 or step == config.dynamics_steps:
                elapsed = time.perf_counter() - dynamics_started
                record = {
                    "stage": "train_end_to_end_ph_world_model",
                    "step": step,
                    "steps": config.dynamics_steps,
                    "learningRate": learning_rate,
                    "stepsPerSecond": step / max(elapsed, 1e-8),
                    "estimatedSeconds": elapsed / step * config.dynamics_steps,
                    **{
                        name: {
                            "loss": float(losses[name]),
                            "gradientNorm": gradient_norms[name],
                            **{key: float(value) for key, value in terms[name].items()},
                        }
                        for name in branches
                    },
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
    dynamics_seconds = time.perf_counter() - dynamics_started
    del fit, optimizers
    if device.type == "cuda":
        torch.cuda.empty_cache()

    evaluation_collection_started = time.perf_counter()
    suite_specs = {
        "policy": (config.test_policy_trajectories, "policy", config.seed + 3_000_000),
        "diagonalOod": (config.test_diagonal_trajectories, "diagonal", config.seed + 4_000_000),
        "reversalOod": (config.test_reversal_trajectories, "reversal", config.seed + 5_000_000),
    }
    evaluation_suites = {
        name: _move_suite(
            collect_end_to_end_suite(
                model_config,
                trajectories=count,
                transitions=config.transitions_per_trajectory,
                seed=seed,
                family=family,
                include_world_states=True,
            ),
            device,
        )
        for name, (count, family, seed) in suite_specs.items()
    }
    audit = _move_suite(
        collect_end_to_end_suite(
            model_config,
            trajectories=config.audit_trajectories,
            transitions=config.transitions_per_trajectory,
            seed=config.seed + 6_000_000,
            family="policy",
            include_world_states=True,
        ),
        device,
    )
    evaluation_collection_seconds = time.perf_counter() - evaluation_collection_started
    evaluation_started = time.perf_counter()
    evaluation: dict[str, Any] = {}
    for suite_name, suite in evaluation_suites.items():
        evaluation[suite_name] = {
            name: _evaluate_branch(
                branch,
                suite,
                audit,
                class_weights,
                config,  # type: ignore[arg-type]
                full_audit=suite_name == "policy",
            )
            for name, branch in branches.items()
        }
    evaluation_seconds = time.perf_counter() - evaluation_started
    decision = _decision(evaluation)

    checkpoint = {
        "kind": "end_to_end_port_hamiltonian_pixel_world_model",
        "seed": config.seed,
        "baseCheckpoint": str(checkpoint_path),
        "modelConfig": model_config.to_dict(),
        "experimentConfig": asdict(config),
        "branches": {name: branch.state_dict() for name, branch in branches.items()},
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    summary = {
        "kind": checkpoint["kind"],
        "config": asdict(config),
        "modelConfig": model_config.to_dict(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": config.seed,
        "trainingDataAudit": {
            "optimizationTensorKeys": ["pixelContexts", "frames", "actions", "actionVectors"],
            "physicalLabelsUsedForOptimization": False,
            "transformerInitializedFromPixelOnlyCheckpoint": True,
            "transformerFrozen": False,
            "allTransformerBlocksTrainable": True,
            "rolloutPixelLossBackpropagatesThroughPHCoreAndTransformerEncoder": True,
            "physicalStatesCollectedOnlyAfterTraining": True,
        },
        "architecture": {
            "visualEncoder": "trainable_direct_pixel_transformer_with_attention_pooling",
            "stateSize": config.state_size,
            "structuredCore": "state_dependent_H_J_R_B",
            "visualDecoder": "latent_conditioned_patch_transformer",
            "baseline": "parameter_matched_neural_ode_with_independent_identical_encoder_decoder_copy",
        },
        "capacity": {
            "portHamiltonianCoreParameters": _parameter_count(ph.core),
            "neuralOdeCoreParameters": _parameter_count(control.core),
            "relativeCoreParameterGap": capacity_gap,
            "portHamiltonianCompleteParameters": _parameter_count(ph),
            "neuralOdeCompleteParameters": _parameter_count(control),
        },
        "parameterChangeFromInitialization": _parameter_change_payload(ph, ph_initial_core),
        "timing": {
            "fitCollectionSeconds": collection_seconds,
            "visualWarmupSeconds": visual_seconds,
            "jointTrainingSeconds": dynamics_seconds,
            "evaluationCollectionSeconds": evaluation_collection_seconds,
            "evaluationSeconds": evaluation_seconds,
            "totalMeasuredSeconds": (
                collection_seconds
                + visual_seconds
                + dynamics_seconds
                + evaluation_collection_seconds
                + evaluation_seconds
            ),
        },
        "evaluation": evaluation,
        "decision": decision,
        "artifacts": [
            "checkpoint.pt",
            "visual-warmup.jsonl",
            "joint-training.jsonl",
            "summary.json",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=131_610_731)
    parser.add_argument("--fit-policy-trajectories", type=int, default=3_072)
    parser.add_argument("--fit-cardinal-trajectories", type=int, default=3_072)
    parser.add_argument("--test-policy-trajectories", type=int, default=512)
    parser.add_argument("--test-diagonal-trajectories", type=int, default=256)
    parser.add_argument("--test-reversal-trajectories", type=int, default=256)
    parser.add_argument("--audit-trajectories", type=int, default=1_024)
    parser.add_argument("--visual-warmup-steps", type=int, default=800)
    parser.add_argument("--visual-warmup-batch-size", type=int, default=32)
    parser.add_argument("--dynamics-steps", type=int, default=5_000)
    parser.add_argument("--dynamics-batch-size", type=int, default=8)
    parser.add_argument("--planner-samples", type=int, default=64)
    parser.add_argument("--planner-steps", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    run_end_to_end_ph_experiment(
        args.checkpoint,
        args.output,
        config=EndToEndPHConfig(
            fit_policy_trajectories=args.fit_policy_trajectories,
            fit_cardinal_trajectories=args.fit_cardinal_trajectories,
            test_policy_trajectories=args.test_policy_trajectories,
            test_diagonal_trajectories=args.test_diagonal_trajectories,
            test_reversal_trajectories=args.test_reversal_trajectories,
            audit_trajectories=args.audit_trajectories,
            visual_warmup_steps=args.visual_warmup_steps,
            visual_warmup_batch_size=args.visual_warmup_batch_size,
            dynamics_steps=args.dynamics_steps,
            dynamics_batch_size=args.dynamics_batch_size,
            planner_samples=args.planner_samples,
            planner_steps=args.planner_steps,
            log_every=args.log_every,
            seed=args.seed,
        ),
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
