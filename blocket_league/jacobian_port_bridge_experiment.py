from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import make_clip, make_excitation_clip
from .env import WorldConfig
from .neural_ph_experiment import NeuralPHBranch
from .neural_port_hamiltonian import NeuralPortHamiltonian
from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .pixel_probe import PLAYER_CLASSES, _soft_centroid, _visual_centroid
from .port_hamiltonian_audit import _entity_token
from .port_hamiltonian_bottleneck import _block5_entity_features
from .position_write_probe import PUCK_CLASSES
from .train_pixel_direct import frames_to_classes, palette_tensor


AXES = ("x", "y")


@dataclass(frozen=True)
class JacobianPortBridgeConfig:
    lens_fit_samples: int = 512
    test_policy_samples: int = 256
    test_cardinal_samples: int = 256
    batch_size: int = 32
    block_count: int = 5
    intervention_strengths: tuple[float, ...] = (1.0, 4.0, 8.0)
    primary_strength: float = 8.0
    random_direction_pairs: int = 16
    seed: int = 121_610_731

    @property
    def test_samples(self) -> int:
        return self.test_policy_samples + self.test_cardinal_samples


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_branch(checkpoint_path: Path, device: torch.device) -> NeuralPHBranch:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "state_dependent_neural_port_hamiltonian":
        raise ValueError("checkpoint is not a state-dependent neural pH experiment")
    config = payload["config"]
    state = payload["branches"]["neuralPortHamiltonian"]
    branch = NeuralPHBranch(
        state["feature_mean"],
        state["feature_scale"],
        state["state_mean"],
        state["state_scale"],
        hidden_size=int(config["hidden_size"]),
        hidden_layers=int(config["hidden_layers"]),
        integration_method=str(config["integration_method"]),
        integration_substeps=int(config["integration_substeps"]),
        resistance_floor=float(config["resistance_floor"]),
        structured=True,
    )
    branch.load_state_dict(state)
    return branch.to(device).eval().requires_grad_(False)


def encoder_jacobian(branch: NeuralPHBranch) -> torch.Tensor:
    """Return the exact, constant Jacobian dE(h)/dh of the affine readout."""

    return (
        branch.state_scale[:, None]
        * branch.encoder.weight
        / branch.feature_scale[None, :]
    )


def _rgb_to_classes(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    video = torch.from_numpy(frames.copy()).permute(0, 1, 4, 2, 3)
    video = video.to(device, non_blocking=True).float().div(127.5).sub(1.0)
    return frames_to_classes(video, palette_tensor(device))


def _context_batch(
    seeds: list[int],
    model: nn.Module,
    device: torch.device,
    *,
    family: str,
) -> torch.Tensor:
    contexts = []
    for seed in seeds:
        arguments = {
            "context_frames": model.config.history_frames,
            "future_frames": 1,
            "image_size": model.config.image_size,
        }
        if family == "policy":
            clip = make_clip(seed, **arguments)
        elif family == "cardinal":
            clip = make_excitation_clip(seed, action_family="cardinal", **arguments)
        else:
            raise ValueError(f"unknown context family {family!r}")
        contexts.append(clip["context"])
    return _rgb_to_classes(np.stack(contexts), device)


def _entity_tokens(model: nn.Module, classes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    player = _visual_centroid(classes[:, -1], PLAYER_CLASSES)
    puck = _visual_centroid(classes[:, -1], PUCK_CLASSES)
    return _entity_token(model, player), _entity_token(model, puck)


def _player_mask(model: nn.Module, classes: torch.Tensor) -> torch.Tensor:
    player, _ = _entity_tokens(model, classes)
    mask = torch.zeros(
        classes.shape[0],
        classes.shape[1],
        model.config.grid_size**2,
        device=classes.device,
    )
    mask[torch.arange(classes.shape[0], device=classes.device), -1, player] = 1.0
    return mask


def _downstream_gradients(
    model: nn.Module,
    classes: torch.Tensor,
    *,
    block_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-example d(next rendered player x,y)/d(player-token write)."""

    write = torch.zeros(
        classes.shape[0], model.config.hidden_size, device=classes.device,
        requires_grad=True,
    )
    logits = model(
        classes,
        intervention_block=block_index,
        intervention=write,
        intervention_mask=_player_mask(model, classes),
    )[:, -1]
    position = _soft_centroid(logits, PLAYER_CLASSES)
    gradients = []
    for axis in range(2):
        gradients.append(
            torch.autograd.grad(
                position[:, axis].sum(), write, retain_graph=axis == 0,
            )[0]
        )
    return torch.stack(gradients, dim=1).detach(), position.detach()


def _unit(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return values / torch.linalg.vector_norm(values, dim=dim, keepdim=True).clamp_min(1e-12)


def _fit_global_lens(
    model: nn.Module,
    *,
    samples: int,
    batch_size: int,
    seed: int,
    block_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gradient_sum = torch.zeros(2, model.config.hidden_size, device=device)
    norm_chunks = []
    started = time.perf_counter()
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        seeds = [seed + (start + index) * 9_973 for index in range(count)]
        classes = _context_batch(seeds, model, device, family="policy")
        gradients, _ = _downstream_gradients(
            model, classes, block_index=block_index,
        )
        gradient_sum += gradients.sum(dim=0)
        norm_chunks.append(torch.linalg.vector_norm(gradients, dim=-1).cpu())
    raw = gradient_sum / samples
    directions = _unit(raw)
    norms = torch.cat(norm_chunks)
    return directions, {
        "samples": samples,
        "source": "policy contexts disjoint from bridge evaluation",
        "method": "mean downstream gradient of next soft player centroid",
        "directionCosine": float(torch.dot(directions[0], directions[1])),
        "meanLocalGradientNorm": {
            axis: float(norms[:, index].mean()) for index, axis in enumerate(AXES)
        },
        "seconds": time.perf_counter() - started,
    }


def lift_player_write(
    direction: torch.Tensor,
    shared_token: torch.Tensor,
) -> torch.Tensor:
    """Lift a player-token write into the concatenated player/puck readout.

    If both rendered entities occupy the same patch, the actual transformer
    write changes both halves of the entity-pair feature.  Keeping this case
    exact avoids claiming a player-only perturbation at overlap states.
    """

    if direction.ndim == 2:
        direction = direction[None].expand(shared_token.shape[0], -1, -1)
    player = direction
    puck = direction * shared_token[:, None, None].to(direction.dtype)
    return torch.cat((player, puck), dim=-1)


def _cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    numerator = (first * second).sum(dim=-1)
    denominator = (
        torch.linalg.vector_norm(first, dim=-1)
        * torch.linalg.vector_norm(second, dim=-1)
    ).clamp_min(1e-12)
    return numerator / denominator


def _sample_summary(values: torch.Tensor) -> dict[str, float | int]:
    flat = values.detach().float().reshape(-1).cpu()
    count = flat.numel()
    mean = float(flat.mean())
    std = float(flat.std(unbiased=count > 1)) if count > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(max(count, 1))
    return {
        "mean": mean,
        "std": std,
        "sampleCi95Low": mean - half_width,
        "sampleCi95High": mean + half_width,
        "positiveFraction": float((flat > 0).float().mean()),
        "samples": count,
    }


def _alignment_summary(
    effect: torch.Tensor,
    port: torch.Tensor,
    *,
    target_coordinates: tuple[int, int],
) -> dict[str, Any]:
    """Summarize [sample, action-axis, state] effect/port alignment."""

    matched = _cosine(effect, port)
    swapped = _cosine(effect, port.flip(dims=(1,))).abs()
    specificity = matched - swapped
    return {
        "matchedCosine": {
            "pooled": _sample_summary(matched),
            **{
                axis: _sample_summary(matched[:, index])
                for index, axis in enumerate(AXES)
            },
        },
        "absoluteSwappedAxisCosine": _sample_summary(swapped),
        "matchedMinusAbsoluteSwapped": _sample_summary(specificity),
        "targetCoordinateSignFraction": float(
            torch.stack(
                (
                    effect[:, 0, target_coordinates[0]],
                    effect[:, 1, target_coordinates[1]],
                ),
                dim=1,
            )
            .gt(0)
            .float()
            .mean()
        ),
        "meanEffectNorm": float(torch.linalg.vector_norm(effect, dim=-1).mean()),
        "meanPortNorm": float(torch.linalg.vector_norm(port, dim=-1).mean()),
    }


def _multi_view_alignment(
    effect: torch.Tensor,
    port: torch.Tensor,
    state_scale: torch.Tensor,
) -> dict[str, Any]:
    return {
        "fullPhysicalState": _alignment_summary(
            effect, port, target_coordinates=(4, 5)
        ),
        "standardizedState": _alignment_summary(
            effect / state_scale[None, None, :],
            port / state_scale[None, None, :],
            target_coordinates=(4, 5),
        ),
        "playerPosition": _alignment_summary(
            effect[..., 0:2], port[..., 0:2], target_coordinates=(0, 1)
        ),
        "playerMomentum": _alignment_summary(
            effect[..., 4:6], port[..., 4:6], target_coordinates=(0, 1)
        ),
        "puckMomentum": _alignment_summary(
            effect[..., 6:8], port[..., 6:8], target_coordinates=(0, 1)
        ),
    }


def _random_directions(
    pairs: int,
    hidden_size: int,
    reference: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    values = torch.randn(pairs * 2, hidden_size, generator=generator, device=device)
    basis = torch.linalg.qr(reference.T, mode="reduced").Q
    values = values - (values @ basis) @ basis.T
    values = _unit(values)
    return values.reshape(pairs, 2, hidden_size)


@torch.no_grad()
def _port_effects(
    branch: NeuralPHBranch,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(branch.core, NeuralPortHamiltonian):
        raise TypeError("D1 requires the structured neural pH branch")
    port = branch.core.port(state)
    instantaneous = WorldConfig().dt * port.transpose(1, 2)
    axes = torch.eye(2, device=state.device, dtype=state.dtype)
    integrated = []
    for axis in range(2):
        action = axes[axis][None].expand(state.shape[0], -1)
        plus = branch.core.integrate(state, action)
        minus = branch.core.integrate(state, -action)
        integrated.append(0.5 * (plus - minus))
    return instantaneous, torch.stack(integrated, dim=1)


@torch.no_grad()
def _next_logits(
    model: nn.Module,
    classes: torch.Tensor,
    *,
    block_index: int,
    direction: torch.Tensor | None = None,
    strength: float = 0.0,
) -> torch.Tensor:
    if direction is None:
        return model(classes)[:, -1]
    return model(
        classes,
        intervention_block=block_index,
        intervention=direction * strength,
        intervention_mask=_player_mask(model, classes),
    )[:, -1]


@torch.no_grad()
def _reencoded_next_state(
    model: nn.Module,
    branch: NeuralPHBranch,
    classes: torch.Tensor,
    next_classes: torch.Tensor,
    *,
    block_count: int,
) -> torch.Tensor:
    history = torch.cat((classes[:, 1:], next_classes[:, None]), dim=1)
    features = _block5_entity_features(model, history, block_count)
    return branch.encode(features)[:, :8]


def _finite_write_batch(
    model: nn.Module,
    branch: NeuralPHBranch,
    classes: torch.Tensor,
    directions: torch.Tensor,
    *,
    strength: float,
    block_index: int,
    block_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_effects = []
    soft_pixel_effects = []
    hard_pixel_effects = []
    for axis in range(2):
        plus_logits = _next_logits(
            model, classes, block_index=block_index,
            direction=directions[axis], strength=strength,
        )
        minus_logits = _next_logits(
            model, classes, block_index=block_index,
            direction=directions[axis], strength=-strength,
        )
        plus_classes = plus_logits.argmax(dim=1)
        minus_classes = minus_logits.argmax(dim=1)
        plus_state = _reencoded_next_state(
            model, branch, classes, plus_classes, block_count=block_count,
        )
        minus_state = _reencoded_next_state(
            model, branch, classes, minus_classes, block_count=block_count,
        )
        state_effects.append(0.5 * (plus_state - minus_state))
        soft_pixel_effects.append(
            0.5
            * (
                _soft_centroid(plus_logits, PLAYER_CLASSES)
                - _soft_centroid(minus_logits, PLAYER_CLASSES)
            )
        )
        hard_pixel_effects.append(
            0.5
            * (
                _visual_centroid(plus_classes, PLAYER_CLASSES)
                - _visual_centroid(minus_classes, PLAYER_CLASSES)
            )
        )
    return (
        torch.stack(state_effects, dim=1),
        torch.stack(soft_pixel_effects, dim=1),
        torch.stack(hard_pixel_effects, dim=1),
    )


def _pixel_effect_summary(effect: torch.Tensor) -> dict[str, Any]:
    target = torch.stack((effect[:, 0, 0], effect[:, 1, 1]), dim=1)
    orthogonal = torch.stack((effect[:, 0, 1], effect[:, 1, 0]), dim=1)
    return {
        "targetAxisPixels": _sample_summary(target),
        "absoluteOrthogonalPixels": _sample_summary(orthogonal.abs()),
        "targetPositiveFraction": float((target > 0).float().mean()),
    }


def _decision(
    differential: dict[str, Any],
    finite: dict[str, Any],
    random_finite: dict[str, Any],
) -> dict[str, Any]:
    differential_momentum = differential["playerMomentum"]
    finite_momentum = finite["playerMomentum"]
    random_momentum = random_finite["playerMomentum"]
    differential_cosine = differential_momentum["matchedCosine"]["pooled"]["mean"]
    differential_specificity = differential_momentum[
        "matchedMinusAbsoluteSwapped"
    ]["mean"]
    finite_cosine = finite_momentum["matchedCosine"]["pooled"]["mean"]
    finite_specificity = finite_momentum["matchedMinusAbsoluteSwapped"]["mean"]
    finite_random_gap = finite_cosine - random_momentum[
        "matchedCosine"
    ]["pooled"]["mean"]
    gates = {
        "differentialMomentumCosineAtLeast0.50": differential_cosine >= 0.50,
        "differentialAxisSpecificityAtLeast0.25": differential_specificity >= 0.25,
        "differentialTargetSignAtLeast0.75": differential_momentum[
            "targetCoordinateSignFraction"
        ] >= 0.75,
        "finiteMomentumCosineAtLeast0.30": finite_cosine >= 0.30,
        "finiteAxisSpecificityAtLeast0.15": finite_specificity >= 0.15,
        "finiteTargetSignAtLeast0.65": finite_momentum[
            "targetCoordinateSignFraction"
        ] >= 0.65,
        "finiteBeatsRandomBy0.20": finite_random_gap >= 0.20,
    }
    differential_pass = all(list(gates.values())[:3])
    finite_pass = all(list(gates.values())[3:])
    if differential_pass and finite_pass:
        outcome = "supported_single_seed"
        text = (
            "The native block-5 Jacobian lens and the learned pH action port "
            "form both an infinitesimal and a finite causal bridge on this seed."
        )
    elif differential_pass:
        outcome = "infinitesimal_only_single_seed"
        text = (
            "The readout differential aligns with the learned port, but the "
            "bridge does not survive the finite render-and-reencode intervention."
        )
    else:
        outcome = "not_supported_single_seed"
        text = (
            "The native block-5 Jacobian lens is not identified with the learned "
            "pH action port on this seed under the preregistered gates."
        )
    return {
        "outcome": outcome,
        "text": text,
        "gates": gates,
        "differentialPass": differential_pass,
        "finiteCausalPass": finite_pass,
        "scope": (
            "Provisional within-seed conclusion. Sample-level confidence intervals "
            "do not quantify training-seed uncertainty."
        ),
    }


def run_jacobian_port_bridge_experiment(
    base_checkpoint_path: Path,
    ph_checkpoint_path: Path,
    output_path: Path,
    *,
    config: JacobianPortBridgeConfig = JacobianPortBridgeConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    if config.block_count != 5:
        raise ValueError("D1 is preregistered at transformer block 5")
    if config.primary_strength not in config.intervention_strengths:
        raise ValueError("primary strength must be part of the intervention sweep")
    if min(config.lens_fit_samples, config.test_samples, config.batch_size) < 1:
        raise ValueError("sample and batch counts must be positive")
    _seed_everything(config.seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    base_payload = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    model = (
        build_pixel_direct_from_checkpoint(base_payload)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    branch = _load_branch(ph_checkpoint_path, device)
    if model.config.hidden_size * 2 != branch.feature_mean.numel():
        raise ValueError("pH readout and transformer hidden size are incompatible")
    block_index = config.block_count - 1
    started = time.perf_counter()
    global_directions, lens_protocol = _fit_global_lens(
        model,
        samples=config.lens_fit_samples,
        batch_size=config.batch_size,
        seed=config.seed + 1_000_000,
        block_index=block_index,
        device=device,
    )
    random_pairs = _random_directions(
        config.random_direction_pairs,
        model.config.hidden_size,
        global_directions,
        seed=config.seed + 7_000_000,
        device=device,
    )
    jacobian = encoder_jacobian(branch)[:8]
    state_scale = branch.state_scale[:8]

    collected: dict[str, list[torch.Tensor]] = {
        "globalDifferential": [],
        "localDifferential": [],
        "instantaneousPort": [],
        "integratedPort": [],
        "randomDifferential": [],
        "sharedToken": [],
    }
    finite_collected = {
        strength: {name: [] for name in ("state", "softPixels", "hardPixels")}
        for strength in config.intervention_strengths
    }
    random_finite_collected = {name: [] for name in ("state", "softPixels", "hardPixels")}
    suite_counts = {
        "policy": config.test_policy_samples,
        "cardinal": config.test_cardinal_samples,
    }
    cursor = 0
    for family, samples in suite_counts.items():
        family_seed = config.seed + (3_000_000 if family == "policy" else 4_000_000)
        for start in range(0, samples, config.batch_size):
            count = min(config.batch_size, samples - start)
            seeds = [family_seed + (start + index) * 9_973 for index in range(count)]
            classes = _context_batch(seeds, model, device, family=family)
            features = _block5_entity_features(model, classes, config.block_count)
            state = branch.encode(features)[:, :8]
            player_token, puck_token = _entity_tokens(model, classes)
            shared = player_token == puck_token
            with torch.enable_grad():
                local_gradients, _ = _downstream_gradients(
                    model, classes, block_index=block_index,
                )
            local_directions = _unit(local_gradients)
            global_lift = lift_player_write(global_directions, shared)
            local_lift = lift_player_write(local_directions, shared)
            global_effect = torch.einsum("of,baf->bao", jacobian, global_lift)
            local_effect = torch.einsum("of,baf->bao", jacobian, local_lift)
            instantaneous, integrated = _port_effects(branch, state)

            random_lifts = torch.stack(
                [lift_player_write(pair, shared) for pair in random_pairs], dim=1,
            )
            random_effect = torch.einsum("of,braf->brao", jacobian, random_lifts)
            collected["globalDifferential"].append(global_effect.cpu())
            collected["localDifferential"].append(local_effect.cpu())
            collected["instantaneousPort"].append(instantaneous.cpu())
            collected["integratedPort"].append(integrated.cpu())
            collected["randomDifferential"].append(random_effect.cpu())
            collected["sharedToken"].append(shared.cpu())

            for strength in config.intervention_strengths:
                state_effect, soft_effect, hard_effect = _finite_write_batch(
                    model,
                    branch,
                    classes,
                    global_directions,
                    strength=strength,
                    block_index=block_index,
                    block_count=config.block_count,
                )
                finite_collected[strength]["state"].append(state_effect.cpu())
                finite_collected[strength]["softPixels"].append(soft_effect.cpu())
                finite_collected[strength]["hardPixels"].append(hard_effect.cpu())

            primary_random = random_pairs[cursor % config.random_direction_pairs]
            state_effect, soft_effect, hard_effect = _finite_write_batch(
                model,
                branch,
                classes,
                primary_random,
                strength=config.primary_strength,
                block_index=block_index,
                block_count=config.block_count,
            )
            random_finite_collected["state"].append(state_effect.cpu())
            random_finite_collected["softPixels"].append(soft_effect.cpu())
            random_finite_collected["hardPixels"].append(hard_effect.cpu())
            cursor += 1
            print(
                json.dumps(
                    {
                        "stage": "jacobian_port_bridge",
                        "family": family,
                        "samples": min(start + count, samples),
                        "total": samples,
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    merged = {name: torch.cat(values) for name, values in collected.items()}
    instantaneous = merged["instantaneousPort"]
    integrated = merged["integratedPort"]
    global_differential = _multi_view_alignment(
        merged["globalDifferential"], instantaneous, state_scale.cpu(),
    )
    local_differential = _multi_view_alignment(
        merged["localDifferential"], instantaneous, state_scale.cpu(),
    )
    random_differential_values = merged["randomDifferential"]
    expanded_port = instantaneous[:, None].expand(
        -1, config.random_direction_pairs, -1, -1,
    )
    random_differential = _multi_view_alignment(
        random_differential_values.reshape(-1, 2, 8),
        expanded_port.reshape(-1, 2, 8),
        state_scale.cpu(),
    )

    finite = {}
    for strength, values in finite_collected.items():
        state_effect = torch.cat(values["state"])
        finite[str(strength)] = {
            "stateAlignment": _multi_view_alignment(
                state_effect, integrated, state_scale.cpu(),
            ),
            "softRenderedPlayerEffect": _pixel_effect_summary(
                torch.cat(values["softPixels"])
            ),
            "hardRenderedPlayerEffect": _pixel_effect_summary(
                torch.cat(values["hardPixels"])
            ),
        }
    random_finite_state = torch.cat(random_finite_collected["state"])
    random_finite = {
        "stateAlignment": _multi_view_alignment(
            random_finite_state, integrated, state_scale.cpu(),
        ),
        "softRenderedPlayerEffect": _pixel_effect_summary(
            torch.cat(random_finite_collected["softPixels"])
        ),
        "hardRenderedPlayerEffect": _pixel_effect_summary(
            torch.cat(random_finite_collected["hardPixels"])
        ),
    }
    primary_finite = finite[str(config.primary_strength)]["stateAlignment"]
    conclusion = _decision(
        global_differential,
        primary_finite,
        random_finite["stateAlignment"],
    )
    result = {
        "kind": "jacobian_lens_port_hamiltonian_bridge",
        "version": 1,
        "singleTrainingSeed": True,
        "question": (
            "Does the frozen transformer's native block-5 causal Jacobian lens "
            "map through E to the independently learned pH action port B(x)?"
        ),
        "baseCheckpoint": str(base_checkpoint_path),
        "baseCheckpointStep": int(base_payload["step"]),
        "phCheckpoint": str(ph_checkpoint_path),
        "phTrainingSeed": int(
            torch.load(ph_checkpoint_path, map_location="cpu", weights_only=False)[
                "config"
            ]["seed"]
        ),
        "config": asdict(config),
        "protocol": {
            "latent": "concatenated player/puck tokens after transformer block 5",
            "readout": "trained affine E(h) into canonical q,p plus hybrid modes",
            "instantaneousTarget": "dt * B(E(h)) * unit-axis action",
            "finiteTarget": (
                "centered pH step [Phi(x,+u)-Phi(x,-u)]/2 using the trained integrator"
            ),
            "finiteObservation": (
                "write h5, render hard next frame, append it, re-extract h5, and apply E"
            ),
            "controls": [
                "wrong action axis",
                f"{config.random_direction_pairs} latent direction pairs orthogonal to the lens",
                "paired plus/minus writes with matched magnitude and model noise",
            ],
            "confidenceIntervals": (
                "normal sample-level intervals conditional on one trained seed"
            ),
        },
        "lens": lens_protocol,
        "sharedPlayerPuckTokenFraction": float(merged["sharedToken"].float().mean()),
        "differentialBridge": {
            "globalCausalLens": global_differential,
            "localPerStateJacobian": local_differential,
            "randomLatentControl": random_differential,
        },
        "finiteCausalBridge": {
            "strengthSweep": finite,
            "randomLatentControlAtPrimaryStrength": random_finite,
        },
        "conclusion": conclusion,
        "timing": {"totalSeconds": time.perf_counter() - started},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_checkpoint", type=Path)
    parser.add_argument("ph_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lens-fit-samples", type=int, default=512)
    parser.add_argument("--test-policy-samples", type=int, default=256)
    parser.add_argument("--test-cardinal-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--strengths", default="1,4,8")
    parser.add_argument("--primary-strength", type=float, default=8.0)
    parser.add_argument("--random-direction-pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=121_610_731)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_jacobian_port_bridge_experiment(
        args.base_checkpoint,
        args.ph_checkpoint,
        args.output,
        config=JacobianPortBridgeConfig(
            lens_fit_samples=args.lens_fit_samples,
            test_policy_samples=args.test_policy_samples,
            test_cardinal_samples=args.test_cardinal_samples,
            batch_size=args.batch_size,
            intervention_strengths=tuple(
                float(value) for value in args.strengths.split(",")
            ),
            primary_strength=args.primary_strength,
            random_direction_pairs=args.random_direction_pairs,
            seed=args.seed,
        ),
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
