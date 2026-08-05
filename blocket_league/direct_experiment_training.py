"""Joint training machinery for the preregistered direct experiment.

All optimization batches are derived from the two tensors ``pixelContexts``
and ``frames``.  The simulator is not imported by this module.  Physical
commands, simulator states, event labels, masks, and coordinates therefore
cannot enter a gradient update through this API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .action_free_latent_effort import (
    LatentEffortConfig,
    LatentEffortInference,
    UnstructuredLatentEffortDynamics,
    latent_effort_statistics,
)
from .direct_action_free_data import weighted_pixel_cross_entropy
from .direct_activation_lens import (
    FrozenSoftPixelActivationLens,
    basis_invariant_response_loss,
    differentiable_attention_backend,
    direct_dynamics_pulse_responses,
    grassmann_response_loss,
)
from .direct_cotangent_bridge import (
    PixelChangeProbeBank,
    activation_observable_covectors,
    cotangent_poisson_bridge,
)
from .direct_jacobian_port_extractor import (
    EmpiricalTangentArtifact,
    EmpiricalTangentConfig,
    FrozenEmpiricalJacobianActivationPort,
)
from .direct_poisson_ph import DirectPoissonPHConfig, DirectPoissonPortHamiltonian
from .direct_ph_ablation_cores import ConstantPortHamiltonian, SkewOnlyPortHamiltonian
from .direct_visual_poisson_ph import (
    DirectVideoLossConfig,
    DirectVisualPoissonPH,
    PersistentOrthogonalPortFrame,
    WholeStreamEncoderConfig,
    WholeStreamFrozenEncoder,
    direct_video_objective,
)
from .experiment_f_contract import Variant
from .latent_patch_renderer import LatentPatchTransformerRenderer
from .tensor_provenance import module_tensor_hash, parameter_count
from .pixel_direct_model import DirectPixelTransformer
from .source_provenance import build_source_manifest
from .runtime_firewall_trace import RuntimeFirewallTrace


def _resolved_source_tree_sha256(value: str | None) -> str:
    if value is None:
        value = build_source_manifest()["treeSha256"]
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("source tree SHA-256 must be 64 lowercase hex characters")
    return value


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    """Publish a complete checkpoint atomically on the target filesystem."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class DirectSystemSpec:
    name: str
    state_size: int
    port_size: int
    dt: float
    poisson_rank: int | None = None

    def __post_init__(self) -> None:
        if self.state_size < 2:
            raise ValueError("direct system state_size must be at least two")
        if self.port_size < 1 or self.port_size > self.state_size:
            raise ValueError("direct system port_size must lie in [1,state_size]")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("direct system dt must be finite and positive")
        resolved_rank = (
            self.state_size - self.state_size % 2
            if self.poisson_rank is None
            else self.poisson_rank
        )
        if (
            type(resolved_rank) is not int
            or resolved_rank < 0
            or resolved_rank > self.state_size
            or resolved_rank % 2
        ):
            raise ValueError("direct system poisson_rank must be even and valid")
        object.__setattr__(self, "poisson_rank", resolved_rank)
    lens_block: int = 4


DIRECT_SYSTEMS = {
    "pendulum": DirectSystemSpec(
        "pendulum", state_size=2, port_size=1, dt=0.05, poisson_rank=2
    ),
    "blocket": DirectSystemSpec(
        "blocket", state_size=8, port_size=2, dt=0.05, poisson_rank=8
    ),
}


@dataclass(frozen=True)
class DirectTrainingConfig:
    steps: int = 30_000
    micro_batch_size: int = 16
    gradient_accumulation: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1_000
    minimum_learning_rate_ratio: float = 0.05
    gradient_clip: float = 1.0
    validation_every: int = 500
    validation_batches: int = 8
    checkpoint_every: int = 500
    log_every: int = 25
    lens_every: int = 1
    lens_batch_size: int = 4
    lens_horizons: tuple[int, ...] = (1, 2, 4)
    state_hidden_size: int = 192
    renderer_hidden_size: int = 192
    renderer_depth: int = 3
    renderer_heads: int = 6
    ph_hidden_size: int = 128
    ph_hidden_layers: int = 3
    coupling_layers: int = 6
    implicit_iterations: int = 32
    implicit_relaxation: float = 0.8
    # Retained as explicit checkpoint/configuration-schema compatibility
    # fields. Neither registered architecture constructs a learned activation
    # write field; both exact Jacobian ports have zero trainable parameters.
    write_hidden_size: int = 16
    write_hidden_layers: int = 2
    port_tangent_channel_rank: int = 16
    port_tangent_neighbors: int = 32
    port_support_floor_ratio: float = 0.02
    probe_ridge: float = 1e-4
    seed: int = 151_910_737

    def __post_init__(self) -> None:
        positive_integers = (
            "steps",
            "micro_batch_size",
            "gradient_accumulation",
            "validation_every",
            "validation_batches",
            "checkpoint_every",
            "log_every",
            "lens_every",
            "lens_batch_size",
            "state_hidden_size",
            "renderer_hidden_size",
            "renderer_depth",
            "renderer_heads",
            "ph_hidden_size",
            "ph_hidden_layers",
            "coupling_layers",
            "implicit_iterations",
            "write_hidden_size",
            "write_hidden_layers",
            "port_tangent_channel_rank",
            "port_tangent_neighbors",
        )
        for name in positive_integers:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.lens_every != 1:
            raise ValueError("registered Jacobian lens frequency must be exactly every step")
        if self.lens_batch_size > self.micro_batch_size:
            raise ValueError("lens_batch_size cannot exceed micro_batch_size")
        if self.checkpoint_every < self.validation_every or (
            self.checkpoint_every % self.validation_every
        ):
            raise ValueError("checkpoint_every must be a multiple of validation_every")
        if (
            type(self.lens_horizons) is not tuple
            or not self.lens_horizons
            or any(type(value) is not int or value < 1 for value in self.lens_horizons)
            or tuple(sorted(set(self.lens_horizons))) != self.lens_horizons
        ):
            raise ValueError("lens_horizons must be a sorted unique positive tuple")
        finite_values = (
            self.learning_rate,
            self.weight_decay,
            self.minimum_learning_rate_ratio,
            self.gradient_clip,
            self.implicit_relaxation,
            self.probe_ridge,
            self.port_support_floor_ratio,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("direct training scalar configuration must be finite")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate/weight_decay are invalid")
        if not 0.0 <= self.minimum_learning_rate_ratio <= 1.0:
            raise ValueError("minimum_learning_rate_ratio must lie in [0,1]")
        if self.gradient_clip <= 0.0 or self.probe_ridge <= 0.0:
            raise ValueError("gradient_clip and probe_ridge must be positive")
        if not 0.0 <= self.port_support_floor_ratio < 1.0:
            raise ValueError("port_support_floor_ratio must lie in [0,1)")
        if not 0.0 < self.implicit_relaxation <= 1.0:
            raise ValueError("implicit_relaxation must lie in (0,1]")
        if self.renderer_hidden_size % self.renderer_heads:
            raise ValueError("renderer_hidden_size must be divisible by renderer_heads")


@dataclass(frozen=True)
class BaselineTrainingConfig:
    steps: int = 30_000
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    validation_every: int = 500
    checkpoint_every: int = 500
    validation_batch_size: int = 64
    log_every: int = 100
    seed: int = 151_910_737

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "batch_size",
            "validation_every",
            "checkpoint_every",
            "validation_batch_size",
            "log_every",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"baseline {name} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("baseline seed must be an integer")
        if not all(
            math.isfinite(value) for value in (self.learning_rate, self.weight_decay)
        ):
            raise ValueError("baseline scalar configuration must be finite")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("baseline learning_rate/weight_decay are invalid")
        if self.checkpoint_every < self.validation_every or (
            self.checkpoint_every % self.validation_every
        ):
            raise ValueError(
                "baseline checkpoint_every must be a multiple of validation_every"
            )


@dataclass(frozen=True)
class DirectModelBundle:
    model: DirectVisualPoissonPH
    write_field: FrozenEmpiricalJacobianActivationPort
    lens: FrozenSoftPixelActivationLens
    probes: PixelChangeProbeBank
    response_frame: PersistentOrthogonalPortFrame
    cotangent_frame: PersistentOrthogonalPortFrame


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_direct_bundle(
    backbone: DirectPixelTransformer,
    system: DirectSystemSpec,
    probes: PixelChangeProbeBank,
    config: DirectTrainingConfig,
    device: torch.device,
    *,
    empirical_tangent: EmpiricalTangentArtifact,
    variant: Variant = "full",
) -> DirectModelBundle:
    lens_block = min(system.lens_block, len(backbone.blocks) - 1)
    encoder = WholeStreamFrozenEncoder(
        backbone,
        WholeStreamEncoderConfig(
            state_size=system.state_size,
            readout_hidden_size=config.state_hidden_size,
            lens_block=lens_block,
        ),
    )
    renderer = LatentPatchTransformerRenderer(
        system.state_size,
        image_size=backbone.config.image_size,
        patch_size=backbone.config.patch_size,
        palette_size=backbone.config.palette_size,
        hidden_size=config.renderer_hidden_size,
        depth=config.renderer_depth,
        heads=config.renderer_heads,
    )
    core_config = DirectPoissonPHConfig(
            state_size=system.state_size,
            port_size=system.port_size,
            poisson_rank=system.poisson_rank,
            hidden_size=config.ph_hidden_size,
            hidden_layers=config.ph_hidden_layers,
            coupling_layers=config.coupling_layers,
            dt=system.dt,
            implicit_iterations=config.implicit_iterations,
            implicit_relaxation=config.implicit_relaxation,
            # A fixed iteration count is differentiable and deterministic.
            # Convergence is still audited from the exposed residual.
            implicit_tolerance=0.0,
    )
    core_class: type[DirectPoissonPortHamiltonian]
    if variant == "skew_only":
        core_class = SkewOnlyPortHamiltonian
    elif variant == "constant_port":
        core_class = ConstantPortHamiltonian
    else:
        core_class = DirectPoissonPortHamiltonian
    core = core_class(core_config)
    inference = LatentEffortInference(
        LatentEffortConfig(
            state_size=system.state_size,
            effort_size=system.port_size,
            hidden_size=128,
            hidden_layers=2,
        )
    )
    model = DirectVisualPoissonPH(encoder, renderer, core, inference).to(device)
    activation = encoder.activation_shape
    write_field = FrozenEmpiricalJacobianActivationPort(
        empirical_tangent,
        history_frames=activation[0],
        patch_count=activation[1],
        hidden_size=activation[2],
        port_size=system.port_size,
        config=EmpiricalTangentConfig(
            channel_rank=config.port_tangent_channel_rank,
            neighbors=config.port_tangent_neighbors,
            support_floor_ratio=config.port_support_floor_ratio,
        ),
    ).to(device).eval().requires_grad_(False)
    write_field.assert_frozen_parameter_free()
    lens = FrozenSoftPixelActivationLens(
        encoder.backbone,
        intervention_block=lens_block,
        horizons=config.lens_horizons,
    ).to(device)
    probes = probes.to(device).eval().requires_grad_(False)
    response_frame = PersistentOrthogonalPortFrame(system.port_size).to(device)
    cotangent_frame = PersistentOrthogonalPortFrame(system.port_size).to(device)
    return DirectModelBundle(
        model=model,
        write_field=write_field,
        lens=lens,
        probes=probes,
        response_frame=response_frame,
        cotangent_frame=cotangent_frame,
    )


def _permute_correspondence_target(
    target: torch.Tensor,
    *,
    shuffled: bool,
) -> torch.Tensor:
    """Apply the registered fixed-point-free batch permutation to a target.

    The operation preserves every target value, norm, rank, horizon, and
    computational dependency.  Only its correspondence with the learned
    state at the same batch row changes.  Returning the original tensor for
    the full model is intentional: the non-ablation path must not acquire an
    otherwise innocuous copy or indexing operation.
    """

    if not shuffled:
        return target
    if target.ndim < 1 or target.shape[0] < 2:
        raise ValueError(
            "a shuffled correspondence target needs a batch dimension of "
            "at least two"
        )
    return target.roll(1, dims=0)


def jacobian_lens_terms(
    bundle: DirectModelBundle,
    pixel_context: torch.Tensor,
    *,
    horizons: tuple[int, ...],
    ridge: float,
    shuffled: bool = False,
    encoded_states: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute both tangent-response and cotangent-Poisson bridges."""

    model, field, lens = bundle.model, bundle.write_field, bundle.lens
    if encoded_states is None:
        states = model.encode(pixel_context)
    else:
        expected = (pixel_context.shape[0], model.core.config.state_size)
        if tuple(encoded_states.shape) != expected:
            raise ValueError(
                f"expected encoded_states shape {expected}, "
                f"got {tuple(encoded_states.shape)}"
            )
        states = encoded_states
    if shuffled and pixel_context.shape[0] < 2:
        raise ValueError(
            "the shuffled-lens ablation requires at least two examples so "
            "that its target permutation has no fixed point"
        )
    source_activation = model.encoder.prefix_activation(pixel_context).detach()
    frozen_activation_covectors = {
        horizon: covector.detach()
        for horizon, covector in activation_observable_covectors(
            lens,
            pixel_context,
            bundle.probes,
            horizons=horizons,
            create_graph=False,
        ).items()
    }
    extraction = field(
        frozen_activation_covectors,
        source_activation,
    )
    write_basis = extraction.jacobian.write_basis
    # The registered tangent bridge lives in the direct latent state, not in
    # the 36k-dimensional pixel simplex.  The frozen side re-encodes each soft
    # autoregressive context; the pH side returns Phi_h itself.  Matching
    # rendered pixels here would be a different (and vastly more expensive)
    # objective than K_h U ~= G_h^pH in the preregistration.
    frozen_response = lens.state_response_jacobians(
        pixel_context,
        write_basis,
        model.encoder.read_suffix_tokens,
        horizons=horizons,
        create_graph=True,
    )
    with differentiable_attention_backend(states):
        direct_response = direct_dynamics_pulse_responses(
            model.step,
            states,
            model.core.config.port_size,
            horizons=horizons,
            create_graph=True,
        )
    frozen_stack = torch.cat(
        tuple(
            _permute_correspondence_target(
                frozen_response.jacobians[horizon], shuffled=shuffled
            )
            for horizon in horizons
        ),
        dim=1,
    )
    direct_stack = torch.cat(
        tuple(direct_response.jacobians[horizon] for horizon in horizons), dim=1
    )
    # One single Procrustes fit is shared by every sample and every horizon.
    # A state-dependent Q(x) can no longer disappear into independent SVDs.
    global_frozen_stack = frozen_stack.reshape(
        1, -1, frozen_stack.shape[-1]
    )
    global_direct_stack = direct_stack.reshape(
        1, -1, direct_stack.shape[-1]
    )
    response_alignment = basis_invariant_response_loss(
        global_frozen_stack, global_direct_stack
    )
    response_subspace = grassmann_response_loss(
        global_frozen_stack, global_direct_stack
    )
    frozen_normalized = global_frozen_stack / torch.linalg.matrix_norm(
        global_frozen_stack, ord="fro", dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)
    direct_oriented = bundle.response_frame(global_direct_stack)
    direct_normalized = direct_oriented / torch.linalg.matrix_norm(
        direct_oriented, ord="fro", dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)
    persistent_response_alignment = (
        frozen_normalized - direct_normalized
    ).square().sum(dim=(-2, -1)).mean()
    proxies = lens.intervention_proxies(pixel_context, write_basis, amplitude=0.05)

    # Every variant pays for and optimizes the exact same cotangent bridge.
    # ``shuffled_lens`` changes only which already-computed Jacobian target is
    # coupled to each learned state; it does not remove a term, alter a norm,
    # or feed mismatched states into the frozen lens itself.
    target_batch_permutation = (
        torch.arange(pixel_context.shape[0], device=pixel_context.device).roll(1)
        if shuffled
        else None
    )
    cotangent = cotangent_poisson_bridge(
        model,
        lens,
        bundle.probes,
        pixel_context,
        horizons=horizons,
        ridge=ridge,
        target_batch_permutation=target_batch_permutation,
        activation_covectors=frozen_activation_covectors,
        prefix_activation=source_activation,
        extracted_write_basis=write_basis,
    )
    learned_port = model.core.port(cotangent.state.float())
    cotangent_total = cotangent.total
    cotangent_alignment = cotangent.port_alignment
    cotangent_compatibility = cotangent.pullback_compatibility
    cotangent_consistency = cotangent.horizon_consistency
    cotangent_isotropy = cotangent.port_isotropy
    tangent_pushforward_alignment = cotangent.tangent_pushforward_alignment
    learned_oriented = bundle.cotangent_frame(learned_port)
    learned_stack = torch.cat(
        tuple(learned_oriented for _ in horizons), dim=-2
    )
    prior_stack = torch.cat(
        tuple(cotangent.poisson_port_priors[horizon] for horizon in horizons),
        dim=-2,
    )
    # Normalize each port column once over the complete validation/training
    # batch and every horizon.  A single constant gain remains free for the
    # post-freeze analytic interface, whereas c(x)-dependent gains can no
    # longer disappear through per-state column normalization.
    learned_scale = torch.linalg.vector_norm(
        learned_stack.reshape(-1, learned_stack.shape[-1]),
        dim=0,
        keepdim=True,
    ).clamp_min(1e-8)
    prior_scale = torch.linalg.vector_norm(
        prior_stack.reshape(-1, prior_stack.shape[-1]),
        dim=0,
        keepdim=True,
    ).clamp_min(1e-8)
    persistent_cotangent_alignment = (
        learned_stack / learned_scale - prior_stack / prior_scale
    ).square().mean()
    signal_floor = torch.relu(
        proxies.first_order_signal.new_tensor(1e-7) - proxies.first_order_signal
    ) / 1e-7
    bridge = (
        persistent_response_alignment
        + persistent_cotangent_alignment
        + 0.10 * response_subspace
        + 0.10 * cotangent_total
    )
    manifold = proxies.manifold_cycle + proxies.current_frame_leakage + signal_floor
    terms = {
        "bridge": bridge,
        "oddness": proxies.odd_symmetry,
        "manifoldCycle": manifold,
    }
    metrics = {
        "responseAlignment": response_alignment,
        "persistentResponseFrameAlignment": persistent_response_alignment,
        "responseSubspace": response_subspace,
        "cotangentAlignment": cotangent_alignment,
        "cotangentCompatibility": cotangent_compatibility,
        "cotangentHorizonConsistency": cotangent_consistency,
        "cotangentIsotropy": cotangent_isotropy,
        "cotangentTangentPushforwardAlignment": tangent_pushforward_alignment,
        "persistentCotangentFrameAlignment": persistent_cotangent_alignment,
        "writeOddness": proxies.odd_symmetry,
        "writeCurrentFrameLeakage": proxies.current_frame_leakage,
        "writeManifoldCycle": proxies.manifold_cycle,
        "writeFirstOrderSignal": proxies.first_order_signal,
        "minimumFrozenResponseSingularValue": torch.linalg.svdvals(
            frozen_stack.detach().float()
        ).amin(),
        "minimumPHResponseSingularValue": torch.linalg.svdvals(
            direct_stack.detach().float()
        ).amin(),
        "extractedPortMinimumSingularValue": extraction.jacobian.singular_values.amin(),
        "extractedPortMaximumOrthonormalityDefect": (
            extraction.jacobian.orthonormality_defect.amax()
        ),
        "extractedPortMinimumProjectedSignalRatio": (
            extraction.projected_signal_ratio.amin()
        ),
    }
    return terms, metrics


def _sample_batch(
    suite: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.randint(0, suite["frames"].shape[0], (batch_size,))
    return (
        suite["pixelContexts"][rows].to(device, non_blocking=True).long(),
        suite["frames"][rows].to(device, non_blocking=True).long(),
    )


def _fixed_validation_batch(
    suite: dict[str, torch.Tensor],
    batch_size: int,
    batch_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic pixels-only validation slice, independent of train RNG."""

    sample_count = suite["frames"].shape[0]
    rows = (
        torch.arange(batch_size, dtype=torch.long) + batch_index * batch_size
    ) % sample_count
    return (
        suite["pixelContexts"][rows].to(device, non_blocking=True).long(),
        suite["frames"][rows].to(device, non_blocking=True).long(),
    )


def _learning_rate(step: int, config: DirectTrainingConfig) -> float:
    if step <= config.warmup_steps:
        multiplier = step / max(config.warmup_steps, 1)
    else:
        progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
        multiplier = config.minimum_learning_rate_ratio + (
            1.0 - config.minimum_learning_rate_ratio
        ) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.learning_rate * multiplier


_FAIL_CLOSED_LENS_MINIMA = frozenset(
    {
        "lensWriteFirstOrderSignal",
        "lensMinimumFrozenResponseSingularValue",
        "lensMinimumPHResponseSingularValue",
        "lensExtractedPortMinimumSingularValue",
        "lensExtractedPortMinimumProjectedSignalRatio",
    }
)
_FAIL_CLOSED_LENS_MAXIMA = frozenset(
    {"lensExtractedPortMaximumOrthonormalityDefect"}
)


def _aggregated_pixels_only_lens_validation(
    bundle: DirectModelBundle,
    suite: dict[str, torch.Tensor],
    train_config: DirectTrainingConfig,
    device: torch.device,
    *,
    variant: Variant,
) -> dict[str, float]:
    """Evaluate the lens on disjoint fixed groups and aggregate fail-closed.

    Every checkpoint candidate sees ``validation_batches`` consecutive,
    non-overlapping groups of ``lens_batch_size`` trajectories.  Losses and
    ordinary diagnostics are averaged.  Signal/rank lower bounds take the
    worst group so one degenerate group cannot disappear inside a mean.
    """

    if variant == "no_jacobian":
        return {}
    group_count = train_config.validation_batches
    group_size = train_config.lens_batch_size
    required_samples = group_count * group_size
    available_samples = int(suite["frames"].shape[0])
    if available_samples < required_samples:
        raise ValueError(
            "pixels-only lens validation requires "
            f"{required_samples} distinct trajectories, got {available_samples}"
        )
    selected_horizons = (
        (1,) if variant == "single_horizon" else train_config.lens_horizons
    )
    grouped: dict[str, list[float]] = {}
    for batch_index in range(group_count):
        lens_contexts, _ = _fixed_validation_batch(
            suite,
            group_size,
            batch_index,
            device,
        )
        # Validation differentiates frozen responses but performs no backward
        # or optimizer update.  Detach each group immediately so eight lens
        # graphs are never retained simultaneously.
        with torch.enable_grad():
            lens_terms, lens_metrics = jacobian_lens_terms(
                bundle,
                lens_contexts[:, 0],
                horizons=selected_horizons,
                ridge=train_config.probe_ridge,
                shuffled=variant == "shuffled_lens",
            )
        group_values = {
            "lensBridge": float(lens_terms["bridge"].detach()),
            "lensOddness": float(lens_terms["oddness"].detach()),
            "lensManifoldCycle": float(lens_terms["manifoldCycle"].detach()),
            **{
                f"lens{name[0].upper()}{name[1:]}": float(value.detach())
                for name, value in lens_metrics.items()
            },
        }
        if grouped and set(group_values) != set(grouped):
            raise AssertionError("lens validation metric schema changed between groups")
        for name, value in group_values.items():
            grouped.setdefault(name, []).append(value)
        del lens_terms, lens_metrics

    aggregated = {
        name: (
            min(values)
            if name in _FAIL_CLOSED_LENS_MINIMA
            else max(values)
            if name in _FAIL_CLOSED_LENS_MAXIMA
            else sum(values) / group_count
        )
        for name, values in grouped.items()
    }
    aggregated["lensValidationGroups"] = float(group_count)
    aggregated["lensValidationContexts"] = float(required_samples)
    return aggregated


@torch.no_grad()
def pixels_only_validation_score(
    bundle: DirectModelBundle,
    suite: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    loss_config: DirectVideoLossConfig,
    train_config: DirectTrainingConfig,
    device: torch.device,
    *,
    variant: Variant,
) -> dict[str, float]:
    bundle.model.eval()
    totals: dict[str, float] = {}
    audit_core = copy.deepcopy(bundle.model.core).to(
        device=device, dtype=torch.float64
    ).eval().requires_grad_(False)
    totals["auditImplicitResidualMax"] = 0.0
    totals["auditChainRuleDefectMax"] = 0.0
    totals["auditBalanceDefectMax"] = 0.0
    totals["auditMinimumPortSingularValue"] = math.inf
    for batch_index in range(train_config.validation_batches):
        contexts, frames = _fixed_validation_batch(
            suite, train_config.micro_batch_size, batch_index, device
        )
        _, metrics = direct_video_objective(
            bundle.model,
            contexts,
            frames,
            class_weights,
            loss_config,
            require_lens_terms=False,
        )
        for name in ("reconstruction", "rolloutPixel", "rolloutLatent", "innovation"):
            totals[name] = totals.get(name, 0.0) + float(metrics[name])
        totals["implicitResidualMax"] = max(
            totals.get("implicitResidualMax", 0.0),
            float(metrics["implicitResidualMax"]),
        )
        totals["chainRuleDefectMax"] = max(
            totals.get("chainRuleDefectMax", 0.0),
            float(metrics["chainRuleDefectMax"]),
        )
        states = bundle.model.encode(contexts)
        efforts = bundle.model.infer_latent_effort(states[:, :-1], states[:, 1:])
        flat_states = states[:, :-1].reshape(-1, states.shape[-1]).double()
        flat_efforts = efforts.reshape(-1, efforts.shape[-1]).double()
        audited = audit_core.audited_step(flat_states, flat_efforts)
        totals["auditImplicitResidualMax"] = max(
            totals["auditImplicitResidualMax"],
            float(audited.implicit_residual_norm.max()),
        )
        totals["auditChainRuleDefectMax"] = max(
            totals["auditChainRuleDefectMax"],
            float(audited.chain_rule_defect.abs().max()),
        )
        totals["auditBalanceDefectMax"] = max(
            totals["auditBalanceDefectMax"],
            float(audited.balance_defect.abs().max()),
        )
        totals["auditMinimumPortSingularValue"] = min(
            totals["auditMinimumPortSingularValue"],
            float(torch.linalg.svdvals(audit_core.port(flat_states)).amin()),
        )
    for name in ("reconstruction", "rolloutPixel", "rolloutLatent", "innovation"):
        totals[name] /= train_config.validation_batches
    totals["score"] = (
        totals["reconstruction"]
        + totals["rolloutPixel"]
        + totals["rolloutLatent"]
        + 0.10 * totals["innovation"]
    )
    lens_eligible = True
    if variant != "no_jacobian":
        lens_values = _aggregated_pixels_only_lens_validation(
            bundle,
            suite,
            train_config,
            device,
            variant=variant,
        )
        totals.update(lens_values)
        totals["score"] += (
            loss_config.jacobian_bridge_weight * totals["lensBridge"]
            + loss_config.oddness_weight * totals["lensOddness"]
            + loss_config.manifold_cycle_weight * totals["lensManifoldCycle"]
        )
        lens_eligible = (
            all(math.isfinite(value) for value in lens_values.values())
            and totals["lensWriteFirstOrderSignal"] >= 1e-7
            and totals["lensMinimumFrozenResponseSingularValue"] >= 1e-6
            and totals["lensMinimumPHResponseSingularValue"] >= 1e-6
            and totals["lensExtractedPortMinimumSingularValue"] >= 1e-8
            and totals["lensExtractedPortMaximumOrthonormalityDefect"] <= 1e-4
            and totals["lensExtractedPortMinimumProjectedSignalRatio"] >= 1e-6
        )
    totals["structureEligible"] = float(
        totals["auditImplicitResidualMax"] <= loss_config.implicit_residual_tolerance
        and totals["auditChainRuleDefectMax"] <= loss_config.chain_rule_tolerance
        and totals["auditBalanceDefectMax"] <= 1e-7
        and totals["auditMinimumPortSingularValue"] >= 1e-5
        and lens_eligible
    )
    del audit_core
    bundle.model.train()
    bundle.model.encoder.backbone.eval()
    return totals


_EVALUATION_CHECKPOINT_KEYS = frozenset(
    {
        "kind",
        "actionChannels",
        "physicalStateChannels",
        "optimizationTensorKeys",
        "system",
        "variant",
        "step",
        "bestValidation",
        "bestStructureEligible",
        "model",
        "writeField",
        "responseFrame",
        "cotangentFrame",
        "probes",
        "probeHash",
        "dataSeal",
        "optimizedParameterNames",
        "trainConfig",
        "lossConfig",
        "backboneHash",
        "sourceTreeSha256",
    }
)


def _validate_direct_checkpoint(
    payload: dict[str, Any],
    bundle: DirectModelBundle,
    *,
    variant: Variant,
    system: DirectSystemSpec,
    train_config: DirectTrainingConfig,
    loss_config: DirectVideoLossConfig,
    data_seal: dict[str, str],
    source_tree_sha256: str | None = None,
    include_training_state: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Positive-schema validation before any checkpoint tensor is loaded."""

    expected_keys = set(_EVALUATION_CHECKPOINT_KEYS)
    if include_training_state:
        expected_keys.update(("optimizer", "rngState"))
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError("direct checkpoint top-level schema is not exact")
    expected_source_tree_sha256 = _resolved_source_tree_sha256(source_tree_sha256)
    if payload["sourceTreeSha256"] != expected_source_tree_sha256:
        raise ValueError("direct checkpoint source-tree provenance mismatch")
    if payload["kind"] != "direct_jacobian_poisson_port_hamiltonian":
        raise ValueError("direct checkpoint kind mismatch")
    if type(payload["actionChannels"]) is not int or payload["actionChannels"] != 0:
        raise ValueError("direct checkpoint did not seal zero action channels")
    if (
        type(payload["physicalStateChannels"]) is not int
        or payload["physicalStateChannels"] != 0
    ):
        raise ValueError("direct checkpoint did not seal zero physical-state channels")
    if (
        type(payload["optimizationTensorKeys"]) is not list
        or payload["optimizationTensorKeys"] != ["pixelContexts", "frames"]
    ):
        raise ValueError("direct checkpoint optimization schema is not pixels-only")
    if type(payload["system"]) is not dict or payload["system"] != asdict(system):
        raise ValueError("direct checkpoint system mismatch")
    if payload["variant"] != variant:
        raise ValueError("direct checkpoint variant mismatch")
    if type(payload["step"]) is not int or payload["step"] < 1:
        raise ValueError("direct checkpoint step is invalid")
    if type(payload["bestValidation"]) not in (int, float) or not math.isfinite(
        float(payload["bestValidation"])
    ):
        raise ValueError("direct checkpoint validation score is invalid")
    if type(payload["bestStructureEligible"]) is not bool:
        raise ValueError("direct checkpoint structural eligibility is invalid")
    if payload["trainConfig"] != asdict(train_config):
        raise ValueError("direct checkpoint training configuration changed")
    if payload["lossConfig"] != asdict(loss_config):
        raise ValueError("direct checkpoint loss configuration changed")
    if payload["backboneHash"] != bundle.model.encoder.sealed_backbone_hash:
        raise ValueError("direct checkpoint backbone seal does not match")
    expected_data_seal_keys = {
        "system",
        "fitAggregateSha256",
        "fitSanitizedTensorSha256",
        "validationAggregateSha256",
        "validationSanitizedTensorSha256",
    }
    data_seal_is_valid = (
        type(data_seal) is dict
        and set(data_seal) == expected_data_seal_keys
        and data_seal.get("system") == system.name
        and all(
            type(data_seal.get(name)) is str
            and len(data_seal[name]) == 64
            and all(character in "0123456789abcdef" for character in data_seal[name])
            for name in expected_data_seal_keys - {"system"}
        )
    )
    if (
        not data_seal_is_valid
        or
        type(payload["dataSeal"]) is not dict
        or set(payload["dataSeal"]) != expected_data_seal_keys
        or payload["dataSeal"] != data_seal
    ):
        raise ValueError("direct checkpoint sanitized data seal does not match")
    if payload["probeHash"] != module_tensor_hash(bundle.probes):
        raise ValueError("direct checkpoint fixed pixel-probe seal does not match")
    expected_names = [name for name, _ in _named_optimized_parameters(bundle)]
    if (
        type(payload["optimizedParameterNames"]) is not list
        or payload["optimizedParameterNames"] != expected_names
        or len(expected_names) != len(set(expected_names))
    ):
        raise ValueError("direct checkpoint optimizer membership does not match exactly")

    modules = {
        "model": bundle.model,
        "writeField": bundle.write_field,
        "responseFrame": bundle.response_frame,
        "cotangentFrame": bundle.cotangent_frame,
        "probes": bundle.probes,
    }
    for field, module in modules.items():
        state = payload[field]
        reference = module.state_dict()
        if type(state) is not dict or set(state) != set(reference):
            raise ValueError(f"direct checkpoint {field} state schema mismatch")
        for name, tensor in state.items():
            expected = reference[name]
            if type(tensor) is not torch.Tensor:
                raise ValueError(f"direct checkpoint {field}.{name} is not a plain tensor")
            if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
                raise ValueError(f"direct checkpoint {field}.{name} shape/dtype mismatch")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"direct checkpoint {field}.{name} is non-finite")
    current_model_state = bundle.model.state_dict()
    for name, tensor in payload["model"].items():
        if name.startswith("encoder.backbone.") and not torch.equal(
            tensor.cpu(), current_model_state[name].detach().cpu()
        ):
            raise ValueError("direct checkpoint attempted to alter the frozen backbone")
    current_probe_state = bundle.probes.state_dict()
    for name, tensor in payload["probes"].items():
        if not torch.equal(tensor.cpu(), current_probe_state[name].detach().cpu()):
            raise ValueError("direct checkpoint attempted to alter the fixed pixel probes")
    if include_training_state:
        if optimizer is None:
            raise ValueError("resume validation requires the live optimizer schema")
        _validate_optimizer_state(
            payload["optimizer"],
            optimizer,
            [parameter for _, parameter in _named_optimized_parameters(bundle)],
            checkpoint_step=payload["step"],
        )
        _validate_safe_rng_state(payload["rngState"])


def _checkpoint_payload(
    bundle: DirectModelBundle,
    optimizer: torch.optim.Optimizer | None,
    *,
    step: int,
    best_validation: float,
    best_structure_eligible: bool,
    variant: Variant,
    system: DirectSystemSpec,
    train_config: DirectTrainingConfig,
    loss_config: DirectVideoLossConfig,
    data_seal: dict[str, str],
    source_tree_sha256: str,
    include_training_state: bool,
) -> dict[str, Any]:
    bundle.model.encoder.assert_backbone_frozen()
    optimized_names = [name for name, _ in _named_optimized_parameters(bundle)]
    payload: dict[str, Any] = {
        "kind": "direct_jacobian_poisson_port_hamiltonian",
        "actionChannels": 0,
        "physicalStateChannels": 0,
        "optimizationTensorKeys": ["pixelContexts", "frames"],
        "system": asdict(system),
        "variant": variant,
        "step": step,
        "bestValidation": best_validation,
        "bestStructureEligible": best_structure_eligible,
        "model": dict(bundle.model.state_dict()),
        "writeField": dict(bundle.write_field.state_dict()),
        "responseFrame": dict(bundle.response_frame.state_dict()),
        "cotangentFrame": dict(bundle.cotangent_frame.state_dict()),
        "probes": dict(bundle.probes.state_dict()),
        "probeHash": module_tensor_hash(bundle.probes),
        "dataSeal": dict(data_seal),
        "optimizedParameterNames": optimized_names,
        "trainConfig": asdict(train_config),
        "lossConfig": asdict(loss_config),
        "backboneHash": bundle.model.encoder.sealed_backbone_hash,
        "sourceTreeSha256": _resolved_source_tree_sha256(source_tree_sha256),
    }
    if include_training_state:
        if optimizer is None:
            raise ValueError("resume checkpoint requires optimizer state")
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        payload["optimizer"] = optimizer.state_dict()
        payload["rngState"] = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": {
                "algorithm": numpy_state[0],
                "state": torch.from_numpy(numpy_state[1].astype(np.int64)),
                "position": int(numpy_state[2]),
                "hasGaussian": int(numpy_state[3]),
                "cachedGaussian": float(numpy_state[4]),
            },
            "python": {
                "version": int(python_state[0]),
                "state": torch.tensor(python_state[1], dtype=torch.int64),
                "gaussian": (
                    None if python_state[2] is None else float(python_state[2])
                ),
            },
        }
    return payload


def _named_optimized_parameters(
    bundle: DirectModelBundle,
) -> list[tuple[str, nn.Parameter]]:
    """Exact ordered optimizer membership, with the backbone excluded by id."""

    backbone_ids = {
        id(parameter) for parameter in bundle.model.encoder.backbone.parameters()
    }
    named = [
        (f"model.{name}", parameter)
        for name, parameter in bundle.model.named_parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    for prefix, module in (
        ("writeField", bundle.write_field),
        ("responseFrame", bundle.response_frame),
        ("cotangentFrame", bundle.cotangent_frame),
    ):
        named.extend(
            (f"{prefix}.{name}", parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )
    names = [name for name, _ in named]
    if len(names) != len(set(names)):
        raise AssertionError("optimized parameter names are not unique")
    if any("encoder.backbone" in name for name in names):
        raise AssertionError("sealed backbone appeared in optimizer membership")
    return named


def _validate_optimizer_state(
    payload: Any,
    optimizer: torch.optim.Optimizer,
    parameters: list[nn.Parameter],
    *,
    checkpoint_step: int,
) -> None:
    """Reject malformed, reordered, or non-finite AdamW resume state."""

    if type(payload) is not dict or set(payload) != {"state", "param_groups"}:
        raise ValueError("resume optimizer schema is not exact")
    reference = optimizer.state_dict()
    groups = payload["param_groups"]
    reference_groups = reference["param_groups"]
    if (
        type(groups) is not list
        or not reference_groups
        or len(groups) != len(reference_groups)
    ):
        raise ValueError("resume optimizer parameter-group count mismatch")
    valid_parameter_ids: list[int] = []
    for group, expected_group in zip(groups, reference_groups, strict=True):
        if type(group) is not dict or set(group) != set(expected_group):
            raise ValueError("resume optimizer parameter-group schema mismatch")
        if type(group["params"]) is not list or group["params"] != expected_group["params"]:
            raise ValueError("resume optimizer parameter IDs/order changed")
        valid_parameter_ids.extend(group["params"])
        for name, expected in expected_group.items():
            if name in {"params", "lr"}:
                continue
            if group[name] != expected:
                raise ValueError(f"resume optimizer hyperparameter {name!r} changed")
        learning_rate = group["lr"]
        if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate):
            raise ValueError("resume optimizer learning rate is non-finite")
        if learning_rate <= 0.0 or learning_rate > expected_group["lr"]:
            raise ValueError("resume optimizer learning rate is outside its schedule")
    if valid_parameter_ids != list(range(len(parameters))):
        raise ValueError("resume optimizer flattened parameter order is not canonical")

    state = payload["state"]
    if type(state) is not dict or not set(state).issubset(valid_parameter_ids):
        raise ValueError("resume optimizer state contains an unknown parameter")
    for parameter_id, values in state.items():
        if type(parameter_id) is not int or type(values) is not dict:
            raise ValueError("resume optimizer state entry is malformed")
        expected_keys = {"step", "exp_avg", "exp_avg_sq"}
        if set(values) != expected_keys:
            raise ValueError("resume AdamW moment schema is not exact")
        parameter = parameters[parameter_id]
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = values[name]
            if (
                type(tensor) is not torch.Tensor
                or tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"resume optimizer {name} is invalid")
        step = values["step"]
        if (
            type(step) is not torch.Tensor
            or step.numel() != 1
            or not bool(torch.isfinite(step).all())
            or float(step) < 1.0
            or float(step) > checkpoint_step
            or not float(step).is_integer()
        ):
            raise ValueError("resume optimizer step is invalid")


def _validate_safe_rng_state(payload: Any) -> None:
    expected = {"torch", "cuda", "numpy", "python"}
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("resume RNG schema is not exact")
    torch_state = payload["torch"]
    if (
        type(torch_state) is not torch.Tensor
        or torch_state.dtype != torch.uint8
        or tuple(torch_state.shape) != tuple(torch.get_rng_state().shape)
    ):
        raise ValueError("resume Torch RNG state is invalid")
    cuda_state = payload["cuda"]
    cuda_reference = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    if (
        type(cuda_state) is not list
        or len(cuda_state) != len(cuda_reference)
        or any(
            type(value) is not torch.Tensor
            or value.dtype != torch.uint8
            or tuple(value.shape) != tuple(reference.shape)
            for value, reference in zip(cuda_state, cuda_reference, strict=True)
        )
    ):
        raise ValueError("resume CUDA RNG state is invalid")
    numpy_state = payload["numpy"]
    if type(numpy_state) is not dict or set(numpy_state) != {
        "algorithm",
        "state",
        "position",
        "hasGaussian",
        "cachedGaussian",
    }:
        raise ValueError("resume NumPy RNG schema is not exact")
    numpy_tensor = numpy_state["state"]
    if (
        numpy_state["algorithm"] != "MT19937"
        or type(numpy_tensor) is not torch.Tensor
        or numpy_tensor.dtype != torch.int64
        or tuple(numpy_tensor.shape) != (624,)
        or bool((numpy_tensor < 0).any())
        or bool((numpy_tensor > 2**32 - 1).any())
        or type(numpy_state["position"]) is not int
        or not 0 <= numpy_state["position"] <= 624
        or type(numpy_state["hasGaussian"]) is not int
        or numpy_state["hasGaussian"] not in {0, 1}
        or type(numpy_state["cachedGaussian"]) not in (int, float)
        or not math.isfinite(float(numpy_state["cachedGaussian"]))
    ):
        raise ValueError("resume NumPy RNG state is invalid")
    python_state = payload["python"]
    if type(python_state) is not dict or set(python_state) != {
        "version",
        "state",
        "gaussian",
    }:
        raise ValueError("resume Python RNG schema is not exact")
    python_tensor = python_state["state"]
    gaussian = python_state["gaussian"]
    if (
        type(python_state["version"]) is not int
        or python_state["version"] != 3
        or type(python_tensor) is not torch.Tensor
        or python_tensor.dtype != torch.int64
        or tuple(python_tensor.shape) != (625,)
        or bool((python_tensor[:-1] < 0).any())
        or bool((python_tensor[:-1] > 2**32 - 1).any())
        or not 0 <= int(python_tensor[-1]) <= 624
        or not (
            gaussian is None
            or (
                type(gaussian) in (int, float)
                and math.isfinite(float(gaussian))
            )
        )
    ):
        raise ValueError("resume Python RNG state is invalid")


def _restore_safe_rng_state(payload: dict[str, Any]) -> None:
    _validate_safe_rng_state(payload)
    torch.set_rng_state(payload["torch"].cpu())
    cuda_state = payload["cuda"]
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)
    numpy_state = payload["numpy"]
    if type(numpy_state) is not dict or set(numpy_state) != {
        "algorithm",
        "state",
        "position",
        "hasGaussian",
        "cachedGaussian",
    }:
        raise ValueError("resume NumPy RNG schema is not exact")
    if numpy_state["algorithm"] != "MT19937":
        raise ValueError("resume NumPy RNG algorithm is not MT19937")
    np.random.set_state(
        (
            "MT19937",
            numpy_state["state"].cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["hasGaussian"]),
            float(numpy_state["cachedGaussian"]),
        )
    )
    python_state = payload["python"]
    if type(python_state) is not dict or set(python_state) != {
        "version",
        "state",
        "gaussian",
    }:
        raise ValueError("resume Python RNG schema is not exact")
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(value) for value in python_state["state"].tolist()),
            python_state["gaussian"],
        )
    )


def train_direct_bundle(
    bundle: DirectModelBundle,
    fit_suite: dict[str, torch.Tensor],
    validation_suite: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    system: DirectSystemSpec,
    output_dir: Path,
    train_config: DirectTrainingConfig,
    loss_config: DirectVideoLossConfig = DirectVideoLossConfig(),
    *,
    variant: Variant = "full",
    data_seal: dict[str, str],
    source_tree_sha256: str | None = None,
    runtime_trace: RuntimeFirewallTrace | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Train representation, pH core, effort head, decoder, and lens jointly."""

    output_dir.mkdir(parents=True, exist_ok=True)
    source_tree_sha256 = _resolved_source_tree_sha256(source_tree_sha256)
    owns_runtime_trace = runtime_trace is None
    if runtime_trace is None:
        runtime_trace = RuntimeFirewallTrace(
            output_dir / "firewall-trace.jsonl",
            stage=f"direct:{system.name}:{variant}",
            source_tree_sha256=source_tree_sha256,
        )
    if variant != "no_jacobian" and train_config.lens_every != 1:
        raise ValueError(
            "the registered Jacobian lens must be sampled on every optimizer step"
        )
    if variant == "no_jacobian":
        loss_config = replace(
            loss_config,
            jacobian_bridge_weight=0.0,
            oddness_weight=0.0,
            manifold_cycle_weight=0.0,
        )
    elif variant == "skew_only":
        loss_config = replace(loss_config, chart_conditioning_weight=0.0)
    device = next(bundle.model.renderer.parameters()).device
    named_parameters = _named_optimized_parameters(bundle)
    parameters = [parameter for _, parameter in named_parameters]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    runtime_trace.record_optimizer(
        phase=f"direct:{variant}",
        named_parameters=dict(named_parameters),
        protected_parameters={
            f"encoder.backbone.{name}": parameter
            for name, parameter in bundle.model.encoder.backbone.named_parameters()
        },
    )
    start_step = 1
    best_validation = math.inf
    best_structure_eligible = False
    last_path = output_dir / "last.pt"
    if resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=True)
        runtime_trace.record_file_read(
            last_path,
            role=f"direct_resume_checkpoint:{variant}",
            serialized_keys=tuple(sorted(payload)) if type(payload) is dict else (),
        )
        _validate_direct_checkpoint(
            payload,
            bundle,
            variant=variant,
            system=system,
            train_config=train_config,
            loss_config=loss_config,
            data_seal=data_seal,
            source_tree_sha256=source_tree_sha256,
            include_training_state=True,
            optimizer=optimizer,
        )
        if int(payload["step"]) >= train_config.steps:
            raise ValueError("resume checkpoint has no registered training step remaining")
        bundle.model.load_state_dict(payload["model"])
        bundle.write_field.load_state_dict(payload["writeField"])
        bundle.response_frame.load_state_dict(payload["responseFrame"])
        bundle.cotangent_frame.load_state_dict(payload["cotangentFrame"])
        bundle.probes.load_state_dict(payload["probes"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"]) + 1
        best_validation = float(payload["bestValidation"])
        best_structure_eligible = bool(payload.get("bestStructureEligible", False))
        _restore_safe_rng_state(payload["rngState"])
    initial_backbone_hash = bundle.model.encoder.sealed_backbone_hash
    runtime_trace.record_backbone_boundary(
        phase=f"direct:{variant}", boundary="start", sha256=initial_backbone_hash
    )
    started = time.perf_counter()
    log_path = output_dir / "train.jsonl"
    log_mode = "a" if start_step > 1 else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        for step in range(start_step, train_config.steps + 1):
            learning_rate = _learning_rate(step, train_config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, float] = {}
            accumulated_counts: dict[str, int] = {}
            use_lens = variant != "no_jacobian" and (
                step == 1 or step % train_config.lens_every == 0
            )
            for accumulation in range(train_config.gradient_accumulation):
                contexts, frames = _sample_batch(
                    fit_suite, train_config.micro_batch_size, device
                )
                runtime_trace.record_gradient_batch(
                    phase=f"direct:{variant}",
                    step=step,
                    tensors={"pixelContexts": contexts, "frames": frames},
                )
                # The first context used by the lens is a view of this exact
                # pixels-only batch.  Share one encoder graph between the
                # Jacobian bridge and prediction loss instead of evaluating
                # the frozen prefix/readout twice.
                encoded_states = bundle.model.encode(contexts)
                lens_terms = None
                lens_metrics: dict[str, torch.Tensor] = {}
                if use_lens and accumulation == 0:
                    selected_horizons = (
                        (1,) if variant == "single_horizon" else train_config.lens_horizons
                    )
                    lens_terms, lens_metrics = jacobian_lens_terms(
                        bundle,
                        contexts[: train_config.lens_batch_size, 0],
                        horizons=selected_horizons,
                        ridge=train_config.probe_ridge,
                        shuffled=variant == "shuffled_lens",
                        encoded_states=encoded_states[
                            : train_config.lens_batch_size, 0
                        ],
                    )
                    if train_config.gradient_accumulation > 1:
                        lens_terms = {
                            name: value * train_config.gradient_accumulation
                            for name, value in lens_terms.items()
                        }
                    micro_loss_config = loss_config
                    require_micro_lens = True
                elif variant != "no_jacobian":
                    micro_loss_config = replace(
                        loss_config,
                        jacobian_bridge_weight=0.0,
                        oddness_weight=0.0,
                        manifold_cycle_weight=0.0,
                    )
                    require_micro_lens = False
                else:
                    micro_loss_config = loss_config
                    require_micro_lens = False
                loss, metrics = direct_video_objective(
                    bundle.model,
                    contexts,
                    frames,
                    class_weights,
                    micro_loss_config,
                    lens_terms=lens_terms,
                    require_lens_terms=require_micro_lens,
                    encoded_states=encoded_states,
                )
                scaled = loss / train_config.gradient_accumulation
                scaled.backward()
                for name, value in {**metrics, **lens_metrics}.items():
                    accumulated[name] = accumulated.get(name, 0.0) + float(value.detach())
                    accumulated_counts[name] = accumulated_counts.get(name, 0) + 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, train_config.gradient_clip
            )
            optimizer.step()
            bundle.model.encoder.assert_backbone_frozen()
            if bundle.model.encoder.sealed_backbone_hash != initial_backbone_hash:
                raise AssertionError("backbone hash changed during direct training")

            if step == 1 or step % train_config.log_every == 0 or step == train_config.steps:
                elapsed = time.perf_counter() - started
                record = {
                    "stage": "joint_direct_jacobian_poisson_ph",
                    "system": system.name,
                    "variant": variant,
                    "step": step,
                    "steps": train_config.steps,
                    "learningRate": learning_rate,
                    "gradientNorm": float(gradient_norm),
                    "lensSampled": use_lens,
                    "seconds": elapsed,
                    "estimatedSeconds": elapsed / max(step - start_step + 1, 1) * (
                        train_config.steps - start_step + 1
                    ),
                    **{
                        name: value / accumulated_counts[name]
                        for name, value in accumulated.items()
                    },
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

            validation = None
            if step % train_config.validation_every == 0 or step == train_config.steps:
                validation = pixels_only_validation_score(
                    bundle,
                    validation_suite,
                    class_weights,
                    loss_config,
                    train_config,
                    device,
                    variant=variant,
                )
                validation_record = {
                    "stage": "pixels_only_validation",
                    "system": system.name,
                    "variant": variant,
                    "step": step,
                    **validation,
                }
                print(json.dumps(validation_record), flush=True)
                log_file.write(json.dumps(validation_record) + "\n")
                log_file.flush()
                candidate_eligible = bool(validation["structureEligible"])
                candidate_rank = (not candidate_eligible, validation["score"])
                current_rank = (not best_structure_eligible, best_validation)
                if candidate_rank < current_rank:
                    best_validation = validation["score"]
                    best_structure_eligible = candidate_eligible
                    _atomic_torch_save(
                        _checkpoint_payload(
                            bundle,
                            optimizer,
                            step=step,
                            best_validation=best_validation,
                            best_structure_eligible=best_structure_eligible,
                            variant=variant,
                            system=system,
                            train_config=train_config,
                            loss_config=loss_config,
                            data_seal=data_seal,
                            source_tree_sha256=source_tree_sha256,
                            include_training_state=False,
                        ),
                        output_dir / "best.pt",
                    )
            if step % train_config.checkpoint_every == 0 or step == train_config.steps:
                _atomic_torch_save(
                    _checkpoint_payload(
                        bundle,
                        optimizer,
                        step=step,
                        best_validation=best_validation,
                        best_structure_eligible=best_structure_eligible,
                        variant=variant,
                        system=system,
                        train_config=train_config,
                        loss_config=loss_config,
                        data_seal=data_seal,
                        source_tree_sha256=source_tree_sha256,
                        include_training_state=True,
                    ),
                    last_path,
                )

    best_path = output_dir / "best.pt"
    best = torch.load(best_path, map_location=device, weights_only=True)
    runtime_trace.record_file_read(
        best_path,
        role=f"direct_selected_checkpoint:{variant}",
        serialized_keys=tuple(sorted(best)) if type(best) is dict else (),
    )
    _validate_direct_checkpoint(
        best,
        bundle,
        variant=variant,
        system=system,
        train_config=train_config,
        loss_config=loss_config,
        data_seal=data_seal,
        source_tree_sha256=source_tree_sha256,
        include_training_state=False,
    )
    bundle.model.load_state_dict(best["model"])
    bundle.write_field.load_state_dict(best["writeField"])
    bundle.response_frame.load_state_dict(best["responseFrame"])
    bundle.cotangent_frame.load_state_dict(best["cotangentFrame"])
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    for parameter in bundle.write_field.parameters():
        parameter.requires_grad_(False)
    for module in (bundle.response_frame, bundle.cotangent_frame):
        module.eval().requires_grad_(False)
    bundle.model.eval()
    bundle.write_field.eval()
    bundle.model.encoder.assert_backbone_frozen()
    runtime_trace.record_backbone_boundary(
        phase=f"direct:{variant}",
        boundary="selected_checkpoint",
        sha256=module_tensor_hash(bundle.model.encoder.backbone),
    )
    runtime_trace_seal = runtime_trace.snapshot().to_dict()
    if owns_runtime_trace:
        runtime_trace.close()
    summary = {
        "system": system.name,
        "variant": variant,
        "bestStep": int(best["step"]),
        "bestValidation": float(best["bestValidation"]),
        "bestStructureEligible": bool(best.get("bestStructureEligible", False)),
        "seconds": time.perf_counter() - started,
        "backboneHashBefore": initial_backbone_hash,
        "backboneHashAfter": module_tensor_hash(bundle.model.encoder.backbone),
        "actionGradientUpdates": 0,
        "physicalStateGradientUpdates": 0,
        "trainableParameters": sum(parameter.numel() for parameter in parameters),
        "sourceTreeSha256": source_tree_sha256,
        "runtimeTrace": runtime_trace_seal,
    }
    _atomic_json_save(summary, output_dir / "training-summary.json")
    return summary


@torch.no_grad()
def encode_pixel_suite(
    model: DirectVisualPoissonPH,
    suite: dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    contexts = suite["pixelContexts"]
    flat = contexts.reshape(-1, *contexts.shape[-3:])
    encoded = []
    for start in range(0, flat.shape[0], batch_size):
        encoded.append(model.encode(flat[start : start + batch_size].to(device).long()).cpu())
    return torch.cat(encoded).reshape(*contexts.shape[:2], -1)


def _matched_baseline_hidden_size(
    target_parameters: int,
    system: DirectSystemSpec,
) -> int:
    candidates = range(8, 1_024)
    return min(
        candidates,
        key=lambda hidden: abs(
            parameter_count(
                UnstructuredLatentEffortDynamics(
                    system.state_size,
                    system.port_size,
                    hidden,
                    dt=system.dt,
                    hidden_layers=2,
                )
            )
            + parameter_count(
                LatentEffortInference(
                    LatentEffortConfig(
                        system.state_size,
                        system.port_size,
                        hidden_size=hidden,
                        hidden_layers=2,
                    )
                )
            )
            - target_parameters
        ),
    )


_BASELINE_EVALUATION_KEYS = frozenset(
    {
        "kind",
        "system",
        "actionChannels",
        "physicalStateChannels",
        "optimizationTensorKeys",
        "trainConfig",
        "hiddenSize",
        "step",
        "bestValidation",
        "bestMetrics",
        "dynamics",
        "inference",
        "dataSeal",
        "fullModelHash",
        "sourceTreeSha256",
    }
)


@torch.no_grad()
def unstructured_baseline_validation_score(
    structured: DirectVisualPoissonPH,
    dynamics: UnstructuredLatentEffortDynamics,
    inference: LatentEffortInference,
    encoded_validation_states: torch.Tensor,
    validation_frames: torch.Tensor,
    state_scale: torch.Tensor,
    class_weights: torch.Tensor,
    config: BaselineTrainingConfig,
    device: torch.device,
) -> dict[str, float | int]:
    """Deterministic full validation using latent and rendered horizon-8 error."""

    if encoded_validation_states.ndim != 3 or validation_frames.ndim != 4:
        raise ValueError("baseline validation tensors have invalid ranks")
    if encoded_validation_states.shape[:2] != validation_frames.shape[:2]:
        raise ValueError("baseline validation state/frame axes differ")
    horizon = 8
    if encoded_validation_states.shape[1] < horizon + 1:
        raise ValueError("baseline validation requires a registered horizon of eight")
    if structured.training or any(parameter.requires_grad for parameter in structured.parameters()):
        raise ValueError("the full structured model must remain frozen during baseline validation")
    dynamics.eval()
    inference.eval()
    latent_total = 0.0
    pixel_total = 0.0
    sample_total = 0
    scale = state_scale.to(device=device, dtype=torch.float32).clamp_min(0.05)
    for start in range(0, encoded_validation_states.shape[0], config.validation_batch_size):
        stop = min(start + config.validation_batch_size, encoded_validation_states.shape[0])
        states = encoded_validation_states[start:stop].to(device).float()
        frames = validation_frames[start:stop].to(device).long()
        efforts = inference(states[:, :horizon], states[:, 1 : horizon + 1])
        current = states[:, 0]
        rollout = []
        for transition in range(horizon):
            current = dynamics(current, efforts[:, transition])
            rollout.append(current)
        prediction = torch.stack(rollout, dim=1)
        latent = ((prediction - states[:, 1 : horizon + 1]) / scale).square().mean()
        logits = structured.render(prediction.flatten(0, 1)).reshape(
            prediction.shape[0], horizon, -1, frames.shape[-2], frames.shape[-1]
        )
        pixel = weighted_pixel_cross_entropy(
            logits, frames[:, 1 : horizon + 1], class_weights
        )
        count = stop - start
        latent_total += float(latent) * count
        pixel_total += float(pixel) * count
        sample_total += count
    if sample_total != encoded_validation_states.shape[0] or sample_total < 1:
        raise AssertionError("baseline validation did not cover its sealed split exactly")
    latent_mean = latent_total / sample_total
    pixel_mean = pixel_total / sample_total
    metrics: dict[str, float | int] = {
        "horizon": horizon,
        "samples": sample_total,
        "latentRolloutMSE": latent_mean,
        "horizon8WeightedPixelCrossEntropy": pixel_mean,
        "selectionScore": latent_mean + pixel_mean,
    }
    if not all(
        math.isfinite(float(value)) for name, value in metrics.items() if name not in {"horizon", "samples"}
    ):
        raise ValueError("baseline validation produced a non-finite metric")
    return metrics


def _baseline_checkpoint_payload(
    dynamics: UnstructuredLatentEffortDynamics,
    inference: LatentEffortInference,
    optimizer: torch.optim.Optimizer,
    *,
    hidden_size: int,
    system: DirectSystemSpec,
    config: BaselineTrainingConfig,
    step: int,
    best_validation: float,
    best_metrics: Mapping[str, float | int],
    data_seal: Mapping[str, str],
    full_model_hash: str,
    source_tree_sha256: str,
    include_training_state: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "unstructured_action_free_latent_effort_baseline",
        "system": asdict(system),
        "actionChannels": 0,
        "physicalStateChannels": 0,
        "optimizationTensorKeys": ["encodedPixelStates"],
        "trainConfig": asdict(config),
        "hiddenSize": hidden_size,
        "step": step,
        "bestValidation": best_validation,
        "bestMetrics": dict(best_metrics),
        "dynamics": dict(dynamics.state_dict()),
        "inference": dict(inference.state_dict()),
        "dataSeal": dict(data_seal),
        "fullModelHash": full_model_hash,
        "sourceTreeSha256": _resolved_source_tree_sha256(source_tree_sha256),
    }
    if include_training_state:
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        payload["optimizer"] = optimizer.state_dict()
        payload["rngState"] = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": {
                "algorithm": numpy_state[0],
                "state": torch.from_numpy(numpy_state[1].astype(np.int64)),
                "position": int(numpy_state[2]),
                "hasGaussian": int(numpy_state[3]),
                "cachedGaussian": float(numpy_state[4]),
            },
            "python": {
                "version": int(python_state[0]),
                "state": torch.tensor(python_state[1], dtype=torch.int64),
                "gaussian": None if python_state[2] is None else float(python_state[2]),
            },
        }
    return payload


def _validate_baseline_training_checkpoint(
    payload: Any,
    dynamics: UnstructuredLatentEffortDynamics,
    inference: LatentEffortInference,
    optimizer: torch.optim.Optimizer,
    *,
    hidden_size: int,
    system: DirectSystemSpec,
    config: BaselineTrainingConfig,
    data_seal: Mapping[str, str],
    full_model_hash: str,
    source_tree_sha256: str | None = None,
    include_training_state: bool,
) -> None:
    expected_source_tree_sha256 = _resolved_source_tree_sha256(source_tree_sha256)
    expected_keys = set(_BASELINE_EVALUATION_KEYS)
    if include_training_state:
        expected_keys.update(("optimizer", "rngState"))
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError("baseline checkpoint schema is not exact")
    if (
        payload["kind"] != "unstructured_action_free_latent_effort_baseline"
        or payload["system"] != asdict(system)
        or payload["actionChannels"] != 0
        or payload["physicalStateChannels"] != 0
        or payload["optimizationTensorKeys"]
        != ["encodedPixelStates"]
        or payload["trainConfig"] != asdict(config)
        or payload["hiddenSize"] != hidden_size
        or payload["dataSeal"] != dict(data_seal)
        or payload["fullModelHash"] != full_model_hash
        or payload["sourceTreeSha256"] != expected_source_tree_sha256
    ):
        raise ValueError("baseline checkpoint provenance mismatch")
    if type(payload["step"]) is not int or not 1 <= payload["step"] <= config.steps:
        raise ValueError("baseline checkpoint step is invalid")
    if type(payload["bestValidation"]) not in (int, float) or not math.isfinite(
        float(payload["bestValidation"])
    ):
        raise ValueError("baseline checkpoint best validation is invalid")
    metrics = payload["bestMetrics"]
    if type(metrics) is not dict or set(metrics) != {
        "horizon",
        "samples",
        "latentRolloutMSE",
        "horizon8WeightedPixelCrossEntropy",
        "selectionScore",
    } or metrics["horizon"] != 8 or type(metrics["samples"]) is not int:
        raise ValueError("baseline checkpoint validation metric schema is invalid")
    if not all(
        type(metrics[name]) in (int, float) and math.isfinite(float(metrics[name]))
        for name in (
            "latentRolloutMSE",
            "horizon8WeightedPixelCrossEntropy",
            "selectionScore",
        )
    ) or float(metrics["selectionScore"]) != float(payload["bestValidation"]):
        raise ValueError("baseline checkpoint validation metrics are inconsistent")
    for name, state, module in (
        ("dynamics", payload["dynamics"], dynamics),
        ("inference", payload["inference"], inference),
    ):
        reference = module.state_dict()
        if type(state) is not dict or set(state) != set(reference):
            raise ValueError(f"baseline checkpoint {name} state schema mismatch")
        for field, tensor in state.items():
            expected = reference[field]
            if (
                type(tensor) is not torch.Tensor
                or tensor.shape != expected.shape
                or tensor.dtype != expected.dtype
                or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
            ):
                raise ValueError(f"baseline checkpoint {name}.{field} tensor is invalid")
    if include_training_state:
        parameters = list(dynamics.parameters()) + list(inference.parameters())
        _validate_optimizer_state(
            payload["optimizer"], optimizer, parameters, checkpoint_step=payload["step"]
        )
        _validate_safe_rng_state(payload["rngState"])


def train_unstructured_action_free_baseline(
    structured: DirectVisualPoissonPH,
    encoded_fit_states: torch.Tensor,
    encoded_validation_states: torch.Tensor,
    validation_frames: torch.Tensor,
    class_weights: torch.Tensor,
    system: DirectSystemSpec,
    output_dir: Path,
    config: BaselineTrainingConfig,
    device: torch.device,
    *,
    data_seal: Mapping[str, str],
    source_tree_sha256: str | None = None,
    runtime_trace: RuntimeFirewallTrace | None = None,
) -> tuple[UnstructuredLatentEffortDynamics, LatentEffortInference, dict[str, Any]]:
    """Matched black-box dynamics with pixels-only best-checkpoint selection."""

    if structured.training or any(parameter.requires_grad for parameter in structured.parameters()):
        raise ValueError("the full model must be frozen before baseline fitting")
    expected_data_seal = {
        "system",
        "fitAggregateSha256",
        "fitSanitizedTensorSha256",
        "validationAggregateSha256",
        "validationSanitizedTensorSha256",
    }
    if type(data_seal) is not dict or set(data_seal) != expected_data_seal:
        raise ValueError("baseline data seal schema is not exact")
    source_tree_sha256 = _resolved_source_tree_sha256(source_tree_sha256)
    seed_everything(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    owns_runtime_trace = runtime_trace is None
    if runtime_trace is None:
        runtime_trace = RuntimeFirewallTrace(
            output_dir / "firewall-trace.jsonl",
            stage=f"baseline:{system.name}",
            source_tree_sha256=source_tree_sha256,
        )
    target = parameter_count(structured.core) + parameter_count(structured.effort_inference)
    hidden = _matched_baseline_hidden_size(target, system)
    dynamics = UnstructuredLatentEffortDynamics(
        system.state_size,
        system.port_size,
        hidden,
        dt=system.dt,
        hidden_layers=2,
    ).to(device)
    inference = LatentEffortInference(
        LatentEffortConfig(
            system.state_size,
            system.port_size,
            hidden_size=hidden,
            hidden_layers=2,
        )
    ).to(device)
    baseline_parameters = parameter_count(dynamics) + parameter_count(inference)
    relative_parameter_gap = abs(target - baseline_parameters) / max(target, 1)
    if relative_parameter_gap > 0.01:
        raise ValueError(
            "unstructured baseline dynamics+inverse capacity differs by more than 1%"
        )
    parameters = list(dynamics.parameters()) + list(inference.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    runtime_trace.record_optimizer(
        phase="baseline:unstructured",
        named_parameters={
            **{
                f"dynamics.{name}": parameter
                for name, parameter in dynamics.named_parameters()
            },
            **{
                f"inference.{name}": parameter
                for name, parameter in inference.named_parameters()
            },
        },
        protected_parameters={
            f"structured.encoder.backbone.{name}": parameter
            for name, parameter in structured.encoder.backbone.named_parameters()
        },
    )
    states = encoded_fit_states.to(device).float()
    if states.ndim != 3 or states.shape[-1] != system.state_size:
        raise ValueError("encoded fit states have an invalid shape")
    state_scale = states.detach().reshape(-1, system.state_size).std(
        dim=0, unbiased=False
    ).clamp_min(0.05)
    full_model_hash = module_tensor_hash(structured)
    runtime_trace.record_backbone_boundary(
        phase="baseline:unstructured",
        boundary="start",
        sha256=module_tensor_hash(structured.encoder.backbone),
    )
    start_step = 1
    best_validation = math.inf
    best_metrics: dict[str, float | int] = {
        "horizon": 8,
        "samples": 0,
        "latentRolloutMSE": math.inf,
        "horizon8WeightedPixelCrossEntropy": math.inf,
        "selectionScore": math.inf,
    }
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    if last_path.exists() != best_path.exists():
        raise ValueError(
            "baseline resume requires the paired best.pt and last.pt checkpoints"
        )
    if last_path.exists():
        last = torch.load(last_path, map_location=device, weights_only=True)
        runtime_trace.record_file_read(
            last_path,
            role="baseline_resume_checkpoint:last",
            serialized_keys=tuple(sorted(last)) if type(last) is dict else (),
        )
        _validate_baseline_training_checkpoint(
            last,
            dynamics,
            inference,
            optimizer,
            hidden_size=hidden,
            system=system,
            config=config,
            data_seal=data_seal,
            full_model_hash=full_model_hash,
            source_tree_sha256=source_tree_sha256,
            include_training_state=True,
        )
        sealed_best = torch.load(best_path, map_location=device, weights_only=True)
        runtime_trace.record_file_read(
            best_path,
            role="baseline_resume_checkpoint:best",
            serialized_keys=(
                tuple(sorted(sealed_best)) if type(sealed_best) is dict else ()
            ),
        )
        _validate_baseline_training_checkpoint(
            sealed_best,
            dynamics,
            inference,
            optimizer,
            hidden_size=hidden,
            system=system,
            config=config,
            data_seal=data_seal,
            full_model_hash=full_model_hash,
            source_tree_sha256=source_tree_sha256,
            include_training_state=False,
        )
        if (
            sealed_best["step"] > last["step"]
            or float(sealed_best["bestValidation"])
            != float(last["bestValidation"])
            or sealed_best["bestMetrics"] != last["bestMetrics"]
        ):
            raise ValueError("baseline best/last validation lineage is inconsistent")
        dynamics.load_state_dict(last["dynamics"], strict=True)
        inference.load_state_dict(last["inference"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        _restore_safe_rng_state(last["rngState"])
        start_step = int(last["step"]) + 1
        best_validation = float(last["bestValidation"])
        best_metrics = dict(last["bestMetrics"])
    started = time.perf_counter()
    log_path = output_dir / "train.jsonl"
    log_mode = "a" if start_step > 1 else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        for step in range(start_step, config.steps + 1):
            dynamics.train()
            inference.train()
            rows = torch.randint(0, states.shape[0], (config.batch_size,), device=device)
            target_states = states[rows]
            runtime_trace.record_gradient_batch(
                phase="baseline:unstructured",
                step=step,
                tensors={"encodedPixelStates": target_states},
            )
            efforts = inference(target_states[:, :-1], target_states[:, 1:])
            current = target_states[:, 0]
            rollout = []
            for transition in range(efforts.shape[1]):
                current = dynamics(current, efforts[:, transition])
                rollout.append(current)
            prediction = torch.stack(rollout, dim=1)
            latent = ((prediction - target_states[:, 1:]) / state_scale).square().mean()
            statistics = latent_effort_statistics(efforts)
            loss = latent + 0.10 * (
                statistics["total"] + 0.25 * statistics["temporal"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if (
                step % config.validation_every == 0
                or step % config.checkpoint_every == 0
                or step == config.steps
            ) and module_tensor_hash(structured) != full_model_hash:
                raise AssertionError("full model changed during baseline optimization")
            if step == 1 or step % config.log_every == 0 or step == config.steps:
                elapsed = time.perf_counter() - started
                record = {
                    "stage": "unstructured_action_free_baseline",
                    "system": system.name,
                    "step": step,
                    "steps": config.steps,
                    "loss": float(loss.detach()),
                    "latent": float(latent.detach()),
                    "gradientNorm": float(gradient_norm),
                    "seconds": elapsed,
                    "estimatedSeconds": elapsed / max(step - start_step + 1, 1)
                    * (config.steps - start_step + 1),
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
            if step % config.validation_every == 0 or step == config.steps:
                metrics = unstructured_baseline_validation_score(
                    structured,
                    dynamics,
                    inference,
                    encoded_validation_states,
                    validation_frames,
                    state_scale,
                    class_weights,
                    config,
                    device,
                )
                score = float(metrics["selectionScore"])
                record = {
                    "stage": "unstructured_pixels_only_validation",
                    "system": system.name,
                    "step": step,
                    **metrics,
                }
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if score < best_validation:
                    best_validation = score
                    best_metrics = dict(metrics)
                    _atomic_torch_save(
                        _baseline_checkpoint_payload(
                            dynamics,
                            inference,
                            optimizer,
                            hidden_size=hidden,
                            system=system,
                            config=config,
                            step=step,
                            best_validation=best_validation,
                            best_metrics=best_metrics,
                            data_seal=data_seal,
                            full_model_hash=full_model_hash,
                            source_tree_sha256=source_tree_sha256,
                            include_training_state=False,
                        ),
                        best_path,
                    )
            if step % config.checkpoint_every == 0 or step == config.steps:
                _atomic_torch_save(
                    _baseline_checkpoint_payload(
                        dynamics,
                        inference,
                        optimizer,
                        hidden_size=hidden,
                        system=system,
                        config=config,
                        step=step,
                        best_validation=best_validation,
                        best_metrics=best_metrics,
                        data_seal=data_seal,
                        full_model_hash=full_model_hash,
                        source_tree_sha256=source_tree_sha256,
                        include_training_state=True,
                    ),
                    last_path,
                )
    if not best_path.exists():
        raise ValueError("baseline training completed without a validation-selected best.pt")
    best = torch.load(best_path, map_location=device, weights_only=True)
    runtime_trace.record_file_read(
        best_path,
        role="baseline_selected_checkpoint",
        serialized_keys=tuple(sorted(best)) if type(best) is dict else (),
    )
    _validate_baseline_training_checkpoint(
        best,
        dynamics,
        inference,
        optimizer,
        hidden_size=hidden,
        system=system,
        config=config,
        data_seal=data_seal,
        full_model_hash=full_model_hash,
        source_tree_sha256=source_tree_sha256,
        include_training_state=False,
    )
    dynamics.load_state_dict(best["dynamics"], strict=True)
    inference.load_state_dict(best["inference"], strict=True)
    dynamics.eval().requires_grad_(False)
    inference.eval().requires_grad_(False)
    runtime_trace.record_backbone_boundary(
        phase="baseline:unstructured",
        boundary="selected_checkpoint",
        sha256=module_tensor_hash(structured.encoder.backbone),
    )
    runtime_trace_seal = runtime_trace.snapshot().to_dict()
    if owns_runtime_trace:
        runtime_trace.close()
    summary = {
        "system": system.name,
        "seed": config.seed,
        "trainConfig": asdict(config),
        "hiddenSize": hidden,
        "structuredParameters": target,
        "baselineParameters": baseline_parameters,
        "relativeParameterGap": relative_parameter_gap,
        "seconds": time.perf_counter() - started,
        "bestStep": int(best["step"]),
        "bestValidation": float(best["bestValidation"]),
        "validationMetrics": dict(best["bestMetrics"]),
        "fullModelHash": full_model_hash,
        "actionGradientUpdates": 0,
        "physicalStateGradientUpdates": 0,
        "sourceTreeSha256": source_tree_sha256,
        "runtimeTrace": runtime_trace_seal,
    }
    _atomic_json_save(summary, output_dir / "summary.json")
    return dynamics, inference, summary


__all__ = [
    "BaselineTrainingConfig",
    "DIRECT_SYSTEMS",
    "DirectModelBundle",
    "DirectSystemSpec",
    "DirectTrainingConfig",
    "Variant",
    "build_direct_bundle",
    "encode_pixel_suite",
    "jacobian_lens_terms",
    "pixels_only_validation_score",
    "seed_everything",
    "train_direct_bundle",
    "train_unstructured_action_free_baseline",
]
