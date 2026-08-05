"""Independent pixels-to-unstructured world-model baseline for Experiment F.

This module deliberately does *not* accept an already encoded latent data set.
The baseline owns its visual readout, renderer, inverse latent-effort head,
state-dependent drift and state-dependent port.  It is therefore a genuine
world model trained from the same sanitized categorical pixels as the direct
Poisson port-Hamiltonian model, rather than a post-hoc dynamics head fitted in
the structured model's representation.

Only the action-free video-transformer tensor values and the mathematical
Jacobian-lens construction are shared.  Every trainable tensor is distinct.
The tangent bridge is retained; the cotangent Poisson bridge and all
Hamiltonian-only penalties are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

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
)
from .direct_jacobian_port_extractor import (
    EmpiricalTangentArtifact,
    EmpiricalTangentConfig,
    FrozenEmpiricalJacobianActivationPort,
)
from .direct_visual_poisson_ph import (
    DirectVideoLossConfig,
    PersistentOrthogonalPortFrame,
    WholeStreamEncoderConfig,
    WholeStreamFrozenEncoder,
    port_frame_regularizers,
    state_effort_independence_loss,
    state_effort_second_moment_independence_loss,
    state_whitening_loss,
)
from .latent_patch_renderer import LatentPatchTransformerRenderer
from .tensor_provenance import module_tensor_hash, parameter_count
from .pixel_direct_model import DirectPixelTransformer


@dataclass(frozen=True)
class IndependentUnstructuredArchitecture:
    """Architecture shared by training and strict post-freeze reconstruction."""

    state_size: int
    port_size: int
    dt: float
    lens_block: int
    state_hidden_size: int
    renderer_hidden_size: int
    renderer_depth: int
    renderer_heads: int
    dynamics_hidden_layers: int
    write_hidden_size: int
    write_hidden_layers: int
    lens_horizons: tuple[int, ...]
    initialization_seed: int

    def __post_init__(self) -> None:
        integers = (
            self.state_size,
            self.port_size,
            self.state_hidden_size,
            self.renderer_hidden_size,
            self.renderer_depth,
            self.renderer_heads,
            self.dynamics_hidden_layers,
            self.write_hidden_size,
            self.write_hidden_layers,
        )
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError("independent-baseline dimensions must be positive integers")
        if type(self.lens_block) is not int or self.lens_block < 0:
            raise ValueError("lens_block must be a non-negative integer")
        if type(self.initialization_seed) is not int:
            raise ValueError("initialization_seed must be an integer")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.port_size > self.state_size:
            raise ValueError("port_size cannot exceed state_size")
        if self.renderer_hidden_size % self.renderer_heads:
            raise ValueError("renderer_hidden_size must be divisible by renderer_heads")
        if (
            type(self.lens_horizons) is not tuple
            or not self.lens_horizons
            or any(type(value) is not int or value < 1 for value in self.lens_horizons)
            or tuple(sorted(set(self.lens_horizons))) != self.lens_horizons
        ):
            raise ValueError("lens_horizons must be sorted unique positive integers")


@dataclass(frozen=True)
class HomologousInitialization:
    """Detached initial tensors copied from the untrained structured model.

    Copying values is not parameter sharing: the baseline modules are created
    independently and retain different tensor identities.  The hashes make
    the controlled common starting point auditable without consulting a
    trained structured checkpoint.
    """

    encoder_readout: Mapping[str, torch.Tensor]
    renderer: Mapping[str, torch.Tensor]
    effort_inference: Mapping[str, torch.Tensor]
    write_field: Mapping[str, torch.Tensor]
    response_frame: Mapping[str, torch.Tensor]
    hashes: Mapping[str, str]
    reference_initialization_seed: int


class IndependentUnstructuredWorldModel(nn.Module):
    """Independent visual bottleneck followed by a black-box latent ODE."""

    def __init__(
        self,
        encoder: WholeStreamFrozenEncoder,
        renderer: LatentPatchTransformerRenderer,
        dynamics: UnstructuredLatentEffortDynamics,
        effort_inference: LatentEffortInference,
    ) -> None:
        super().__init__()
        if encoder.state_size != dynamics.state_size:
            raise ValueError("encoder and unstructured dynamics state dimensions differ")
        if renderer.state_size != dynamics.state_size:
            raise ValueError("renderer and unstructured dynamics state dimensions differ")
        if effort_inference.config.state_size != dynamics.state_size:
            raise ValueError("inverse head and dynamics state dimensions differ")
        if effort_inference.config.effort_size != dynamics.effort_size:
            raise ValueError("inverse head and dynamics effort dimensions differ")
        self.encoder = encoder
        self.renderer = renderer
        self.dynamics = dynamics
        self.effort_inference = effort_inference

    @property
    def state_size(self) -> int:
        return self.dynamics.state_size

    @property
    def port_size(self) -> int:
        return self.dynamics.effort_size

    def encode(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.encoder(contexts)

    def infer_latent_effort(
        self, present: torch.Tensor, successor: torch.Tensor
    ) -> torch.Tensor:
        return self.effort_inference(present, successor)

    def step(self, state: torch.Tensor, latent_effort: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=state.device.type, enabled=False):
            return self.dynamics.integrate(state.float(), latent_effort.float())

    def render(self, state: torch.Tensor) -> torch.Tensor:
        return self.renderer(state)


@dataclass(frozen=True)
class IndependentUnstructuredBundle:
    model: IndependentUnstructuredWorldModel
    write_field: FrozenEmpiricalJacobianActivationPort
    lens: FrozenSoftPixelActivationLens
    probes: PixelChangeProbeBank
    response_frame: PersistentOrthogonalPortFrame
    architecture: IndependentUnstructuredArchitecture
    dynamics_hidden_size: int
    target_trainable_parameters: int
    trainable_parameters: int
    relative_parameter_gap: float
    homologous_initialization_hashes: Mapping[str, str]


def _detached_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _encoder_readout_state(encoder: WholeStreamFrozenEncoder) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for prefix, module in (("pool_score", encoder.pool_score), ("readout", encoder.readout)):
        for name, tensor in module.state_dict().items():
            values[f"{prefix}.{name}"] = tensor.detach().cpu().clone()
    return values


def _load_encoder_readout_state(
    encoder: WholeStreamFrozenEncoder, state: Mapping[str, torch.Tensor]
) -> None:
    expected = _encoder_readout_state(encoder)
    if set(state) != set(expected):
        raise ValueError("homologous encoder-readout state schema mismatch")
    pool_prefix = "pool_score."
    readout_prefix = "readout."
    encoder.pool_score.load_state_dict(
        {name.removeprefix(pool_prefix): state[name] for name in state if name.startswith(pool_prefix)},
        strict=True,
    )
    encoder.readout.load_state_dict(
        {name.removeprefix(readout_prefix): state[name] for name in state if name.startswith(readout_prefix)},
        strict=True,
    )


def capture_homologous_initialization(
    *,
    encoder: WholeStreamFrozenEncoder,
    renderer: nn.Module,
    effort_inference: nn.Module,
    write_field: nn.Module,
    response_frame: nn.Module,
    reference_initialization_seed: int,
) -> HomologousInitialization:
    """Capture only common *initial* tensors from a reference architecture."""

    if type(reference_initialization_seed) is not int:
        raise ValueError("reference initialization seed must be an integer")
    encoder.assert_backbone_frozen()
    modules = {
        "encoderReadout": nn.ModuleDict(
            {"pool_score": encoder.pool_score, "readout": encoder.readout}
        ),
        "renderer": renderer,
        "effortInference": effort_inference,
        "writeField": write_field,
        "responseFrame": response_frame,
    }
    return HomologousInitialization(
        encoder_readout=_encoder_readout_state(encoder),
        renderer=_detached_state(renderer),
        effort_inference=_detached_state(effort_inference),
        write_field=_detached_state(write_field),
        response_frame=_detached_state(response_frame),
        hashes={name: module_tensor_hash(module) for name, module in modules.items()},
        reference_initialization_seed=reference_initialization_seed,
    )


def _homologous_hashes(bundle: IndependentUnstructuredBundle) -> dict[str, str]:
    model = bundle.model
    modules = {
        "encoderReadout": nn.ModuleDict(
            {"pool_score": model.encoder.pool_score, "readout": model.encoder.readout}
        ),
        "renderer": model.renderer,
        "effortInference": model.effort_inference,
        "writeField": bundle.write_field,
        "responseFrame": bundle.response_frame,
    }
    return {name: module_tensor_hash(module) for name, module in modules.items()}


def unstructured_dynamics_parameter_count(
    state_size: int,
    port_size: int,
    hidden_size: int,
    hidden_layers: int,
) -> int:
    """Closed-form trainable count for two independent state-conditioned MLPs."""

    if min(state_size, port_size, hidden_size, hidden_layers) < 1:
        raise ValueError("unstructured dimensions must be positive")

    def mlp(output_size: int) -> int:
        return (
            state_size * hidden_size
            + hidden_size
            + (hidden_layers - 1) * (hidden_size * hidden_size + hidden_size)
            + hidden_size * output_size
            + output_size
        )

    return mlp(state_size) + mlp(state_size * port_size)


def matched_dynamics_hidden_size(
    *,
    target_trainable_parameters: int,
    fixed_trainable_parameters: int,
    state_size: int,
    port_size: int,
    hidden_layers: int,
    maximum_hidden_size: int = 4_096,
) -> tuple[int, int, float]:
    """Choose the closest total parameter budget and enforce the locked 1%."""

    if target_trainable_parameters < 1 or fixed_trainable_parameters < 0:
        raise ValueError("capacity targets must be non-negative and non-zero")
    candidates = range(1, maximum_hidden_size + 1)
    hidden = min(
        candidates,
        key=lambda value: abs(
            fixed_trainable_parameters
            + unstructured_dynamics_parameter_count(
                state_size, port_size, value, hidden_layers
            )
            - target_trainable_parameters
        ),
    )
    observed = fixed_trainable_parameters + unstructured_dynamics_parameter_count(
        state_size, port_size, hidden, hidden_layers
    )
    gap = abs(observed - target_trainable_parameters) / target_trainable_parameters
    if gap > 0.01:
        raise ValueError(
            "independent unstructured world-model capacity differs by more than 1%"
        )
    return hidden, observed, gap


def _named_trainable_parameters(
    model: IndependentUnstructuredWorldModel,
    write_field: nn.Module,
    response_frame: PersistentOrthogonalPortFrame,
) -> list[tuple[str, nn.Parameter]]:
    backbone_ids = {id(value) for value in model.encoder.backbone.parameters()}
    named = [
        (f"model.{name}", value)
        for name, value in model.named_parameters()
        if value.requires_grad and id(value) not in backbone_ids
    ]
    named.extend(
        (f"writeField.{name}", value)
        for name, value in write_field.named_parameters()
        if value.requires_grad
    )
    named.extend(
        (f"responseFrame.{name}", value)
        for name, value in response_frame.named_parameters()
        if value.requires_grad
    )
    names = [name for name, _ in named]
    if len(names) != len(set(names)) or any("encoder.backbone" in name for name in names):
        raise AssertionError("independent baseline optimizer membership is invalid")
    return named


def build_independent_unstructured_bundle(
    backbone: DirectPixelTransformer,
    architecture: IndependentUnstructuredArchitecture,
    *,
    empirical_tangent: EmpiricalTangentArtifact,
    probes: PixelChangeProbeBank,
    tangent_config: EmpiricalTangentConfig,
    target_trainable_parameters: int,
    homologous_initialization: HomologousInitialization,
    device: torch.device,
) -> IndependentUnstructuredBundle:
    """Build a fully independent and total-capacity-matched visual baseline."""

    if architecture.lens_block >= len(backbone.blocks):
        raise ValueError("independent baseline lens block is outside the backbone")
    # A private deterministic fork makes baseline construction independent of
    # job ordering.  Common tensors are then overwritten by the exact captured
    # reference initialization; no trained structured tensor is consulted.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(architecture.initialization_seed)
        encoder = WholeStreamFrozenEncoder(
            backbone,
            WholeStreamEncoderConfig(
                state_size=architecture.state_size,
                readout_hidden_size=architecture.state_hidden_size,
                lens_block=architecture.lens_block,
            ),
        )
        renderer = LatentPatchTransformerRenderer(
            architecture.state_size,
            image_size=backbone.config.image_size,
            patch_size=backbone.config.patch_size,
            palette_size=backbone.config.palette_size,
            hidden_size=architecture.renderer_hidden_size,
            depth=architecture.renderer_depth,
            heads=architecture.renderer_heads,
        )
        inference = LatentEffortInference(
            LatentEffortConfig(
                state_size=architecture.state_size,
                effort_size=architecture.port_size,
                hidden_size=128,
                hidden_layers=2,
            )
        )
        activation = encoder.activation_shape
        write_field = FrozenEmpiricalJacobianActivationPort(
            empirical_tangent,
            history_frames=activation[0],
            patch_count=activation[1],
            hidden_size=activation[2],
            port_size=architecture.port_size,
            config=tangent_config,
        ).eval().requires_grad_(False)
        write_field.assert_frozen_parameter_free()
        response_frame = PersistentOrthogonalPortFrame(architecture.port_size)

        _load_encoder_readout_state(encoder, homologous_initialization.encoder_readout)
        renderer.load_state_dict(homologous_initialization.renderer, strict=True)
        inference.load_state_dict(homologous_initialization.effort_inference, strict=True)
        write_field.load_state_dict(homologous_initialization.write_field, strict=True)
        response_frame.load_state_dict(
            homologous_initialization.response_frame, strict=True
        )
        if (
            homologous_initialization.reference_initialization_seed
            != architecture.initialization_seed
        ):
            raise ValueError(
                "independent and reference initialization seeds are not identical"
            )

        # Count all independent trainables except the not-yet-built dynamics.
        placeholder = UnstructuredLatentEffortDynamics(
            architecture.state_size,
            architecture.port_size,
            1,
            dt=architecture.dt,
            hidden_layers=architecture.dynamics_hidden_layers,
        )
        temporary_model = IndependentUnstructuredWorldModel(
            encoder, renderer, placeholder, inference
        )
        fixed = sum(
            value.numel()
            for name, value in _named_trainable_parameters(
                temporary_model, write_field, response_frame
            )
            if not name.startswith("model.dynamics.")
        )
        hidden, observed, gap = matched_dynamics_hidden_size(
            target_trainable_parameters=target_trainable_parameters,
            fixed_trainable_parameters=fixed,
            state_size=architecture.state_size,
            port_size=architecture.port_size,
            hidden_layers=architecture.dynamics_hidden_layers,
        )
        dynamics = UnstructuredLatentEffortDynamics(
            architecture.state_size,
            architecture.port_size,
            hidden,
            dt=architecture.dt,
            hidden_layers=architecture.dynamics_hidden_layers,
        )
        model = IndependentUnstructuredWorldModel(
            encoder, renderer, dynamics, inference
        )
        lens = FrozenSoftPixelActivationLens(
            backbone,
            intervention_block=architecture.lens_block,
            horizons=architecture.lens_horizons,
        )

    model = model.to(device)
    write_field = write_field.to(device)
    response_frame = response_frame.to(device)
    lens = lens.to(device)
    probes = probes.to(device).eval().requires_grad_(False)
    bundle = IndependentUnstructuredBundle(
        model=model,
        write_field=write_field,
        lens=lens,
        probes=probes,
        response_frame=response_frame,
        architecture=architecture,
        dynamics_hidden_size=hidden,
        target_trainable_parameters=target_trainable_parameters,
        trainable_parameters=observed,
        relative_parameter_gap=gap,
        homologous_initialization_hashes=dict(homologous_initialization.hashes),
    )
    if sum(value.numel() for _, value in independent_named_parameters(bundle)) != observed:
        raise AssertionError("independent baseline parameter-count derivation drifted")
    if _homologous_hashes(bundle) != dict(homologous_initialization.hashes):
        raise AssertionError("independent baseline common initialization is not homologous")
    bundle.model.encoder.assert_backbone_frozen()
    return bundle


def independent_named_parameters(
    bundle: IndependentUnstructuredBundle,
) -> list[tuple[str, nn.Parameter]]:
    return _named_trainable_parameters(
        bundle.model, bundle.write_field, bundle.response_frame
    )


def independent_evaluation_modules(
    bundle: IndependentUnstructuredBundle,
) -> dict[str, nn.Module]:
    """Return the exact checkpoint modules, explicitly excluding the backbone.

    The frozen transformer is reconstructed from its separately authenticated
    action-free checkpoint.  Duplicating it inside this baseline artifact
    would obscure whether the baseline had silently changed its visual prior.
    """

    return {
        "encoderPoolScore": bundle.model.encoder.pool_score,
        "encoderReadout": bundle.model.encoder.readout,
        "renderer": bundle.model.renderer,
        "dynamics": bundle.model.dynamics,
        "effortInference": bundle.model.effort_inference,
        "writeField": bundle.write_field,
        "responseFrame": bundle.response_frame,
    }


def independent_tangent_lens_terms(
    bundle: IndependentUnstructuredBundle,
    pixel_context: torch.Tensor,
    *,
    horizons: tuple[int, ...],
    encoded_states: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Match frozen-transformer and unstructured latent pulse Jacobians."""

    model = bundle.model
    states = model.encode(pixel_context) if encoded_states is None else encoded_states
    expected = (pixel_context.shape[0], model.state_size)
    if tuple(states.shape) != expected:
        raise ValueError(f"expected encoded states shape {expected}")
    source_activation = model.encoder.prefix_activation(pixel_context).detach()
    activation_covectors = {
        horizon: value.detach()
        for horizon, value in activation_observable_covectors(
            bundle.lens,
            pixel_context,
            bundle.probes,
            horizons=horizons,
            create_graph=False,
        ).items()
    }
    extraction = bundle.write_field(activation_covectors, source_activation)
    write_basis = extraction.jacobian.write_basis
    frozen = bundle.lens.state_response_jacobians(
        pixel_context,
        write_basis,
        model.encoder.read_suffix_tokens,
        horizons=horizons,
        create_graph=True,
    )
    with differentiable_attention_backend(states):
        direct = direct_dynamics_pulse_responses(
            model.step,
            states,
            model.port_size,
            horizons=horizons,
            create_graph=True,
        )
    frozen_stack = torch.cat(tuple(frozen.jacobians[value] for value in horizons), dim=1)
    direct_stack = torch.cat(tuple(direct.jacobians[value] for value in horizons), dim=1)
    global_frozen = frozen_stack.reshape(1, -1, frozen_stack.shape[-1])
    global_direct = direct_stack.reshape(1, -1, direct_stack.shape[-1])
    alignment = basis_invariant_response_loss(global_frozen, global_direct)
    subspace = grassmann_response_loss(global_frozen, global_direct)
    frozen_normalized = global_frozen / torch.linalg.matrix_norm(
        global_frozen, ord="fro", dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)
    oriented = bundle.response_frame(global_direct)
    direct_normalized = oriented / torch.linalg.matrix_norm(
        oriented, ord="fro", dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)
    persistent = (frozen_normalized - direct_normalized).square().sum((-2, -1)).mean()
    proxies = bundle.lens.intervention_proxies(
        pixel_context, write_basis, amplitude=0.05
    )
    signal_floor = torch.relu(
        proxies.first_order_signal.new_tensor(1e-7) - proxies.first_order_signal
    ) / 1e-7
    terms = {
        "bridge": persistent + 0.10 * subspace,
        "oddness": proxies.odd_symmetry,
        "manifoldCycle": (
            proxies.manifold_cycle + proxies.current_frame_leakage + signal_floor
        ),
    }
    metrics = {
        "responseAlignment": alignment,
        "persistentResponseFrameAlignment": persistent,
        "responseSubspace": subspace,
        "writeOddness": proxies.odd_symmetry,
        "writeCurrentFrameLeakage": proxies.current_frame_leakage,
        "writeManifoldCycle": proxies.manifold_cycle,
        "writeFirstOrderSignal": proxies.first_order_signal,
        "minimumFrozenResponseSingularValue": torch.linalg.svdvals(
            frozen_stack.detach().float()
        ).amin(),
        "minimumUnstructuredResponseSingularValue": torch.linalg.svdvals(
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


def independent_video_objective(
    bundle: IndependentUnstructuredBundle,
    pixel_contexts: torch.Tensor,
    frames: torch.Tensor,
    class_weights: torch.Tensor,
    config: DirectVideoLossConfig,
    *,
    lens_terms: Mapping[str, torch.Tensor] | None,
    require_lens_terms: bool = True,
    encoded_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply every registered non-Hamiltonian loss to the independent WM."""

    if pixel_contexts.ndim != 5:
        raise ValueError("pixel_contexts must be [batch,time,history,height,width]")
    if frames.shape[:2] != pixel_contexts.shape[:2] or frames.ndim != 4:
        raise ValueError("contexts and frames must share [batch,time] axes")
    model = bundle.model
    states = model.encode(pixel_contexts) if encoded_states is None else encoded_states
    expected = (*pixel_contexts.shape[:2], model.state_size)
    if tuple(states.shape) != expected:
        raise ValueError(f"expected encoded states shape {expected}")
    if states.shape[1] < 2:
        raise ValueError("at least one transition is required")
    efforts = model.infer_latent_effort(states[:, :-1], states[:, 1:])
    reconstruction = weighted_pixel_cross_entropy(
        model.render(states), frames, class_weights
    )
    current = states[:, 0]
    rolled: list[torch.Tensor] = []
    for index in range(efforts.shape[1]):
        current = model.step(current, efforts[:, index])
        rolled.append(current)
    rollout = torch.stack(rolled, dim=1)
    scale = states.detach().reshape(-1, model.state_size).std(
        dim=0, unbiased=False
    ).clamp_min(0.05)
    rollout_latent = ((rollout - states[:, 1:]) / scale).square().mean()
    horizons = tuple(
        value for value in config.rollout_horizons if value <= rollout.shape[1]
    )
    if not horizons:
        raise ValueError("no registered rollout horizon fits this batch")
    indices = torch.tensor(tuple(value - 1 for value in horizons), device=frames.device)
    rollout_pixel = weighted_pixel_cross_entropy(
        model.render(rollout[:, indices]), frames[:, indices + 1], class_weights
    )
    effort_terms = latent_effort_statistics(
        efforts, target_variance=config.innovation_target_variance
    )
    linear_independence = state_effort_independence_loss(states[:, :-1], efforts)
    second_independence = state_effort_second_moment_independence_loss(
        states[:, :-1], efforts
    )
    independence = linear_independence + second_independence
    innovation = (
        effort_terms["total"]
        + 0.25 * effort_terms["temporal"]
        + independence
    )
    whitening = state_whitening_loss(states)
    frame, holonomy, port_rank = port_frame_regularizers(
        model.dynamics, states.float()
    )
    active_lens = any(
        value > 0.0
        for value in (
            config.jacobian_bridge_weight,
            config.oddness_weight,
            config.manifold_cycle_weight,
        )
    )
    if require_lens_terms and active_lens and lens_terms is None:
        raise RuntimeError("independent baseline requires its tangent Jacobian lens")
    lens_terms = {} if lens_terms is None else dict(lens_terms)
    required = {"bridge", "oddness", "manifoldCycle"}
    if require_lens_terms and active_lens and not required.issubset(lens_terms):
        raise RuntimeError(
            f"missing independent tangent-lens terms: {sorted(required - set(lens_terms))}"
        )
    zero = states.new_zeros(())
    bridge = lens_terms.get("bridge", zero)
    oddness = lens_terms.get("oddness", zero)
    manifold = lens_terms.get("manifoldCycle", zero)
    total = (
        config.reconstruction_weight * reconstruction
        + config.rollout_pixel_weight * rollout_pixel
        + config.rollout_latent_weight * rollout_latent
        + config.innovation_weight * innovation
        + config.jacobian_bridge_weight * bridge
        + config.oddness_weight * oddness
        + config.manifold_cycle_weight * manifold
        + config.whitening_weight * whitening
        + config.port_frame_weight * (frame + port_rank)
        + config.port_holonomy_weight * holonomy
    )
    metrics = {
        "total": total,
        "reconstruction": reconstruction,
        "rolloutPixel": rollout_pixel,
        "rolloutLatent": rollout_latent,
        "innovation": innovation,
        "effortMean": effort_terms["mean"],
        "effortVariance": effort_terms["variance"],
        "effortDecorrelation": effort_terms["decorrelation"],
        "effortTemporal": effort_terms["temporal"],
        "stateEffortIndependence": independence,
        "stateEffortLinearIndependence": linear_independence,
        "stateEffortSecondMomentIndependence": second_independence,
        "whitening": whitening,
        "portFrameTransport": frame,
        "portFrameHolonomy": holonomy,
        "portRankOrientation": port_rank,
        "jacobianBridge": bridge,
        "writeOddness": oddness,
        "manifoldCycle": manifold,
    }
    return total, metrics


def freeze_independent_bundle(bundle: IndependentUnstructuredBundle) -> None:
    for module in (
        bundle.model,
        bundle.write_field,
        bundle.lens,
        bundle.probes,
        bundle.response_frame,
    ):
        module.eval().requires_grad_(False)
    bundle.model.encoder.assert_backbone_frozen()


__all__ = [
    "HomologousInitialization",
    "IndependentUnstructuredArchitecture",
    "IndependentUnstructuredBundle",
    "IndependentUnstructuredWorldModel",
    "build_independent_unstructured_bundle",
    "capture_homologous_initialization",
    "freeze_independent_bundle",
    "independent_evaluation_modules",
    "independent_named_parameters",
    "independent_tangent_lens_terms",
    "independent_video_objective",
    "matched_dynamics_hidden_size",
    "unstructured_dynamics_parameter_count",
]
